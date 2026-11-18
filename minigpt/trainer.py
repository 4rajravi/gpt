"""Optimiser, training loop and loss estimation."""

import re
import time

import torch

from .history import TrainingHistory


class Trainer:
    """Runs AdamW training, logs every iteration and records a TrainingHistory."""

    def __init__(self, model, dataset, cfg, tracer, history=None):
        """
        Args:
            model (LanguageModel): The model to train (already moved to cfg.device).
            dataset (TextDataset): Source of train/val batches.
            cfg (GPTConfig): Learning rate, number of iterations and evaluation settings.
            tracer (Tracer): Prints calculations on traced iterations.
            history (TrainingHistory, optional): Where metrics are recorded; a new one if omitted.
        """
        self.model = model
        self.dataset = dataset
        self.cfg = cfg
        self.tracer = tracer
        self.history = history if history is not None else TrainingHistory()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    @torch.no_grad()
    def estimate_loss(self):
        """Average the loss over eval_iters batches of each split (less noisy than one batch).

        Dropout is switched off (eval mode) during the estimate and tracing is suppressed.

        Returns:
            dict[str, float]: {"train": mean train loss, "val": mean validation loss}.
        """
        self.model.eval()
        out = {}
        with self.tracer.active(False):
            for split in ("train", "val"):
                losses = torch.zeros(self.cfg.eval_iters)
                for k in range(self.cfg.eval_iters):
                    x, y = self.dataset.get_batch(split)
                    _, loss = self.model(x, y)
                    losses[k] = loss.item()
                out[split] = losses.mean().item()
        self.model.train()
        return out

    def train_step(self, batch=None):
        """Run one iteration: batch → forward → loss → backward → optimizer step.

        If the tracer is enabled, the gradients and the weight update are printed too.

        Args:
            batch (tuple[Tensor, Tensor], optional): (x, y) to train on, each (B, T).
                A random training batch is sampled if omitted.

        Returns:
            tuple[float, float]: (loss before the update, total gradient norm).
        """
        x, y = batch if batch is not None else self.dataset.get_batch("train")
        _, loss = self.model(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._grad_norm()
        if self.tracer.enabled:
            self._trace_gradients(grad_norm)
            name, watched = self._watched_parameter()
            before = watched.detach().clone()
        self.optimizer.step()
        if self.tracer.enabled:
            self._trace_update(name, watched, before)
        return loss.item(), grad_norm

    def train(self, log_every=1, trace_every=0):
        """Train for cfg.max_iters iterations, printing and recording metrics.

        Args:
            log_every (int): Print the loss every N iterations (1 = every iteration).
            trace_every (int): Print a full calculation trace every N iterations (0 = never).

        Returns:
            TrainingHistory: The recorded per-step and per-evaluation metrics.
        """
        cfg = self.cfg
        self.history.num_params = self.model.num_parameters()
        run_start = time.perf_counter()
        for it in range(cfg.max_iters):
            if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
                losses = self.estimate_loss()
                self.history.record_eval(it, losses["train"], losses["val"])
                print(f"── step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            trace = trace_every > 0 and it % trace_every == 0
            if trace:
                print(f"\n========== full calculation trace for iteration {it} ==========")
            start = time.perf_counter()
            with self.tracer.active(trace):
                loss, grad_norm = self.train_step()
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.history.record_step(it, loss, grad_norm, elapsed_ms)

            if it % log_every == 0:
                print(f"iter {it:5d} | loss {loss:.4f} | grad norm {grad_norm:.4f} | {elapsed_ms:6.1f} ms")
            self.history.total_seconds = time.perf_counter() - run_start
        return self.history

    # -- helpers ---------------------------------------------------------------

    def _grad_norm(self):
        """Compute the global gradient norm.

        Returns:
            float: sqrt(sum of squared gradients over every parameter).
        """
        norms = [p.grad.detach().norm() for p in self.model.parameters() if p.grad is not None]
        return torch.stack(norms).norm().item()

    def _watched_parameter(self):
        """Pick the parameter whose before/after values get printed: the output layer's weights.

        Returns:
            tuple[str, nn.Parameter]: (parameter name, parameter) of the last 2-D weight matrix.
        """
        return next((n, p) for n, p in reversed(list(self.model.named_parameters())) if p.dim() == 2)

    def _is_detailed(self, param_name):
        """Check whether a parameter belongs to a layer/head the tracer shows in detail.

        Args:
            param_name (str): Name from named_parameters(), e.g. "blocks.1.sa.heads.0.key.weight".

        Returns:
            bool: True if its gradient should be printed individually.
        """
        m = re.match(r"blocks\.(\d+)\.(?:sa\.heads\.(\d+)\.)?", param_name)
        return m is None or self.tracer.focus(int(m[1]), int(m[2] or 0)) is self.tracer

    def _trace_gradients(self, grad_norm):
        """Print the gradient norm of every (detailed) parameter after loss.backward().

        Args:
            grad_norm (float): The global gradient norm, printed at the end.
        """
        tr = self.tracer
        with tr.scope("loss.backward(): the chain rule fills p.grad for every parameter"):
            hidden = 0
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                if not self._is_detailed(name):
                    hidden += 1
                    continue
                tr.note(f"{name:<44} {str(tuple(p.shape)):<12} |grad| = {p.grad.norm().item():.6f}")
            if hidden:
                tr.note(f"… plus {hidden} parameter tensors in hidden layers/heads")
            tr.value("total gradient norm", grad_norm, "sqrt(Σ |grad|²) over all parameters")

    def _trace_update(self, name, p, before):
        """Print one weight matrix before and after optimizer.step(), with its gradient and change.

        Args:
            name (str): Parameter name, used as the label.
            p (nn.Parameter): The parameter after the update.
            before (Tensor): Copy of the parameter taken before the update.
        """
        tr = self.tracer
        lr = self.optimizer.param_groups[0]["lr"]
        with tr.scope(f"optimizer.step(): AdamW moves every weight against its gradient (lr={lr})"):
            tr.note("m = β1·m + (1-β1)·g     v = β2·v + (1-β2)·g²     w ← w − lr·(m̂ / (√v̂ + ε) + λ·w)")
            tr.tensor(f"{name}  (before)", before)
            tr.tensor(f"{name}.grad", p.grad)
            tr.tensor(f"{name}  (after)", p)
            tr.tensor("change = after − before", p.detach() - before)
            if int(self.optimizer.state[p]["step"]) == 1:
                tr.note(f"first step: m̂/√v̂ = g/|g| = ±1, so each weight moves by ≈ {lr} "
                        "in the opposite direction of its gradient")
