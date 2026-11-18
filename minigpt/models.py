"""Language models: a shared base class, the bigram baseline and the GPT."""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import Block


class LanguageModel(nn.Module):
    """Shared behaviour: forward → (logits, loss), cross-entropy, sampling.

    Subclasses only implement `compute_logits`.
    """

    def __init__(self, cfg, tracer):
        """
        Args:
            cfg (GPTConfig): Model hyperparameters (vocab_size must be set).
            tracer (Tracer): Prints the calculations when enabled.
        """
        super().__init__()
        self.cfg = cfg
        self.tracer = tracer

    @staticmethod
    def create(kind, cfg, tracer):
        """Build a model by name.

        Args:
            kind (str): "gpt" or "bigram".
            cfg (GPTConfig): Model hyperparameters.
            tracer (Tracer): Passed to every module.

        Returns:
            LanguageModel: A freshly initialised model on the CPU.
        """
        return MODEL_TYPES[kind](cfg, tracer)

    def num_parameters(self):
        """Count trainable numbers in the model.

        Returns:
            int: Total number of parameter elements.
        """
        return sum(p.numel() for p in self.parameters())

    def compute_logits(self, idx):
        """Score every possible next token at every position.

        Args:
            idx (Tensor): (B, T) input token ids.

        Returns:
            Tensor: (B, T, vocab_size) unnormalised scores (logits).
        """
        raise NotImplementedError

    def forward(self, idx, targets=None):
        """Compute logits and, if targets are given, the loss.

        Args:
            idx (Tensor): (B, T) input token ids.
            targets (Tensor, optional): (B, T) correct next-token ids.

        Returns:
            tuple[Tensor, Tensor | None]: (logits of shape (B, T, vocab_size), scalar loss or None).
        """
        logits = self.compute_logits(idx)
        loss = None if targets is None else self.compute_loss(logits, targets)
        return logits, loss

    def compute_loss(self, logits, targets):
        """Cross-entropy between the predicted distributions and the true next tokens.

        Args:
            logits (Tensor): (B, T, V) scores.
            targets (Tensor): (B, T) correct next-token ids.

        Returns:
            Tensor: Scalar mean of -log p(target) over all B*T positions.
        """
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)
        # Same maths as F.cross_entropy, written out step by step. (F.cross_entropy also
        # crashes with a bus error on Apple MPS in torch 2.0, which this avoids.)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = nll.mean()

        tr = self.tracer
        if tr.enabled:
            with tr.scope("Cross-entropy loss"):
                tr.tensor("logits.view(B*T, V)", logits, f"flatten ({B},{T},{V}) → ({B * T},{V})")
                tr.tensor("targets.view(B*T)", targets, "the correct next token at every position")
                tr.tensor("probs = softmax(logits)", log_probs.exp(), "exp(logits) / Σ exp(logits) — rows sum to 1")
                tr.tensor("p[target]", (-nll).exp(), "probability assigned to the correct token")
                tr.tensor("-log p[target]", nll, "loss at each position")
                tr.value("loss", loss, "mean of -log p[target] over all B*T positions")
                tr.value("random-guess baseline", math.log(V), f"-ln(1/{V}) — what an untrained model should score")
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, decode=None, trace_tokens=0):
        """Extend a sequence one sampled token at a time (autoregressive generation).

        Runs in eval mode (dropout off) and restores the previous mode afterwards.

        Args:
            idx (Tensor): (B, T) starting token ids.
            max_new_tokens (int): How many tokens to append.
            decode (Callable[[list[int]], str], optional): Turns ids into text for the trace.
            trace_tokens (int): Print the probability distribution for the first N sampled tokens.

        Returns:
            Tensor: (B, T + max_new_tokens) the input followed by the sampled tokens.
        """
        was_training = self.training
        self.eval()
        tr = self.tracer
        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            with tr.active(False):
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if step < trace_tokens:
                with tr.active(True):
                    self._trace_sample(step, idx_cond, logits, probs, idx_next, decode)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train(was_training)
        return idx

    @torch.no_grad()
    def next_token_probs(self, idx):
        """Probability of every possible next token after the given context.

        Args:
            idx (Tensor): (1, T) context token ids (cropped to the last block_size tokens).

        Returns:
            Tensor: (vocab_size,) probabilities on the CPU, summing to 1.
        """
        was_training = self.training
        self.eval()
        with self.tracer.active(False):
            logits, _ = self(idx[:, -self.cfg.block_size:])
        self.train(was_training)
        return F.softmax(logits[0, -1], dim=-1).cpu()

    def _trace_sample(self, step, idx_cond, logits, probs, idx_next, decode):
        """Print one generation step: context, logits, probabilities, top-5 and the sampled token.

        Args:
            step (int): Index of the token being generated.
            idx_cond (Tensor): (1, T) context fed to the model.
            logits (Tensor): (1, V) scores of the last position.
            probs (Tensor): (1, V) softmax of the logits.
            idx_next (Tensor): (1, 1) the sampled token id.
            decode (Callable[[list[int]], str] | None): Turns ids into text, if available.
        """
        tr = self.tracer
        label = (lambda i: repr(decode([i]))) if decode else str
        with tr.scope(f"Generate token #{step + 1}"):
            context = idx_cond[0].tolist()
            tr.note(f"context ({len(context)} token{'s' * (len(context) != 1)}): "
                    + (repr(decode(context)) if decode else str(context)))
            tr.tensor("logits[:, -1, :]", logits, "only the last position's scores predict the next token")
            tr.tensor("probs = softmax(logits)", probs)
            top_p, top_i = probs[0].topk(min(5, probs.shape[-1]))
            tr.note("top 5 candidates:")
            for p, i in zip(top_p.tolist(), top_i.tolist()):
                tr.note(f"  {label(i):>6}  p={p:.4f}  {'█' * max(1, round(p * 40))}")
            tr.note(f"torch.multinomial(probs) sampled → {label(idx_next[0].item())}")


class BigramLanguageModel(LanguageModel):
    """bigram.py: each token directly looks up the scores for the next token."""

    def __init__(self, cfg, tracer):
        """
        Args:
            cfg (GPTConfig): Only vocab_size is used.
            tracer (Tracer): Prints the calculations when enabled.
        """
        super().__init__(cfg, tracer)
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.vocab_size)

    def compute_logits(self, idx):
        """Look up row idx of a (vocab × vocab) table: the scores for the next token.

        Args:
            idx (Tensor): (B, T) input token ids.

        Returns:
            Tensor: (B, T, vocab_size) logits; each position only depends on its own token.
        """
        tr = self.tracer
        with tr.scope("Bigram forward pass"):
            tr.tensor("idx", idx, "(B,T) input token ids")
            logits = self.token_embedding_table(idx)
            tr.tensor("logits = token_embedding_table[idx]", logits,
                      "row idx of the table = scores for the next token   (B,T,vocab)")
        return logits


class GPTLanguageModel(LanguageModel):
    """gpt.py: token + position embeddings → N transformer blocks → LayerNorm → linear head."""

    def __init__(self, cfg, tracer):
        """
        Args:
            cfg (GPTConfig): Model hyperparameters.
            tracer (Tracer): Passed to every block, head and layer.
        """
        super().__init__(cfg, tracer)
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i, tracer) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Initialise Linear/Embedding weights from N(0, 0.02) and biases to zero (as in GPT-2).

        Args:
            module (nn.Module): Called once for every sub-module by `self.apply`.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def compute_logits(self, idx):
        """Run the full transformer.

        Args:
            idx (Tensor): (B, T) input token ids, T <= block_size.

        Returns:
            Tensor: (B, T, vocab_size) logits for the next token at every position.
        """
        B, T = idx.shape
        tr = self.tracer
        with tr.scope("GPT forward pass"):
            tr.tensor("idx", idx, "(B,T) input token ids")
            tok_emb = self.token_embedding_table(idx)
            tr.tensor("tok_emb = token_embedding_table[idx]", tok_emb, "what each token is   (B,T,C)")
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
            tr.tensor("pos_emb = position_embedding_table[0..T-1]", pos_emb, "where each token is   (T,C)")
            x = tok_emb + pos_emb
            tr.tensor("x = tok_emb + pos_emb", x, "(T,C) is broadcast over the batch   (B,T,C)")

            for block in self.blocks:
                x = block(x)
                if tr.focus(block.layer_idx) is not tr:
                    tr.tensor(f"x after Block {block.layer_idx}", x,
                              "(details hidden; raise --detail-layers to print them)", show=False)

            x = self.ln_f(x)
            tr.tensor("x = ln_f(x)", x, "final LayerNorm")
            logits = self.lm_head(x)
            tr.tensor("logits = lm_head(x)", logits, "(B,T,C) @ (C,vocab) + bias → (B,T,vocab)")
        return logits

    @torch.no_grad()
    def attention_maps(self, idx):
        """Collect every head's attention weights for one input sequence.

        Args:
            idx (Tensor): (1, T) token ids, T <= block_size.

        Returns:
            Tensor: (n_layer, n_head, T, T) on the CPU; row t shows how much token t attends to each earlier token.
        """
        heads = [head for block in self.blocks for head in block.sa.heads]
        was_training = self.training
        self.eval()
        for head in heads:
            head.keep_attention = True
        try:
            with self.tracer.active(False):
                self(idx)
        finally:
            for head in heads:
                head.keep_attention = False
            self.train(was_training)
        return torch.stack([torch.stack([h.attention[0] for h in b.sa.heads]) for b in self.blocks]).cpu()


MODEL_TYPES = {"bigram": BigramLanguageModel, "gpt": GPTLanguageModel}
