"""Runs the whole pipeline once with every intermediate calculation printed."""

from dataclasses import asdict

import torch

from .data import TextDataset
from .generator import TextGenerator
from .models import GPTLanguageModel, LanguageModel
from .trainer import Trainer


class Walkthrough:
    """Hyperparameters → data → model → one training step → generation, all traced."""

    def __init__(self, text, tokenizer, cfg, kind, tracer):
        """
        Args:
            text (str): Full training text.
            tokenizer (CharTokenizer): Tokenizer built from `text`.
            cfg (GPTConfig): Hyperparameters (the "tiny" preset keeps the output readable).
            kind (str): "gpt" or "bigram".
            tracer (Tracer): Configured with how many rows/cols/layers/heads to print.
        """
        self.text = text
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tracer = tracer
        self.dataset = TextDataset(text, tokenizer, cfg, tracer)
        self.model = LanguageModel.create(kind, cfg, tracer).to(cfg.device)
        self.trainer = Trainer(self.model, self.dataset, cfg, tracer)

    def run(self, generate_tokens=3, sample_length=100):
        """Print every stage in order.

        Args:
            generate_tokens (int): Generation steps whose probabilities are traced.
            sample_length (int): Characters generated at the end.
        """
        with self.tracer.active(True):
            self.show_config()
            self.show_data()
            self.show_model()
            self.show_training_step()
            self.show_generation(generate_tokens, sample_length)

    def show_config(self):
        """STEP 0: print every hyperparameter (and head_size for the GPT)."""
        tr = self.tracer
        tr.banner("STEP 0 │ Hyperparameters")
        for key, value in asdict(self.cfg).items():
            tr.note(f"{key:<14} = {value}")
        if isinstance(self.model, GPTLanguageModel):
            tr.note(f"{'head_size':<14} = n_embd // n_head = {self.cfg.n_embd} // {self.cfg.n_head} "
                    f"= {self.cfg.head_size}")

    def show_data(self):
        """STEP 1: print the text, the vocabulary, an encode/decode example and the train/val split."""
        tr, tok = self.tracer, self.tokenizer
        tr.banner("STEP 1 │ Text → characters → integers")
        tr.value("characters in the dataset", f"{len(self.text):,}")
        tr.note("first 200 characters:\n" + "-" * 40 + "\n" + self.text[:200] + "\n" + "-" * 40)
        tr.note(f"vocabulary: {tok.vocab_size} unique characters {''.join(tok.chars)!r}")
        tr.note("stoi (first 15): " + ", ".join(f"{c!r}→{i}" for c, i in list(tok.stoi.items())[:15]))
        sample = "hii there"
        ids = tok.encode(sample)
        tr.note(f"encode({sample!r}) = {ids}")
        tr.note(f"decode({ids}) = {tok.decode(ids)!r}")
        tr.tensor("data = torch.tensor(encode(text))", self.dataset.data, rows=1, cols=16)
        train, val = self.dataset.splits["train"], self.dataset.splits["val"]
        tr.note(f"train = data[:90%] → {len(train):,} tokens   |   val = data[90%:] → {len(val):,} tokens")

    def show_model(self):
        """STEP 2: print every parameter tensor with its shape and size."""
        tr = self.tracer
        tr.banner(f"STEP 2 │ Build {type(self.model).__name__} (weights start random)")
        for name, p in self.model.named_parameters():
            tr.note(f"{name:<44} {str(tuple(p.shape)):<12} {p.numel():>9,}")
        tr.note(f"{'total':<57} {self.model.num_parameters():>9,} parameters")

    def show_training_step(self):
        """STEP 3: trace batch → forward → loss → backward → update, then re-check the loss."""
        tr = self.tracer
        tr.banner("STEP 3 │ One training step: batch → forward → loss → backward → update")
        batch = self.dataset.get_batch("train")
        loss_before, _ = self.trainer.train_step(batch)

        self.model.eval()
        with tr.active(False), torch.no_grad():
            _, loss_after = self.model(*batch)
        self.model.train()
        tr.banner("RESULT │ Did the update help?")
        tr.note(f"loss on this same batch: {loss_before:.4f} before the update → {loss_after.item():.4f} after")

    def show_generation(self, generate_tokens, sample_length):
        """STEP 4: trace a few sampling steps and print a short generated text.

        Args:
            generate_tokens (int): Sampling steps to trace.
            sample_length (int): Characters to generate.
        """
        tr = self.tracer
        tr.banner("STEP 4 │ Generate text (the model has trained on one batch, so expect gibberish)")
        generator = TextGenerator(self.model, self.tokenizer, self.cfg.device)
        text = generator.generate(max_new_tokens=sample_length, trace_tokens=generate_tokens)
        tr.note(f"{sample_length} generated characters:\n" + "-" * 40 + text + "\n" + "-" * 40)
        tr.note("run `python main.py train` to train properly, then `python main.py generate`.")
