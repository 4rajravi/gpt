"""Hyperparameters, presets and device selection."""

from dataclasses import dataclass

import torch

PRESETS = {
    # Small enough to read every number in the trace. Runs on CPU in seconds.
    "tiny": dict(batch_size=4, block_size=8, n_embd=16, n_head=2, n_layer=2, dropout=0.0,
                 max_iters=2000, learning_rate=1e-3, eval_interval=200, eval_iters=50, device="cpu"),
    # A laptop-friendly GPT: trains in a few minutes and already looks Shakespeare-ish.
    "small": dict(batch_size=32, block_size=64, n_embd=128, n_head=4, n_layer=4, dropout=0.1,
                  max_iters=3000, learning_rate=1e-3, eval_interval=250, eval_iters=100),
    # Exactly the hyperparameters from gpt.py in the lecture (~10.8M parameters).
    "lecture": dict(batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2,
                    max_iters=5000, learning_rate=3e-4, eval_interval=500, eval_iters=200),
}


@dataclass
class GPTConfig:
    """All hyperparameters of a model and its training run.

    Attributes:
        batch_size (int): B — independent sequences processed in parallel.
        block_size (int): T — maximum context length the model can see.
        n_embd (int): C — size of every token's embedding vector.
        n_head (int): Attention heads per block (head_size = n_embd // n_head).
        n_layer (int): Number of transformer blocks.
        dropout (float): Probability of zeroing activations during training.
        max_iters (int): Training iterations.
        learning_rate (float): AdamW learning rate.
        eval_interval (int): Evaluate train/val loss every N iterations.
        eval_iters (int): Batches averaged per evaluation.
        device (str): "cpu", "cuda", "mps" or "auto" (picks the fastest available).
        seed (int): Random seed for reproducibility.
        vocab_size (int): Number of distinct tokens; filled in from the tokenizer.
    """

    batch_size: int = 32
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    max_iters: int = 3000
    learning_rate: float = 1e-3
    eval_interval: int = 250
    eval_iters: int = 100
    device: str = "auto"
    seed: int = 1337
    vocab_size: int = 0

    def __post_init__(self):
        """Validate the head split and resolve device="auto".

        Raises:
            ValueError: If n_embd is not divisible by n_head.
        """
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.device == "auto":
            self.device = self.pick_device()

    @property
    def head_size(self):
        """int: Dimension of each attention head (n_embd // n_head)."""
        return self.n_embd // self.n_head

    @classmethod
    def from_preset(cls, name, **overrides):
        """Start from a named preset and replace any values that were explicitly given.

        Args:
            name (str): One of PRESETS ("tiny", "small", "lecture").
            **overrides: Field values to change; None means "keep the preset's value".

        Returns:
            GPTConfig: The resulting config.
        """
        values = {**PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}}
        return cls(**values)

    @staticmethod
    def pick_device():
        """Choose the fastest available device.

        Returns:
            str: "cuda" if an NVIDIA GPU is available, else "mps" on Apple silicon, else "cpu".
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
