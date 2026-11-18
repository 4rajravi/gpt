"""Save and restore a trained model together with its config and vocabulary."""

import os
from dataclasses import asdict

import torch

from .config import GPTConfig
from .data import CharTokenizer
from .models import LanguageModel


class Checkpoint:
    """Everything needed to rebuild a model later: weights, hyperparameters and characters."""

    @staticmethod
    def save(path, model, kind, cfg, tokenizer):
        """Write a checkpoint file.

        Args:
            path (str): Destination, e.g. "checkpoints/gpt.pt"; parent folders are created.
            model (LanguageModel): Trained model whose state_dict is saved.
            kind (str): Model type ("gpt" or "bigram"), needed to rebuild the right class.
            cfg (GPTConfig): Hyperparameters the model was built with.
            tokenizer (CharTokenizer): Vocabulary, so ids decode to the same characters.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "kind": kind,
            "config": asdict(cfg),
            "vocab": tokenizer.chars,
            "state_dict": model.state_dict(),
        }, path)

    @staticmethod
    def load(path, tracer, device="auto"):
        """Rebuild a model from a checkpoint file.

        Args:
            path (str): Checkpoint written by `save`.
            tracer (Tracer): Tracer to attach to the rebuilt model.
            device (str): Where to put the model ("auto" picks the fastest available).

        Returns:
            tuple[LanguageModel, CharTokenizer, GPTConfig]: (model with trained weights, tokenizer, config).
        """
        ckpt = torch.load(path, map_location="cpu")
        cfg = GPTConfig(**{**ckpt["config"], "device": device})
        tokenizer = CharTokenizer(ckpt["vocab"])
        model = LanguageModel.create(ckpt["kind"], cfg, tracer)
        model.load_state_dict(ckpt["state_dict"])
        return model.to(cfg.device), tokenizer, cfg
