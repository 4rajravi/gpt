"""Prompt → text using a trained model."""

import torch


class TextGenerator:
    """Wraps a model and its tokenizer so you can work with strings instead of ids."""

    def __init__(self, model, tokenizer, device):
        """
        Args:
            model (LanguageModel): Trained model.
            tokenizer (CharTokenizer): The tokenizer the model was trained with.
            device (str): Device the model lives on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt="", max_new_tokens=500, trace_tokens=0):
        """Continue a prompt with sampled characters.

        Args:
            prompt (str): Starting text. Empty means start from a single newline (token 0), as in the lecture.
            max_new_tokens (int): Number of characters to generate.
            trace_tokens (int): Print the probabilities behind the first N sampled characters.

        Returns:
            str: The prompt followed by the generated text.
        """
        ids = self.tokenizer.encode(prompt) if prompt else [0]
        context = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(context, max_new_tokens,
                                  decode=self.tokenizer.decode, trace_tokens=trace_tokens)
        return self.tokenizer.decode(out[0].tolist())
