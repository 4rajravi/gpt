"""Raw text, character-level tokenizer and random batch sampling."""

import os
import urllib.request

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class Corpus:
    """Loads the training text, downloading Tiny Shakespeare the first time."""

    def __init__(self, path="data/input.txt", url=TINY_SHAKESPEARE_URL):
        """
        Args:
            path (str): Local text file; downloaded from `url` if it does not exist.
            url (str): Where to download the text from.
        """
        self.path = path
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            print(f"Downloading {url} -> {path}")
            urllib.request.urlretrieve(url, path)
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()


class CharTokenizer:
    """Maps every unique character to an integer id and back."""

    def __init__(self, chars):
        """
        Args:
            chars (Iterable[str]): The vocabulary; a character's position is its id.
        """
        self.chars = list(chars)
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        """Build the vocabulary from every distinct character in a text, sorted.

        Args:
            text (str): Training text.

        Returns:
            CharTokenizer: Tokenizer covering all characters of `text`.
        """
        return cls(sorted(set(text)))

    def encode(self, s):
        """Convert a string to token ids.

        Args:
            s (str): Text to encode.

        Returns:
            list[int]: One id per character.

        Raises:
            ValueError: If `s` contains characters that are not in the vocabulary.
        """
        unknown = sorted(set(s) - self.stoi.keys())
        if unknown:
            raise ValueError(f"characters not in the vocabulary: {unknown}")
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """Convert token ids back to a string.

        Args:
            ids (Iterable[int]): Token ids.

        Returns:
            str: The decoded text.
        """
        return "".join(self.itos[i] for i in ids)


class TextDataset:
    """Encodes the whole text, splits it 90/10 into train/val and samples batches."""

    def __init__(self, text, tokenizer, cfg, tracer, train_frac=0.9):
        """
        Args:
            text (str): Full training text.
            tokenizer (CharTokenizer): Used to encode the text.
            cfg (GPTConfig): Supplies block_size, batch_size and device.
            tracer (Tracer): Prints each sampled batch when enabled.
            train_frac (float): Fraction of the text used for training; the rest is validation.
        """
        self.tokenizer = tokenizer
        self.block_size = cfg.block_size
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.tracer = tracer
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(train_frac * len(self.data))
        self.splits = {"train": self.data[:n], "val": self.data[n:]}

    def get_batch(self, split):
        """Sample batch_size random windows of block_size tokens, plus their next-token targets.

        Args:
            split (str): "train" or "val".

        Returns:
            tuple[Tensor, Tensor]: (x, y), both (B, T) long tensors on cfg.device;
            y[b, t] is the token that follows x[b, t] in the text.
        """
        data = self.splits[split]
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        if self.tracer.enabled:
            self._trace(split, data, ix, x, y)
        return x.to(self.device), y.to(self.device)

    def _trace(self, split, data, ix, x, y):
        """Print the sampled offsets, x, y and the context → target pairs of the first row.

        Args:
            split (str): "train" or "val".
            data (Tensor): The split's token ids.
            ix (Tensor): (B,) start offsets.
            x (Tensor): (B, T) inputs.
            y (Tensor): (B, T) targets.
        """
        tr = self.tracer
        with tr.scope(f"get_batch('{split}')"):
            tr.tensor("ix", ix, f"{self.batch_size} random start offsets in [0, {len(data) - self.block_size})")
            tr.tensor("x", x, "x[b] = data[ix[b] : ix[b]+block_size]   (B,T)")
            tr.tensor("y", y, "y[b] = data[ix[b]+1 : ix[b]+block_size+1]  ← x shifted left by one")
            tr.note(f"x[0] as text: {self.tokenizer.decode(x[0].tolist())!r}")
            tr.note("every position is its own training example (inputs → next-token target):")
            for t in range(min(self.block_size, 8)):
                context = x[0, :t + 1].tolist()
                target = y[0, t].item()
                tr.note(f"  {str(context):<34} {self.tokenizer.decode(context)!r:<12} → "
                        f"{target:>2} {self.tokenizer.decode([target])!r}")
