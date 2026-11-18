"""Records every number a training run produces so it can be saved, reloaded and plotted."""

import json
import os
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class TrainingHistory:
    """Per-iteration and per-evaluation metrics of one training run.

    Attributes:
        model (str): Model type, e.g. "gpt" or "bigram".
        preset (str): Config preset name, e.g. "small".
        config (dict): The full GPTConfig used for the run.
        num_params (int): Number of trainable parameters.
        total_seconds (float): Wall-clock duration of the whole run.
        steps (dict): Lists keyed "iter", "loss", "grad_norm", "ms" — one entry per iteration.
        evals (dict): Lists keyed "iter", "train", "val" — one entry per evaluation.
    """

    model: str = ""
    preset: str = ""
    config: dict = field(default_factory=dict)
    num_params: int = 0
    total_seconds: float = 0.0
    steps: dict = field(default_factory=lambda: {"iter": [], "loss": [], "grad_norm": [], "ms": []})
    evals: dict = field(default_factory=lambda: {"iter": [], "train": [], "val": []})

    @property
    def label(self):
        """str: Short human-readable run name, e.g. "gpt-small"."""
        return f"{self.model}-{self.preset}"

    def record_step(self, it, loss, grad_norm, ms):
        """Store the metrics of one training iteration.

        Args:
            it (int): Iteration number.
            loss (float): Training loss on this iteration's batch.
            grad_norm (float): Total gradient norm before the optimizer step.
            ms (float): Time the iteration took, in milliseconds.
        """
        for key, value in (("iter", it), ("loss", loss), ("grad_norm", grad_norm), ("ms", ms)):
            self.steps[key].append(value)

    def record_eval(self, it, train_loss, val_loss):
        """Store one evaluation of the averaged train and validation loss.

        Args:
            it (int): Iteration at which the evaluation ran.
            train_loss (float): Mean loss over eval_iters training batches.
            val_loss (float): Mean loss over eval_iters validation batches.
        """
        self.evals["iter"].append(it)
        self.evals["train"].append(train_loss)
        self.evals["val"].append(val_loss)

    def save(self, path):
        """Write the history to a JSON file, creating parent folders as needed.

        Args:
            path (str): Destination file, e.g. "results/gpt-small/history.json".
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path):
        """Read a history previously written by `save`.

        Args:
            path (str): JSON file to read.

        Returns:
            TrainingHistory: The restored history.
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


class ConsoleLog:
    """Context manager that copies everything printed to stdout into a log file as well.

    Example:
        with ConsoleLog("results/gpt-small/train.log"):
            print("goes to the screen and to the file")
    """

    def __init__(self, path):
        """
        Args:
            path (str): Log file to (over)write.
        """
        self.path = path
        self._file = None
        self._stdout = None

    def __enter__(self):
        """Start copying stdout to the file.

        Returns:
            ConsoleLog: This object.
        """
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        """Restore the original stdout and close the file.

        Returns:
            bool: False, so exceptions raised inside the block still propagate.
        """
        sys.stdout = self._stdout
        self._file.close()
        return False

    def write(self, text):
        """Write text to both the terminal and the log file.

        Args:
            text (str): Text to write.

        Returns:
            int: Number of characters written.
        """
        self._stdout.write(text)
        self._file.write(text)
        return len(text)

    def flush(self):
        """Flush both the terminal and the log file."""
        self._stdout.flush()
        self._file.flush()
