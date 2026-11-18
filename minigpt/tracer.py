"""Prints every intermediate calculation so you can watch the model think."""

from contextlib import contextmanager

import torch


class Tracer:
    """Pretty-prints tensors, scalars and notes while the model runs.

    All output methods return immediately when the tracer is disabled, so the
    trace calls sprinkled through the model cost almost nothing during training.
    `focus(layer, head)` hands back a muted tracer for layers/heads beyond the
    ones you asked to see in detail, which keeps bigger models readable.
    """

    def __init__(self, enabled=False, max_rows=4, max_cols=8, detail_layers=1, detail_heads=1):
        """
        Args:
            enabled (bool): Start with printing switched on.
            max_rows (int): Matrix rows printed per tensor.
            max_cols (int): Matrix columns printed per tensor.
            detail_layers (int): Transformer blocks traced in full (the rest get a one-line summary).
            detail_heads (int): Attention heads per block traced in full.
        """
        self.enabled = enabled
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.detail_layers = detail_layers
        self.detail_heads = detail_heads
        self._depth = 0
        self._muted = None

    # -- control ---------------------------------------------------------------

    @contextmanager
    def active(self, on=True):
        """Temporarily switch tracing on or off inside a `with` block.

        Args:
            on (bool): Whether tracing is enabled inside the block.

        Yields:
            Tracer: This tracer.
        """
        previous, self.enabled = self.enabled, on
        try:
            yield self
        finally:
            self.enabled = previous

    def focus(self, layer, head=0):
        """Choose the tracer a given layer/head should log to.

        Args:
            layer (int): Transformer block index.
            head (int): Attention head index within the block.

        Returns:
            Tracer: This tracer if the layer/head is within the detail limits, else a muted one.
        """
        if layer < self.detail_layers and head < self.detail_heads:
            return self
        if self._muted is None:
            self._muted = Tracer(enabled=False)
        return self._muted

    @contextmanager
    def scope(self, title):
        """Print a section header and indent everything logged inside it.

        Args:
            title (str): Section name, e.g. "Block 0".
        """
        self._emit(f"┌─ {title}")
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    # -- output ----------------------------------------------------------------

    def banner(self, title):
        """Print a full-width heading between ═ rules.

        Args:
            title (str): Heading text.
        """
        if self.enabled:
            line = "═" * 80
            print(f"\n{line}\n {title}\n{line}")

    def note(self, message):
        """Print a line of free text at the current indentation.

        Args:
            message (str): Text to print (may contain newlines).
        """
        self._emit(message)

    def value(self, name, value, formula=""):
        """Print a single named number.

        Args:
            name (str): Label, e.g. "loss".
            value (float | int | str | Tensor): The value; 1-element tensors are converted with .item().
            formula (str): Optional explanation printed after an arrow.
        """
        if not self.enabled:
            return
        if isinstance(value, torch.Tensor):
            value = value.item()
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        self._emit(f"▸ {name} = {shown}" + (f"   ← {formula}" if formula else ""))

    def tensor(self, name, t, formula="", show=True, rows=None, cols=None):
        """Print a tensor's shape, statistics and a slice of its values.

        Args:
            name (str): Label, usually the expression that produced the tensor.
            t (Tensor): Tensor to show (any device; it is copied to the CPU).
            formula (str): Optional explanation printed after an arrow.
            show (bool): Print the values; False prints only shape and statistics.
            rows (int, optional): Override max_rows for this tensor.
            cols (int, optional): Override max_cols for this tensor.
        """
        if not self.enabled:
            return
        t = t.detach().cpu()
        self._emit(f"▸ {name}  shape={tuple(t.shape)}" + (f"   ← {formula}" if formula else ""))
        if t.numel() == 0:
            return
        self._emit("    " + self._stats(t))
        if show:
            for line in self._preview(t, rows or self.max_rows, cols or self.max_cols):
                self._emit("    " + line)

    # -- formatting helpers ----------------------------------------------------

    def _emit(self, message=""):
        """Print text with the current indentation prefix, if enabled.

        Args:
            message (str): Text to print; each line gets the prefix.
        """
        if not self.enabled:
            return
        prefix = "│ " * self._depth
        for line in str(message).split("\n"):
            print(prefix + line)

    @staticmethod
    def _stats(t):
        """Summarise a tensor in one line.

        Args:
            t (Tensor): CPU tensor.

        Returns:
            str: min/max for integer tensors; mean/std/min/max (ignoring -inf) for float tensors.
        """
        if not t.is_floating_point():
            return f"min={t.min().item()}  max={t.max().item()}"
        finite = t[torch.isfinite(t)]
        if finite.numel() == 0:
            return "all values are infinite"
        stats = (f"mean={finite.mean().item():+.4f}  std={finite.std().item():.4f}  "
                 f"min={finite.min().item():+.4f}  max={finite.max().item():+.4f}")
        masked = t.numel() - finite.numel()
        return stats + (f"  ({masked} entries are -inf)" if masked else "")

    def _preview(self, t, rows, cols):
        """Format the top-left corner of a tensor as aligned text rows.

        Tensors with more than 2 dimensions are shown at index 0 of the leading dimensions.

        Args:
            t (Tensor): CPU tensor.
            rows (int): Maximum rows to show.
            cols (int): Maximum columns to show.

        Returns:
            list[str]: Lines of text, including a "(showing [...])" note when the view is cut.
        """
        lead = []
        while t.dim() > 2:
            lead.append("0")
            t = t[0]
        as_int = not t.is_floating_point()
        if t.dim() == 0:
            return [self._fmt(t.item(), as_int)]
        matrix = t.unsqueeze(0) if t.dim() == 1 else t
        if t.dim() == 1:
            rows, cols = 1, cols * 2 if as_int else cols + 4
        r, c = matrix.shape
        cells = [[self._fmt(v, as_int) for v in row] for row in matrix[:rows, :cols].tolist()]
        width = max(len(s) for row in cells for s in row)

        lines = []
        if lead or r > rows or c > cols:
            dims = lead + ([f":{min(r, rows)}"] if t.dim() == 2 else []) + [f":{min(c, cols)}"]
            lines.append(f"(showing [{', '.join(dims)}])")
        for row in cells:
            lines.append("[" + " ".join(s.rjust(width) for s in row) + (" …" if c > cols else "") + "]")
        if r > rows:
            lines.append(f" … {r - rows} more rows")
        return lines

    @staticmethod
    def _fmt(v, as_int):
        """Format one number.

        Args:
            v (float | int): The value.
            as_int (bool): Format as an integer (token ids, masks).

        Returns:
            str: e.g. "42", "+0.1234" or "-inf".
        """
        if as_int:
            return str(int(v))
        if v == float("-inf"):
            return "-inf"
        if v == float("inf"):
            return "inf"
        return f"{v:+.4f}"
