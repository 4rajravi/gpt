"""Charts of training results and of what the trained model learned (requires matplotlib)."""

import math
import os

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from .models import GPTLanguageModel

COLORS = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
              grid="#e1e0d9", axis="#c3c2b7", soft="#b7d3f6",
              series=["#2a78d6", "#eb6834", "#1baf7a"],
              ramp=["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# A list of families lets matplotlib fall back per glyph (Helvetica has no ↵ or →; DejaVu does).
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


class TrainingPlots:
    """Draws the project's charts and saves them as PNG files."""

    PROMPTS = ("ROMEO:", "To be, or not to b", "My lord, I am a")

    def __init__(self, out_dir="docs/images"):
        """
        Args:
            out_dir (str): Folder the PNG files are written to (created if missing).
        """
        self.out_dir = out_dir
        self.c = COLORS
        self.cmap = LinearSegmentedColormap.from_list("attention", COLORS["ramp"])
        os.makedirs(out_dir, exist_ok=True)

    def make_all(self, histories, model, tokenizer, prompt):
        """Draw every chart that the available data allows.

        Args:
            histories (list[TrainingHistory]): Saved training runs.
            model (LanguageModel): Trained model (on the CPU) for the attention/probability/parameter charts.
            tokenizer (CharTokenizer): The model's tokenizer.
            prompt (str): Text whose attention patterns are drawn.

        Returns:
            list[str]: Paths of the saved PNG files.
        """
        paths = [self.loss_curves(h) for h in histories]
        paths += [self.grad_norm(h) for h in histories if h.model == "gpt"]
        if len(histories) > 1:
            paths.append(self.model_comparison(histories))
        if isinstance(model, GPTLanguageModel):
            paths += [self.attention_grid(model, tokenizer, prompt), self.sharpest_head(model, tokenizer, prompt)]
        paths += [self.next_char_probs(model, tokenizer, self.PROMPTS), self.parameter_breakdown(model)]
        return paths

    # -- training charts -------------------------------------------------------

    def loss_curves(self, history):
        """Per-iteration batch loss plus the averaged train and validation loss of one run.

        Args:
            history (TrainingHistory): The run to plot.

        Returns:
            str: Path of the saved PNG.
        """
        c = self.c
        fig, ax = self._figure()
        steps, evals = history.steps, history.evals
        ax.plot(steps["iter"], steps["loss"], color=c["soft"], linewidth=1, label="batch loss (every iteration)")
        for key, color, name in (("train", c["series"][0], "train"), ("val", c["series"][1], "val")):
            ax.plot(evals["iter"], evals[key], color=color, linewidth=2, label=f"{name} loss (avg of eval batches)")
            self._end_label(ax, evals["iter"][-1], evals[key][-1], f"{name} {evals[key][-1]:.3f}", color)
        self._baseline(ax, history)
        ax.set_xlabel("iteration")
        ax.set_ylabel("cross-entropy loss")
        ax.set_xlim(0, steps["iter"][-1] * 1.14)
        self._title(ax, f"{history.label}: val loss {evals['val'][0]:.2f} → {evals['val'][-1]:.2f}",
                    f"{history.num_params:,} parameters · {len(steps['iter']):,} iterations · "
                    f"{history.total_seconds / 60:.1f} min")
        self._legend(ax)
        return self._save(fig, f"loss_{history.label}")

    def model_comparison(self, histories):
        """Validation loss of several runs on one chart (e.g. bigram vs GPT).

        Args:
            histories (list[TrainingHistory]): Runs to compare (at most 3 series).

        Returns:
            str: Path of the saved PNG.
        """
        c = self.c
        fig, ax = self._figure()
        ordered = sorted(histories, key=lambda h: h.model != "gpt")
        for h, color in zip(ordered[:3], c["series"][::2] + c["series"][1:2]):
            ax.plot(h.evals["iter"], h.evals["val"], color=color, linewidth=2,
                    label=f"{h.label} ({h.num_params:,} params)")
            self._end_label(ax, h.evals["iter"][-1], h.evals["val"][-1], f"{h.model} {h.evals['val'][-1]:.3f}", color)
        self._baseline(ax, histories[0])
        ax.set_xlabel("iteration")
        ax.set_ylabel("validation loss")
        ax.set_xlim(0, max(h.evals["iter"][-1] for h in histories) * 1.14)
        self._title(ax, "Validation loss: the transformer vs the bigram baseline",
                    "same data, same batches; lower is better")
        self._legend(ax)
        return self._save(fig, "model_comparison")

    def grad_norm(self, history):
        """Total gradient norm at every iteration, with a 50-step moving average.

        Args:
            history (TrainingHistory): The run to plot.

        Returns:
            str: Path of the saved PNG.
        """
        c = self.c
        fig, ax = self._figure(h=3.8)
        it = np.array(history.steps["iter"])
        g = np.array(history.steps["grad_norm"])
        window = 50
        smooth = np.convolve(g, np.ones(window) / window, mode="valid")
        ax.plot(it, g, color=c["soft"], linewidth=1, label="every iteration")
        ax.plot(it[window - 1:], smooth, color=c["series"][0], linewidth=2, label=f"{window}-iteration average")
        ax.set_ylim(0, np.percentile(g, 99.5) * 1.1)
        ax.set_xlabel("iteration")
        ax.set_ylabel("gradient norm")
        peak = int(smooth.argmax())
        self._title(ax, f"{history.label}: total gradient norm",
                    f"{window}-iteration average peaks at {smooth[peak]:.2f} (iteration {it[peak + window - 1]}) "
                    f"and ends at {smooth[-1]:.2f} · y-axis cut at the 99.5th percentile")
        self._legend(ax, loc="lower right")
        return self._save(fig, f"grad_norm_{history.label}")

    # -- model charts ----------------------------------------------------------

    def attention_grid(self, model, tokenizer, prompt):
        """Heatmap of every head in every layer for one prompt.

        Args:
            model (GPTLanguageModel): Trained GPT on the CPU.
            tokenizer (CharTokenizer): The model's tokenizer.
            prompt (str): Text to run through the model (cropped to block_size).

        Returns:
            str: Path of the saved PNG.
        """
        c = self.c
        maps, labels = self._attention(model, tokenizer, prompt)
        L, H, T, _ = maps.shape
        fig, axes = plt.subplots(L, H, figsize=(2.2 * H + 1, 2.2 * L + 0.9), facecolor=c["surface"],
                                 squeeze=False, layout="constrained")
        for l in range(L):
            for h in range(H):
                ax = axes[l][h]
                image = ax.imshow(maps[l, h], cmap=self.cmap, vmin=0, vmax=1)
                self._heatmap_ticks(ax, labels, fontsize=6, show_x=l == L - 1, show_y=h == 0)
                ax.set_title(f"layer {l} · head {h}", fontsize=9, color=c["ink2"], pad=4)
        self._colorbar(fig, image, axes)
        fig.suptitle(f"Attention weights of all {L * H} heads for {prompt!r}\n"
                     "row = token doing the looking, column = earlier token it looks at",
                     x=0.02, ha="left", fontsize=12, color=c["ink"])
        return self._save(fig, "attention_all_heads")

    def sharpest_head(self, model, tokenizer, prompt):
        """Large heatmap of the head whose attention is most focused (lowest entropy), with weights labelled.

        Args:
            model (GPTLanguageModel): Trained GPT on the CPU.
            tokenizer (CharTokenizer): The model's tokenizer.
            prompt (str): Text to run through the model.

        Returns:
            str: Path of the saved PNG.
        """
        c = self.c
        maps, labels = self._attention(model, tokenizer, prompt)
        L, H, T, _ = maps.shape
        entropy = -(maps * (maps + 1e-9).log()).sum(-1).mean(-1)  # (L, H)
        l, h = divmod(int(entropy.argmin()), H)
        weights = maps[l, h]

        fig, ax = plt.subplots(figsize=(7, 6.4), facecolor=c["surface"])
        image = ax.imshow(weights, cmap=self.cmap, vmin=0, vmax=1)
        self._heatmap_ticks(ax, labels, fontsize=10)
        for i in range(T):
            for j in range(i + 1):
                if weights[i, j] >= 0.3:
                    ink = c["surface"] if weights[i, j] > 0.55 else c["ink"]
                    ax.text(j, i, f"{weights[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6, color=ink)
        ax.set_xlabel("attends to (earlier token)", color=c["ink2"])
        ax.set_ylabel("token doing the looking", color=c["ink2"])
        self._colorbar(fig, image, ax)
        self._title(ax, f"Most focused head: layer {l}, head {h}",
                    f"mean attention entropy {entropy[l, h]:.2f} nats (uniform over {T} tokens would be "
                    f"≤ {math.log(T):.2f}); weights ≥ .30 labelled")
        return self._save(fig, "attention_sharpest_head")

    def next_char_probs(self, model, tokenizer, prompts, top_k=8):
        """Bar charts of the most likely next characters after a few prompts.

        Args:
            model (LanguageModel): Trained model on the CPU.
            tokenizer (CharTokenizer): The model's tokenizer.
            prompts (Sequence[str]): Contexts to complete.
            top_k (int): Bars per prompt.

        Returns:
            str: Path of the saved PNG.
        """
        c = self.c
        fig, axes = self._figure(w=4 * len(prompts), h=3.8, ncols=len(prompts))
        for ax, prompt in zip(np.atleast_1d(axes), prompts):
            probs = model.next_token_probs(torch.tensor([tokenizer.encode(prompt)]))
            top_p, top_i = probs.topk(top_k)
            names = [self._char_label(tokenizer.decode([i])) for i in top_i.tolist()][::-1]
            values = top_p.tolist()[::-1]
            ax.barh(range(top_k), values, color=c["series"][0], height=0.6)
            for y, v in enumerate(values):
                ax.text(v + 0.01, y, f"{v:.1%}", va="center", fontsize=8, color=c["ink2"])
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(names, fontfamily="DejaVu Sans Mono", fontsize=10, color=c["ink"])
            ax.set_xlim(0, min(1.0, max(values) * 1.3))
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=c["grid"], linewidth=0.8)
            ax.set_xlabel("probability")
            ax.set_title(f"after {prompt!r}", loc="left", fontsize=10, color=c["ink"], fontfamily="DejaVu Sans Mono")
        fig.suptitle("What the model predicts next  (↵ = newline, ␣ = space)",
                     x=0.01, ha="left", fontsize=12, color=c["ink"])
        fig.tight_layout()
        return self._save(fig, "next_char_probs")

    def parameter_breakdown(self, model):
        """Horizontal bars of where the model's parameters live.

        Args:
            model (LanguageModel): Any model.

        Returns:
            str: Path of the saved PNG.
        """
        c = self.c
        groups = {}
        for name, p in model.named_parameters():
            group = self._param_group(name)
            groups[group] = groups.get(group, 0) + p.numel()
        items = sorted(groups.items(), key=lambda kv: kv[1])
        total = sum(groups.values())

        fig, ax = self._figure(h=0.55 * len(items) + 1.3)
        ax.barh([k for k, _ in items], [v for _, v in items], color=c["series"][0], height=0.6)
        for y, (_, v) in enumerate(items):
            ax.text(v + total * 0.01, y, f"{v:,}  ({v / total:.1%})", va="center", fontsize=9, color=c["ink2"])
        ax.set_xlim(0, max(groups.values()) * 1.3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=c["grid"], linewidth=0.8)
        ax.tick_params(axis="y", labelcolor=c["ink"], labelsize=10)
        ax.set_xlabel("parameters")
        self._title(ax, f"Where the {total:,} parameters are", type(model).__name__)
        return self._save(fig, "parameter_breakdown")

    # -- helpers ---------------------------------------------------------------

    def _attention(self, model, tokenizer, prompt):
        """Run a prompt through the model and collect its attention maps.

        Args:
            model (GPTLanguageModel): Trained GPT.
            tokenizer (CharTokenizer): The model's tokenizer.
            prompt (str): Input text.

        Returns:
            tuple[Tensor, list[str]]: ((n_layer, n_head, T, T) weights, printable label per token).
        """
        ids = tokenizer.encode(prompt)[: model.cfg.block_size]
        maps = model.attention_maps(torch.tensor([ids]))
        return maps, [self._char_label(ch) for ch in tokenizer.decode(ids)]

    @staticmethod
    def _char_label(ch):
        """Make whitespace visible in axis labels.

        Args:
            ch (str): One character.

        Returns:
            str: "↵" for newline, "␣" for space, otherwise the character itself.
        """
        return {"\n": "↵", " ": "␣"}.get(ch, ch)

    @staticmethod
    def _param_group(name):
        """Map a parameter name to the component it belongs to.

        Args:
            name (str): Name from named_parameters().

        Returns:
            str: Human-readable component name.
        """
        if name.startswith("token_embedding"):
            return "token embedding"
        if name.startswith("position_embedding"):
            return "position embedding"
        if ".sa." in name:
            return "attention (Q, K, V, proj)"
        if ".ffwd." in name:
            return "feed-forward (MLP)"
        if name.startswith("lm_head"):
            return "output layer (lm_head)"
        return "LayerNorm"

    def _figure(self, w=8, h=4.5, ncols=1):
        """Create a styled figure.

        Args:
            w (float): Width in inches.
            h (float): Height in inches.
            ncols (int): Number of side-by-side axes.

        Returns:
            tuple[Figure, Axes | ndarray]: The figure and its axes.
        """
        fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=self.c["surface"])
        for ax in np.atleast_1d(axes):
            self._style(ax)
        return fig, axes

    def _style(self, ax):
        """Apply the chart style: surface colour, recessive axes and a hairline y-grid.

        Args:
            ax (Axes): Axes to style.
        """
        c = self.c
        ax.set_facecolor(c["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
        ax.tick_params(colors=c["axis"], labelcolor=c["ink2"], labelsize=9)
        ax.grid(axis="y", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(c["ink2"])
        ax.yaxis.label.set_color(c["ink2"])

    def _title(self, ax, title, subtitle=""):
        """Left-aligned title with an optional smaller line under it in secondary ink.

        Args:
            ax (Axes): Axes to title.
            title (str): Main title.
            subtitle (str): Smaller explanatory line under the title.
        """
        ax.set_title(title, loc="left", fontsize=12, color=self.c["ink"], pad=26 if subtitle else 10)
        if subtitle:
            ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 9),
                        textcoords="offset points", fontsize=9.5, color=self.c["ink2"], va="bottom")

    def _legend(self, ax, loc="center right"):
        """Frameless legend in secondary ink.

        Args:
            ax (Axes): Axes whose labelled lines go in the legend.
            loc (str): Matplotlib legend location.
        """
        ax.legend(frameon=False, fontsize=9, labelcolor=self.c["ink2"], loc=loc)

    def _end_label(self, ax, x, y, text, color):
        """Mark the last point of a line and write its value next to it.

        Args:
            ax (Axes): Target axes.
            x (float): Last x value.
            y (float): Last y value.
            text (str): Label text.
            color (str): Series colour of the marker.
        """
        ax.plot([x], [y], "o", color=color, markersize=6, markeredgecolor=self.c["surface"], markeredgewidth=2)
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=self.c["ink2"])

    def _baseline(self, ax, history):
        """Dashed line at ln(vocab_size): the loss of uniform random guessing.

        Args:
            ax (Axes): Target axes.
            history (TrainingHistory): Supplies vocab_size from its config.
        """
        vocab = history.config.get("vocab_size", 65)
        ax.axhline(math.log(vocab), color=self.c["muted"], linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"random guess = ln({vocab}) = {math.log(vocab):.2f}", (1, math.log(vocab)),
                    xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=self.c["muted"])

    def _heatmap_ticks(self, ax, labels, fontsize, show_x=True, show_y=True):
        """Put one character label per row/column of an attention heatmap.

        Args:
            ax (Axes): Heatmap axes.
            labels (list[str]): One label per token.
            fontsize (float): Label size.
            show_x (bool): Draw column labels.
            show_y (bool): Draw row labels.
        """
        c = self.c
        ticks = range(len(labels))
        ax.set_xticks(ticks, labels if show_x else [], fontsize=fontsize, fontfamily="DejaVu Sans Mono", color=c["ink"])
        ax.set_yticks(ticks, labels if show_y else [], fontsize=fontsize, fontfamily="DejaVu Sans Mono", color=c["ink"])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _colorbar(self, fig, image, axes):
        """Add a styled colour bar for attention weights.

        Args:
            fig (Figure): The figure.
            image (AxesImage): Any heatmap drawn with the shared colour scale.
            axes (Axes | ndarray): Axes the bar should sit next to.
        """
        bar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=self.c["axis"], labelcolor=self.c["ink2"], labelsize=8)
        bar.set_label("attention weight", color=self.c["ink2"], fontsize=9)

    def _save(self, fig, name):
        """Write the figure to out_dir and close it.

        Args:
            fig (Figure): Figure to save.
            name (str): File name without extension.

        Returns:
            str: Path of the saved PNG.
        """
        path = os.path.join(self.out_dir, f"{name}.png")
        fig.savefig(path, dpi=150, facecolor=self.c["surface"], bbox_inches="tight")
        plt.close(fig)
        return path
