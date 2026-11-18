"""Transformer building blocks: attention head, multi-head attention, MLP, block."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of causal self-attention.

    Attributes:
        keep_attention (bool): When True, forward() stores its attention weights in `attention`.
        attention (Tensor | None): (B, T, T) weights from the last forward pass with keep_attention on.
    """

    def __init__(self, cfg, head_size, layer_idx, head_idx, tracer):
        """
        Args:
            cfg (GPTConfig): Supplies n_embd, block_size and dropout.
            head_size (int): Dimension of the key/query/value vectors (hs).
            layer_idx (int): Index of the block this head belongs to (for tracing).
            head_idx (int): Index of this head within its block (for tracing).
            tracer (Tracer): Prints the head's calculations when enabled.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.tracer = tracer
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)
        self.keep_attention = False
        self.attention = None

    def forward(self, x):
        """Let every token gather information from itself and the tokens before it.

        Args:
            x (Tensor): (B, T, C) token representations.

        Returns:
            Tensor: (B, T, head_size) attention-weighted sum of the value vectors.
        """
        B, T, C = x.shape
        tr = self.tracer.focus(self.layer_idx, self.head_idx)
        with tr.scope(f"Head {self.head_idx} (layer {self.layer_idx})"):
            k = self.key(x)
            tr.tensor("k = key(x)", k, "what each token contains   (B,T,C) @ (C,hs) → (B,T,hs)")
            q = self.query(x)
            tr.tensor("q = query(x)", q, "what each token is looking for   (B,T,hs)")

            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            tr.tensor("wei = q @ kᵀ · hs^-0.5", wei,
                      f"affinity of every token pair, scaled by 1/√{k.shape[-1]}   (B,T,T)")
            tr.tensor("tril[:T, :T]", self.tril[:T, :T].int(), "1 = allowed to look, 0 = future token")
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            tr.tensor("wei.masked_fill(tril == 0, -inf)", wei, "a token may not look at the future")
            wei = F.softmax(wei, dim=-1)
            tr.tensor("wei = softmax(wei, dim=-1)", wei, "-inf → 0, each row becomes weights summing to 1")
            tr.tensor("wei.sum(dim=-1)", wei.sum(dim=-1), "check: every row sums to 1")
            if self.keep_attention:
                self.attention = wei.detach()
            wei = self.dropout(wei)
            if self.training and self.dropout.p > 0:
                tr.tensor("dropout(wei)", wei, f"randomly zero {self.dropout.p:.0%} of weights (training only)")

            v = self.value(x)
            tr.tensor("v = value(x)", v, "what each token communicates if attended to   (B,T,hs)")
            out = wei @ v
            tr.tensor("out = wei @ v", out, "weighted average of past values   (B,T,T) @ (B,T,hs) → (B,T,hs)")
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected back to n_embd."""

    def __init__(self, cfg, layer_idx, tracer):
        """
        Args:
            cfg (GPTConfig): Supplies n_head, head_size, n_embd and dropout.
            layer_idx (int): Index of the block this module belongs to.
            tracer (Tracer): Prints the calculations when enabled.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.tracer = tracer
        self.heads = nn.ModuleList(
            [Head(cfg, cfg.head_size, layer_idx, h, tracer) for h in range(cfg.n_head)]
        )
        self.proj = nn.Linear(cfg.head_size * cfg.n_head, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """Run all heads on the same input and mix their outputs.

        Args:
            x (Tensor): (B, T, C) token representations.

        Returns:
            Tensor: (B, T, C) combined output of all heads.
        """
        tr = self.tracer.focus(self.layer_idx)
        n = len(self.heads)
        with tr.scope(f"MultiHeadAttention ({n} heads × head_size {self.heads[0].key.out_features})"):
            outs = [head(x) for head in self.heads]
            if tr.detail_heads < n:
                hidden = f"head {n - 1}" if tr.detail_heads == n - 1 else f"heads {tr.detail_heads}..{n - 1}"
                tr.note(f"({hidden} ran the same steps with their own weights; "
                        f"use --detail-heads {n} to print them)")
            out = torch.cat(outs, dim=-1)
            tr.tensor("concat(heads)", out, f"{n} × (B,T,{outs[0].shape[-1]}) → (B,T,{out.shape[-1]})")
            out = self.dropout(self.proj(out))
            tr.tensor("proj(concat)", out, "linear layer mixes the heads' outputs   (B,T,C)")
        return out


class FeedForward(nn.Module):
    """Per-token MLP: expand to 4·n_embd, ReLU, project back."""

    def __init__(self, cfg, layer_idx, tracer):
        """
        Args:
            cfg (GPTConfig): Supplies n_embd and dropout.
            layer_idx (int): Index of the block this module belongs to.
            tracer (Tracer): Prints the calculations when enabled.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.tracer = tracer
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """Transform every token independently ("thinking" about what attention gathered).

        Args:
            x (Tensor): (B, T, C) token representations.

        Returns:
            Tensor: (B, T, C) transformed representations.
        """
        tr = self.tracer.focus(self.layer_idx)
        with tr.scope("FeedForward"):
            h = self.fc(x)
            tr.tensor("h = fc(x)", h, "expand each token C → 4C   (B,T,4C)")
            h = self.relu(h)
            if tr.enabled:
                zeroed = (h == 0).float().mean().item()
                tr.tensor("h = relu(h)", h, f"max(0, h): {zeroed:.0%} of values were negative → 0")
            out = self.dropout(self.proj(h))
            tr.tensor("proj(h)", out, "project back 4C → C   (B,T,C)")
        return out


class Block(nn.Module):
    """Transformer block: communication (attention) then computation (feed-forward)."""

    def __init__(self, cfg, layer_idx, tracer):
        """
        Args:
            cfg (GPTConfig): Model hyperparameters.
            layer_idx (int): Position of this block in the stack (0 = first).
            tracer (Tracer): Prints the calculations when enabled.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.tracer = tracer
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.sa = MultiHeadAttention(cfg, layer_idx, tracer)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg, layer_idx, tracer)

    def forward(self, x):
        """Apply x + attention(LN(x)), then x + MLP(LN(x)) (pre-norm residual block).

        Args:
            x (Tensor): (B, T, C) residual stream.

        Returns:
            Tensor: (B, T, C) updated residual stream.
        """
        tr = self.tracer.focus(self.layer_idx)
        with tr.scope(f"Block {self.layer_idx}"):
            h = self.ln1(x)
            tr.tensor("ln1(x)", h, "LayerNorm: each token's C features → mean 0, std 1 (then scale+shift)")
            x = x + self.sa(h)
            tr.tensor("x = x + sa(ln1(x))", x, "residual connection: add attention output to the stream")
            h = self.ln2(x)
            tr.tensor("ln2(x)", h, "LayerNorm again before the MLP")
            x = x + self.ffwd(h)
            tr.tensor("x = x + ffwd(ln2(x))", x, "residual connection → output of this block")
        return x
