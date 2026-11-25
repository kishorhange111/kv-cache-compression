"""KV caches with a fixed token budget, and low-bit KV caches.

Eviction: every decode step appends the new token's (k, v) to the layer's cache, attends over
it, and if the cache is then over ``budget`` tokens, keeps the ``budget`` with the highest
*priority*. Policies differ only in that priority:

| policy | priority (higher = kept) | source |
|---|---|---|
| ``window`` | position: keep the most recent | sliding-window attention |
| ``sink`` | the first ``n_sink`` tokens always, then recency | StreamingLLM (Xiao et al. 2023) |
| ``h2o`` | the newest ``budget/2``, then *accumulated* attention | H2O (Zhang et al. 2023) |
| ``tova`` | attention from the current query, averaged over heads | TOVA (Oren et al. 2024) |

``h2o`` is per head, as in the paper; ``tova`` keeps the same tokens for every head in a layer.

Quantisation (``QuantCache``) keeps every token but stores k and v in ``bits`` bits with an
asymmetric min/max scale per group. Values are quantised dequantised on write, so the
attention maths sees exactly the rounding a packed kernel would. Memory is counted
analytically by ``bytes_per_token``.
- ``key_mode="token"``: k is grouped per token (over head_dim), like v.
- ``key_mode="channel"`` (KIVI, Liu et al. 2024): k is grouped per channel over blocks of
  ``group`` tokens, because key outliers live in fixed channels. The newest, incomplete
  block stays in full precision.
"""

from __future__ import annotations

import torch

INF = float("inf")


def quantize(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Asymmetric round-to-nearest along ``dim`` (one scale and zero point per slice)."""
    lo, hi = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    scale = (hi - lo).clamp(min=1e-8) / (2**bits - 1)
    return ((x - lo) / scale).round_().clamp_(0, 2**bits - 1) * scale + lo


class Cache:
    """Full-precision cache for one layer. ``policy=None`` never evicts."""

    def __init__(self, policy: str | None = None, budget: int | None = None, n_sink: int = 4):
        if policy not in (None, "window", "sink", "h2o", "tova"):
            raise ValueError(f"unknown policy {policy!r}")
        if policy is not None and (budget is None or budget < 2):
            raise ValueError("an evicting cache needs a budget of at least 2")
        if policy == "sink" and budget <= n_sink:
            raise ValueError("the budget must exceed n_sink")
        self.policy, self.budget, self.n_sink = policy, budget, n_sink
        self.k = self.v = self.pos = self.score = None

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k, v, pos: int):
        b, h = k.shape[:2]
        p = torch.full((b, h, 1), pos, dtype=torch.long, device=k.device)
        s = torch.zeros((b, h, 1), dtype=torch.float32, device=k.device)
        if self.k is None:
            self.k, self.v, self.pos, self.score = k, v, p, s
        else:
            self.k, self.v = torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)
            self.pos, self.score = torch.cat([self.pos, p], 2), torch.cat([self.score, s], 2)

    def keys(self):
        return self.k

    def values(self):
        return self.v

    def priority(self, att: torch.Tensor, now: int) -> torch.Tensor:
        pos = self.pos.float()
        if self.policy == "window":
            return pos
        if self.policy == "sink":
            return torch.where(self.pos < self.n_sink, INF, pos)
        if self.policy == "h2o":
            recent = self.pos > now - self.budget // 2
            return torch.where(recent, INF, self.score)
        return att.mean(1, keepdim=True).expand_as(self.score)  # tova

    def observe(self, att: torch.Tensor, now: int):
        """att: (b, h, T) attention of the current query over the cache."""
        self.score += att.float()
        if self.policy is None or len(self) <= self.budget:
            return
        keep = self.priority(att.float(), now).topk(self.budget, dim=-1).indices
        g = keep.unsqueeze(-1).expand(-1, -1, -1, self.k.shape[-1])
        self.k, self.v = self.k.gather(2, g), self.v.gather(2, g)
        self.pos, self.score = self.pos.gather(2, keep), self.score.gather(2, keep)


class QuantCache(Cache):
    def __init__(self, bits: int, key_mode: str = "token", group: int = 32):
        super().__init__()
        if key_mode not in ("token", "channel"):
            raise ValueError(key_mode)
        self.bits, self.key_mode, self.group = bits, key_mode, group
        self.n_quant = 0  # tokens of k already quantised per channel (channel mode)

    def append(self, k, v, pos: int):
        super().append(k, quantize(v, self.bits, -1), pos)
        if self.key_mode == "token":
            self.k[:, :, -1:] = quantize(k, self.bits, -1)
        elif len(self) - self.n_quant == self.group:  # a full block of residual keys
            blk = self.k[:, :, self.n_quant :]
            self.k[:, :, self.n_quant :] = quantize(blk, self.bits, 2)
            self.n_quant = len(self)

    def bytes_per_token(self, head_dim: int, n_heads: int, scale_bytes: int = 2) -> float:
        """Average bytes per token per layer for k and v, including scales and zero points.
        Channel-mode residual keys are counted at fp16."""
        v = n_heads * (head_dim * self.bits / 8 + 2 * scale_bytes)
        if self.key_mode == "token":
            k = v
        else:
            T = max(1, len(self))
            k_q = (
                n_heads
                * head_dim
                * (self.n_quant * self.bits / 8 + 2 * scale_bytes * (self.n_quant // self.group))
                / T
            )
            k = k_q + n_heads * head_dim * 2 * (T - self.n_quant) / T
        return k + v


def fp16_bytes_per_token(head_dim: int, n_heads: int) -> int:
    return 2 * 2 * head_dim * n_heads
