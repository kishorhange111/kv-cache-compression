"""GPT-2 re-implemented on top of Hugging Face weights, with an explicit decode step.

Writing the attention by hand gives the cache full control over which (k, v) a query sees.
GPT-2 uses learned absolute positions added at the input, so an evicted token's
position is already baked into the cached keys and values. Unlike RoPE models, nothing needs
re-indexing after an eviction.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cache import Cache


class GPT2:
    def __init__(self, hf_model) -> None:
        c = hf_model.config
        self.n_layer, self.n_head, self.d = c.n_layer, c.n_head, c.n_embd
        self.hd = self.d // self.n_head
        self.eps = c.layer_norm_epsilon
        sd = {k: v.detach() for k, v in hf_model.state_dict().items()}
        self.wte, self.wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
        self.lnf = (sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"])
        self.layers = []
        for i in range(self.n_layer):
            p = f"transformer.h.{i}."
            self.layers.append(
                {
                    n: sd[p + n]
                    for n in (
                        "ln_1.weight",
                        "ln_1.bias",
                        "attn.c_attn.weight",
                        "attn.c_attn.bias",
                        "attn.c_proj.weight",
                        "attn.c_proj.bias",
                        "ln_2.weight",
                        "ln_2.bias",
                        "mlp.c_fc.weight",
                        "mlp.c_fc.bias",
                        "mlp.c_proj.weight",
                        "mlp.c_proj.bias",
                    )
                }
            )
        self.device, self.dtype = self.wte.device, self.wte.dtype

    def _ln(self, x, w, b):
        return F.layer_norm(x, (self.d,), w, b, self.eps)

    def _mlp(self, L, x):
        h = self._ln(x, L["ln_2.weight"], L["ln_2.bias"])
        h = F.gelu(h @ L["mlp.c_fc.weight"] + L["mlp.c_fc.bias"], approximate="tanh")
        return h @ L["mlp.c_proj.weight"] + L["mlp.c_proj.bias"]

    def _qkv(self, L, x):
        h = self._ln(x, L["ln_1.weight"], L["ln_1.bias"])
        qkv = h @ L["attn.c_attn.weight"] + L["attn.c_attn.bias"]
        b, t = x.shape[:2]
        return qkv.view(b, t, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)  # 3,b,h,t,hd

    def _head(self, x):
        return self._ln(x, *self.lnf) @ self.wte.t()

    @torch.no_grad()
    def forward(self, idx, mask: torch.Tensor | None = None):
        """Parallel forward. ``mask`` (t, t) bool, True = may attend; default causal."""
        t = idx.shape[1]
        x = self.wte[idx] + self.wpe[:t]
        if mask is None:
            mask = torch.ones(t, t, dtype=torch.bool, device=idx.device).tril()
        for L in self.layers:
            q, k, v = self._qkv(L, x)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            y = y.transpose(1, 2).reshape(*x.shape)
            x = x + y @ L["attn.c_proj.weight"] + L["attn.c_proj.bias"]
            x = x + self._mlp(L, x)
        return self._head(x)

    @torch.no_grad()
    def step(self, tok: torch.Tensor, pos: int, caches: list[Cache]) -> torch.Tensor:
        """One token per sequence (b,) at position ``pos`` -> logits (b, vocab)."""
        x = (self.wte[tok] + self.wpe[pos])[:, None]
        for L, cache in zip(self.layers, caches, strict=True):
            q, k, v = self._qkv(L, x)
            cache.append(k, v, pos)
            s = (q @ cache.keys().transpose(-1, -2)).float() / math.sqrt(self.hd)
            att = s.softmax(-1)  # (b, h, 1, T)
            y = (att.to(v.dtype) @ cache.values()).transpose(1, 2).reshape(*x.shape)
            cache.observe(att[:, :, 0], pos)
            x = x + y @ L["attn.c_proj.weight"] + L["attn.c_proj.bias"]
            x = x + self._mlp(L, x)
        return self._head(x)[:, 0]


def streaming_logits(model: GPT2, idx: torch.Tensor, make_cache) -> torch.Tensor:
    """Teacher-forced token-by-token decode of ``idx`` (b, t) with fresh caches."""
    caches = [make_cache() for _ in range(model.n_layer)]
    return torch.stack([model.step(idx[:, t], t, caches) for t in range(idx.shape[1])], 1)
