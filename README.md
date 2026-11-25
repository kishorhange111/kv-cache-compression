# kv-cache-compression

[![CI](https://github.com/kishorhange111/kv-cache-compression/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/kv-cache-compression/actions/workflows/ci.yml)

At long context, decode is limited by reading the KV cache, and the KV cache is what limits
batch size. This repo implements the two main ways to shrink it, **evicting tokens** and
**quantising them**, on a hand-written GPT-2 decode loop. It then measures what each one
costs, on ordinary text and on a task built to expose what eviction throws away.

## Policies ([`cache.py`](src/kvc/cache.py))

Every decode step appends the new (k, v), attends, and if the cache is over budget *B*,
keeps the *B* tokens with the highest priority:

| policy | kept | paper |
|---|---|---|
| `window` | the most recent *B* | sliding window |
| `sink` | the first 4 tokens always, then the most recent | StreamingLLM (Xiao et al. 2023) |
| `h2o` | the newest *B*/2, then the most *accumulated* attention, per head | H2O (Zhang et al. 2023) |
| `tova` | the most attention from the current query, averaged over heads | TOVA (Oren et al. 2024) |
| `intN k-per-token` | everything, k and v stored in N bits with a min/max scale per token | — |
| `intN k-per-channel` | everything, v per token, but k grouped **per channel** over blocks of 32 tokens, with the newest block in fp16 | KIVI (Liu et al. 2024) |

KIVI's observation is that key outliers sit in fixed channels, so per-token groups waste
their range on them. Grouping keys per channel instead keeps 2-bit keys usable.

[`model.py`](src/kvc/model.py) re-implements GPT-2's forward pass on the Hugging Face
weights, which gives the cache full control of what each query sees. GPT-2's positions are
absolute and added at the input, so evicted tokens need no re-indexing, unlike RoPE models.

The tests pin down each piece:
- the re-implementation matches Hugging Face logits;
- full-cache decoding matches the parallel forward;
- `window` decoding equals a banded attention mask, and `sink` equals sink-plus-band;
- H2O keeps the recent half plus per-head heavy hitters;
- TOVA drops the least-attended token for every head;
- quantisation error is at most half a step;
- KIVI's residual keys stay exact until a block fills;
- memory accounting is correct.

## Evaluation ([`evaluate.py`](src/kvc/evaluate.py))

GPT-2 (124M) runs at 1,024 tokens, teacher-forced and token by token through each cache,
with budgets *B* ∈ {64, 128, 256, 512}. There are two tasks:
- **Perplexity** on held-out text, over all positions and over the positions past *B*,
  which are the only ones eviction can affect.
- **Long-range copy**: `filler · R · filler · R`, where R is 64 random tokens and the two
  copies are 832 positions apart. GPT-2's induction heads predict the second R almost
  perfectly if the first is still cached. This checks whether a policy keeps
  information that is *old but needed*, which perplexity on ordinary text barely measures.

```
pytest -q                                         # 15 tests, CPU (tiny random GPT-2)
python -m kvc.evaluate --out results --device cuda
```

**Results are pending** from the Colab run.
