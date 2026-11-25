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

## Results (Tesla T4; [`results/summary.md`](results/summary.md), raw rows in [`results/runs.jsonl`](results/runs.jsonl))

GPT-2 (124M), 1,024 tokens, 8 sequences per task, held-out Python stdlib text. KV memory
is relative to the full fp16 cache, and the quantized rows include scales.

| cache | KV memory | ppl | ppl past budget | long-range copy |
|---|---|---|---|---|
| full | 100% | 7.27 | – | **97.6%** |
| window B=64 / 512 | 6.2% / 50% | 410.14 / 14.85 | 481.31 / 23.15 | 0.0% / 0.0% |
| sink B=64 / 512 | 6.2% / 50% | 18.98 / 7.76 | 18.14 / 6.32 | 0.0% / 0.0% |
| H2O B=64 / 512 | 6.2% / 50% | 21.83 / 7.68 | 21.06 / 6.18 | 0.0% / 2.4% |
| TOVA B=64 / 256 / 512 | 6.2% / 25% / 50% | 18.59 / 9.77 / 7.72 | 17.75 / 8.77 / 6.25 | 0.0% / 0.0% / 10.1% |
| int8, keys per token | 53.1% | 7.27 | – | 97.6% |
| int4, keys per token | 28.1% | 7.79 | – | 95.8% |
| **int4, keys per channel** | **30.7%** | **7.30** | – | **97.2%** |
| int2, keys per token | 15.6% | 36.76 | – | 1.4% |
| **int2, keys per channel** | **18.4%** | **8.80** | – | **88.3%** |

The full B ∈ {64, 128, 256, 512} sweep is in `results/summary.md`.

1. **Perplexity hides what eviction loses.** At half the memory, sinks, H2O and TOVA all
   come within 0.5 ppl of the full cache, and past the budget they even score *lower*
   (6.2–6.3 vs 7.27), because those positions are later in the document and easier. Yet
   all of them lose nearly the entire long-range copy (≤10% vs 97.6%). A token that was
   unimportant when it was evicted can become essential later, and a score-based policy
   can't know that in advance.
2. **Attention sinks are most of what makes eviction usable.** A plain sliding window of
   64 tokens reaches ppl 410. Keeping 4 initial tokens as well (StreamingLLM) brings it to
   19. Among the eviction policies TOVA is consistently best, and it is the only one
   that recovers some copying (10% at B = 512).
3. **Quantization keeps the information.** 4-bit KV with keys quantized *per channel* uses
   30.7% of the memory, costs 0.03 ppl and keeps 97.2% copy accuracy. That is the same
   memory as a B = 256 eviction cache, which copies 0%.
4. **Per-channel keys matter, as KIVI argues.** Key outliers sit in fixed channels, so a
   per-token scale wastes its range. At 2 bits, per-channel keys give 8.80 ppl and 88.3%
   copy, against 36.76 ppl and 1.4% per token.
