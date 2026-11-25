"""Compare KV-cache policies on GPT-2 at 1,024 tokens.

    python -m kvc.evaluate --out results --device cuda

Two tasks, both teacher-forced and token by token through the cache:

- **Perplexity** on held-out text (Python standard-library source by default, or
  ``--data wikitext``). It is reported over the whole sequence, and over the positions past
  the budget only, since those are the only positions eviction can affect.
- **Long-range copy**: ``filler · R · filler · R``, where R is 64 random tokens and the two
  copies are 832 tokens apart (at 1,024). GPT-2's induction heads can predict the second R almost
  perfectly, *if* the first R is still in the cache. Accuracy on the second R measures
  whether a policy keeps information that is old but needed later. Perplexity on ordinary
  text mostly measures whether it keeps the recent context.

Finished configurations are kept in ``<out>/runs.jsonl``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sysconfig
import time
import zlib

import torch
import torch.nn.functional as F

from .cache import Cache, QuantCache, fp16_bytes_per_token
from .model import GPT2, streaming_logits


def configs(budgets=(64, 128, 256, 512)):
    yield "full", None, lambda: Cache()
    for pol in ("window", "sink", "h2o", "tova"):
        for b in budgets:
            yield f"{pol} B={b}", b, (lambda pol=pol, b=b: Cache(pol, b))
    for bits, mode in ((8, "token"), (4, "token"), (4, "channel"), (2, "token"), (2, "channel")):
        yield f"int{bits} k-per-{mode}", None, (lambda bits=bits, mode=mode: QuantCache(bits, mode))


def nll(logits, idx):
    return F.cross_entropy(logits[:, :-1].float().transpose(1, 2), idx[:, 1:], reduction="none")


def copy_batch(
    text_tokens: torch.Tensor, n: int, length: int, gen: torch.Generator, vocab: int = 50257
):
    """(n, length) rows of filler · R · filler · R, the start of the second R, and |R|.

    At length 1,024: 128 filler tokens, |R| = 64, and the copies are 832 positions apart."""
    fill1, r_len = length // 8, length // 16
    fill2 = length - fill1 - 2 * r_len
    lo, hi = min(1000, vocab // 4), min(30000, vocab)
    rows = []
    for _ in range(n):
        i = int(torch.randint(0, len(text_tokens) - length, (1,), generator=gen))
        f = text_tokens[i : i + fill1 + fill2]
        r = torch.randint(lo, hi, (r_len,), generator=gen)
        rows.append(torch.cat([f[:fill1], r, f[fill1:], r]))
    return torch.stack(rows), fill1 + r_len + fill2, r_len


def text_tokens(data: str):
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if data == "wikitext":
        from datasets import load_dataset

        text = "\n".join(
            load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"]
        )
    else:
        root = sysconfig.get_paths()["stdlib"]
        files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
        held = [f for f in files if zlib.crc32(os.path.relpath(f, root).encode()) % 20 == 0]
        text = "\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in held)
    return torch.tensor(tok(text[:3_000_000])["input_ids"])


def evaluate_config(model, make_cache, budget, text, copy, copy_start, n_ctx):
    t0 = time.time()
    lg = streaming_logits(model, text, make_cache)
    loss = nll(lg, text)
    rec = {"ppl": math.exp(float(loss.mean()))}
    if budget is not None and budget < n_ctx - 1:
        rec["ppl_past_budget"] = math.exp(float(loss[:, budget:].mean()))
    caches = [make_cache() for _ in range(model.n_layer)]
    preds = []
    for t in range(copy.shape[1] - 1):
        preds.append(model.step(copy[:, t], t, caches).argmax(-1))
    pred = torch.stack(preds, 1)
    target = copy[:, copy_start + 1 :]
    rec["copy_acc"] = float((pred[:, copy_start:] == target).float().mean())
    c = caches[0]
    hd, nh = model.hd, model.n_head
    full = fp16_bytes_per_token(hd, nh) * n_ctx
    if isinstance(c, QuantCache):
        rec["kv_mem_frac"] = c.bytes_per_token(hd, nh) * n_ctx / full
    else:
        rec["kv_mem_frac"] = min(n_ctx, budget or n_ctx) / n_ctx
    rec["seconds"] = time.time() - t0
    return rec


def report(recs, out, meta):
    lines = [
        f"{meta}\n",
        "| cache | KV memory | ppl | ppl past budget | long-range copy acc |",
        "|---|---|---|---|---|",
    ]
    for r in recs:
        pb = r.get("ppl_past_budget")
        lines.append(
            f"| {r['name']} | {100 * r['kv_mem_frac']:.1f}% | {r['ppl']:.2f} | "
            f"{'–' if pb is None else f'{pb:.2f}'} | {100 * r['copy_acc']:.1f}% |"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main(argv=None, model=None, tokens=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", choices=["stdlib", "wikitext"], default="stdlib")
    ap.add_argument("--n", type=int, default=8, help="sequences per task")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256, 512])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.jsonl")
    recs = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    done = {r["name"] for r in recs}
    if model is None:
        from transformers import GPT2LMHeadModel

        dt = torch.float16 if a.device == "cuda" else torch.float32
        model = GPT2(GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=dt).to(a.device))
    tokens = text_tokens(a.data) if tokens is None else tokens
    gen = torch.Generator().manual_seed(0)
    starts = torch.randint(0, len(tokens) - a.ctx, (a.n,), generator=gen)
    text = torch.stack([tokens[s : s + a.ctx] for s in starts]).to(a.device)
    copy, copy_start, r_len = copy_batch(tokens, a.n, a.ctx, gen, model.wte.shape[0])
    copy = copy.to(a.device)
    for name, budget, make in configs(a.budgets):
        if name in done:
            continue
        r = {
            "name": name,
            "budget": budget,
            **evaluate_config(model, make, budget, text, copy, copy_start, a.ctx),
        }
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        recs.append(r)
        print(
            f"[{time.strftime('%H:%M:%S')}] {name}: ppl {r['ppl']:.2f} "
            f"copy {r['copy_acc']:.3f} ({r['seconds']:.0f}s)",
            flush=True,
        )
    dev = torch.cuda.get_device_name() if a.device == "cuda" else "cpu"
    meta = (
        f"GPT-2 (124M), {a.ctx} tokens, {a.n} sequences per task, data={a.data}, {dev}. "
        f"Copy task: {r_len} random tokens repeated {copy_start - a.ctx // 8} positions later."
    )
    print(report(recs, a.out, meta))


if __name__ == "__main__":
    main()
