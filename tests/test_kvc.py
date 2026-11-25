import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from kvc.cache import Cache, QuantCache, fp16_bytes_per_token, quantize
from kvc.evaluate import copy_batch, main
from kvc.model import GPT2, streaming_logits

T = 40


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_positions=256,
        vocab_size=300,
        bos_token_id=0,
        eos_token_id=0,
    )
    hf = GPT2LMHeadModel(cfg).eval()
    return hf, GPT2(hf)


@pytest.fixture(scope="module")
def idx():
    return torch.randint(0, 300, (3, T), generator=torch.Generator().manual_seed(1))


def test_reimplementation_matches_hugging_face(models, idx):
    hf, m = models
    with torch.no_grad():
        torch.testing.assert_close(m.forward(idx), hf(idx).logits, atol=1e-4, rtol=1e-4)


def test_full_cache_decode_matches_parallel_forward(models, idx):
    _, m = models
    torch.testing.assert_close(
        streaming_logits(m, idx, Cache), m.forward(idx), atol=1e-4, rtol=1e-4
    )


def _mask(keep):
    i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    return (j <= i) & keep(i, j)


def test_window_equals_banded_attention(models, idx):
    _, m = models
    B = 8  # the query attends the B cached tokens plus itself
    got = streaming_logits(m, idx, lambda: Cache("window", B))
    want = m.forward(idx, _mask(lambda i, j: j >= i - B))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_attention_sinks_equal_sink_plus_band_mask(models, idx):
    _, m = models
    B, S = 10, 4
    got = streaming_logits(m, idx, lambda: Cache("sink", B, n_sink=S))
    want = m.forward(idx, _mask(lambda i, j: (j < S) | (j >= i - (B - S))))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(got, streaming_logits(m, idx, lambda: Cache("window", B)))


def _filled(policy, budget, n=8):
    c = Cache(policy, budget)
    for t in range(n):
        c.append(torch.full((1, 2, 1, 3), float(t)), torch.zeros(1, 2, 1, 3), t)
    return c


def test_h2o_keeps_recent_and_heavy_hitters():
    c = _filled("h2o", 4, n=6)  # positions 0..5
    c.score[0, 0] = torch.tensor([0.0, 9.0, 0.1, 0.2, 0.0, 0.0])
    c.score[0, 1] = torch.tensor([5.0, 0.0, 0.1, 6.0, 0.0, 0.0])
    c.budget = 4
    c.observe(torch.zeros(1, 2, 6), now=5)
    kept = [sorted(c.pos[0, h].tolist()) for h in range(2)]
    assert kept == [[1, 3, 4, 5], [0, 3, 4, 5]]  # recent {4, 5} plus the top-2 older, per head


def test_tova_drops_least_attended_token_for_all_heads():
    c = _filled("tova", 5, n=6)
    att = torch.tensor([[0.3, 0.1, 0.2, 0.05, 0.15, 0.2], [0.1, 0.2, 0.2, 0.15, 0.2, 0.15]])
    c.observe(att[None], now=5)  # head means: 3 has 0.1 (lowest)
    for h in range(2):
        assert sorted(c.pos[0, h].tolist()) == [0, 1, 2, 4, 5]


def test_bad_arguments():
    with pytest.raises(ValueError):
        Cache("lru", 8)
    with pytest.raises(ValueError):
        Cache("sink", 4, n_sink=4)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantization_error_is_at_most_half_a_step(bits):
    x = torch.randn(4, 16, 64, dtype=torch.float64)
    q = quantize(x, bits, -1)
    step = (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)) / (2**bits - 1)
    assert torch.all((q - x).abs() <= step / 2 + 1e-12)
    assert q.unique().numel() <= 4 * 16 * 2**bits


def test_kivi_keeps_residual_keys_exact_until_a_group_fills():
    c = QuantCache(2, "channel", group=4)
    ks = [torch.randn(1, 2, 1, 8) for _ in range(6)]
    for t, k in enumerate(ks):
        c.append(k, torch.randn(1, 2, 1, 8), t)
    assert c.n_quant == 4
    torch.testing.assert_close(c.k[:, :, 4:], torch.cat(ks[4:], 2))  # residual: full precision
    assert not torch.allclose(c.k[:, :, :4], torch.cat(ks[:4], 2))
    assert c.k[0, 0, :4, 0].unique().numel() <= 4  # 2 bits per channel over the group


def test_int8_cache_is_nearly_lossless_and_int2_is_not(models, idx):
    _, m = models
    ref = m.forward(idx)
    e8 = (streaming_logits(m, idx, lambda: QuantCache(8)) - ref).abs().max()
    e2 = (streaming_logits(m, idx, lambda: QuantCache(2)) - ref).abs().max()
    assert e8 < 0.1 * e2  # 8-bit error is a small fraction of 2-bit error


def test_memory_accounting():
    full = fp16_bytes_per_token(64, 12)
    assert QuantCache(8).bytes_per_token(64, 12) / full == pytest.approx((64 + 4) / 128)
    assert QuantCache(4).bytes_per_token(64, 12) / full == pytest.approx((32 + 4) / 128)


def test_copy_batch_layout():
    tok = torch.arange(5000)
    x, start, r = copy_batch(tok, 3, 1024, torch.Generator().manual_seed(0))
    assert x.shape == (3, 1024) and r == 64 and start == 1024 - 64 and start - 128 == 832
    torch.testing.assert_close(x[:, 128 : 128 + r], x[:, start:])


def test_end_to_end_tiny_and_resume(models, tmp_path):
    _, m = models
    tokens = torch.randint(0, 300, (4000,), generator=torch.Generator().manual_seed(2))
    args = [
        "--device",
        "cpu",
        "--out",
        str(tmp_path),
        "--n",
        "2",
        "--ctx",
        "160",
        "--budgets",
        "16",
    ]
    main(args, model=m, tokens=tokens)
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 1 + 4 + 5
    main(args, model=m, tokens=tokens)
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == len(lines)
    assert "long-range copy" in (tmp_path / "summary.md").read_text(encoding="utf-8")
