"""The perturbation model: two weight-shared branches, mutation as a localized edit.

ITW (in-the-wild) is the wild type with the null edit; MUT carries the mutation. The two
branches are the *same modules* applied twice, so nothing can drift between them, and the head
reads ``f_MUT - f_ITW``.

Three properties are structural rather than emergent, and the unit tests pin all three:

1. **A null edit gives exactly zero.** With no mutation, ``delta`` is zero everywhere and the
   BLOSUM pair is ``(a, a)`` on both branches, so every intermediate is identical and the
   difference is the zero vector. The head is therefore **bias-free**: ``MLP(0) = 0`` exactly.
   With biases it would emit a constant offset for a no-op edit, which is wrong on its face.
2. **The ITW branch never sees the structure refinement.** Step 3 is gated on ``delta != 0``,
   and ITW's delta is identically zero, so ``t' = t`` by construction rather than by
   arithmetic coincidence.
3. **The distance bias is shared across heads and monotone-free.** Nothing forces the learned
   bias to decay with distance; whether it does is a finding, not an assumption.

One deliberate departure from ProtAttBA, whose gate and pooling this reuses in spirit: their
``MutilHeadSelfAttn`` masks padded keys with ``masked_fill(mask == 0, 1e-10)`` instead of
``-inf``, so padding keeps a real share of the softmax mass and a prediction depends on its
batch's padding width. That is reproduced faithfully in the reproduction, and **not** here --
this module masks with ``-inf``. Copying a bug into new code to match a baseline would make
every ablation below depend on batch composition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

#: 16 bins up to 20 A plus one "beyond" bin, per the spec.
DIST_BINS = 16
DIST_MAX = 20.0
N_DIST_BINS = DIST_BINS + 1


@dataclass
class PerturbConfig:
    """Every flag an ablation touches. Defaults are the full model of section 1."""

    seq_dim: int = 256            # PCA-reduced ESM width
    struct_dim: int = 128         # ProteinMPNN encoder width
    n_heads: int = 4
    head_dim: int = 64            # n_heads * head_dim == model width
    dropout: float = 0.2

    delta_scope: str = "site"     # {site, all}
    use_blosum: bool = True
    blosum_mode: str = "blosum"   # {blosum, learned}
    dist_bias: bool = True

    # ablation switches, all default-off so the default config is the full model
    delta_film: bool = True       # step 3
    struct_film: bool = True      # step 4 (False -> concat + linear)
    film_reversed: bool = False   # step 4 direction swapped
    use_structure: bool = True    # False -> MPNN tokens zeroed
    shuffle_structure: bool = False
    late_fusion: bool = False
    use_delta: bool = True        # False -> blosum_only / full_mutant_tokens
    full_mutant_tokens: bool = False
    two_branch: bool = True       # False -> single_branch
    concat_head: bool = False     # True -> head on [f_MUT; f_ITW]
    cross_chain: bool = True      # False -> self-attention within each chain
    dist_bias_only: bool = False  # attention from geometry alone, no Q.K
    site_pool: bool = False

    @property
    def model_dim(self) -> int:
        return self.n_heads * self.head_dim


class RotaryEmbedding(nn.Module):
    """Standard RoPE, applied to queries and keys before the dot product."""

    def __init__(self, dim: int):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(n, device=device, dtype=self.inv_freq.dtype)
        f = torch.einsum("i,j->ij", t, self.inv_freq)
        cos, sin = f.cos(), f.sin()
        return (torch.cat([cos, cos], -1)[None, None], torch.cat([sin, sin], -1)[None, None])


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-b, a], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class LocalGate(nn.Module):
    """ProtAttBA's local-pattern gate: LayerNorm -> conv(k=1) -> softmax -> reweight."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Conv1d(dim, 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a = self.score(h.transpose(1, 2)).squeeze(1)
        a = a.masked_fill(~mask.bool(), float("-inf"))
        return x * torch.softmax(a, -1).unsqueeze(-1)


class AttnPool(nn.Module):
    """ProtAttBA's convolutional pooling: the same scoring, collapsed to one vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Conv1d(dim, 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a = self.score(h.transpose(1, 2)).squeeze(1)
        a = a.masked_fill(~mask.bool(), float("-inf"))
        # A row with no valid position would give all -inf; guard it rather than emit NaN.
        empty = (~mask.bool()).all(dim=-1, keepdim=True)
        a = a.masked_fill(empty, 0.0)
        return torch.bmm(torch.softmax(a, -1).unsqueeze(1), x).squeeze(1)


class CrossAttention(nn.Module):
    """One layer of cross-chain attention with RoPE and a learned distance bias."""

    def __init__(self, cfg: PerturbConfig):
        super().__init__()
        d = cfg.model_dim
        self.cfg = cfg
        self.ln_q, self.ln_k, self.ln_v = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.rope = RotaryEmbedding(cfg.head_dim)
        self.ln_out, self.ln_ffn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Linear(d, d)
        self.drop = nn.Dropout(cfg.dropout)
        # shared across heads, per the spec
        self.bias = nn.Embedding(N_DIST_BINS, 1) if cfg.dist_bias else None
        if self.bias is not None:
            nn.init.zeros_(self.bias.weight)

    def forward(self, q_in, kv_in, kv_mask, dist_bins=None):
        b, lq, _ = q_in.shape
        h, hd = self.cfg.n_heads, self.cfg.head_dim
        q = self.q(self.ln_q(q_in)).view(b, lq, h, hd).transpose(1, 2)
        k = self.k(self.ln_k(kv_in)).view(b, kv_in.shape[1], h, hd).transpose(1, 2)
        v = self.v(self.ln_v(kv_in)).view(b, kv_in.shape[1], h, hd).transpose(1, 2)

        cq, sq = self.rope(lq, q.device)
        ck, sk = self.rope(kv_in.shape[1], k.device)
        q, k = apply_rope(q, cq, sq), apply_rope(k, ck, sk)

        if self.cfg.dist_bias_only:
            # geometry alone decides attention; no content term at all
            logits = torch.zeros(b, h, lq, kv_in.shape[1], device=q.device, dtype=q.dtype)
        else:
            logits = (q @ k.transpose(-2, -1)) / (hd ** 0.5)

        if self.bias is not None and dist_bins is not None:
            logits = logits + self.bias(dist_bins).squeeze(-1).unsqueeze(1)

        # -inf, not ProtAttBA's 1e-10: padding must not keep softmax mass (see module docstring)
        logits = logits.masked_fill(~kv_mask.bool()[:, None, None, :], float("-inf"))
        attn = self.drop(torch.softmax(logits, dim=-1))
        self.last_attn = attn.detach()

        out = (attn @ v).transpose(1, 2).reshape(b, lq, h * hd)
        out = self.ln_out(out)
        return out + self.ln_ffn(F.relu(self.ffn(out)))


class Branch(nn.Module):
    """One forward pass over an (antibody, antigen) pair. Shared by ITW and MUT."""

    def __init__(self, cfg: PerturbConfig):
        super().__init__()
        self.cfg = cfg
        d, s = cfg.model_dim, cfg.struct_dim

        self.proj = nn.Linear(cfg.seq_dim, s)                      # step 1: W_p 256 -> 128
        self.g1 = nn.Linear(s, s); self.b1 = nn.Linear(s, s)       # step 3 FiLM, 128 -> 128
        if cfg.film_reversed:
            self.g2 = nn.Linear(cfg.seq_dim, s); self.b2 = nn.Linear(cfg.seq_dim, s)
            self.lift = nn.Linear(s, d)
        else:
            self.g2 = nn.Linear(s, cfg.seq_dim); self.b2 = nn.Linear(s, cfg.seq_dim)
            self.lift = nn.Linear(cfg.seq_dim, d) if cfg.seq_dim != d else nn.Identity()
        if not cfg.struct_film:
            self.concat_mix = nn.Linear(cfg.seq_dim + s, cfg.seq_dim)

        # FiLMs start as the identity: (1 + 0) * x + 0
        for lin in (self.g1, self.b1, self.g2, self.b2):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

        if cfg.use_blosum:
            n_in = 40 if cfg.blosum_mode == "blosum" else 40
            self.mut_mlp = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 32))
            self.mut_mix = nn.Linear(cfg.seq_dim + 32, cfg.seq_dim)

        self.gate_ab, self.gate_ag = LocalGate(d), LocalGate(d)
        self.attn_ab, self.attn_ag = CrossAttention(cfg), CrossAttention(cfg)
        self.pool_ab, self.pool_ag = AttnPool(d), AttnPool(d)

    def encode(self, seq, struct, delta, blosum, site_mask):
        """Steps 1-5 for one chain."""
        cfg = self.cfg
        t = struct
        if cfg.delta_film and delta is not None:
            active = (delta.abs().sum(-1, keepdim=True) > 0).float()
            t = torch.where(active > 0, (1 + self.g1(delta)) * t + self.b1(delta), t)

        if cfg.struct_film:
            if cfg.film_reversed:
                h = (1 + self.g2(seq)) * t + self.b2(seq)
                h = self.lift(h)
            else:
                h = (1 + self.g2(t)) * seq + self.b2(t)
                h = self.lift(h)
        else:
            h = self.lift(self.concat_mix(torch.cat([seq, t], dim=-1)))

        if cfg.use_blosum and blosum is not None:
            m = self.mut_mlp(blosum) * site_mask.unsqueeze(-1)
            h_seq = h if h.shape[-1] == cfg.seq_dim else h
            h = self.mut_mix(torch.cat([h_seq, m], dim=-1)) if h.shape[-1] == cfg.seq_dim else h
        return h

    def forward(self, batch, mutated: bool):
        cfg = self.cfg
        out = {}
        feats = []
        for side, other in (("ab", "ag"), ("ag", "ab")):
            seq = batch[f"seq_{side}_mt" if (mutated and cfg.full_mutant_tokens)
                        else f"seq_{side}_wt"]
            struct = batch[f"struct_{side}"]
            if not cfg.use_structure:
                struct = torch.zeros_like(struct)
            elif cfg.shuffle_structure:
                idx = torch.randperm(struct.shape[1], device=struct.device)
                struct = struct[:, idx]

            if mutated and cfg.use_delta and not cfg.full_mutant_tokens:
                d_full = self.proj(batch[f"seq_{side}_mt"]) - self.proj(seq)
                if cfg.delta_scope == "site":
                    d_full = d_full * batch[f"site_{side}"].unsqueeze(-1)
                delta = d_full
            else:
                delta = torch.zeros(seq.shape[0], seq.shape[1], cfg.struct_dim,
                                    device=seq.device, dtype=seq.dtype)

            bl = batch[f"blosum_{side}_mt" if mutated else f"blosum_{side}_wt"]
            out[side] = self.encode(seq, struct, delta, bl, batch[f"site_{side}"])
            feats.append(side)

        h_ab = self.gate_ab(out["ab"], batch["mask_ab"])
        h_ag = self.gate_ag(out["ag"], batch["mask_ag"])

        if cfg.cross_chain:
            a = self.attn_ab(h_ab, h_ag, batch["mask_ag"], batch.get("dist_bins"))
            g = self.attn_ag(h_ag, h_ab, batch["mask_ab"],
                             batch["dist_bins"].transpose(1, 2) if batch.get("dist_bins") is not None else None)
        else:
            a = self.attn_ab(h_ab, h_ab, batch["mask_ab"], None)
            g = self.attn_ag(h_ag, h_ag, batch["mask_ag"], None)

        if cfg.site_pool:
            pool_ab = batch["mask_ab"] * batch["near_ab"]
            pool_ag = batch["mask_ag"] * batch["near_ag"]
        else:
            pool_ab, pool_ag = batch["mask_ab"], batch["mask_ag"]
        return torch.cat([self.pool_ab(a, pool_ab), self.pool_ag(g, pool_ag)], dim=-1)


class PerturbModel(nn.Module):
    """ITW and MUT through one shared branch; head on the difference."""

    def __init__(self, cfg: PerturbConfig | None = None):
        super().__init__()
        self.cfg = cfg or PerturbConfig()
        self.branch = Branch(self.cfg)
        width = self.cfg.model_dim * 2
        n_in = width * 2 if self.cfg.concat_head else width
        # Bias-free, so a null edit maps to exactly zero (see module docstring).
        self.head = nn.Sequential(
            nn.Linear(n_in, 128, bias=False), nn.ReLU(), nn.Dropout(self.cfg.dropout),
            nn.Linear(128, 1, bias=False),
        )

    def forward(self, batch) -> torch.Tensor:
        f_mut = self.branch(batch, mutated=True)
        if not self.cfg.two_branch:
            return self.head(f_mut).squeeze(-1)
        f_itw = self.branch(batch, mutated=False)
        z = torch.cat([f_mut, f_itw], -1) if self.cfg.concat_head else (f_mut - f_itw)
        return self.head(z).squeeze(-1)


def distance_bins(dist: torch.Tensor) -> torch.Tensor:
    """Ca distances to bin indices: 16 equal bins over [0, 20) A, plus one bin for >= 20 A.

    The interior edges are ``linspace(0, 20, 17)[1:]``, i.e. 16 boundaries ending at 20, so
    ``bucketize`` returns 0..16 and only distances at or beyond 20 A reach the final bin.

    Dropping the trailing edge instead -- ``[1:-1]``, which is what this did first -- leaves 15
    boundaries, caps the output at 15, and silently merges "beyond 20 A" into the 18.75-20 A
    bin. The bias would then have no separate parameter for non-contacts at all, which is the
    one thing the last bin exists for. Caught by the unit test, not by anything raising.
    """
    edges = torch.linspace(0, DIST_MAX, DIST_BINS + 1, device=dist.device)[1:]
    return torch.bucketize(dist, edges).clamp(max=N_DIST_BINS - 1)
