"""Parameter-economy perturbation model, ~51k trainable parameters.

Same idea as ``model.py`` -- two weight-shared branches, the mutation as a localized edit, the
head on the difference -- rebuilt against a parameter budget roughly 16x smaller. The point of
the budget is that with 940 training rows, capacity is not the binding constraint and a large
head mostly buys memorisation surface.

Where the parameters went, and why each choice is cheap rather than arbitrary:

* **Low-rank projection instead of a dense one.** Each modality goes 128 -> 16 -> 64 with no
  nonlinearity between the factors, which is 3,152 parameters against 8,256 for a dense
  128->64. The rank-16 bottleneck is an ablation (``rank_8`` / ``rank_32`` / ``dense_proj``).
* **One FiLM, not two.** Only the sequence delta modulates structure. The reverse direction
  costs another 8.3k and is the ``struct_film_on`` ablation rather than a default.
* **One attention block, shared across both directions.** ``W_Q, W_K, W_V, W_O`` are the same
  tensors for antibody-attends-antigen and antigen-attends-antibody. Unsharing them is the
  ``separate_attn_dirs`` ablation (+16.6k). There is no FFN.
* **Masked mean pooling, not a learned pooler.** Zero parameters, and required test (e) pins
  that it is invariant to how far the batch happened to be padded.

Three properties are structural, not emergent, and the tests pin all three:

1. A null edit gives **exactly** zero: both branches compute identical tensors, the difference
   is the zero vector, and the head is bias-free so it maps zero to zero.
2. The ITW branch never sees the structure FiLM: step 2 is gated on ``delta != 0`` and ITW's
   delta is identically zero.
3. Both branches receive the same crop, mask, chain ids and residue indices, because all four
   come from the wild-type structure and are passed in once.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

DIST_BINS = 16
DIST_MAX = 20.0
N_DIST_BINS = DIST_BINS + 1
N_CHAIN_TYPES = 3


@dataclass
class PerturbV2Config:
    """Widths are fixed by section 1 and are not tuned; the flags are the ablations."""

    pca_dim: int = 128        # per modality, after the fold-local PCA
    rank: int = 16            # low-rank bottleneck
    width: int = 64           # model width
    n_heads: int = 2
    dropout: float = 0.2

    delta_scope: str = "site"
    use_blosum: bool = True
    blosum_mode: str = "blosum"      # {blosum, learned}
    dist_bias: bool = True

    delta_film: bool = True          # step 2
    struct_film_on: bool = False     # second FiLM, structure -> sequence
    film_as_concat: bool = False     # step 2 as concat + linear
    fuse_gelu: bool = True           # step 3 nonlinearity
    use_structure: bool = True
    shuffle_structure: bool = False
    use_delta: bool = True
    full_mutant_tokens: bool = False
    two_branch: bool = True
    concat_head: bool = False
    chain_embedding: bool = True
    learned_diag: bool = True
    dense_proj: bool = False
    cross_chain: bool = True
    dist_bias_only: bool = False
    separate_attn_dirs: bool = False
    with_ffn: bool = False
    pooled_hadamard: bool = False
    pairwise_hadamard: bool = False
    array_index_rope: bool = False

    #: How the token sequence collapses to one vector. ``mean`` is section 1's unweighted
    #: masked mean over the whole crop, which costs no parameters and dilutes the edit: on
    #: our rows the mutated residue's ESM embedding moves 3.23 while off-site tokens move
    #: 0.33, but averaged over a ~42-token crop the difference between the two pooled
    #: vectors is only 2.5% of the pooled vector itself (p10-p90 1.6-4.8%). The head then
    #: has to find ddG in a 2.5% perturbation whose remaining 97.5% is complex identity.
    #: ``site`` pools over the mutated residues instead -- after cross-attention those
    #: tokens already carry interface context, so this is not the same as discarding it --
    #: and ``site_mean`` concatenates both, keeping the global view at the cost of doubling
    #: the head's input. A side with no mutation has no site and falls back to the crop.
    pool: str = "mean"               # {mean, site, site_mean}

    #: Input-side regularisation, both off by default and both training-only. The model
    #: carries 85 parameters per training row and reaches train R2 0.83 against test R2
    #: 0.036 on fold 4, so dropout 0.2 in two places and weight decay 1e-2 are not holding
    #: it. These are the two knobs the earlier fusion sweeps had and the v2 rewrite dropped.
    #:
    #: Both are sampled ONCE PER ROW and applied identically to the ITW and MUT branches.
    #: Sampling them independently would inject noise straight into the difference the head
    #: reads, which is only 2.5% of the pooled vector to begin with -- it would regularise
    #: by destroying the signal, and it would break the null-edit-is-zero property during
    #: training. Shared, the difference is untouched and the property still holds exactly.
    input_noise: float = 0.0         # Gaussian noise, as a fraction of each channel's sd
    feature_dropout: float = 0.0     # probability of zeroing a whole PCA channel

    #: Weight init for the FiLM that carries the mutation. Zero makes the FiLM an exact
    #: identity at step 0, so the ESM delta contributes NOTHING to the representation until
    #: gradient descent lifts it off zero -- while ``branch.mut`` ships with ordinary init
    #: and injects BLOSUM from the very first step. Measured at initialisation: delta
    #: contributes 0.000000 and BLOSUM 0.048477. The delta path has to climb out of zero
    #: against a path already explaining variance, which is the likeliest reason gamma
    #: carries 41x less gradient per parameter than attention while mut carries 2x more.
    #:
    #: The BIAS stays zero whatever this is set to, so gamma(0) == 0 and a null edit still
    #: leaves both branches identical -- required test (a) is unaffected.
    film_init: float = 0.0

    #: Draw the SAME dropout mask in both branches. The head reads f_mut - f_itw, and with
    #: independent masks the attention dropout disagrees between the two calls: on a null
    #: edit, where the true difference is exactly zero, the branches still differ by 0.42
    #: against a real edit of 0.60. That is 71% as much noise as signal, injected into a
    #: difference that is already only 2.5% of the pooled vector -- and it is present in
    #: every step of every run trained so far. Sharing the mask regularises each branch
    #: exactly as before while leaving their difference clean.
    shared_branch_dropout: bool = True

    @property
    def head_dim(self) -> int:
        return self.width // self.n_heads


def distance_bins(dist: torch.Tensor) -> torch.Tensor:
    """16 equal bins over [0, 20) A plus one bin for >= 20 A.

    The interior edges are ``linspace(0, 20, 17)[1:]``, i.e. 16 boundaries ending at 20, so
    only distances at or beyond 20 A reach the final bin. Dropping the trailing edge instead
    leaves 15 boundaries and silently merges "beyond 20 A" into the 18.75-20 A bin, which
    would leave the bias no parameter for a non-contact at all.
    """
    edges = torch.linspace(0, DIST_MAX, DIST_BINS + 1, device=dist.device)[1:]
    return torch.bucketize(dist, edges).clamp(max=N_DIST_BINS - 1)


class Reduce(nn.Module):
    """PCA output -> width. Learned diagonal, then a low-rank (or dense) projection."""

    def __init__(self, cfg: PerturbV2Config):
        super().__init__()
        self.cfg = cfg
        if cfg.learned_diag:
            self.scale = nn.Parameter(torch.ones(cfg.pca_dim))
            self.shift = nn.Parameter(torch.zeros(cfg.pca_dim))
        if cfg.dense_proj:
            self.proj = nn.Linear(cfg.pca_dim, cfg.width)
        else:
            self.down = nn.Linear(cfg.pca_dim, cfg.rank, bias=False)
            self.up = nn.Linear(cfg.rank, cfg.width)

    def forward(self, x):
        if self.cfg.learned_diag:
            x = x * self.scale + self.shift
        return self.proj(x) if self.cfg.dense_proj else self.up(self.down(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, pos: torch.Tensor):
        """``pos`` is (B, L) original residue indices, so a crop does not renumber anything."""
        f = pos.float().unsqueeze(-1) * self.inv_freq
        cos, sin = f.cos(), f.sin()
        return torch.cat([cos, cos], -1).unsqueeze(1), torch.cat([sin, sin], -1).unsqueeze(1)


def _rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-b, a], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class CrossAttention(nn.Module):
    """One layer, pre-LN, residual, distance bias on the logits. No FFN unless asked."""

    def __init__(self, cfg: PerturbV2Config):
        super().__init__()
        w = cfg.width
        self.cfg = cfg
        self.ln_q, self.ln_kv = nn.LayerNorm(w), nn.LayerNorm(w)
        self.q, self.k, self.v, self.o = (nn.Linear(w, w, bias=False) for _ in range(4))
        self.rope = RotaryEmbedding(cfg.head_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.bias = nn.Embedding(N_DIST_BINS, 1) if cfg.dist_bias else None
        if self.bias is not None:
            nn.init.zeros_(self.bias.weight)
        self.ffn = (nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))
                    if cfg.with_ffn else None)

    def forward(self, q_in, kv_in, kv_mask, q_pos, kv_pos, dist_bins=None):
        b, lq, w = q_in.shape
        h, hd = self.cfg.n_heads, self.cfg.head_dim
        qh = self.q(self.ln_q(q_in)).view(b, lq, h, hd).transpose(1, 2)
        kv = self.ln_kv(kv_in)
        kh = self.k(kv).view(b, kv_in.shape[1], h, hd).transpose(1, 2)
        vh = self.v(kv).view(b, kv_in.shape[1], h, hd).transpose(1, 2)

        cq, sq = self.rope(q_pos)
        ck, sk = self.rope(kv_pos)
        qh, kh = apply_rope(qh, cq, sq), apply_rope(kh, ck, sk)

        if self.cfg.dist_bias_only:
            logits = torch.zeros(b, h, lq, kv_in.shape[1], device=qh.device, dtype=qh.dtype)
        else:
            logits = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
        if self.bias is not None and dist_bins is not None:
            logits = logits + self.bias(dist_bins).squeeze(-1).unsqueeze(1)

        # -inf, never a small finite number: a padded key must carry no softmax mass, or a
        # prediction depends on how wide its batch happened to be padded.
        logits = logits.masked_fill(~kv_mask.bool()[:, None, None, :], float("-inf"))
        attn = self.drop(torch.softmax(logits, dim=-1))
        self.last_attn = attn.detach()
        out = self.o((attn @ vh).transpose(1, 2).reshape(b, lq, w))
        out = q_in + out
        return out + self.ffn(out) if self.ffn is not None else out


class BranchV2(nn.Module):
    """Steps 0-6 for one (antibody, antigen) crop. Shared by ITW and MUT."""

    def __init__(self, cfg: PerturbV2Config):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.red_seq, self.red_str = Reduce(cfg), Reduce(cfg)
        self.chain = nn.Embedding(N_CHAIN_TYPES, w) if cfg.chain_embedding else None
        if self.chain is not None:
            nn.init.zeros_(self.chain.weight)

        if cfg.film_as_concat:
            self.film_concat = nn.Linear(2 * w, w)
        else:
            self.gamma, self.beta = nn.Linear(w, w), nn.Linear(w, w)
            for lin in (self.gamma, self.beta):
                if cfg.film_init > 0:
                    nn.init.normal_(lin.weight, std=cfg.film_init)
                else:
                    nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)   # always: keeps gamma(0) == 0, so test (a) holds
        if cfg.struct_film_on:
            self.gamma2, self.beta2 = nn.Linear(w, w), nn.Linear(w, w)
            for lin in (self.gamma2, self.beta2):
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

        self.fuse = nn.Linear(2 * w, w)
        if cfg.use_blosum:
            self.mut = (nn.Linear(40, w) if cfg.blosum_mode == "blosum"
                        else nn.Embedding(20 * 20, w))
        self.attn = CrossAttention(cfg)
        self.attn_rev = CrossAttention(cfg) if cfg.separate_attn_dirs else None

    def encode(self, seq_pca, str_pca, delta, blosum, site, chain_type):
        cfg = self.cfg
        seq = self.red_seq(seq_pca)
        t = self.red_str(str_pca)
        if self.chain is not None:
            seq = seq + self.chain(chain_type)

        if cfg.film_as_concat:
            t = self.film_concat(torch.cat([t, delta], dim=-1))
        elif cfg.delta_film:
            active = (delta.abs().sum(-1, keepdim=True) > 0).float()
            t = torch.where(active > 0, (1 + self.gamma(delta)) * t + self.beta(delta), t)
        if cfg.struct_film_on:
            seq = (1 + self.gamma2(t)) * seq + self.beta2(t)

        h = self.fuse(torch.cat([seq, t], dim=-1))
        if cfg.fuse_gelu:
            h = F.gelu(h)
        if cfg.use_blosum and blosum is not None:
            add = (self.mut(blosum) if cfg.blosum_mode == "blosum"
                   else self.mut(blosum.long().argmax(-1)))
            h = h + add * site.unsqueeze(-1)
        return h

    def forward(self, batch, mutated: bool, perturb: dict | None = None):
        cfg = self.cfg
        feats = {}

        def px(x, key):
            """Apply this row's shared noise and channel mask; identical in both branches.

            Noise is added before the mask so a dropped channel is exactly zero rather than
            pure noise. Both operations commute with the WT/MUT difference -- the additive
            noise cancels and the multiplicative mask factors out -- so the edit the head
            reads is masked but never contaminated.
            """
            if perturb is None:
                return x
            n, m = perturb[key]
            if n is not None:
                x = x + n
            return x if m is None else x * m

        for side in ("ab", "ag"):
            seq_key = (f"seq_{side}_mt" if (mutated and cfg.full_mutant_tokens)
                       else f"seq_{side}_wt")
            seq_pca = px(batch[seq_key], f"seq_{side}")
            str_pca = px(batch[f"struct_{side}"], f"struct_{side}")
            if not cfg.use_structure:
                str_pca = torch.zeros_like(str_pca)
            elif cfg.shuffle_structure:
                str_pca = str_pca[:, torch.randperm(str_pca.shape[1], device=str_pca.device)]

            if mutated and cfg.use_delta and not cfg.full_mutant_tokens:
                d = (self.red_seq(px(batch[f"seq_{side}_mt"], f"seq_{side}"))
                     - self.red_seq(px(batch[f"seq_{side}_wt"], f"seq_{side}")))
                if cfg.delta_scope == "site":
                    d = d * batch[f"site_{side}"].unsqueeze(-1)
            else:
                d = torch.zeros(seq_pca.shape[0], seq_pca.shape[1], cfg.width,
                                device=seq_pca.device, dtype=seq_pca.dtype)

            bl = batch[f"blosum_{side}_mt" if mutated else f"blosum_{side}_wt"]
            feats[side] = self.encode(seq_pca, str_pca, d, bl,
                                      batch[f"site_{side}"], batch[f"chain_{side}"])

        h_ab, h_ag = feats["ab"], feats["ag"]
        pos_ab = (torch.arange(h_ab.shape[1], device=h_ab.device).expand(h_ab.shape[0], -1)
                  if cfg.array_index_rope else batch["res_ab"])
        pos_ag = (torch.arange(h_ag.shape[1], device=h_ag.device).expand(h_ag.shape[0], -1)
                  if cfg.array_index_rope else batch["res_ag"])
        db = batch.get("dist_bins")

        if cfg.pairwise_hadamard:
            # cross-attention minus the learned alignment: fixed contact weights, 0 params
            d = batch["dist"].clamp(max=40.0)
            wgt = torch.exp(-(d ** 2) / (2 * 6.0 ** 2)) * batch["mask_ag"].unsqueeze(1)
            wgt = wgt / wgt.sum(-1, keepdim=True).clamp(min=1e-6)
            a = h_ab * (wgt @ h_ag)
            wgt_t = wgt.transpose(1, 2)
            wgt_t = wgt_t / wgt_t.sum(-1, keepdim=True).clamp(min=1e-6)
            g = h_ag * (wgt_t @ h_ab)
        elif cfg.pooled_hadamard:
            a, g = h_ab, h_ag
        elif cfg.cross_chain:
            # both updates from the PRE-update tensors, then applied
            rev = self.attn_rev or self.attn
            a = self.attn(h_ab, h_ag, batch["mask_ag"], pos_ab, pos_ag, db)
            g = rev(h_ag, h_ab, batch["mask_ab"], pos_ag, pos_ab,
                    db.transpose(1, 2) if db is not None else None)
        else:
            a = self.attn(h_ab, h_ab, batch["mask_ab"], pos_ab, pos_ab, None)
            g = self.attn(h_ag, h_ag, batch["mask_ag"], pos_ag, pos_ag, None)

        f_ab = self.pool(a, batch["mask_ab"], batch["site_ab"])
        f_ag = self.pool(g, batch["mask_ag"], batch["site_ag"])
        if cfg.pooled_hadamard:
            return torch.cat([f_ab * f_ag, f_ab + f_ag], dim=-1)
        return torch.cat([f_ab, f_ag], dim=-1)

    def pool(self, x: torch.Tensor, mask: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        """Collapse one side's tokens to one vector; see ``PerturbV2Config.pool``."""
        mode = self.cfg.pool
        if mode == "mean":
            return masked_mean(x, mask)
        # A side with no mutation on it has an all-zero site mask. Pooling over nothing
        # would divide by the clamp and return zeros, which silently deletes that whole
        # side for every antibody-only or antigen-only row -- most of the dataset. Fall
        # back to the crop for exactly those rows, per row, not per batch.
        w = mask * site
        w = torch.where(w.sum(1, keepdim=True) < 0.5, mask, w)
        f_site = masked_mean(x, w)
        if mode == "site":
            return f_site
        if mode == "site_mean":
            return torch.cat([f_site, masked_mean(x, mask)], dim=-1)
        raise ValueError(f"unknown pool {mode!r}")


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real tokens only, so padding width cannot change the answer."""
    m = mask.unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class PerturbV2(nn.Module):
    def __init__(self, cfg: PerturbV2Config | None = None):
        super().__init__()
        self.cfg = cfg or PerturbV2Config()
        self.branch = BranchV2(self.cfg)
        # site_mean returns two pooled vectors per side rather than one
        f = self.cfg.width * 2 * (2 if self.cfg.pool == "site_mean" else 1)
        n_in = f * 2 if self.cfg.concat_head else f
        # bias-free, so a null edit maps to exactly zero
        self.head = nn.Sequential(
            nn.Linear(n_in, self.cfg.width, bias=False), nn.GELU(),
            nn.Dropout(self.cfg.dropout), nn.Linear(self.cfg.width, 1, bias=False))

    def make_perturb(self, batch) -> dict | None:
        """One sample of input noise and channel dropout per row, shared by both branches.

        Returns None outside training or when both are off, so inference and every existing
        test are bit-identical to before.
        """
        cfg = self.cfg
        if not self.training or (cfg.input_noise <= 0 and cfg.feature_dropout <= 0):
            return None
        out = {}
        for key in ("seq_ab", "seq_ag", "struct_ab", "struct_ag"):
            ref = batch[f"{key}_wt"] if key.startswith("seq") else batch[key]
            n = None
            if cfg.input_noise > 0:
                # Relative to each channel's own spread. These are PCA components, so their
                # variances fall off by orders of magnitude across the 256 dimensions; a
                # single absolute sd would drown the leading components' neighbours and do
                # nothing at all to the tail.
                sd = ref.std(dim=(0, 1), keepdim=True)
                n = torch.randn_like(ref) * (cfg.input_noise * sd)
            m = None
            if cfg.feature_dropout > 0:
                # whole PCA channels, shared across tokens, inverted so the scale is kept
                keep = torch.rand(ref.shape[0], 1, ref.shape[2], device=ref.device,
                                  dtype=ref.dtype) >= cfg.feature_dropout
                m = keep.to(ref.dtype) / (1.0 - cfg.feature_dropout)
            out[key] = (n, m)
        return out

    def forward(self, batch) -> torch.Tensor:
        p = self.make_perturb(batch)
        share = self.training and self.cfg.shared_branch_dropout
        # Rewinding the generator is what makes the two branches draw the same masks. It
        # works because both calls issue the same sequence of random ops -- same modules,
        # same shapes, and nothing stochastic is conditional on `mutated`. The null-edit
        # test below runs in train mode precisely so that stops being an assumption.
        state = torch.get_rng_state() if share else None
        dev = (torch.cuda.get_rng_state_all()
               if share and torch.cuda.is_available() else None)

        f_mut = self.branch(batch, mutated=True, perturb=p)
        if not self.cfg.two_branch:
            return self.head(f_mut).squeeze(-1)
        if state is not None:
            torch.set_rng_state(state)
            if dev is not None:
                torch.cuda.set_rng_state_all(dev)
        f_itw = self.branch(batch, mutated=False, perturb=p)
        z = torch.cat([f_mut, f_itw], -1) if self.cfg.concat_head else (f_mut - f_itw)
        return self.head(z).squeeze(-1)
