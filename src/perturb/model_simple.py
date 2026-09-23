"""The perturbation model with everything that did not earn its place taken out.

The ladder said the architecture responds to removing shortcuts, not to adding machinery:
dropping BLOSUM gained more per-complex Pearson (+0.039) than clipping, the dropout fix,
site pooling and regularisation combined, and it did so on fewer parameters. This is that
observation taken to its conclusion. Gone: cross-attention, RoPE, the learned distance
bias, the chain embedding, the low-rank Reduce, BLOSUM. What is left is the edit.

    seq_pca  (B, L, 128) --LayerNorm--> Linear --> x_seq
    str_pca  (B, L, 128) --LayerNorm--> Linear --> x_str

    delta = x_seq(mutant) - x_seq(wild type)          the sequence modality
    t     = (1 + gamma(delta)) * x_str + beta(delta)   structure, modulated by the edit

    pool over the MUTATED residues only
    concat the two modalities, both sides
    MLP -> ddG

Two decisions worth stating, because neither is forced by the sketch.

**The structure modality enters as ``t - x_str``, not ``t``.** What the FiLM did to the
structure is a function of the edit; the structure itself is a function of the complex. On
940 rows over 53 complexes the model reaches train R2 0.83 against test R2 0.036, and it
gets there by learning complexes, so anything handed to the head that identifies the
complex without describing the edit is a liability. Passing the difference means every
input to the head is zero for a null edit, and with a bias-free head the prediction is then
exactly zero -- the same property the two-branch model had, kept without the second branch.

**Pooling is over the mutated residues only.** ``site_mean`` was the best configuration in
the ladder (+0.172) and plain masked-mean over a ~42-token crop leaves the edit at 2.5% of
the pooled vector. Here there is no attention to carry information from the site to the
rest of the crop, so pooling anywhere else would mostly average tokens the edit never
touched. A side with no mutation on it has no site and contributes zeros.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SimpleConfig:
    pca_dim: int = 128       # per modality, after the fold-local PCA
    width: int = 64          # d, the shared per-token width
    hidden: int = 64         # MLP hidden width
    dropout: float = 0.2
    use_structure: bool = True
    film: bool = True        # False: concatenate delta and structure instead of modulating
    input_noise: float = 0.0
    feature_dropout: float = 0.0
    #: gamma/beta init. Zero makes the FiLM an identity at step 0, which in the previous
    #: model meant the ESM delta contributed nothing until gradient descent lifted it off
    #: zero. The bias stays zero regardless, so gamma(0) == 0 and a null edit is still zero.
    film_init: float = 0.02


class Project(nn.Module):
    """LayerNorm then Linear, per token.

    The LayerNorm is the reason this can use one width for both modalities: the PCA
    components of ESM and of ProteinMPNN differ in scale by orders of magnitude, and
    normalising per token puts them on comparable footing before either is projected.
    """

    def __init__(self, cfg: SimpleConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.pca_dim)
        self.lin = nn.Linear(cfg.pca_dim, cfg.width)

    def forward(self, x):
        return self.lin(self.norm(x))


def site_mean(x: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
    """Mean over the mutated residues. A side with none contributes exactly zeros."""
    w = site.unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp(min=1.0)


class PerturbSimple(nn.Module):
    def __init__(self, cfg: SimpleConfig | None = None):
        super().__init__()
        self.cfg = cfg or SimpleConfig()
        c = self.cfg
        self.proj_seq = Project(c)
        self.proj_str = Project(c) if c.use_structure else None

        if c.use_structure and c.film:
            self.gamma, self.beta = nn.Linear(c.width, c.width), nn.Linear(c.width, c.width)
            for lin in (self.gamma, self.beta):
                if c.film_init > 0:
                    nn.init.normal_(lin.weight, std=c.film_init)
                else:
                    nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)      # keeps gamma(0) == 0, so a null edit is zero

        # two modalities per side, two sides; without structure it is the delta alone
        per_side = c.width * (2 if c.use_structure else 1)
        # bias-free throughout, so an all-zero input maps to exactly zero
        self.mlp = nn.Sequential(
            nn.Linear(per_side * 2, c.hidden, bias=False), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(c.hidden, 1, bias=False))

    def perturb(self, batch):
        """One sample of input noise and channel dropout per row, shared by WT and MUT.

        Sampling separately would inject noise straight into the delta, which is the only
        thing this model reads. Additive noise cancels in the difference and a channel mask
        factors out of it, so both regularise the representation without touching the edit.
        """
        c = self.cfg
        if not self.training or (c.input_noise <= 0 and c.feature_dropout <= 0):
            return None
        out = {}
        for key in ("seq_ab", "seq_ag", "struct_ab", "struct_ag"):
            ref = batch[f"{key}_wt"] if key.startswith("seq") else batch[key]
            n = None
            if c.input_noise > 0:
                sd = ref.std(dim=(0, 1), keepdim=True)
                n = torch.randn_like(ref) * (c.input_noise * sd)
            m = None
            if c.feature_dropout > 0:
                keep = torch.rand(ref.shape[0], 1, ref.shape[2], device=ref.device,
                                  dtype=ref.dtype) >= c.feature_dropout
                m = keep.to(ref.dtype) / (1.0 - c.feature_dropout)
            out[key] = (n, m)
        return out

    def side(self, batch, side: str, p) -> torch.Tensor:
        c = self.cfg

        def px(x, key):
            if p is None:
                return x
            n, m = p[key]
            if n is not None:
                x = x + n
            return x if m is None else x * m

        site = batch[f"site_{side}"]
        seq_wt = px(batch[f"seq_{side}_wt"], f"seq_{side}")
        seq_mt = px(batch[f"seq_{side}_mt"], f"seq_{side}")
        delta = self.proj_seq(seq_mt) - self.proj_seq(seq_wt)

        if not c.use_structure:
            return site_mean(delta, site)

        x_str = self.proj_str(px(batch[f"struct_{side}"], f"struct_{side}"))
        if c.film:
            t = (1 + self.gamma(delta)) * x_str + self.beta(delta)
        else:
            t = x_str * delta                       # a plain interaction, for the ablation
        # what the EDIT did to the structure, not the structure itself: see the docstring
        struct_term = t - x_str if c.film else t
        return torch.cat([site_mean(delta, site), site_mean(struct_term, site)], dim=-1)

    def forward(self, batch) -> torch.Tensor:
        p = self.perturb(batch)
        z = torch.cat([self.side(batch, "ab", p), self.side(batch, "ag", p)], dim=-1)
        return self.mlp(z).squeeze(-1)


@dataclass
class MLPConfig:
    """The shortest path from ESM to ddG, for the floor it sets.

    Pool the delta at the mutated residues and regress. No per-token projection, no
    structure, no FiLM, no attention, no crop geometry -- one vector per row.

    It exists to price everything else. If the full perturbation model does not clearly
    beat this, the machinery between the embedding and the answer is not earning its place.
    """

    pca_dim: int = 256       # per modality, after the fold-local PCA
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.2
    input_noise: float = 0.0
    feature_dropout: float = 0.0
    #: Concatenate the project's handcrafted columns (chem + ProteinMPNN log-likelihood
    #: ratios) to the pooled delta. The forest that beats every net here uses these, so
    #: giving them to the net separates "the features are better" from "the model class is
    #: better". They are standardised with fold-local statistics before they are seen,
    #: because d_volume runs to hundreds while d_charge is a small integer, and a single
    #: LayerNorm over a vector that is 512 embedding dimensions and 26 chemistry columns
    #: would normalise the chemistry into irrelevance.
    chem_dim: int = 0


class PerturbMLP(nn.Module):
    def __init__(self, cfg: MLPConfig | None = None):
        super().__init__()
        c = self.cfg = cfg or MLPConfig()
        # Both sides concatenated. A side with no mutation contributes zeros, so the head
        # can tell "no antigen-side mutation" from "an antigen-side mutation of zero size".
        d = c.pca_dim * 2
        # The embedding half is normalised on its own; the chemistry arrives already
        # standardised and is concatenated after, so neither rescales the other.
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        d += c.chem_dim
        blocks: list[nn.Module] = []
        for _ in range(c.layers):
            # bias-free, so an all-zero input (a null edit) maps to exactly zero
            blocks += [nn.Linear(d, c.hidden, bias=False), nn.GELU(), nn.Dropout(c.dropout)]
            d = c.hidden
        blocks += [nn.Linear(d, 1, bias=False)]
        self.net = nn.Sequential(*blocks)

    def forward(self, batch) -> torch.Tensor:
        c = self.cfg
        parts = []
        for side in ("ab", "ag"):
            wt, mt = batch[f"seq_{side}_wt"], batch[f"seq_{side}_mt"]
            if self.training and (c.input_noise > 0 or c.feature_dropout > 0):
                # shared across WT and MUT so it cancels in the delta, as elsewhere
                if c.input_noise > 0:
                    n = torch.randn_like(wt) * (c.input_noise * wt.std(dim=(0, 1),
                                                                      keepdim=True))
                    wt, mt = wt + n, mt + n
                if c.feature_dropout > 0:
                    keep = (torch.rand(wt.shape[0], 1, wt.shape[2], device=wt.device,
                                       dtype=wt.dtype) >= c.feature_dropout)
                    m = keep.to(wt.dtype) / (1.0 - c.feature_dropout)
                    wt, mt = wt * m, mt * m
            parts.append(site_mean(mt - wt, batch[f"site_{side}"]))
        z = self.norm(torch.cat(parts, dim=-1))
        if c.chem_dim:
            z = torch.cat([z, batch["chem"]], dim=-1)
        return self.net(z).squeeze(-1)


@dataclass
class TwoTowerConfig:
    """Process the wild-type and mutant representations, then combine them.

    Everything so far subtracted first and processed the difference. This does the reverse:
    each sequence is pooled and passed through a shared processing layer on its own, and
    only then are the two brought together -- by concatenation, or by subtraction as the
    ablation.

    The two arms are not equivalent, and the difference is the thing this session kept
    running into. ``subtract`` can only see what the edit CHANGED: identical inputs give
    exactly zero, so the complex cannot be identified through it. ``concat`` hands the head
    the wild-type vector as well, which is constant across every mutation of a complex --
    and predicting a complex's mean scores pooled +0.672 here, better than any model in the
    repo, while scoring 0.000 per complex. So concat is expected to look better on pooled
    Pearson and that will not mean it is better.
    """

    pca_dim: int = 128       # per side, after the fold-local PCA
    proc: int = 128          # the shared processing layer
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    combine: str = "concat"  # {concat, subtract}
    chem_dim: int = 0
    input_noise: float = 0.0
    feature_dropout: float = 0.0


class PerturbTwoTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig | None = None):
        super().__init__()
        c = self.cfg = cfg or TwoTowerConfig()
        if c.combine not in ("concat", "subtract"):
            raise ValueError(f"unknown combine {c.combine!r}")
        # Shared between the two towers: the same function must be applied to both, or the
        # subtraction compares two different things and the concat arm can tell them apart
        # by which tower they came from.
        self.norm = nn.LayerNorm(c.pca_dim * 2, elementwise_affine=False)
        self.proc = nn.Linear(c.pca_dim * 2, c.proc, bias=False)

        d = c.proc * (2 if c.combine == "concat" else 1) + c.chem_dim
        blocks: list[nn.Module] = []
        for _ in range(c.layers):
            blocks += [nn.Linear(d, c.hidden, bias=False), nn.GELU(), nn.Dropout(c.dropout)]
            d = c.hidden
        blocks += [nn.Linear(d, 1, bias=False)]
        self.mlp = nn.Sequential(*blocks)

    def perturb(self, batch):
        """Shared between the towers, so it cancels under `subtract`."""
        c = self.cfg
        if not self.training or (c.input_noise <= 0 and c.feature_dropout <= 0):
            return None
        out = {}
        for side in ("ab", "ag"):
            ref = batch[f"seq_{side}_wt"]
            n = (torch.randn_like(ref) * (c.input_noise * ref.std(dim=(0, 1), keepdim=True))
                 if c.input_noise > 0 else None)
            m = None
            if c.feature_dropout > 0:
                keep = (torch.rand(ref.shape[0], 1, ref.shape[2], device=ref.device,
                                   dtype=ref.dtype) >= c.feature_dropout)
                m = keep.to(ref.dtype) / (1.0 - c.feature_dropout)
            out[side] = (n, m)
        return out

    def tower(self, batch, which: str, p) -> torch.Tensor:
        """Pool each side at the mutated residues, concat the sides, process. Shared."""
        parts = []
        for side in ("ab", "ag"):
            x = batch[f"seq_{side}_{which}"]
            if p is not None:
                n, m = p[side]
                if n is not None:
                    x = x + n
                if m is not None:
                    x = x * m
            parts.append(site_mean(x, batch[f"site_{side}"]))
        return self.proc(self.norm(torch.cat(parts, dim=-1)))

    def forward(self, batch) -> torch.Tensor:
        p = self.perturb(batch)
        p_mt = self.tower(batch, "mt", p)
        p_wt = self.tower(batch, "wt", p)
        z = (torch.cat([p_mt, p_wt], dim=-1) if self.cfg.combine == "concat"
             else p_mt - p_wt)
        if self.cfg.chem_dim:
            z = torch.cat([z, batch["chem"]], dim=-1)
        return self.mlp(z).squeeze(-1)


@dataclass
class SiteTokenConfig:
    """The mutated token, its original, and the wild-type binding site.

    The residue axis collapses to three vectors:

        tok_wt     the wild-type residue's embedding, at the mutated position
        tok_mt     the mutant residue's embedding, at the same position
        site_pool  the wild-type binding site, pooled over the crop, per side

    A multi-point row averages its mutated tokens into tok_wt and tok_mt rather than being
    dropped: 272 rows, 29% of the data, and 7 complexes have no single-point row at all.
    Averaging is not obviously right -- ddG is not additive in the number of mutations
    (corr(k, ddG) = -0.151 overall, -0.292 among multi-point rows), so a sum would be
    wrong and a mean is a choice -- but it keeps every row and every complex.

    The head is given tok_wt and tok_mt separately rather than their difference, so it can
    learn what to do with them instead of having subtraction imposed.

    ``use_site_pool`` is the one thing to watch. The binding-site pool is IDENTICAL for
    every mutation of a complex, so it is a complex-identity channel -- and predicting a
    complex's mean scores pooled +0.672 on these labels, better than any model here, while
    scoring 0.000 per complex. It gives the head context for where the mutation sits; it
    also gives it a shortcut. Turning it off is a one-line ablation and worth running.
    """

    pca_dim: int = 128
    #: A shared Linear applied to every token after the PCA, before anything is pooled.
    #: 0 leaves the PCA components as they are. Projecting first means the subtraction
    #: below happens in a learned space rather than in PCA coordinates, and the same map
    #: is used for the mutated tokens and for the binding-site pools so they stay
    #: comparable.
    proj: int = 0
    #: A narrower projection for the binding-site pools only. 0 reuses ``proj``. The pools
    #: are identical for every mutation of a complex, so their width is the width of the
    #: complex-identity channel: narrowing them shrinks the shortcut while leaving the
    #: edit pathway at full width. That is a different intervention from trimming the
    #: model uniformly, and the evidence favours it -- shrinking this architecture
    #: uniformly has twice made it worse (+0.205 -> +0.170 at width 32, and the MLP gained
    #: going 32k -> 82k), so capacity per se is not what is limiting it.
    pool_proj: int = 0
    #: Hand the head tok_mt - tok_wt instead of both vectors. The difference is what the
    #: edit did; the pair lets the head also read which residue was there to begin with.
    subtract: bool = False
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    use_site_pool: bool = True
    #: The project's handcrafted columns (chem + ProteinMPNN log-likelihood ratios),
    #: concatenated after the blocks above are normalised. They arrive already
    #: standardised with training-fold statistics, so they are NOT passed through another
    #: LayerNorm: doing so would renormalise 26 columns per row and destroy the relative
    #: scale between, say, d_volume and n_to_ala that the fold-local standardisation set.
    chem_dim: int = 0
    #: Let each sequence token attend over the ProteinMPNN tokens of its own side, after
    #: the projection. The two modalities share the crop, so token i of one is residue i
    #: of the other -- but attention is over ALL of them, so residue i's sequence vector
    #: can read residue j's structure. That is spatial context a per-token map cannot
    #: express. K and V come from the WILD-TYPE structure for both branches, so the
    #: subtraction still isolates the edit: identical sequences give identical attention
    #: output and a null edit is still exactly zero.
    cross_attn: bool = False
    n_heads: int = 4
    #: A separate Linear for the ProteinMPNN tokens. 0 reuses ``proj``, which is free
    #: because both modalities are 128-d PCA -- but it forces one map to serve two
    #: representations that a dot-product attention is meant to compare, so the query and
    #: key spaces cannot differ. Its own map costs 8,192 and lets them.
    mpnn_proj: int = 0
    #: How the attention's residual is mixed: ``res_pre`` * q_in + ``res_post`` * attn(q_in).
    #: The default 1.0 / 1.0 is the ordinary residual. Weighting them shifts how much of
    #: the token survives attention untouched -- and because the head reads a DIFFERENCE,
    #: it also rescales the two terms of that difference relative to each other:
    #:     res_pre * (seq_mt - seq_wt)  +  res_post * (attn(seq_mt) - attn(seq_wt))
    #: the raw sequence edit, and what the edit changed about what it reads.
    res_pre: float = 1.0
    res_post: float = 1.0
    input_noise: float = 0.0
    feature_dropout: float = 0.0


class CrossAttn(nn.Module):
    """One cross-attention layer, pre-LN, residual. Queries from seq, keys/values from MPNN."""

    def __init__(self, w: int, n_heads: int, dropout: float,
                 res_pre: float = 1.0, res_post: float = 1.0):
        super().__init__()
        assert w % n_heads == 0, f"width {w} must divide by {n_heads} heads"
        self.h, self.hd = n_heads, w // n_heads
        self.ln_q = nn.LayerNorm(w, elementwise_affine=False)
        self.ln_kv = nn.LayerNorm(w, elementwise_affine=False)
        self.q, self.k, self.v, self.o = (nn.Linear(w, w, bias=False) for _ in range(4))
        self.drop = nn.Dropout(dropout)
        self.res_pre, self.res_post = res_pre, res_post

    def forward(self, q_in, kv_in, kv_mask):
        b, lq, w = q_in.shape
        qh = self.q(self.ln_q(q_in)).view(b, lq, self.h, self.hd).transpose(1, 2)
        kv = self.ln_kv(kv_in)
        kh = self.k(kv).view(b, kv.shape[1], self.h, self.hd).transpose(1, 2)
        vh = self.v(kv).view(b, kv.shape[1], self.h, self.hd).transpose(1, 2)
        logits = (qh @ kh.transpose(-2, -1)) / (self.hd ** 0.5)
        # -inf, never a small finite number: a padded key must carry no softmax mass, or
        # the answer depends on how wide the batch happened to be padded.
        logits = logits.masked_fill(~kv_mask.bool()[:, None, None, :], float("-inf"))
        attn = self.drop(torch.softmax(logits, dim=-1))
        out = self.o((attn @ vh).transpose(1, 2).reshape(b, lq, w))
        return self.res_pre * q_in + self.res_post * out


class PerturbSiteToken(nn.Module):
    def __init__(self, cfg: SiteTokenConfig | None = None):
        super().__init__()
        c = self.cfg = cfg or SiteTokenConfig()
        w = c.proj or c.pca_dim
        self.proj = nn.Linear(c.pca_dim, c.proj, bias=False) if c.proj else None
        pw = c.pool_proj or w
        self.pool_proj = (nn.Linear(c.pca_dim, c.pool_proj, bias=False)
                          if c.pool_proj and c.use_site_pool else None)
        # ProteinMPNN's PCA output is 128-d like the sequence's, so `proj` maps both into
        # one space and no separate input layer is needed before the dot product.
        self.attn = (CrossAttn(w, c.n_heads, c.dropout, c.res_pre, c.res_post)
                     if c.cross_attn else None)
        # K and V must arrive at the attention's width, so this maps to w, not to a free
        # choice -- mpnn_proj is a flag for WHETHER it is separate, not for how wide.
        self.mpnn_proj = (nn.Linear(c.pca_dim, w, bias=False)
                          if c.cross_attn and c.mpnn_proj else None)

        # Each block is normalised on its OWN, then concatenated. A single LayerNorm over
        # the concatenation would normalise across blocks that are not the same kind of
        # quantity: the edit is a DIFFERENCE of two token vectors and is small, while a
        # binding-site pool is an absolute vector and is not. Jointly normalised, the pools
        # set the scale and the edit -- the only part that varies between mutations of the
        # same complex -- is compressed toward nothing.
        n_tok = 1 if c.subtract else 2
        self.norm_tok = nn.ModuleList(
            [nn.LayerNorm(w, elementwise_affine=False) for _ in range(n_tok)])
        self.norm_pool = nn.ModuleList(
            [nn.LayerNorm(pw, elementwise_affine=False) for _ in range(2)]
        ) if c.use_site_pool else None
        d = w * n_tok + (pw * 2 if c.use_site_pool else 0) + c.chem_dim
        blocks: list[nn.Module] = []
        for _ in range(c.layers):
            blocks += [nn.Linear(d, c.hidden), nn.GELU(), nn.Dropout(c.dropout)]
            d = c.hidden
        blocks += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*blocks)

    def forward(self, batch) -> torch.Tensor:
        c = self.cfg
        noise = mask = None
        if self.training and (c.input_noise > 0 or c.feature_dropout > 0):
            ref = batch["seq_ab_wt"]
            if c.input_noise > 0:
                noise = c.input_noise * ref.std(dim=(0, 1), keepdim=True)
            if c.feature_dropout > 0:
                keep = (torch.rand(ref.shape[0], 1, ref.shape[2], device=ref.device,
                                   dtype=ref.dtype) >= c.feature_dropout)
                mask = keep.to(ref.dtype) / (1.0 - c.feature_dropout)

        def px(x):
            # One draw per row, applied to WT and MUT alike, so the pair stays comparable
            if noise is not None:
                x = x + torch.randn_like(x) * noise
            if mask is not None:
                x = x * mask
            return x

        def tok_proj(x):
            return self.proj(x) if self.proj is not None else x

        def pool_map(x):
            # its own map when pool_proj is set, otherwise the same one the tokens use
            if self.pool_proj is not None:
                return self.pool_proj(x)
            return tok_proj(x)

        # Exactly one side carries the mutation, so the other contributes zeros and the
        # sum picks out the mutated token without needing to know which side it was on.
        def tokens(which: str, side: str):
            x = tok_proj(px(batch[f"seq_{side}_{which}"]))
            if self.attn is not None:
                st = px(batch[f"struct_{side}"])
                kv = (self.mpnn_proj(st) if self.mpnn_proj is not None
                      else tok_proj(st))               # wild-type structure, both branches
                x = self.attn(x, kv, batch[f"mask_{side}"])
            return site_mean(x, batch[f"site_{side}"])

        tok_wt = sum(tokens("wt", s) for s in ("ab", "ag"))
        tok_mt = sum(tokens("mt", s) for s in ("ab", "ag"))
        raw = [tok_mt - tok_wt] if c.subtract else [tok_wt, tok_mt]
        parts = [n(x) for n, x in zip(self.norm_tok, raw)]

        if c.use_site_pool:
            for i, s in enumerate(("ab", "ag")):
                m = batch[f"mask_{s}"].unsqueeze(-1)
                x = pool_map(px(batch[f"seq_{s}_wt"]))
                pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
                parts.append(self.norm_pool[i](pooled))
        if c.chem_dim:
            parts.append(batch["chem"])
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)
