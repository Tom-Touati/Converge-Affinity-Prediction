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


def ordinal_logits(z: torch.Tensor, b0: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    """(B,) scores -> (B, K) threshold logits with monotone thresholds.

    logit_k = z + b_k where b_0 = b0 and each later b is strictly smaller, so the implied
    probabilities are non-increasing in k by construction.
    """
    b = torch.cat([b0, b0 - torch.cumsum(torch.nn.functional.softplus(gap), 0)])
    return z.unsqueeze(-1) + b


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
    #: Which modality asks the questions.
    #:
    #: ``seq_to_struct``  sequence queries the structure: "given this residue, what of the
    #:                    local geometry matters". The query changes with the mutation.
    #: ``struct_to_seq``  the structure queries the sequence: "given this position, what is
    #:                    sitting here now". The KEYS and VALUES change with the mutation.
    #:
    #: The reversal is not symmetric, and one consequence is worth knowing before reading
    #: any result. Under ``struct_to_seq`` the query is the wild-type structure, which is
    #: IDENTICAL in both branches, so the residual term cancels exactly in the difference
    #: the head reads:
    #:     (r_pre * s + r_post * A_mt) - (r_pre * s + r_post * A_wt) = r_post * (A_mt - A_wt)
    #: res_pre therefore has no effect on the edit signal in this direction -- it only
    #: rescales a constant the head sees. Tuning it here would be tuning nothing.
    attn_direction: str = "seq_to_struct"
    #: Modulate the sequence by the structure with a FiLM instead of attending to it:
    #:     x = (1 + gamma(structure)) * sequence + beta(structure)
    #: The difference from attention is what it can express. FiLM is per RESIDUE -- position
    #: i's structure scales position i's sequence and nothing else -- while attention lets
    #: residue i read residue j. So FiLM cannot carry spatial context, only a local gate.
    #: It is also much cheaper: two 64x64 maps against attention's four plus an input map.
    #:
    #: The edit survives it cleanly. The structure is the wild type in both branches, so
    #: gamma and beta are identical there and the difference is
    #:     (1 + gamma(s)) * (seq_mt - seq_wt)
    #: -- beta cancels outright and the edit is gated, never mixed with anything constant.
    film_struct: bool = False
    #: Reduce raw embeddings inside the network instead of by a fold-local PCA. 0 keeps
    #: the PCA path. Set to a group width (128) and the model builds a GroupedReduce per
    #: modality, sized from the ACTUAL input width, which differs between ESM (1280),
    #: AntiBERTy (512) and ProteinMPNN (128).
    group_reduce: int = 0
    group_out: int = 16
    #: the true input widths, which only the trainer knows; set from the batch at build time
    in_seq: int = 0
    in_str: int = 0
    #: Pool both modalities across residues, then let the model learn how much of each to
    #: use, per row, from all three inputs together:
    #:     g_seq, g_str = sigmoid(W [z_seq ; z_str ; chem])
    #:     head sees  [g_seq * z_seq ; g_str * z_str ; chem]
    #: The gates are vectors, not scalars, so the choice is per dimension rather than one
    #: number per modality.
    #:
    #: Worth stating plainly: z_str is pooled from the WILD-TYPE structure, so it is
    #: constant across every mutation of a complex -- the same complex-identity channel
    #: that the binding-site pools were, and those turned out to be worth +0.003. What
    #: makes this different is that the GATE is computed from z_seq, which does carry the
    #: edit, so the edit decides how much structure to admit rather than the structure
    #: being added unconditionally.
    gated_fusion: bool = False
    #: Pool ProteinMPNN over the mutated residues and CONCATENATE it with the sequence
    #: edit and the chem columns, with no gate, no FiLM and no attention. Until now the
    #: structure could only enter through one of those three, so the plainest way of
    #: combining the two modalities had never actually been run -- which makes it the
    #: missing control for all three of them.
    concat_struct: bool = False
    #: Pool the structure over the whole binding AREA (the crop mask) instead of over the
    #: mutated residues. The two modalities then answer different questions: the sequence
    #: says what the substitution is, the structure says what kind of pocket it sits in.
    #: ProteinMPNN encodes local geometry per residue, so a mean over the interface is a
    #: reasonable description of the environment, where a mean over one residue is not.
    #:
    #: Watch this one. The area pool is nearly constant across mutations of a complex --
    #: the crop moves a little with the site, but most of it does not -- and a channel that
    #: identifies the complex scores +0.672 pooled and +0.000 per complex (ERROR_ANALYSIS
    #: section 8). Expect pooled r to rise; per-complex r is the number that matters.
    struct_area_pool: bool = False
    #: Number of ORDERED thresholds for an ordinal head; 0 keeps plain regression.
    #: With CLASS_EDGES = (-0.5, +0.5) this is 2: stabilising / neutral / destabilising.
    #:
    #: The head stays a single scalar and the thresholds are biases on top of it (CORAL).
    #: That is the point: one shared direction means the predicted ORDER cannot contradict
    #: itself between thresholds, which a 3-way softmax can do freely -- a softmax is happy
    #: to call something both stabilising and destabilising before neutral.
    ordinal: int = 0
    #: Antibody tokens attend to ANTIGEN tokens and vice versa -- across the interface,
    #: rather than from sequence to structure. This is ProtAttBA's mechanism, and it is the
    #: one arrangement in the family that lets a mutation's representation depend on what
    #: it is binding to. Uses the same CrossAttn block; set cross_attn as well.
    ab_ag_attn: bool = False
    #: The sequence EDIT queries the structure.
    #:
    #: Every other attention variant here makes raw sequence tokens the queries and pools
    #: afterwards, so the thing being attended is a residue, not a mutation. Here the query
    #: is `proj(mut) - proj(wt)` at each mutated position, and the keys and values are the
    #: ProteinMPNN tokens over the binding site. The question asked is therefore "given this
    #: edit, what does the surrounding structure say about it" rather than "given this
    #: residue, what does the structure say".
    #:
    #: Pooling is a sum over the mutated positions divided by their count AND by a learned
    #: positive scalar: mean / alpha. The division by count makes multi-point and
    #: single-point rows comparable; alpha lets the network set how loudly this block speaks
    #: against the chem columns it is concatenated with. It is deliberately NOT followed by
    #: a LayerNorm, which would undo it.
    delta_xattn: bool = False
    #: Initial value of the learned pooling denominator, BEFORE softplus.
    #:
    #: Not zero, which is the obvious choice and the wrong one. softplus(0) = 0.693 left the
    #: attended edit at sd 0.216 against the chem block's 0.799 -- a 3.7x mismatch at the
    #: concatenation, with the quieter block being the only one that varies between
    #: mutations of the same complex. -1.56 gives softplus = 0.187, which puts the two at
    #: parity on fold 0 (measured by probe_fusion_scale.py). Training is free to move it;
    #: this only decides where it starts, and starting a block 3.7x too quiet means its
    #: gradient is small exactly when it most needs to learn.
    pool_alpha_init: float = -1.56
    #: Two branches instead of one difference.
    #:
    #: delta_xattn makes `proj(mut) - proj(wt)` the query, which forces subtraction before
    #: the structure is ever consulted. Here the wild-type and the mutant each get their own
    #: pass -- same projection, same attention, same keys -- and a linear layer combines the
    #: two pooled vectors. Subtraction is a special case of that map, so this is strictly
    #: more general, and the network decides whether the difference is the right summary
    #: rather than being told.
    #:
    #: The two branches SHARE their weights on purpose. They are the same modality, and a
    #: siamese pair is what makes their outputs comparable; giving each its own projection
    #: would let them drift into different spaces and make the combination meaningless.
    #: This is a different question from keeping sequence and structure apart, which they
    #: still are.
    two_branch: bool = False
    #: Drop the sequence entirely: no ESM, no attention, no edit. The head sees only the
    #: ProteinMPNN pool at the mutated residues plus the chem and geometry columns.
    #:
    #: This is the reference every sequence-based number here should be read against. The
    #: project has repeatedly found that 26 hand-computed columns beat 1280-dimensional
    #: embeddings, and six probes across three protein language models failed to clear
    #: zero -- but "the sequence arm adds little" has never been priced directly, with
    #: everything else held fixed. This prices it.
    drop_seq: bool = False
    #: How the two branches are combined: "mix" is a learned Linear over their
    #: concatenation, "sub" is the plain difference of the two pooled vectors.
    #:
    #: Three points on one axis, and they are not the same experiment. delta_xattn subtracts
    #: BEFORE the structure is consulted, so the attention only ever sees an edit. "sub"
    #: subtracts AFTER, so each branch reads the structure in its own context and only then
    #: are they differenced. "mix" does neither and lets a linear layer decide -- subtraction
    #: is [I, -I] and therefore reachable, so "sub" is the constrained version of "mix" and
    #: the comparison prices what that freedom is worth.
    branch_combine: str = "mix"
    #: Two stacked attentions, with the chem/geometry columns as the QUERY of the second.
    #:
    #:   1. structure attends to sequence, per token
    #:   2. one GELU layer over that
    #:   3. a single token, projected from the chem and geometry columns, attends to the
    #:      result -- so the question asked of the structure is "given THIS kind of edit,
    #:      what here matters", and the answer is one vector
    #:
    #: The second attention also does the pooling: there is one query, so its output is
    #: already one vector per row and no site_mean is needed. That is a real difference from
    #: every other path here, where pooling is a fixed mean the network cannot shape.
    #:
    #: The columns stop being a block bolted onto the head and become the thing that decides
    #: what to read. Whether that is better is the point of running it.
    chem_query: bool = False
    #: Restrict the chem/geometry query to the MUTATED residues instead of the whole crop.
    #:
    #: The first version of chem_query masked its attention with the crop mask, so the
    #: query pooled over all ~85 cropped residues of which 1-5 are mutated. That scored
    #: +0.168 per complex against the gated fusion's +0.344, and the reason was visible in
    #: the predictions: within a complex they moved with sd 0.410 against the truth's 1.359,
    #: the least of any model that could still see the sequence, and next door to the
    #: no-sequence reference's 0.069. Attention over a crop is a pool, and a pool over 85
    #: tokens cannot say much about the three that changed.
    #:
    #: Restricting the keys keeps the idea -- the columns decide what to read -- and puts it
    #: back where site_mean was reading. A side with no mutated residue contributes nothing,
    #: as it does under site_mean; without that the softmax would see every key masked and
    #: return NaN, which is the usual way this kind of mask fails.
    chem_query_site: bool = False
    #: Rotary position encoding on the cross-attention, keyed on residue index.
    #:
    #: Every attention block here has so far been permutation invariant: the crop arrives
    #: as an unordered bag of residues, so "the residue two along the chain" and "the
    #: residue at the far end of the domain" are indistinguishable to it. Contact geometry
    #: is largely local, and the mutated site's immediate neighbours are the residues whose
    #: packing it disturbs, so sequence separation is information the block is currently
    #: throwing away.
    #:
    #: Rotation is applied to the residue INDEX, not the crop slot -- see rotary(). It is
    #: also skipped wherever the query has no position of its own (the chem/geometry token)
    #: or lives in a different molecule from the keys (antibody attending to antigen, where
    #: a difference of residue indices means nothing).
    rope: bool = False
    #: 10000 is the language-model value, chosen for thousands of tokens. Measured on these
    #: crops the same-chain separations run to a median of 29 and a 90th percentile of 75,
    #: and at that base 7 of the 16 frequency channels turn less than half a radian across
    #: the whole range -- constants, carrying nothing -- while 5 more wrap and alias. Only 4
    #: were tuned to anything this data contains. At 200 the series spans 1 to ~170 instead
    #: of 1 to 47,000: 8 channels usefully tuned and none dead.
    rope_base: float = 200.0
    #: How far apart to place consecutive chains along the rotary axis.
    #:
    #: The residue index restarts at 0 in every chain, so an antibody crop holds an H30 and
    #: an L30 -- 94 slots apart in the crop -- that would otherwise be handed the SAME
    #: phase, an encoding claiming zero separation between residues in different molecules.
    #: Each chain is pushed a fixed distance along the axis: longer than any chain here, and
    #: far short of the slowest component's ~47,000-residue wavelength, so cross-chain pairs
    #: land at a large consistent separation rather than a false zero.
    rope_chain_span: int = 1024
    #: How the mutant and wild-type sequence embeddings combine into the "delta" query.
    #:
    #: ``sub``  mt - wt, a DIFFERENCE. Zero wherever the model did not change anything,
    #:          large wherever it did, sign carries the direction of the change. This is
    #:          what every other block in the project reads.
    #: ``mul``  mt * wt, elementwise -- a per-CHANNEL similarity rather than a change. Large
    #:          positive where both embeddings agree in sign and magnitude on that channel
    #:          (the mutation preserved whatever that channel encodes), near zero where
    #:          either side is near zero on it, and negative where they disagree in sign
    #:          (the mutation flipped it). A channel large in wt and near-zero in mt gives a
    #:          near-zero product either way -- "this channel no longer applies" and "this
    #:          channel was never active" are NOT distinguished, which a subtraction does
    #:          distinguish (it would show a large negative). The standard alternative to a
    #:          difference for comparing two vectors of the same space (NLI-style
    #:          ``[u, v, u*v, u-v]``), tried here on its own rather than alongside the sub.
    edit_op: str = "sub"
    #: Give each input its OWN first linear layer instead of one shared map.
    #:
    #: A shared projection forces the antibody, the antigen and ProteinMPNN into a single
    #: basis before anything is compared, which is a strong assumption to make for free:
    #: an antibody variable domain, an antigen and an inverse-folding encoding are not the
    #: same distribution, and the first layer is exactly where that should be learned
    #: rather than imposed. Sharing it also means a gradient from one modality moves the
    #: representation of the other.
    #:
    #: Structure additionally used to FALL BACK to the sequence projection whenever
    #: mpnn_proj was unset, which is the same violation by another route.
    split_proj: bool = True
    input_noise: float = 0.0
    feature_dropout: float = 0.0


class GroupedReduce(nn.Module):
    """1280 -> 160 as ten independent 128 -> 16 maps, not one dense 1280 -> 160.

    Replaces the fold-local PCA with something the network learns. A dense layer would be
    1280 x 160 = 204,800 parameters against 752 training rows, which is more weights in one
    layer than the whole rest of the model. Block-diagonal, it is 10 x 128 x 16 = 20,480:
    a tenth of the cost, and each block sees a contiguous slice of the embedding.

    The slicing is arbitrary -- ESM dimensions carry no group structure -- so this is a
    parameter-saving constraint, not a claim about the representation. What it buys over
    PCA is that the reduction is fitted to the LABEL rather than to variance, and that it
    is not refit per fold.
    """

    def __init__(self, in_dim: int, group: int = 128, out_per_group: int = 16):
        super().__init__()
        if in_dim % group:
            raise ValueError(f"{in_dim} does not divide into groups of {group}")
        self.n, self.group, self.out = in_dim // group, group, out_per_group
        self.w = nn.Parameter(torch.empty(self.n, group, out_per_group))
        nn.init.normal_(self.w, std=group ** -0.5)

    @property
    def out_dim(self) -> int:
        return self.n * self.out

    def forward(self, x):
        b, l, _ = x.shape
        # (b, l, n, group) x (n, group, out) -> (b, l, n, out), then flattened
        return torch.einsum("blng,ngo->blno",
                            x.view(b, l, self.n, self.group), self.w).reshape(b, l, -1)


def rotary(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate each head's channel pairs by an angle proportional to residue index.

    ``x`` is (batch, heads, length, head_dim) and ``pos`` is (batch, length). The dot
    product of a rotated query and a rotated key depends only on the DIFFERENCE of their
    positions, which is the property worth having here: what should matter is how far
    apart two residues are in the chain, not where the crop happened to start.

    ``pos`` must be the true residue index, never the crop slot. The crop is a selection
    -- interface, site neighbourhood, mutated +/-2 -- so consecutive slots are routinely
    tens of residues apart, and rotating by slot would encode a spacing that does not
    exist.
    """
    b, h, l, d = x.shape
    half = d // 2
    inv = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.float().unsqueeze(-1) * inv                    # (b, l, half)
    cos, sin = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class CrossAttn(nn.Module):
    """One cross-attention layer, pre-LN, residual. Queries from seq, keys/values from MPNN."""

    def __init__(self, w: int, n_heads: int, dropout: float,
                 res_pre: float = 1.0, res_post: float = 1.0,
                 rope: bool = False, rope_base: float = 10000.0):
        super().__init__()
        assert w % n_heads == 0, f"width {w} must divide by {n_heads} heads"
        self.rope, self.rope_base = rope, rope_base
        assert not rope or (w // n_heads) % 2 == 0, "rope needs an even head dimension"
        self.h, self.hd = n_heads, w // n_heads
        self.ln_q = nn.LayerNorm(w, elementwise_affine=False)
        self.ln_kv = nn.LayerNorm(w, elementwise_affine=False)
        self.q, self.k, self.v, self.o = (nn.Linear(w, w, bias=False) for _ in range(4))
        self.drop = nn.Dropout(dropout)
        self.res_pre, self.res_post = res_pre, res_post

    def forward(self, q_in, kv_in, kv_mask, q_pos=None, kv_pos=None):
        b, lq, w = q_in.shape
        qh = self.q(self.ln_q(q_in)).view(b, lq, self.h, self.hd).transpose(1, 2)
        kv = self.ln_kv(kv_in)
        kh = self.k(kv).view(b, kv.shape[1], self.h, self.hd).transpose(1, 2)
        vh = self.v(kv).view(b, kv.shape[1], self.h, self.hd).transpose(1, 2)
        # Both sides must carry a position or neither does: rotating only the keys leaves
        # a query with no phase to be measured against, which is not a relative encoding.
        if self.rope and q_pos is not None and kv_pos is not None:
            qh = rotary(qh, q_pos, self.rope_base)
            kh = rotary(kh, kv_pos, self.rope_base)
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
        # With group_reduce the raw embedding is reduced here instead of by a PCA, and
        # the two modalities have different widths (ESM 1280 / AntiBERTy 512 / MPNN 128),
        # so each gets its own block-diagonal map sized from the real input.
        self.gr_seq = self.gr_str = None
        seq_in = c.pca_dim
        if c.group_reduce:
            self.gr_seq = GroupedReduce(c.in_seq or c.pca_dim, c.group_reduce, c.group_out)
            self.gr_str = GroupedReduce(c.in_str or c.pca_dim, c.group_reduce, c.group_out)
            seq_in = self.gr_seq.out_dim
        w = c.proj or seq_in
        if c.proj and c.split_proj:
            # one per side: the antibody and the antigen get their own first layer
            self.proj = None
            self.proj_side = nn.ModuleDict(
                {k: nn.Linear(seq_in, c.proj, bias=False) for k in ("ab", "ag")})
        else:
            self.proj = nn.Linear(seq_in, c.proj, bias=False) if c.proj else None
            self.proj_side = None
        pw = c.pool_proj or w
        self.pool_proj = (nn.Linear(seq_in, c.pool_proj, bias=False)
                          if c.pool_proj and c.use_site_pool else None)
        # ProteinMPNN's PCA output is 128-d like the sequence's, so `proj` maps both into
        # one space and no separate input layer is needed before the dot product.
        self.attn = (CrossAttn(w, c.n_heads, c.dropout, c.res_pre, c.res_post,
                               rope=c.rope, rope_base=c.rope_base)
                     if (c.cross_attn or c.ab_ag_attn or c.delta_xattn) else None)
        if c.film_struct:
            self.g_str, self.b_str = nn.Linear(w, w), nn.Linear(w, w)
            for lin in (self.g_str, self.b_str):
                nn.init.normal_(lin.weight, std=0.02)   # off zero, so it acts from step 0
                nn.init.zeros_(lin.bias)
        # K and V must arrive at the attention's width, so this maps to w, not to a free
        # choice -- mpnn_proj is a flag for WHETHER it is separate, not for how wide.
        str_in = self.gr_str.out_dim if self.gr_str is not None else c.pca_dim
        # Always separate when structure is read at all: falling back to the sequence
        # projection is exactly the weight sharing this is meant to avoid.
        # gated_fusion belongs here and was missing: without it no separate structure
        # projection is built, ProteinMPNN falls through to the SEQUENCE's Linear, and an
        # antibody sequence, an antigen sequence and an inverse-folding encoding are all
        # forced through one 128 -> 64 map. The parameter count is what exposed it --
        # 49,793 instead of 57,985, exactly one Linear(128, 64) short.
        uses_struct = (c.cross_attn or c.film_struct or c.concat_struct
                       or c.delta_xattn or c.gated_fusion)
        self.mpnn_proj = (nn.Linear(str_in, w, bias=False)
                          if uses_struct and (c.mpnn_proj or c.split_proj) else None)

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
        if c.delta_xattn:
            d = w + c.chem_dim
            if c.chem_query:
                # the columns enter as the second attention's query, so they are
                # NOT concatenated again -- doing both would feed them twice
                d = w
            # softplus keeps the denominator positive: a learned scalar free to cross
            # zero would flip the sign of this whole block part-way through training
            self.pool_alpha = nn.Parameter(torch.full((1,), c.pool_alpha_init))
            if c.two_branch and c.branch_combine == "mix":
                # combines the wild-type and mutant pooled vectors; a subtraction is
                # [I, -I] and therefore reachable, but not imposed
                self.branch_mix = nn.Linear(w * 2, w, bias=False)
            if c.chem_query:
                self.post_attn = nn.Linear(w, w)              # the GELU layer of step 2
                self.chem_tok = nn.Linear(c.chem_dim, w)      # columns -> one query token
                self.attn2 = CrossAttn(w, c.n_heads, c.dropout,
                                       c.res_pre, c.res_post)   # query has no position
        if c.concat_struct:
            # its own norm, for the same reason the other blocks have one: the edit is
            # a difference of two token vectors and is small, the structure pool is an
            # absolute vector and is not
            self.norm_str = nn.LayerNorm(w, elementwise_affine=False)
            d += w
        if c.drop_seq:
            # the last word on the width: every block above has already added its own
            # contribution, and an earlier override was silently undone by concat_struct's
            # `d += w`. Only the structure pool and the columns survive here.
            d = (w if c.concat_struct else 0) + c.chem_dim
        if c.gated_fusion:
            # both modalities pooled, plus chem, all seen by the gate
            zs = w
            d = zs * 2 + c.chem_dim
            self.gate = nn.Linear(d, zs * 2)
            self.norm_str = nn.LayerNorm(zs, elementwise_affine=False)
        blocks: list[nn.Module] = []
        for _ in range(c.layers):
            blocks += [nn.Linear(d, c.hidden), nn.GELU(), nn.Dropout(c.dropout)]
            d = c.hidden
        blocks += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*blocks)
        if c.ordinal:
            # One free bias and then strictly decreasing steps, so
            # P(y > edge_0) >= P(y > edge_1) holds for EVERY input rather than being
            # something the optimiser is merely encouraged to discover.
            self.ord_b0 = nn.Parameter(torch.zeros(1))
            self.ord_gap = nn.Parameter(torch.full((c.ordinal - 1,), 0.5))

    def _struct_vec(self, batch, px, tok_proj) -> torch.Tensor:
        """One vector per row describing the wild-type structure at the mutation.

        Pooled over the mutated residues by default, or over the whole crop -- the binding
        area -- when ``struct_area_pool`` is set. Only the wild-type structure exists; there
        is no mutant structure, so this term is the same for both branches and cancels from
        nothing.
        """
        c = self.cfg
        key = "mask" if c.struct_area_pool else "site"
        z = 0
        for side in ("ab", "ag"):
            st = px(batch[f"struct_{side}"])
            if self.gr_str is not None:
                st = self.gr_str(st)
            stp = self.mpnn_proj(st) if self.mpnn_proj is not None else tok_proj(st, side)
            z = z + site_mean(stp, batch[f"{key}_{side}"])
        return z

    def rope_pos(self, batch, side):
        """Residue index, offset per chain so two chains cannot share a phase."""
        r = batch[f"res_{side}"]
        ch = batch.get(f"chain_{side}")
        return r if ch is None else r + ch * self.cfg.rope_chain_span

    def forward(self, batch) -> torch.Tensor:
        c = self.cfg
        # The scale and the dropout mask are per WIDTH. They used to be taken once from
        # seq_ab_wt and applied to everything, which held only while the PCA made the
        # sequence and the structure both 128 wide. Without it the sequence is 1280 and
        # the structure 128, and every configuration that reads structure -- attention or
        # FiLM -- died on the mismatch. A plain no-PCA run never touched it, so this
        # surfaced only once cross-attention was asked for without the PCA.
        draws: dict[int, tuple] = {}

        def px(x):
            # The feature-dropout mask is drawn once per width and reused, so WT and MUT
            # lose the same columns and it cancels in their difference.
            if not (self.training and (c.input_noise > 0 or c.feature_dropout > 0)):
                return x
            w = x.shape[-1]
            if w not in draws:
                n = (c.input_noise * x.std(dim=(0, 1), keepdim=True)
                     if c.input_noise > 0 else None)
                m = None
                if c.feature_dropout > 0:
                    keep = (torch.rand(x.shape[0], 1, w, device=x.device, dtype=x.dtype)
                            >= c.feature_dropout)
                    m = keep.to(x.dtype) / (1.0 - c.feature_dropout)
                draws[w] = (n, m)
            n, m = draws[w]
            if n is not None:
                x = x + torch.randn_like(x) * n
            if m is not None:
                x = x * m
            return x

        def tok_proj(x, side="ab"):
            if self.gr_seq is not None:
                x = self.gr_seq(x)
            if self.proj_side is not None:
                return self.proj_side[side](x)
            return self.proj(x) if self.proj is not None else x

        def pool_map(x, side="ab"):
            # its own map when pool_proj is set, otherwise the same one the tokens use
            if self.pool_proj is not None:
                return self.pool_proj(x)
            return tok_proj(x, side)

        # Exactly one side carries the mutation, so the other contributes zeros and the
        # sum picks out the mutated token without needing to know which side it was on.
        def tokens(which: str, side: str):
            seq = tok_proj(px(batch[f"seq_{side}_{which}"]), side)
            if c.film_struct:
                st = px(batch[f"struct_{side}"])
                if self.gr_str is not None:
                    st = self.gr_str(st)
                stp = (self.mpnn_proj(st) if self.mpnn_proj is not None else tok_proj(st, side))
                if c.struct_area_pool:
                    # Modulate every sequence token by ONE summary of the binding area,
                    # broadcast over the crop. Without this the FiLM only ever sees the
                    # structure at the mutated residue -- everything else is discarded by
                    # the site_mean below, so the area flag reached this branch and did
                    # nothing at all.
                    m = batch[f"mask_{side}"].unsqueeze(-1)
                    stp = ((stp * m).sum(1) / m.sum(1).clamp(min=1.0)).unsqueeze(1)
                seq = (1 + self.g_str(stp)) * seq + self.b_str(stp)
                return site_mean(seq, batch[f"site_{side}"])
            if c.ab_ag_attn:
                # across the interface: this side's tokens ask the partner chain's tokens.
                # Both branches use the SAME `which`, so a mutation changes the query on the
                # mutated side and the keys on the other -- which is the point.
                other = "ag" if side == "ab" else "ab"
                seq_o = tok_proj(px(batch[f"seq_{other}_{which}"]), other)
                x = self.attn(seq, seq_o, batch[f"mask_{other}"])
                return site_mean(x, batch[f"site_{side}"])
            if self.attn is None:
                return site_mean(seq, batch[f"site_{side}"])
            st = px(batch[f"struct_{side}"])
            if self.gr_str is not None:
                st = self.gr_str(st)
            stp = (self.mpnn_proj(st) if self.mpnn_proj is not None
                   else tok_proj(st, side))            # wild-type structure, both branches
            if c.attn_direction == "struct_to_seq":
                # the structure asks; the sequence answers, and it is the answer that the
                # mutation changes
                x = self.attn(stp, seq, batch[f"mask_{side}"],
                              self.rope_pos(batch, side), self.rope_pos(batch, side))
            else:
                x = self.attn(seq, stp, batch[f"mask_{side}"],
                              self.rope_pos(batch, side), self.rope_pos(batch, side))
            return site_mean(x, batch[f"site_{side}"])

        if c.delta_xattn:
            # The QUERY is the edit itself, not a residue. Built over the whole crop so the
            # batch stays rectangular, then restricted to the mutated positions after the
            # attention -- gathering variable-length queries per row would need ragged
            # tensors for no gain.
            # `which` is the list of branches: one pass over the difference, or two passes
            # over the wild-type and the mutant separately, combined afterwards.
            which = ("wt", "mt") if c.two_branch else ("delta",)
            acc = {k: 0 for k in which}
            n_sites = 0
            for side in ("ab", "ag"):
                st = px(batch[f"struct_{side}"])
                if self.gr_str is not None:
                    st = self.gr_str(st)
                # one set of keys and values per side, shared by both branches, so the two
                # passes differ only in what they ask
                kv = self.mpnn_proj(st) if self.mpnn_proj is not None else tok_proj(st, side)
                site = batch[f"site_{side}"].unsqueeze(-1)
                if c.chem_query:
                    # 1. structure attends to sequence, per token
                    # 2. one GELU layer over the result
                    # 3. a single token from the chem/geometry columns attends to THAT,
                    #    which both asks the question and does the pooling
                    ch_q = self.chem_tok(batch["chem"]).unsqueeze(1)      # (B, 1, w)
                    for k in which:
                        seq = tok_proj(px(batch[f"seq_{side}_{k}"]), side)
                        a1 = self.attn(kv, seq, batch[f"mask_{side}"],
                                       self.rope_pos(batch, side), self.rope_pos(batch, side))
                        a1 = torch.nn.functional.gelu(self.post_attn(a1))
                        km = batch[f"mask_{side}"].bool()
                        if c.chem_query_site:
                            km = km & site.squeeze(-1).bool()
                        # A row mutated only on the other side has no key left here. Feeding
                        # an all-masked softmax returns NaN, so those rows attend over the
                        # crop and their output is then zeroed: the side contributed nothing,
                        # which is exactly what site_mean does with an empty site mask.
                        has = km.any(-1, keepdim=True)
                        a2 = self.attn2(ch_q, a1, km | ~has)              # columns ask
                        acc[k] = acc[k] + a2.squeeze(1) * has
                    continue
                for k in which:
                    if k == "delta":
                        p_mt = tok_proj(px(batch[f"seq_{side}_mt"]), side)
                        p_wt = tok_proj(px(batch[f"seq_{side}_wt"]), side)
                        seq = p_mt * p_wt if c.edit_op == "mul" else p_mt - p_wt
                    else:
                        seq = tok_proj(px(batch[f"seq_{side}_{k}"]), side)
                    if c.attn_direction == "struct_to_seq":
                        # the structure asks and the sequence answers. The two branches then
                        # differ only in what is being ANSWERED with, since the query is the
                        # same wild-type structure either way. Both directions are pooled at
                        # the mutated positions, which is legitimate because the query and
                        # the keys share one crop -- position j means the same residue in
                        # both.
                        out = self.attn(kv, seq, batch[f"mask_{side}"],
                                        self.rope_pos(batch, side), self.rope_pos(batch, side))
                    else:
                        out = self.attn(seq, kv, batch[f"mask_{side}"],
                                        self.rope_pos(batch, side), self.rope_pos(batch, side))
                    acc[k] = acc[k] + (out * site).sum(1)
                n_sites = n_sites + site.sum(1)
            if not c.two_branch:
                pooled = acc["delta"]
            elif c.branch_combine == "sub":
                # differenced after each branch has read the structure, so the subtraction
                # happens in a space the attention has already shaped
                pooled = acc["mt"] - acc["wt"]
            else:
                pooled = torch.cat([acc["wt"], acc["mt"]], dim=-1)
            # sum over the mutated positions, divided by their count and by a learned
            # positive scalar. The count makes a 6-point row comparable to a 1-point row;
            # alpha sets how loudly this speaks next to the chem columns.
            if c.chem_query:
                # the second attention emitted exactly one vector per row, so there is
                # nothing to average over -- dividing by a site count here would only
                # rescale by how many residues the mutation happened to touch
                z = pooled / torch.nn.functional.softplus(self.pool_alpha)
            else:
                z = pooled / (n_sites.clamp(min=1.0)
                              * torch.nn.functional.softplus(self.pool_alpha))
            if c.two_branch and c.branch_combine == "mix":
                z = self.branch_mix(z)
            if c.chem_dim and not c.chem_query:
                z = torch.cat([z, batch["chem"]], dim=-1)
            return self.mlp(z).squeeze(-1)

        tok_wt = sum(tokens("wt", s) for s in ("ab", "ag"))
        tok_mt = sum(tokens("mt", s) for s in ("ab", "ag"))
        if c.gated_fusion:
            z_seq = self.norm_tok[0](tok_mt - tok_wt)
            z_str = self.norm_str(self._struct_vec(batch, px, tok_proj))
            ch = batch["chem"] if c.chem_dim else z_seq[:, :0]
            g = torch.sigmoid(self.gate(torch.cat([z_seq, z_str, ch], dim=-1)))
            zs = z_seq.shape[-1]
            z = torch.cat([g[:, :zs] * z_seq, g[:, zs:] * z_str, ch], dim=-1)
            return self.mlp(z).squeeze(-1)

        if c.drop_seq:
            parts = []          # no ESM, no edit -- the reference case
        else:
            raw = [tok_mt - tok_wt] if c.subtract else [tok_wt, tok_mt]
            parts = [n(x) for n, x in zip(self.norm_tok, raw)]

        if c.use_site_pool:
            for i, s in enumerate(("ab", "ag")):
                m = batch[f"mask_{s}"].unsqueeze(-1)
                x = pool_map(px(batch[f"seq_{s}_wt"]), s)
                pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
                parts.append(self.norm_pool[i](pooled))
        if c.concat_struct:
            parts.append(self.norm_str(self._struct_vec(batch, px, tok_proj)))
        if c.chem_dim:
            parts.append(batch["chem"])
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)
