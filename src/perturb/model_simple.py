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


class PerturbMLP(nn.Module):
    def __init__(self, cfg: MLPConfig | None = None):
        super().__init__()
        c = self.cfg = cfg or MLPConfig()
        # Both sides concatenated. A side with no mutation contributes zeros, so the head
        # can tell "no antigen-side mutation" from "an antigen-side mutation of zero size".
        d = c.pca_dim * 2
        blocks: list[nn.Module] = [nn.LayerNorm(d)]
        for _ in range(c.layers):
            # bias-free, so an all-zero input (a null edit) maps to exactly zero
            blocks += [nn.Linear(d, c.hidden, bias=False), nn.GELU(), nn.Dropout(c.dropout)]
            d = c.hidden
        blocks += [nn.Linear(d, 1, bias=False)]
        self.net = nn.Sequential(*blocks)
        # LayerNorm has an affine bias that would break the null-edit property; drop it.
        self.net[0].elementwise_affine = False
        self.net[0].weight = None
        self.net[0].bias = None

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
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)
