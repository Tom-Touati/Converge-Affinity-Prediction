import subprocess, sys
code = r'''
import sys, torch
sys.path.insert(0, "/content/perturb")
from model_simple import PerturbSiteToken, SiteTokenConfig
base = dict(pca_dim=128, proj=64, subtract=True, use_site_pool=False, hidden=128,
            layers=1, dropout=0.35, input_noise=0.25, feature_dropout=0.35,
            chem_dim=26, mpnn_proj=64, struct_area_pool=True)
def batch(B=4, L=40):
    b = {"chem": torch.randn(B, 26)}
    for side in ("ab", "ag"):
        for w in ("wt", "mt"):
            b[f"seq_{side}_{w}"] = torch.randn(B, L, 128)
        b[f"struct_{side}"] = torch.randn(B, L, 128)
        s = torch.zeros(B, L); s[:, 3] = 1
        b[f"site_{side}"] = s
        m = torch.zeros(B, L); m[:, :30] = 1      # binding area = first 30 of the crop
        b[f"mask_{side}"] = m
    return b
for name, extra in (("area_concat", dict(concat_struct=True)),
                    ("area_gated",  dict(gated_fusion=True)),
                    ("area_film",   dict(film_struct=True)),
                    ("area_xattn",  dict(cross_attn=True, n_heads=4,
                                         attn_direction="seq_to_struct"))):
    m = PerturbSiteToken(SiteTokenConfig(**base, **extra)).eval()
    b = batch()
    out = m(b)
    # the area pool must depend on residues OUTSIDE the mutated site, or the flag did nothing
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["struct_ab"][:, 10:30] = 0.0            # inside the area, outside the site
    moved = bool((out - m(b2)).abs().max() > 1e-6)
    print(f"{name:<13} ok out={tuple(out.shape)} "
          f"{sum(p.numel() for p in m.parameters()):>7,}p  "
          f"reads-area={moved}")
'''
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print(r.stdout.strip()); print(r.stderr.strip()[-600:] if r.returncode else "rc 0")
