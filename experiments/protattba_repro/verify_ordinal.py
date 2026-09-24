"""Does the ordinal head train, and are its thresholds genuinely ordered?"""
import subprocess, sys
code = r'''
import sys, torch
sys.path.insert(0, "/content/perturb")
from model_simple import PerturbSiteToken, SiteTokenConfig, ordinal_logits
torch.manual_seed(0)
cfg = SiteTokenConfig(pca_dim=128, proj=64, subtract=True, use_site_pool=False,
                      hidden=128, layers=1, dropout=0.35, input_noise=0.25,
                      feature_dropout=0.35, chem_dim=26, mpnn_proj=64,
                      gated_fusion=True, ordinal=2)
m = PerturbSiteToken(cfg)
B, L = 32, 40
b = {"chem": torch.randn(B, 26)}
for side in ("ab", "ag"):
    for w in ("wt", "mt"):
        b[f"seq_{side}_{w}"] = torch.randn(B, L, 128)
    b[f"struct_{side}"] = torch.randn(B, L, 128)
    s = torch.zeros(B, L); s[:, 3] = 1
    b[f"site_{side}"] = s
    b[f"mask_{side}"] = torch.ones(B, L)
y = torch.randn(B) * 1.5 + 0.9
edges = torch.tensor([-0.5, 0.5])
bce = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.1)
first = None
for step in range(60):
    z = m(b)
    lg = ordinal_logits(z, m.ord_b0, m.ord_gap)
    loss = bce(lg, (y.unsqueeze(-1) > edges).float())
    opt.zero_grad(); loss.backward(); opt.step()
    if first is None: first = float(loss)
print(f"params {sum(p.numel() for p in m.parameters()):,}")
print(f"loss {first:.4f} -> {float(loss):.4f}  (learns: {float(loss) < first})")
m.eval()
with torch.no_grad():
    z = m(b); lg = ordinal_logits(z, m.ord_b0, m.ord_gap); pr = torch.sigmoid(lg)
print(f"thresholds b = {ordinal_logits(torch.zeros(1), m.ord_b0, m.ord_gap).squeeze().tolist()}")
print(f"P(y>-0.5) >= P(y>+0.5) for every row: {bool((pr[:,0] >= pr[:,1]).all())}")
score = pr.sum(-1)
rank_match = bool(torch.equal(torch.argsort(score), torch.argsort(z)))
print(f"ordinal score ranks rows identically to the scalar: {rank_match}")
'''
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print(r.stdout.strip()); print(r.stderr.strip()[-500:] if r.returncode else "rc 0")
