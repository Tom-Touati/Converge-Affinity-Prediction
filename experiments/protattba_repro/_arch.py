import sys, types, torch, torch.nn as nn
from argparse import Namespace
sys.path.insert(0, ".")
from paths_local import add_upstream_to_path
add_upstream_to_path()

# stub the encoder: we cache embeddings, so it is not part of the trained model
import model as um
class Stub(nn.Module):
    @classmethod
    def from_pretrained(cls,*a,**k): return cls()
um.EsmModel = Stub
from model import SeqBindModel

args = Namespace(hidden_size=1280, num_heads=4, dropout=0.1, out_dim=1,
                 freeze_backbone=True, model_locate="")
m = SeqBindModel(args)

print("=== top-level children of SeqBindModel ===")
for name, mod in m.named_children():
    p = sum(x.numel() for x in mod.parameters())
    print(f"  {name:26s} {type(mod).__name__:20s} {p:>11,d} params")

print("\n=== one MutilHeadSelfAttn, layer by layer ===")
for name, mod in m.wt_ab_attn.named_children():
    p = sum(x.numel() for x in mod.parameters())
    extra = ""
    if isinstance(mod, nn.Linear): extra = f"  {mod.in_features}->{mod.out_features}"
    if isinstance(mod, nn.LayerNorm): extra = f"  {tuple(mod.normalized_shape)}"
    print(f"  {name:10s} {type(mod).__name__:18s} {p:>10,d}{extra}")

print("\n=== one AttnTransform / AttnMean ===")
for lbl, blk in (("AttnTransform", m.wt_ab_conv_transformer), ("AttnMean", m.wt_ab_mean)):
    print(f"  {lbl}:")
    for name, mod in blk.named_children():
        p = sum(x.numel() for x in mod.parameters())
        k = f"  kernel={mod.kernel_size} {mod.in_channels}->{mod.out_channels}" if isinstance(mod, nn.Conv1d) else ""
        print(f"    {name:12s} {type(mod).__name__:14s} {p:>9,d}{k}")

print("\n=== OutHead ===")
for name, mod in m.out_head.named_children():
    p = sum(x.numel() for x in mod.parameters())
    extra = f"  {mod.in_features}->{mod.out_features}" if isinstance(mod, nn.Linear) else ""
    print(f"  {name:10s} {type(mod).__name__:14s} {p:>10,d}{extra}")

tot = sum(p.numel() for p in m.parameters())
print(f"\nTRAINABLE TOTAL (encoder excluded, we cache it): {tot:,}")
print(f"distinct nn.Module leaves: {sum(1 for _ in m.modules())}")
