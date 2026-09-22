"""Standalone per-residue ESM-2 650M extraction on the VM. No src/ imports."""
import pickle, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, EsmModel

ROOT = "/content/protattba_repro"
TABLE = f"{ROOT}/project_sequences.parquet"
ESM   = f"{ROOT}/model/esm2_650m"
OUT   = f"{ROOT}/esm2_650m_tokens.npy"
IDX   = f"{ROOT}/esm2_650m_index.pkl"
COLS  = ("ab_wt", "ag_wt", "ab_mt", "ag_mt")

frame = pd.read_parquet(TABLE)
seqs = set()
for c in COLS:
    seqs.update(s for s in frame[c].astype(str) if s and s.lower() != "nan")
seqs = sorted(seqs, key=lambda s: (-len(s), s))
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{len(frame)} rows, {len(seqs)} sequences, {sum(map(len,seqs))} residues, device {dev}", flush=True)

tok = AutoTokenizer.from_pretrained(ESM)
model = EsmModel.from_pretrained(ESM).to(dev).eval()
H = model.config.hidden_size
ids = [tok(s, return_tensors="np")["input_ids"][0].astype(np.int32) for s in seqs]
lens = np.array([len(t) for t in ids]); total = int(lens.sum())
print(f"{total} tokens, cache {total*H*2/1e9:.2f} GB fp16", flush=True)

store = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16, shape=(total, H))
off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
pad = tok.pad_token_id
t0, done, i, nb = time.time(), 0, 0, 0
while i < len(seqs):
    size = max(1, 32768 // int(lens[i])); j = min(i+size, len(seqs))
    ch = ids[i:j]; w = max(len(c) for c in ch)
    x = np.full((len(ch), w), pad, np.int64); m = np.zeros((len(ch), w), np.int64)
    for r,c in enumerate(ch): x[r,:len(c)]=c; m[r,:len(c)]=1
    with torch.no_grad():
        o = model(input_ids=torch.from_numpy(x).to(dev),
                  attention_mask=torch.from_numpy(m).to(dev)).last_hidden_state.half().cpu().numpy()
    for r,c in enumerate(ch): store[off[i+r]:off[i+r]+len(c)] = o[r,:len(c)]
    done += int(lens[i:j].sum()); i, nb = j, nb+1
    if nb % 10 == 0 or i == len(seqs):
        el = time.time()-t0
        print(f"  {i}/{len(seqs)} seqs {done}/{total} tok {el/60:.1f} min {done/el:.0f} tok/s", flush=True)
store.flush()
index = {ids[k].tobytes(): (int(off[k]), int(lens[k])) for k in range(len(seqs))}
assert len(index) == len(seqs)
pickle.dump({"index": index, "hidden": H, "total_tokens": total,
             "checkpoint": "facebook/esm2_t33_650M_UR50D"}, open(IDX, "wb"))
print(f"DONE {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
