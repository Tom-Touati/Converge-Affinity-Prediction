"""Guard the one deviation that could silently change the reproduction: the encoder stack.

ProtAttBA's environment is python 3.10 with torch 2.1.2 and transformers 4.40.2. Colab now
ships python 3.13 with torch 2.11 and transformers 5.16, and their 2024 pins have no wheels
for 3.13, so the remote run cannot use their versions (see ``NEEDED`` in colab_job.py).

That leaves a question worth answering rather than assuming: does ESM2-650M produce the same
per-residue embeddings under transformers 5.16 as under 4.40.2? If it does not, every number
downstream is measuring a different encoder and the reproduction is meaningless.

Two modes:

``--write``   Run locally, under their pins, against the cache ``extract_embeddings.py`` built.
              Writes a fingerprint of three sequences -- short, median and longest -- as the
              reference.
``--reference`` Run on the VM. Recomputes those three sequences from the checkpoint with
              whatever stack is installed and compares against the reference, failing loudly
              on disagreement.

The fingerprint is deliberately more than a checksum: per-sequence mean, std, min, max and the
first eight components of the first and last real token. A checksum would only say "different"
and float reassociation across library versions guarantees some difference; these say *how*
different, so a 1e-5 reordering can be told apart from a changed implementation.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import S1131_CSV
from paths_local import EMBEDDINGS, EMBEDDINGS_INDEX, ESM2_DIR

HERE = Path(__file__).resolve().parent

#: Loose enough for float reassociation across CUDA/CPU and library versions, tight enough
#: that a different ESM2 implementation or checkpoint could not pass.
TOL_ABS = 2e-3


def pick_sequences() -> list[str]:
    """Three sequences spanning the length range, chosen deterministically from the csv."""
    pool = sorted(set(pd.read_csv(S1131_CSV).a.astype(str)), key=lambda s: (len(s), s))
    return [pool[0], pool[len(pool) // 2], pool[-1]]


def fingerprint(emb: np.ndarray) -> dict:
    emb = np.asarray(emb, dtype=np.float64)
    return {
        "n_tokens": int(emb.shape[0]),
        "hidden": int(emb.shape[1]),
        "mean": float(emb.mean()),
        "std": float(emb.std()),
        "min": float(emb.min()),
        "max": float(emb.max()),
        "first_token_head": [float(x) for x in emb[0, :8]],
        "last_token_head": [float(x) for x in emb[-1, :8]],
    }


def from_cache(seqs: list[str]) -> list[dict]:
    """Read the fingerprints out of the local cache, via the same token-id keying it uses."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ESM2_DIR))
    store = np.load(EMBEDDINGS, mmap_mode="r")
    with open(EMBEDDINGS_INDEX, "rb") as f:
        index = pickle.load(f)["index"]

    out = []
    for s in seqs:
        ids = tok(s, return_tensors="np")["input_ids"][0].astype(np.int32)
        hit = index.get(ids.tobytes())
        if hit is None:
            raise KeyError(f"sequence of length {len(s)} is not in the local cache")
        off, n = hit
        fp = fingerprint(store[off:off + n])
        fp["seq_len"] = len(s)
        out.append(fp)
    return out


def from_model(seqs: list[str]) -> list[dict]:
    """Recompute from the checkpoint with whatever stack is installed here."""
    import torch
    from transformers import AutoTokenizer, EsmModel

    tok = AutoTokenizer.from_pretrained(str(ESM2_DIR))
    model = EsmModel.from_pretrained(str(ESM2_DIR)).eval()
    out = []
    for s in seqs:
        enc = tok(s, return_tensors="pt")
        with torch.no_grad():
            emb = model(**enc).last_hidden_state[0].float().cpu().numpy()
        fp = fingerprint(emb)
        fp["seq_len"] = len(s)
        out.append(fp)
    return out


def compare(ref: list[dict], got: list[dict]) -> bool:
    ok = True
    scalar_keys = ("mean", "std", "min", "max")
    for i, (a, b) in enumerate(zip(ref, got)):
        print(f"\n  sequence {i}  len {a['seq_len']}  tokens {a['n_tokens']}")
        if a["n_tokens"] != b["n_tokens"] or a["hidden"] != b["hidden"]:
            print(f"    SHAPE MISMATCH reference {a['n_tokens']}x{a['hidden']} "
                  f"vs here {b['n_tokens']}x{b['hidden']}")
            ok = False
            continue
        for k in scalar_keys:
            d = abs(a[k] - b[k])
            flag = "" if d < TOL_ABS else "   <-- EXCEEDS TOLERANCE"
            print(f"    {k:<6s} reference {a[k]:+.6f}   here {b[k]:+.6f}   |diff| {d:.2e}{flag}")
            ok &= d < TOL_ABS
        for k in ("first_token_head", "last_token_head"):
            d = float(np.abs(np.array(a[k]) - np.array(b[k])).max())
            flag = "" if d < TOL_ABS else "   <-- EXCEEDS TOLERANCE"
            print(f"    {k:<16s} max |diff| over 8 components {d:.2e}{flag}")
            ok &= d < TOL_ABS
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="PATH", help="write the reference from the local cache")
    g.add_argument("--reference", metavar="PATH", help="compare against a written reference")
    args = ap.parse_args()

    seqs = pick_sequences()
    print(f"three sequences, lengths {[len(s) for s in seqs]}")

    if args.write:
        import torch
        import transformers
        payload = {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "fingerprints": from_cache(seqs),
        }
        Path(args.write).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.write} under torch {torch.__version__}, "
              f"transformers {transformers.__version__}")
        return

    ref = json.loads(Path(args.reference).read_text())
    import torch
    import transformers
    print(f"reference built under python {ref['python']}, torch {ref['torch']}, "
          f"transformers {ref['transformers']}")
    print(f"this runtime   python {__import__('sys').version.split()[0]}, "
          f"torch {torch.__version__}, transformers {transformers.__version__}")

    ok = compare(ref["fingerprints"], from_model(seqs))
    print()
    if ok:
        print(f"ESM2-650M agrees across the two stacks to better than {TOL_ABS:g} absolute.\n"
              "The library drift forced by Colab's python 3.13 does not change the encoder.")
    else:
        print("!!! STAGE_FAILED: ESM2-650M does not agree across stacks; the remote run would "
              "be measuring a different encoder than the paper's")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
