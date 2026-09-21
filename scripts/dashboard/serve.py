#!/usr/bin/env python3
"""Live view of a sweep still running on a Colab VM.

`colab download` works while the kernel is BUSY, which is what makes this possible: a
background thread pulls rank_fusion_sweep.csv and each finished config's history.csv off
the VM every POLL_SECONDS, and the page renders whatever has landed. Nothing here talks to
the training process, so polling cannot disturb it.

    python scripts/dashboard/serve.py [--port 8777] [--session cb]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import erroranalysis

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
DATA = HERE / "_data"
REMOTE = "/content/converge_bind/reports"
POLL_SECONDS = 45

STATE: dict = {"sweep": [], "history": {}, "last_poll": None, "error": None, "polls": 0}
LOCK = threading.Lock()


def pull(session: str, remote: str, local: pathlib.Path) -> bool:
    """One `colab download`, routed through WSL because the CLI is Linux-only."""
    local.parent.mkdir(parents=True, exist_ok=True)
    wsl_local = "/mnt/" + str(local).replace(":", "").replace("\\", "/")
    wsl_local = wsl_local[:6].lower() + wsl_local[6:]
    cmd = ["wsl.exe", "-e", "bash", "-lic",
           f"timeout 120 colab download -s {session} {remote} {wsl_local}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return "Downloaded" in (r.stdout or "") + (r.stderr or "")
    except Exception:
        return False


def poll_loop(session: str):
    while True:
        try:
            if pull(session, f"{REMOTE}/rank_fusion_sweep.csv", DATA / "sweep.csv"):
                rows = list(csv.DictReader(io.StringIO(
                    (DATA / "sweep.csv").read_text(encoding="utf-8", errors="replace"))))
                with LOCK:
                    STATE["sweep"] = rows
                for r in rows:                       # one history per finished config
                    name = r.get("name")
                    if not name:
                        continue
                    f = DATA / f"{name}_history.csv"
                    if pull(session, f"{REMOTE}/{name}/history.csv", f):
                        with LOCK:
                            STATE["history"][name] = summarise(f)
            with LOCK:
                STATE["last_poll"] = time.strftime("%H:%M:%S")
                STATE["polls"] += 1
                STATE["error"] = None
        except Exception as e:                        # a failed poll must never kill the loop
            with LOCK:
                STATE["error"] = f"{type(e).__name__}: {e}"
        time.sleep(POLL_SECONDS)


def summarise(path: pathlib.Path) -> dict:
    """Mean train/val curves per epoch, averaged over seeds, kept per fold."""
    out: dict = {}
    try:
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8",
                                                              errors="replace"))))
    except Exception:
        return out
    agg: dict = {}
    for r in rows:
        try:
            fold, ep = int(float(r["fold"])), int(float(r["epoch"]))
        except (KeyError, ValueError):
            continue
        b = agg.setdefault(fold, {}).setdefault(ep, {"v": [], "t": [], "l": []})
        for key, dest in (("val_rho", "v"), ("train_rho", "t"), ("loss", "l")):
            try:
                val = float(r.get(key, ""))
                if val == val:                        # drop NaN
                    b[dest].append(val)
            except ValueError:
                pass
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    for fold, eps in agg.items():
        ks = sorted(eps)
        out[str(fold)] = {
            "epoch": ks,
            "val": [mean(eps[k]["v"]) for k in ks],
            "train": [mean(eps[k]["t"]) for k in ks],
            "loss": [mean(eps[k]["l"]) for k in ks],
        }
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                        # keep the console readable
        pass

    def do_GET(self):
        if self.path.startswith("/api/state"):
            with LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        elif self.path.startswith("/api/runs"):
            body = json.dumps(erroranalysis.available_runs()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        elif self.path.startswith("/api/error"):
            run = self.path.split("run=", 1)[1].split("&")[0] if "run=" in self.path else ""
            try:
                body = json.dumps(erroranalysis.analyse(run)).encode()
            except Exception as e:                    # a bad run name must not kill the server
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        else:
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--session", default="cb")
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=poll_loop, args=(a.session,), daemon=True).start()
    print(f"serving http://localhost:{a.port}  (polling session {a.session} "
          f"every {POLL_SECONDS}s)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
