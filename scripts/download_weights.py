#!/usr/bin/env python3
"""
Robust weight downloader for Boogu-Image-0.1 (Base / Turbo / Edit).

The Boogu checkpoints are public (no token needed). This downloader uses curl with
resume (-C -), stall-abort (--speed-limit/--speed-time) and per-file sha256 verification
against the repo's LFS hashes, so it survives flaky CDN connections that hang or write
truncated/corrupt files. Files are fetched in parallel.

Usage:
    python scripts/download_weights.py base            # -> ./models/Boogu-Image-0.1-Base
    python scripts/download_weights.py turbo edit      # several at once
    python scripts/download_weights.py all
    python scripts/download_weights.py Boogu/Boogu-Image-0.1-Base   # any repo id

Optional: set HF_TOKEN in the environment for gated/private repos (not needed for Boogu).
"""
import os, sys, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi

ALIASES = {
    "base":  "Boogu/Boogu-Image-0.1-Base",
    "turbo": "Boogu/Boogu-Image-0.1-Turbo",
    "edit":  "Boogu/Boogu-Image-0.1-Edit",
}
TOKEN = os.environ.get("HF_TOKEN") or None
WORKERS = int(os.environ.get("BOOGU_DL_WORKERS", "5"))
MODELS_DIR = os.environ.get("BOOGU_MODELS_DIR", "models")


def sha256(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def lfs_sha(s):
    l = getattr(s, "lfs", None)
    return (l.get("sha256") if isinstance(l, dict) else getattr(l, "sha256", None))


def fetch(repo, out, name, exp):
    fp = os.path.join(out, *name.split("/"))
    os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    auth = ["-H", f"Authorization: Bearer {TOKEN}"] if TOKEN else []
    for attempt in range(1, 61):
        rc = subprocess.call(["curl", "-L", *auth, "-C", "-",
                              "--connect-timeout", "30", "--speed-limit", "50000",
                              "--speed-time", "30", "--retry", "0", "-s", "-o", fp, url])
        if rc == 0:
            if exp and sha256(fp) != exp:
                print(f"  sha mismatch {name} -> restart", flush=True)
                os.remove(fp); continue
            return True
        import time as _t
        _t.sleep(15 if rc in (6, 7) else 4)   # 6=DNS,7=connect: transient, back off
    print(f"  GIVING UP {name}", flush=True)
    return False


def download(repo):
    out = os.path.join(MODELS_DIR, repo.split("/")[-1])
    api = HfApi(token=TOKEN)
    info = api.model_info(repo, files_metadata=True)
    todo = []
    for s in info.siblings:
        exp = lfs_sha(s)
        fp = os.path.join(out, *s.rfilename.split("/"))
        ok = os.path.exists(fp) and ((sha256(fp) == exp) if exp else (os.path.getsize(fp) == (s.size or 0)))
        if not ok:
            todo.append((s.rfilename, exp))
    print(f"{repo}: {len(todo)} file(s) to fetch -> {out}", flush=True)
    fails = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, repo, out, n, e): n for n, e in todo}
        for fu in as_completed(futs):
            if not fu.result():
                fails += 1
    print(f"{repo}: {'OK' if not fails else f'{fails} FAILED'}", flush=True)
    return fails == 0


def main():
    args = sys.argv[1:] or ["base"]
    repos = []
    for a in args:
        if a == "all":
            repos += list(ALIASES.values())
        else:
            repos.append(ALIASES.get(a.lower(), a))
    ok = all(download(r) for r in dict.fromkeys(repos))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
