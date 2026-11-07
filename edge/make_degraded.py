#!/usr/bin/env python3
"""Write the corrupted test images the server evaluated (protocol P29, supplement 2) as lossless PNG files, for the board.

The server never stored these images: core.open_image and detector.py --corrupt build them on the fly with corrupt.apply
from the PIL-decoded source (no EXIF rotation; no test item asks for it). This script repeats that computation with the
same corrupt.py on the same machine, so the board reads exactly the pixels the server systems saw.

    python make_degraded.py --pb <dir with corrupt.py> --items ed_test@dark2.jsonl td_test@dark2.jsonl --out-dir <dir> [--workers 16]
        ->  <dir>/images_ed/<source file name>.png ... and <dir>/degraded.md5 (md5 of each PNG file)
"""
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool

from PIL import Image


def work(job):
    path, kind, severity, out = job
    import corrupt

    if not os.path.exists(out):
        image = corrupt.apply(Image.open(path).convert("RGB"), kind, severity, path)
        image.save(out + ".part", format="PNG", compress_level=1)
        os.replace(out + ".part", out)
    return out, hashlib.md5(open(out, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb", required=True, help="directory of the frozen pipeline code (corrupt.py)")
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    sys.path.insert(0, args.pb)
    jobs = {}
    for p in args.items:
        for line in open(p):
            it = json.loads(line)
            assert it.get("corrupt") and not it.get("exif_transpose"), it["id"]
            d = os.path.join(args.out_dir, f"images_{it['scenario']}")
            os.makedirs(d, exist_ok=True)
            jobs[it["image"]] = (it["image"], it["corrupt"]["kind"], it["corrupt"]["severity"], os.path.join(d, os.path.basename(it["image"]) + ".png"))
    with Pool(args.workers, initializer=sys.path.insert, initargs=(0, args.pb)) as pool:
        sums = pool.map(work, list(jobs.values()), chunksize=4)
    with open(os.path.join(args.out_dir, "degraded.md5"), "w") as f:
        for out, md5 in sorted(sums):
            f.write(f"{md5}  {os.path.relpath(out, args.out_dir)}\n")
    print(f"DEGRADED_DONE {len(sums)} images in {args.out_dir}")


if __name__ == "__main__":
    main()
