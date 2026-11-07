"""Near-duplicate audit of the benchmark splits (no model involved).

Every annotated image gets a 256-bit difference hash of a 17x16 grey thumbnail. Two images are near-duplicates when
their hashes differ in at most --max-bits bits (same frame re-saved, resized, or a neighbouring frame of one sequence).
Reports, per scenario, how many test / validation images have a near-duplicate among the fit images (and fit images of
one cross-fitting fold in the other fold), and writes the pairs and a group id per image for grouped re-analysis.

    python dupaudit.py --out-dir <dir> [--max-bits 26] [--workers 32]
"""
import argparse
import collections
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

import bench

Image.MAX_IMAGE_PIXELS = None


def dhash(path):
    try:
        image = Image.open(path)
        image.draft("L", (136, 128))  # JPEG: decode at reduced size
        grey = np.asarray(image.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
        return path, np.packbits((grey[:, 1:] > grey[:, :-1]).reshape(-1)).tobytes()
    except Exception as exc:  # unreadable file: reported, never silently dropped
        return path, repr(exc)


def split_of(scenario, split, path):
    if split == "test":
        return "test"
    return "val" if bench.is_val(path) else "fit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-bits", type=int, default=26, help="of 256; about 10%% of the bits")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    import head  # fold_of

    os.makedirs(args.out_dir, exist_ok=True)
    images = []  # (scenario, split, path)
    for split in ("train", "test"):
        images += [("ed", split_of("ed", split, img["image"]), img["image"]) for img in bench.load_ed(split)]
    for split, imgs in bench.load_td().items():
        images += [("td", split_of("td", split, img["image"]), img["image"]) for img in imgs]
    with Pool(args.workers) as pool:
        hashed = dict(pool.imap_unordered(dhash, [p for _, _, p in images], chunksize=64))
    failed = [p for p, h in hashed.items() if isinstance(h, str)]
    print(f"{len(images)} images hashed, {len(failed)} unreadable")
    for scenario in ("ed", "td"):
        rows = [(split, path) for sc, split, path in images if sc == scenario and path not in failed]
        bits = np.unpackbits(np.frombuffer(b"".join(hashed[p] for _, p in rows), dtype=np.uint8).reshape(len(rows), 32), axis=1).astype(np.float32)
        # Hamming distance = a(1-b) + (1-a)b, computed in blocks
        parent = list(range(len(rows)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        pairs = []
        ones = bits.sum(1)
        for start in range(0, len(rows), 2000):
            block = bits[start : start + 2000]
            dist = ones[start : start + 2000, None] + ones[None, :] - 2 * block @ bits.T
            for i, j in zip(*np.nonzero(dist <= args.max_bits)):
                i += start
                if i < j:
                    pairs.append((int(i), int(j), int(dist[i - start, j])))
                    parent[find(i)] = find(j)
        splits = [s for s, _ in rows]
        partners = collections.defaultdict(set)  # image index -> splits of its near-duplicates
        for i, j, _ in pairs:
            partners[i].add((splits[j], j))
            partners[j].add((splits[i], i))
        n = collections.Counter(splits)
        has = lambda i, target: any(s == target for s, _ in partners[i])
        print(f"## {scenario}: {len(rows)} images (fit {n['fit']}, val {n['val']}, test {n['test']}); {len(pairs)} near-duplicate pairs at <= {args.max_bits} of 256 bits")
        for src, dst in (("test", "fit"), ("test", "val"), ("val", "fit"), ("test", "test"), ("fit", "fit")):
            hit = sum(1 for i, s in enumerate(splits) if s == src and has(i, dst))
            print(f"  {src} images with a near-duplicate in {dst}: {hit} of {n[src]} ({100 * hit / max(1, n[src]):.1f}%)")
        for limit in (3, 7, 15):  # how the test-to-fit overlap depends on the strictness of "near-duplicate"
            close = {i for i, j, d in pairs if d <= limit and splits[i] == "test" and splits[j] == "fit"} | {j for i, j, d in pairs if d <= limit and splits[j] == "test" and splits[i] == "fit"}
            print(f"  test images with a fit image within {limit} bits: {len(close)} of {n['test']} ({100 * len(close) / max(1, n['test']):.1f}%)")
        fold = [head.fold_of(p, 2) if s == "fit" else -1 for s, p in rows]
        cross = sum(1 for i, s in enumerate(splits) if s == "fit" and any(t == "fit" and fold[j] != fold[i] for t, j in partners[i]))
        print(f"  fit images with a near-duplicate in the other cross-fitting fold: {cross} of {n['fit']} ({100 * cross / max(1, n['fit']):.1f}%)")
        hist = collections.Counter(min(d // 4 * 4, 24) for _, _, d in pairs)
        print("  distance histogram (bits: pairs): " + ", ".join(f"{k}-{k + 3}: {v}" for k, v in sorted(hist.items())))
        with open(os.path.join(args.out_dir, f"{scenario}_pairs.jsonl"), "w") as f:
            for i, j, d in sorted(pairs, key=lambda t: t[2]):
                f.write(json.dumps({"a": rows[i][1], "a_split": splits[i], "b": rows[j][1], "b_split": splits[j], "bits": d}, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out_dir, f"{scenario}_groups.json"), "w") as f:
            json.dump({rows[i][1]: int(find(i)) for i in range(len(rows))}, f, ensure_ascii=False)
    print("DUPAUDIT_DONE")


if __name__ == "__main__":
    main()
