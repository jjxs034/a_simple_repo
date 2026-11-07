"""Groups of near-duplicate training images for cross-fitting: images of one group must fall into the same fold,
otherwise the detector that is supposed not to have seen an image has seen its near-copy.

Uses the pairs written by dupaudit.py, training images only (no test image takes part). Writes a json
{image path: name of the first member of its group} for the images that have a near-duplicate; head.fold_of reads it
through the environment variable PB_FOLD_GROUPS.

    python foldgroups.py --pairs <dupaudit out dir> --out fold_groups.json [--max-bits 26]
"""
import argparse
import collections
import json
import os

import head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-bits", type=int, default=26)
    args = parser.parse_args()
    out = {}
    for sc in ("ed", "td"):
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pairs = []
        for line in open(os.path.join(args.pairs, f"{sc}_pairs.jsonl")):
            p = json.loads(line)
            if p["a_split"] == p["b_split"] == "fit" and p["bits"] <= args.max_bits:
                pairs.append((p["a"], p["b"]))
                parent[find(p["a"])] = find(p["b"])
        groups = collections.defaultdict(list)
        for x in parent:
            groups[find(x)].append(x)
        for members in groups.values():
            first = min(os.path.basename(m) for m in members)
            for m in members:
                out[m] = first
        split_before = sum(head.fold_of(a, 2) != head.fold_of(b, 2) for a, b in pairs)
        sizes = sorted((len(m) for m in groups.values()), reverse=True)
        print(f"{sc}: {len(pairs)} near-duplicate pairs among training images; {len(groups)} groups covering {len(parent)} images, largest {sizes[:5]}; "
              f"{split_before} pairs ({100 * split_before / max(1, len(pairs)):.0f}%) were split across the two folds by the image-level assignment")
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"{args.out}: {len(out)} images")


if __name__ == "__main__":
    main()
