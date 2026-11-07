#!/usr/bin/env python3
"""Protocol P30 (the public replication on InsPLAD-det): accuracy of each system on the test questions, overall and by
question type, paired differences with a 95% interval from resampling test images (2,000 draws, seed 0, as in the paper),
and the input length of each model. Standard library only.

    python p30_report.py --items pl_test.jsonl --system NAME=FILE[,FILE...] ... --pair A-B ... --out p30_summary.json
"""
import argparse
import collections
import json
import random


def load(paths):
    rows = {}
    for p in paths.split(","):
        for line in open(p):
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def interval(diffs_by_image, reps=2000, seed=0):
    keys = list(diffs_by_image)
    sums = [sum(diffs_by_image[k]) for k in keys]
    counts = [len(diffs_by_image[k]) for k in keys]
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        s = n = 0
        for _ in keys:
            j = rng.randrange(len(keys))
            s += sums[j]; n += counts[j]
        stats.append(100 * s / n)
    stats.sort()
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--system", nargs="+", required=True, help="NAME=FILE[,FILE...]")
    ap.add_argument("--pair", nargs="*", default=[], help="A-B: paired difference A minus B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = {it["id"]: it for it in map(json.loads, open(args.items))}
    systems = {}
    for spec in args.system:
        name, paths = spec.split("=", 1)
        systems[name] = load(paths)
    out = {"n_questions": len(items), "n_images": len({it["image"] for it in items.values()}), "systems": {}, "pairs": {}}
    for name, rows in systems.items():
        missing = [q for q in items if q not in rows]
        acc = lambda qs: round(100 * sum(rows[q]["correct"] for q in qs) / len(qs), 4) if qs else None
        by_type = {t: acc([q for q in items if items[q]["qtype"] == t and q in rows]) for t in ("cls", "exist", "count")}
        lengths = [rows[q]["input_tokens"] for q in items if q in rows and "input_tokens" in rows[q]]
        visual = [rows[q]["visual_tokens"] for q in items if q in rows and "visual_tokens" in rows[q]]
        out["systems"][name] = {"answered": len(items) - len(missing), "accuracy": acc([q for q in items if q in rows]), "by_type": by_type,
                                "input_tokens_mean": round(sum(lengths) / len(lengths), 1) if lengths else None, "input_tokens_max": max(lengths) if lengths else None,
                                "visual_tokens_mean": round(sum(visual) / len(visual), 1) if visual else None}
        print(f"{name:12s} {out['systems'][name]['accuracy']}  by type {by_type}  answered {len(items) - len(missing)}/{len(items)}  "
              f"input {out['systems'][name]['input_tokens_mean']} (max {out['systems'][name]['input_tokens_max']})")
    for pair in args.pair:
        a, b = pair.split("-", 1)
        common = [q for q in items if q in systems[a] and q in systems[b]]
        by_image = collections.defaultdict(list)
        for q in common:
            by_image[items[q]["image"]].append(systems[a][q]["correct"] - systems[b][q]["correct"])
        d = 100 * sum(sum(v) for v in by_image.values()) / len(common)
        lo, hi = interval(by_image)
        out["pairs"][pair] = {"n": len(common), "diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                              "a_right_b_wrong": sum(systems[a][q]["correct"] > systems[b][q]["correct"] for q in common),
                              "a_wrong_b_right": sum(systems[a][q]["correct"] < systems[b][q]["correct"] for q in common)}
        print(f"{pair:28s} {d:+.1f} ({lo:+.1f} to {hi:+.1f})  n {len(common)}")
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
