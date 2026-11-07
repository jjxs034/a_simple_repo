#!/usr/bin/env python3
"""Compare two sets of per-question answers on the same questions (protocol P29, criteria 1, 2 and 4): accuracy of each
per scenario, the paired difference with a 95% interval from resampling images (the convention of the paper), and the
share of questions answered alike. Standard library only, so it runs on the server and on the board.

    python3 p29_compare.py --a NAME=FILE --b NAME=FILE --items items.jsonl ... [--json out.json]

FILE: jsonl with id, scenario and correct (results/*/shard*.jsonl, simulation, board outputs); several files can be
joined with commas. pred is compared when both sides give letters or both give classes.
"""
import argparse
import collections
import json
import random


def load(spec):
    name, paths = spec.split("=", 1)
    rows = {}
    for path in paths.split(","):
        for line in open(path):
            r = json.loads(line)
            rows[r["id"]] = r
    return name, rows


def interval(pairs, by_image, reps=2000, seed=0):
    """95% interval of mean(a) - mean(b) when whole images are resampled."""
    groups = collections.defaultdict(list)
    for qid, (a, b) in pairs.items():
        groups[by_image[qid]].append(a - b)
    keys = list(groups)
    sums = [sum(groups[k]) for k in keys]
    counts = [len(groups[k]) for k in keys]
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
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--items", nargs="+", required=True, help="question files, to know the image of each question")
    ap.add_argument("--json")
    args = ap.parse_args()
    by_image = {}
    for path in args.items:
        for line in open(path):
            it = json.loads(line)
            by_image[it["id"]] = it["image"]
    na, A = load(args.a)
    nb, B = load(args.b)
    common = [q for q in A if q in B and q in by_image]
    out = {"a": na, "b": nb, "only_a": len(set(A) - set(B)), "only_b": len(set(B) - set(A)), "scenarios": {}}
    for sc in sorted({A[q]["scenario"] for q in common}) + ["all"]:
        qs = [q for q in common if sc == "all" or A[q]["scenario"] == sc]
        acc_a = 100 * sum(A[q]["correct"] for q in qs) / len(qs)
        acc_b = 100 * sum(B[q]["correct"] for q in qs) / len(qs)
        lo, hi = interval({q: (A[q]["correct"], B[q]["correct"]) for q in qs}, by_image)
        same = sum(A[q]["correct"] == B[q]["correct"] for q in qs)
        out["scenarios"][sc] = {"n": len(qs), "acc_" + na: round(acc_a, 4), "acc_" + nb: round(acc_b, 4), "diff": round(acc_a - acc_b, 4),
                                "ci95": [round(lo, 4), round(hi, 4)], "same_correctness": round(100 * same / len(qs), 4),
                                "a_right_b_wrong": sum(A[q]["correct"] > B[q]["correct"] for q in qs), "a_wrong_b_right": sum(A[q]["correct"] < B[q]["correct"] for q in qs)}
        s = out["scenarios"][sc]
        print(f"{sc:>3}: n {s['n']:5d}  {na} {acc_a:5.1f}  {nb} {acc_b:5.1f}  diff {acc_a - acc_b:+.1f} ({lo:+.1f} to {hi:+.1f})  "
              f"same correctness {s['same_correctness']:.1f}%  ({s['a_right_b_wrong']} only {na} right, {s['a_wrong_b_right']} only {nb} right)")
    is_letter = lambda p: isinstance(p, str) and len(p) == 1 and p in "ABCDEFGH"  # float results store class names or counts, not letters
    common_letters = [q for q in common if is_letter(A[q].get("pred")) and is_letter(B[q].get("pred"))]
    if common_letters:  # both sides answered with option letters: how often is it the same letter
        out["same_letter"] = round(100 * sum(A[q]["pred"] == B[q]["pred"] for q in common_letters) / len(common_letters), 4)
        print(f"same answer letter: {out['same_letter']:.1f}% of {len(common_letters)}")
    print(f"questions only in {na}: {out['only_a']}, only in {nb}: {out['only_b']}")
    if args.json:
        json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
