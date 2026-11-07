"""Does the benchmark's train-test near-duplication change the conclusions? Uses stored per-question outputs only.

Test questions are stratified by how close their image is to any image seen in training or validation
(dupaudit.py pairs): near-identical (<= 7 of 256 hash bits), similar scene (8 to --max-bits), or no near-duplicate.
Accuracy of every method per stratum, and paired differences between methods per stratum (intervals resample images).

    python dupeffect.py --bench <bench dir> --pairs <dupaudit out dir> \
        --row '检测器直接答题=<results>/detector_v2_test.jsonl' --row '大模型 整图576=<results>/sft_uniform_fit:u576' \
        --row '大模型核验=<results>/verify_v1_all.jsonl:mllm' \
        --pair '大模型 整图576' '检测器直接答题'
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

STRATA = ["无近重复", "相似场景（8 位以上）", "几乎相同（≤7 位）"]


def load_row(spec):
    """'name=<results dir or file>[+<another seed>...][:policy]' -> (name, {question id: correct}). With several seeds
    joined by '+', the value is the mean correctness over the seeds (questions answered by every seed only)."""
    name, src = spec.split("=", 1)
    paths, _, policy = src.partition(":")
    seeds = []
    for path in paths.split("+"):
        ok = {}
        files = glob.glob(os.path.join(path, "*shard*.jsonl")) if os.path.isdir(path) else [path]
        for p in files:
            for line in open(p):
                r = json.loads(line)
                if not policy or policy in (r.get("policy"), r.get("scorer")):
                    ok[r["id"]] = r["correct"]
        seeds.append(ok)
    if len(seeds) == 1:
        return name, seeds[0]
    return name, {i: sum(s[i] for s in seeds) / len(seeds) for i in seeds[0] if all(i in s for s in seeds)}


def clustered_ci(items, values, reps=2000, seed=0):
    by_image = collections.defaultdict(list)
    for it, v in zip(items, values):
        by_image[it["image"]].append(v)
    if len(by_image) < 2:
        return float("nan"), float("nan")
    sums = np.array([sum(v) for v in by_image.values()], dtype=float)
    counts = np.array([len(v) for v in by_image.values()], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(reps, len(sums)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--row", action="append", required=True)
    parser.add_argument("--pair", action="append", nargs=2, default=[])
    parser.add_argument("--max-bits", type=int, default=26)
    args = parser.parse_args()
    rows = collections.OrderedDict(load_row(spec) for spec in args.row)
    for sc in ("ed", "td"):
        items = [json.loads(line) for line in open(os.path.join(args.bench, f"{sc}_test.jsonl"))]
        nearest = {}  # test image -> smallest distance to a fit or validation image
        for line in open(os.path.join(args.pairs, f"{sc}_pairs.jsonl")):
            p = json.loads(line)
            for a, b in (("a", "b"), ("b", "a")):
                if p[a + "_split"] == "test" and p[b + "_split"] in ("fit", "val") and p["bits"] <= args.max_bits:
                    nearest[p[a]] = min(nearest.get(p[a], 999), p["bits"])
        stratum = lambda it: 0 if it["image"] not in nearest else (2 if nearest[it["image"]] <= 7 else 1)
        groups = [[it for it in items if stratum(it) == k] for k in range(3)] + [items]
        print(f"## {sc}：测试题按「图像与训练或验证图像的相近程度」分层的准确率（%）")
        print()
        print("| 方法 | " + " | ".join(STRATA) + " | 全部 |")
        print("|---|" + "---:|" * 4)
        print("| 题数（图像数） | " + " | ".join(f"{len(g)}（{len({it['image'] for it in g})}）" for g in groups) + " |")
        for name, ok in rows.items():
            missing = sum(it["id"] not in ok for it in items)
            assert missing == 0, f"{name}: {missing} test questions of {sc} have no output"
            print(f"| {name} | " + " | ".join(f"{100 * np.mean([ok[it['id']] for it in g]):.1f}" if g else "—" for g in groups) + " |")
        print()
        if args.pair:
            print(f"## {sc}：分层的配对差值（百分点；95% 区间按图像重采样）")
            print()
            print("| 对比 | " + " | ".join(STRATA) + " | 全部 |")
            print("|---|" + "---|" * 4)
            for a, b in args.pair:
                cells = []
                for g in groups:
                    d = [rows[a][it["id"]] - rows[b][it["id"]] for it in g]
                    lo, hi = clustered_ci(g, d)
                    cells.append(f"{100 * np.mean(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）" if d else "—")
                print(f"| {a} − {b} | " + " | ".join(cells) + " |")
            print()


if __name__ == "__main__":
    main()
