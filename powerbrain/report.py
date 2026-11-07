"""Tables for one evaluation run: accuracy per policy, paired differences, degeneracy check.

    python report.py --results <dir> [--pairs o1-u8192 o1-ol ...] [--out report.md]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random

DEFAULT_PAIRS = ["u8192-u256", "o1-u256", "o1-u576", "o1-u8192", "o1-ol", "os2-ol", "o1-os2", "o0.5-o1", "o2-o1", "ol-u8192", "r1-u256"]
BIN_LABELS = ["<0.2%", "0.2–0.5%", "0.5–1%", "1–5%", "≥5%"]
QTYPES = ["cls", "exist", "count"]


def bootstrap_ci(values, reps=2000, seed=0):
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def balanced(sub):
    """Mean over question types of the mean per-answer-class accuracy: insensitive to the answer prior,
    so always giving the most frequent answer scores at chance."""
    per_type = []
    for q in QTYPES:
        by_truth = collections.defaultdict(list)
        for r in sub:
            if r["qtype"] == q:
                by_truth[r["truth"]].append(r["correct"])
        if by_truth:
            per_type.append(mean([mean(v) for v in by_truth.values()]))
    return mean(per_type)


def load(results_dir):
    rows = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(results_dir, "*shard*.jsonl"))):
        for line in open(path):
            row = json.loads(line)
            rows[row["policy"]][row["id"]] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = load(args.results)
    policies = sorted(rows, key=lambda p: mean([r["visual_tokens"] for r in rows[p].values()]))
    lines = [f"# {os.path.basename(os.path.normpath(args.results))}", ""]

    for scenario in ("ed", "td"):
        lines += [f"## {scenario}：准确率", "", "| 策略 | 样本 | 平均视觉令牌 | 准确率 | 95% 区间 | 类别均衡准确率 | " + " | ".join(QTYPES) + " | 平均耗时 s |", "|---|---:|---:|---:|---|---:|" + "---:|" * (len(QTYPES) + 1)]
        for p in policies:
            sub = [r for r in rows[p].values() if r["scenario"] == scenario]
            if not sub:
                continue
            acc = [r["correct"] for r in sub]
            lo, hi = bootstrap_ci(acc)
            per_type = [f"{100 * mean([r['correct'] for r in sub if r['qtype'] == q]):.1f}" for q in QTYPES]
            lines.append(
                f"| {p} | {len(sub)} | {mean([r['visual_tokens'] for r in sub]):.0f} | {100 * mean(acc):.1f}% | {100 * lo:.1f}–{100 * hi:.1f} | {100 * balanced(sub):.1f}% | "
                + " | ".join(per_type) + f" | {mean([r['seconds'] for r in sub]):.3f} |"
            )
        lines += ["", f"## {scenario}：配对差值（百分点）", "", "| 对比 | 全部 | " + " | ".join(QTYPES) + " | n |", "|---|---|" + "---|" * len(QTYPES) + "---:|"]
        for pair in args.pairs:
            a, b = pair.split("-", 1)
            if a not in rows or b not in rows:
                continue
            cells, n_all = [], 0
            for q in [None] + QTYPES:
                ids = [i for i, r in rows[a].items() if i in rows[b] and r["scenario"] == scenario and (q is None or r["qtype"] == q)]
                if not ids:
                    cells.append("—")
                    continue
                diff = [rows[a][i]["correct"] - rows[b][i]["correct"] for i in ids]
                lo, hi = bootstrap_ci(diff)
                cells.append(f"{100 * mean(diff):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）")
                if q is None:
                    n_all = len(ids)
            if n_all:
                lines.append(f"| {a} − {b} | " + " | ".join(cells) + f" | {n_all} |")
        lines += ["", f"## {scenario}：按证据面积占比分层的准确率", "", "| 策略 | " + " | ".join(BIN_LABELS) + " |", "|---|" + "---:|" * len(BIN_LABELS)]
        for p in policies:
            cells = []
            for b in range(len(BIN_LABELS)):
                acc = [r["correct"] for r in rows[p].values() if r["scenario"] == scenario and r["bin"] == b]
                cells.append(f"{100 * mean(acc):.1f} ({len(acc)})" if acc else "—")
            if any(c != "—" for c in cells):
                lines.append(f"| {p} | " + " | ".join(cells) + " |")
        lines += ["", f"## {scenario}：退化检查（各题型中出现最多的那个预测所占比例；括号内为真实答案中最多者的比例）", "", "| 策略 | " + " | ".join(QTYPES) + " |", "|---|" + "---:|" * len(QTYPES)]
        for p in policies:
            cells = []
            for q in QTYPES:
                sub = [r for r in rows[p].values() if r["scenario"] == scenario and r["qtype"] == q]
                if not sub:
                    cells.append("—")
                    continue
                pred = collections.Counter(r["pred"] for r in sub).most_common(1)[0][1] / len(sub)
                truth = collections.Counter(r["truth"] for r in sub).most_common(1)[0][1] / len(sub)
                cells.append(f"{100 * pred:.0f}% ({100 * truth:.0f}%)")
            if any(c != "—" for c in cells):
                lines.append(f"| {p} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
