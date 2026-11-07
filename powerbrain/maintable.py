"""The main results table from stored per-question outputs: accuracy with interval, answer-balanced accuracy, accuracy
per question form, visual and input tokens, forward time, and the paired difference to a reference row. Intervals
resample images. A row may pool several seeds of one system ('+' between result folders): its accuracy is the mean
over seeds (with their standard deviation) and paired differences use each question's mean correctness.

    python maintable.py --bench <bench dir> --ref '本文' --row '本文=<res>/a+<res>/b:g:e' --row '整图 576=<res>/u:u576' ... [--json numbers.json]
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

import core
import dupeffect

QTYPES = ("cls", "exist", "count")


def load(spec):
    name, src = spec.split("=", 1)
    paths, _, policy = src.partition(":")
    seeds = []
    for path in paths.split("+"):
        rows = {}
        for p in (glob.glob(os.path.join(path, "*shard*.jsonl")) if os.path.isdir(path) else [path]):
            for line in open(p):
                r = json.loads(line)
                if not policy or policy in (r.get("policy"), r.get("scorer")):
                    rows[r["id"]] = r
        seeds.append(rows)
    return name, seeds


def balanced(items, ok):
    per_type = []
    for q in QTYPES:
        by_truth = collections.defaultdict(list)
        for it in items:
            if it["qtype"] == q:
                by_truth[it["options"][core.LETTERS.index(it["answer"])]].append(ok[it["id"]])
        if by_truth:
            per_type.append(np.mean([np.mean(v) for v in by_truth.values()]))
    return float(np.mean(per_type))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--files", nargs="+", default=["test"])
    parser.add_argument("--row", action="append", required=True)
    parser.add_argument("--ref", help="name of the row the paired differences are taken against")
    parser.add_argument("--json")
    args = parser.parse_args()
    rows = collections.OrderedDict(load(spec) for spec in args.row)
    numbers = {}
    for sc in ("ed", "td"):
        items = [json.loads(line) for name in args.files for line in open(os.path.join(args.bench, f"{sc}_{name}.jsonl"))]
        print(f"## {sc}：主表（{len(items)} 题，{len({it['image'] for it in items})} 幅图像；区间按图像重采样，95%）")
        print()
        print("| 系统 | 准确率 | 类别均衡 | 类别题 | 是否存在 | 计数 | 看图比例 | 视觉令牌 | 输入令牌 | 前向耗时 ms | 对「" + (args.ref or "—") + "」的差值 |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        mean_ok = {}
        for name, seeds in rows.items():
            if any(it["id"] not in s for s in seeds for it in items):
                continue
            mean_ok[name] = {it["id"]: float(np.mean([s[it["id"]]["correct"] for s in seeds])) for it in items}
        for name, seeds in rows.items():
            if name not in mean_ok:
                print(f"| {name} | 缺 {sum(it['id'] not in s for s in seeds for it in items)} 题 |" + " |" * 9)
                continue
            ok = mean_ok[name]
            v = [ok[it["id"]] for it in items]
            lo, hi = dupeffect.clustered_ci(items, v)
            per_seed = [100 * np.mean([s[it["id"]]["correct"] for it in items]) for s in seeds]
            acc = f"{100 * np.mean(v):.1f}（{100 * lo:.1f}–{100 * hi:.1f}）" + (f"，{len(seeds)} 个种子标准差 {np.std(per_seed, ddof=1):.2f}" if len(seeds) > 1 else "")
            first = seeds[0]
            tok = [first[it["id"]].get("visual_tokens") for it in items]
            inp = [first[it["id"]].get("input_tokens") for it in items]
            sec = [first[it["id"]].get("seconds") for it in items]
            fmt = lambda x, f: f.format(np.mean(x)) if all(t is not None for t in x) else "—"
            # look-on-doubt rows (lookrows.py): share of the questions for which the vision encoder runs, over all seeds
            looked = [s[it["id"]].get("looked") for s in seeds for it in items]
            share = 100 * float(np.mean(looked)) if all(t is not None for t in looked) else None
            shown = f"{share:.1f}%" if share is not None else "—"
            diff, diff_numbers = "—", None
            if args.ref and args.ref in mean_ok and name != args.ref:
                d = [mean_ok[args.ref][it["id"]] - ok[it["id"]] for it in items]
                dlo, dhi = dupeffect.clustered_ci(items, d)
                diff = f"{100 * np.mean(d):+.1f}（{100 * dlo:+.1f} 到 {100 * dhi:+.1f}）"
                diff_numbers = [round(100 * float(np.mean(d)), 4), round(100 * dlo, 4), round(100 * dhi, 4)]
            q = {t: 100 * np.mean([ok[it["id"]] for it in items if it["qtype"] == t]) for t in QTYPES}
            bal = 100 * balanced(items, ok)
            print(f"| {name} | {acc} | {bal:.1f} | {q['cls']:.1f} | {q['exist']:.1f} | {q['count']:.1f} | {shown} | {fmt(tok, '{:.0f}')} | {fmt(inp, '{:.0f}')} | "
                  f"{1000 * float(np.median(sec)):.0f} | {diff} |" if all(t is not None for t in sec) else
                  f"| {name} | {acc} | {bal:.1f} | {q['cls']:.1f} | {q['exist']:.1f} | {q['count']:.1f} | {shown} | — | — | — | {diff} |")
            numbers[f"{sc}/{name}"] = {"accuracy": round(100 * float(np.mean(v)), 4), "ci": [round(100 * lo, 4), round(100 * hi, 4)], "balanced": round(bal, 4),
                                       **{k: round(float(x), 4) for k, x in q.items()}, "seeds": [round(float(x), 4) for x in per_seed],
                                       "sd": round(float(np.std(per_seed, ddof=1)), 4) if len(per_seed) > 1 else None,
                                       "ref_minus_this": diff_numbers,  # paired difference (reference row minus this row) with its interval
                                       "looked": round(share, 2) if share is not None else None,
                                       "visual_tokens": round(float(np.mean(tok)), 1) if all(t is not None for t in tok) else None,
                                       "input_tokens": round(float(np.mean(inp)), 1) if all(t is not None for t in inp) else None}
        print()
    if args.json:
        json.dump(numbers, open(args.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
