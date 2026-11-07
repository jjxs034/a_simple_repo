"""Report of the open-ended evaluation (P19): accuracy per question kind with intervals over images, paired differences
to a reference row, and the detector-rule control (thresholds tuned on the open-ended validation questions).

    python openformreport.py --bench <dir> --detections ed_detections.jsonl td_detections.jsonl --ref 本文 \
        --row '本文=<results>/f_open_ours:g:e' --row '整图 576 令牌=<results>/f_open_w576:u576' ...
"""
import argparse
import collections
import json
import os

import numpy as np

import detector
import dupeffect
import openform

KINDS = [("cls", "类别"), ("exist", "是否存在"), ("count", "计数"), ("inventory", "清点")]
BASE = ("cls", "exist", "count")


def rule_rows(bench, scenario, detections):
    """The detector answering by rule. Base forms: the rule of the multiple-choice benchmark (same decision, the options
    are only its output space). Inventory: per class, the number of detections at or above one threshold."""
    read = lambda split: [json.loads(line) for line in open(os.path.join(bench, f"{scenario}_{split}_open.jsonl"))]
    val, test = read("val"), read("test")
    ok = {}
    for q in BASE:
        tune = [it for it in val if it["qtype"] == q and it["image"] in detections]
        sub = [it for it in test if it["qtype"] == q]
        thr = max((detector.accuracy(tune, detections, t / 20)[0], t / 20) for t in range(1, 19))[1] if q != "cls" and tune else 0.25
        ok.update({it["id"]: bool(o) for it, o in zip(sub, detector.accuracy(sub, detections, thr)[1])})

    def inventory_ok(items, thr):
        out = []
        for it in items:
            have = collections.Counter(d[5] for d in detections.get(it["image"], []) if d[4] >= thr)
            truth = {c: openform.cap(n) for c, n in openform.counts_of(it).items()}
            out.append({c: openform.cap(n) for c, n in have.items()} == truth)
        return out

    tune = [it for it in val if it["qtype"] == "inventory"]
    thr = max((np.mean(inventory_ok(tune, t / 20)), t / 20) for t in range(1, 19))[1]
    sub = [it for it in test if it["qtype"] == "inventory"]
    ok.update({it["id"]: o for it, o in zip(sub, inventory_ok(sub, thr))})
    return ok


def cell(items, ok):
    values = [float(ok[it["id"]]) for it in items]
    lo, hi = dupeffect.clustered_ci(items, values)
    return f"{100 * np.mean(values):.1f}（{100 * lo:.1f}–{100 * hi:.1f}）"


def diff(items, a, b):
    d = [float(a[it["id"]]) - float(b[it["id"]]) for it in items]
    lo, hi = dupeffect.clustered_ci(items, d)
    return f"{100 * np.mean(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--detections", nargs="+", required=True)
    parser.add_argument("--row", action="append", default=[])
    parser.add_argument("--ref")
    args = parser.parse_args()
    detections = {}
    for path in args.detections:
        for line in open(path):
            rec = json.loads(line)
            detections[rec["image"]] = rec["detections"]
    rows = collections.OrderedDict(dupeffect.load_row(spec) for spec in args.row)
    for sc in ("ed", "td"):
        items = [json.loads(line) for line in open(os.path.join(args.bench, f"{sc}_test_open.jsonl"))]
        base = [it for it in items if it["qtype"] in BASE]
        inv = [it for it in items if it["qtype"] == "inventory"]
        table = collections.OrderedDict([("检测分支按规则（无语言模型）", rule_rows(args.bench, sc, detections))] + list(rows.items()))
        print(f"## {sc}：开放式评测（不带选项；准确率 %，括号内为按图像重采样的 95% 区间；基础题 {len(base)} 道，清点题 {len(inv)} 道）")
        print()
        print("| 系统 | 三类基础题合计 | " + " | ".join(name for _, name in KINDS) + (f" | 基础题：{args.ref} 减该行 | 清点题：{args.ref} 减该行 |" if args.ref else " |"))
        print("|---|---|" + "---:|" * 3 + "---|" + ("---|---|" if args.ref else ""))
        for name, ok in table.items():
            missing = sum(it["id"] not in ok for it in items)
            if missing:
                print(f"| {name} | 缺 {missing} 题 |" + " |" * (4 + (2 if args.ref else 0)))
                continue
            line = f"| {name} | {cell(base, ok)} | " + " | ".join(f"{100 * np.mean([ok[it['id']] for it in items if it['qtype'] == q]):.1f}" for q in BASE) + f" | {cell(inv, ok)} |"
            if args.ref:
                ref = table[args.ref]
                line += (" — | — |" if name == args.ref else f" {diff(base, ref, ok)} | {diff(inv, ref, ok)} |")
            print(line)
        print()


if __name__ == "__main__":
    main()
