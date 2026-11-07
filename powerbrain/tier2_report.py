"""Tier 2 routed system on the whole ED test set (III-C; runs of tier2_loco.sh). For each held-out class X, PowerBrain is the
model trained without X (f_loco_one_X, evidence with X dropped from the candidates), and the class rule of II-B routes a
question that names X (its target is X: presence and count questions) to the un-adapted model reading the native-resolution
image (results/base, u8192); every other question stays with PowerBrain, including the questions that ask which class is
present, since they name no class. Accuracy with 95% intervals that resample test images (dupeffect.clustered_ci), paired
differences, the share routed, and the time per question weighted by that share from the v7 timing medians.

    python tier2_report.py --bench <bench dir> --results <results dir> --classes TaDiao ShiGongJiXie \
        --ms-ours 180 --ms-base 1075 > tier2_report.md
"""
import argparse
import glob
import json
import os

import numpy as np

import dupeffect


def records(path, policy):
    """{question id: correct} of the shard files of a result folder (shard*.jsonl only, not other runs kept beside them)."""
    ok = {}
    for p in sorted(glob.glob(os.path.join(path, "shard*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            if r["policy"] == policy:
                ok[r["id"]] = r["correct"]
    return ok


def acc(items, ok):
    v = [ok[i["id"]] for i in items]
    lo, hi = dupeffect.clustered_ci(items, v)
    return f"{100 * np.mean(v):.1f}（{100 * lo:.1f}–{100 * hi:.1f}）"


def diff(items, a, b):
    d = [a[i["id"]] - b[i["id"]] for i in items]
    lo, hi = dupeffect.clustered_ci(items, d)
    return f"{100 * np.mean(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--ms-ours", type=float, required=True, help="median time per question of PowerBrain (4B), ED, v7")
    parser.add_argument("--ms-base", type=float, required=True, help="median time of the un-adapted model at native resolution, ED, v7")
    args = parser.parse_args()
    items = [json.loads(line) for line in open(os.path.join(args.bench, "ed_test.jsonl"))]
    base = records(os.path.join(args.results, "base"), "u8192")
    assert all(i["id"] in base for i in items)
    for x in args.classes:
        loco = records(os.path.join(args.results, f"f_loco_one_{x}_all"), "g:e")
        assert all(i["id"] in loco for i in items), x
        named = [i for i in items if i.get("target") == x]
        other = [i for i in items if i.get("target") != x]
        routed = {i["id"]: (base if i.get("target") == x else loco)[i["id"]] for i in items}
        share = len(named) / len(items)
        print(f"## 留出 {x}：ED 测试集 {len(items)} 题，点名 {x} 的 {len(named)} 题（{100 * share:.1f}%）交给 Tier 2")
        print()
        print("| 系统 | 全部 | 点名留出类的题 | 其余的题 |")
        print("|---|---:|---:|---:|")
        for name, ok in (("分流后整系统（Tier 2 + 本文）", routed), ("本文（训练时没有该类）", loco), ("未适配模型，原始分辨率", base)):
            print(f"| {name} | {acc(items, ok)} | {acc(named, ok)} | {acc(other, ok)} |")
        print()
        print(f"- 分流后整系统 − 本文（训练时没有该类）：全部 {diff(items, routed, loco)}；点名留出类的题 {diff(named, routed, loco)}")
        print(f"- 分流后整系统 − 未适配模型，原始分辨率：全部 {diff(items, routed, base)}；其余的题 {diff(other, routed, base)}")
        t = share * args.ms_base + (1 - share) * args.ms_ours
        print(f"- 每题时间（按分流比例加权的中位数，ED）：{t:.0f} ms（本文 {args.ms_ours:.0f}，Tier 2 {args.ms_base:.0f}）")
        print()


if __name__ == "__main__":
    main()
