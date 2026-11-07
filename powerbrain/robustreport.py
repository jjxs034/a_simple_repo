"""Accuracy under synthetic image corruptions (corrupt.py, chain_robust.sh): every system on the clean test images and
on each corrupted copy, the mean over the corruptions, and paired differences pooled over the corruptions.

    python robustreport.py --bench <bench dir> --results <results dir> --runs fog:2 dark:2 ... \
        [--system '名称=<干净结果>|<退化结果，{C} 处填退化名>' ...] [--pair A B ...]

Without --system the rows of the first robustness test (P13) are used; result paths are relative to --results.
"""
import argparse
import json
import os

import numpy as np

import dupeffect

SYSTEMS = [  # (name, clean results, corrupted results with {C})
    ("检测分支按规则", "detector_v2_test.jsonl", "detector_c{C}.jsonl"),
    ("检测分支 + 小汇总网络", "aggregator_v1.jsonl", "aggregator_c{C}.jsonl"),
    ("只读清单的语言模型（不给图像）", "sft_pack_noimg:g:e", "sft_pack_noimg_c{C}:g:e"),
    ("一体化模型（缩略图 + 清单）", "sft_pack_text:g:e", "sft_pack_text_c{C}:g:e"),
    ("整图大模型 576 令牌", "sft_uniform_fit:u576", "sft_uniform_fit_c{C}:u576"),
    ("整图大模型原始分辨率", "sft_uniform_fit:u8192", "sft_uniform_fit_c{C}:u8192"),
]
PAIRS = [("一体化模型（缩略图 + 清单）", "检测分支 + 小汇总网络"), ("一体化模型（缩略图 + 清单）", "检测分支按规则"),
         ("一体化模型（缩略图 + 清单）", "只读清单的语言模型（不给图像）"), ("整图大模型 576 令牌", "检测分支按规则")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--system", action="append", default=[])
    parser.add_argument("--pair", action="append", nargs=2, default=[])
    args = parser.parse_args()
    tags = [r.replace(":", "") for r in args.runs]
    systems = [(spec.split("=", 1)[0], *spec.split("=", 1)[1].split("|")) for spec in args.system] or SYSTEMS
    pairs = [tuple(p) for p in args.pair] or (PAIRS if not args.system else [])

    def load(spec):
        try:
            return dupeffect.load_row("x=" + os.path.join(args.results, spec))[1]
        except (FileNotFoundError, OSError):
            return {}

    clean = {name: load(c) for name, c, _ in systems}
    dirty = {name: {t: load(d.format(C=t)) for t in tags} for name, _, d in systems}
    for sc in ("ed", "td"):
        items = [json.loads(line) for line in open(os.path.join(args.bench, f"{sc}_test.jsonl"))]
        print(f"## {sc}：干净图像与 {len(tags)} 种合成退化下的准确率（%；{len(items)} 题；各行读同一幅退化图像，测试时不重调任何阈值）")
        print()
        print("| 系统 | 干净 | " + " | ".join(tags) + " | 退化平均 | 平均下降 |")
        print("|---|---:|" + "---:|" * (len(tags) + 2))
        for name, _, _ in systems:
            if any(it["id"] not in clean[name] for it in items):
                print(f"| {name} | 缺干净结果 |" + " |" * (len(tags) + 2))
                continue
            base = 100 * np.mean([clean[name][it["id"]] for it in items])
            cells, vals = [], []
            for t in tags:
                ok = dirty[name][t]
                if any(it["id"] not in ok for it in items):
                    cells.append("—")
                    continue
                vals.append(100 * np.mean([ok[it["id"]] for it in items]))
                cells.append(f"{vals[-1]:.1f}")
            mean = f"{np.mean(vals):.1f}" if len(vals) == len(tags) else "—"
            drop = f"{base - np.mean(vals):.1f}" if len(vals) == len(tags) else "—"
            print(f"| {name} | {base:.1f} | " + " | ".join(cells) + f" | {mean} | {drop} |")
        print()
        print(f"## {sc}：配对差值，{len(tags)} 种退化合并（百分点；95% 区间按图像重采样）")
        print()
        for a, b in pairs:
            pooled_items, d, dd = [], [], []
            complete = all(it["id"] in dirty[a][t] and it["id"] in dirty[b][t] for t in tags for it in items) and all(it["id"] in clean[a] and it["id"] in clean[b] for it in items)
            if not complete:
                continue
            for t in tags:
                for it in items:
                    pooled_items.append(it)
                    d.append(dirty[a][t][it["id"]] - dirty[b][t][it["id"]])
                    # difference of the drops: (clean - corrupted) of b minus that of a; positive = a loses less
                    dd.append((clean[b][it["id"]] - dirty[b][t][it["id"]]) - (clean[a][it["id"]] - dirty[a][t][it["id"]]))
            lo, hi = dupeffect.clustered_ci(pooled_items, d)
            lo2, hi2 = dupeffect.clustered_ci(pooled_items, dd)
            print(f"- {a} − {b}：退化图像上 {100 * np.mean(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）；前者少掉 {100 * np.mean(dd):+.1f}（{100 * lo2:+.1f} 到 {100 * hi2:+.1f}）")
        print()


if __name__ == "__main__":
    main()
