"""Accuracy of several systems by stratum of the test questions, with paired differences per stratum
(stored per-question outputs only; intervals resample images).

  --by size      area of the question's evidence boxes relative to the image (five bins)
  --by pixels    megapixels of the source image (camera resolution groups)
  --by qtype     question form

    python strata.py --bench <bench dir> --by size --row 'name=<results>[:policy]' ... --pair A B
"""
import argparse
import collections
import json
import os

import numpy as np

import dupeffect

SIZE_BINS = ["<0.2%", "0.2–0.5%", "0.5–1%", "1–5%", "≥5%"]


def pixel_group(item):
    mp = item["width"] * item["height"] / 1e6
    return "≤1.5 百万像素" if mp <= 1.5 else ("1.5–4 百万像素" if mp <= 4 else ("4–6 百万像素" if mp <= 6 else ">6 百万像素"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--by", choices=["size", "pixels", "qtype"], required=True)
    parser.add_argument("--files", nargs="+", default=["test"])
    parser.add_argument("--row", action="append", required=True)
    parser.add_argument("--pair", action="append", nargs=2, default=[])
    args = parser.parse_args()
    rows = collections.OrderedDict(dupeffect.load_row(spec) for spec in args.row)
    key = {"size": lambda it: SIZE_BINS[it["bin"]], "pixels": pixel_group, "qtype": lambda it: it["qtype"]}[args.by]
    order = {"size": SIZE_BINS, "pixels": ["≤1.5 百万像素", "1.5–4 百万像素", "4–6 百万像素", ">6 百万像素"]}.get(args.by)
    title = {"size": "题目证据占画面的面积比例", "pixels": "原图像素数", "qtype": "问法"}[args.by]
    for sc in ("ed", "td"):
        items = [json.loads(line) for name in args.files for line in open(os.path.join(args.bench, f"{sc}_{name}.jsonl"))]
        groups = collections.OrderedDict()
        for k in (order or sorted({key(it) for it in items})):
            sub = [it for it in items if key(it) == k]
            if sub:
                groups[k] = sub
        groups["全部"] = items
        print(f"## {sc}：按{title}分层的准确率（%）")
        print()
        print("| 系统 | " + " | ".join(f"{k}（{len(v)} 题，{len({it['image'] for it in v})} 图）" for k, v in groups.items()) + " |")
        print("|---|" + "---:|" * len(groups))
        for name, ok in rows.items():
            if any(it["id"] not in ok for it in items):
                print(f"| {name} | 缺 {sum(it['id'] not in ok for it in items)} 题 |" + " |" * (len(groups) - 1))
                continue
            print(f"| {name} | " + " | ".join(f"{100 * np.mean([ok[it['id']] for it in sub]):.1f}" for sub in groups.values()) + " |")
        print()
        for a, b in args.pair:
            if any(it["id"] not in rows[a] or it["id"] not in rows[b] for it in items):
                continue
            cells = []
            for k, sub in groups.items():
                d = [rows[a][it["id"]] - rows[b][it["id"]] for it in sub]
                lo, hi = dupeffect.clustered_ci(sub, d)
                cells.append(f"{k} {100 * np.mean(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）")
            print(f"- {a} − {b}：" + "；".join(cells))
        print()


if __name__ == "__main__":
    main()
