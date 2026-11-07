"""95% intervals of every point of Fig. 3: the accuracy of each system on the clean test images and under each synthetic
degradation, with the interval that resamples test images (dupeffect.clustered_ci, 2,000 resamples, seed 0, as every
other interval of the paper). Same systems and result files as robustreport.py, which gives the points themselves.

    python robust_ci.py --bench <bench dir> --results <results dir> --runs fog:2 dark:2 ... \
        --system '名称=<干净结果>|<退化结果，{C} 处填退化名>' ... [--json out.json]
"""
import argparse
import json
import os

import numpy as np

import dupeffect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--system", action="append", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()
    tags = [r.replace(":", "") for r in args.runs]
    systems = [(spec.split("=", 1)[0], *spec.split("=", 1)[1].split("|")) for spec in args.system]

    def load(spec):
        return dupeffect.load_row("x=" + os.path.join(args.results, spec))[1]

    out = {}
    for sc in ("ed", "td"):
        items = [json.loads(line) for line in open(os.path.join(args.bench, f"{sc}_test.jsonl"))]
        print(f"## {sc}：各点的准确率与 95% 区间（%；{len(items)} 题；区间按图像重采样 2,000 次）")
        print()
        print("| 系统 | 干净 | " + " | ".join(tags) + " | 最大半宽 |")
        print("|---|---:|" + "---:|" * (len(tags) + 1))
        for name, clean, dirty in systems:
            cells, half = [], []
            for tag, spec in [("clean", clean)] + [(t, dirty.format(C=t)) for t in tags]:
                ok = load(spec)
                missing = [it["id"] for it in items if it["id"] not in ok]
                assert not missing, (name, tag, len(missing))
                vals = [ok[it["id"]] for it in items]
                acc = 100 * np.mean(vals)
                lo, hi = (100 * v for v in dupeffect.clustered_ci(items, vals))
                out.setdefault(sc, {}).setdefault(name, {})[tag] = [round(acc, 2), round(lo, 2), round(hi, 2)]
                cells.append(f"{acc:.1f}（{lo:.1f}–{hi:.1f}）")
                half.append(max(acc - lo, hi - acc))
            print(f"| {name} | " + " | ".join(cells) + f" | {max(half):.1f} |")
        print()
    if args.json:
        json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
