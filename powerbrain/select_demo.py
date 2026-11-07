"""Problem (12) on the validation images: for every configuration of Omega, the accuracy alpha_omega(z) under each condition
z (clean, low light), the latency predicted per question by the latency equation of II-D with the coefficients calibrated on the board
(numbers_board.tex), and the expected accuracy under mixes of the conditions. Aggregates only.

    python select_demo.py --bench <bench dir> --results <results dir> \
        --config 'Tier 0, nano=f_listn2b_s0_val|f_listn2b_s0_val_cdark2|list' ... --mix 0.9 0.5 --json out.json > out.md

A configuration is 'name=<clean results>|<low-light results>|list or thumb[|nano or medium]' (policy g:e; detector nano by
default); the thumbnail tokens and the input length come from the records themselves. The medium detector's time per image
on the board is given with --t-det-medium ED TD (ms, medians over all test images, runs/p29/det_{ed,td}_m.jsonl of
2026-10-29); without it, its configurations get the latency without the detector (marked '+ T_D').
"""
import argparse
import glob
import json
import os

import numpy as np

import dupeffect

# numbers_board.tex (P29/P31 calibration on the RK3588, 2026-10-25): prefill beta, vision encoder gamma, call overhead t0,
# detector median per image (nano detector, test images)
BETA1, BETA2, GAMMA1, GAMMA2 = 3.4319, 0.000870746, 5.43727, 0.0243489
T0 = {"list": 42.08, "thumb": 229.96}
T_DET = {"nano": {"ed": 274.0, "td": 265.0}, "medium": None}


def records(path):
    out = {}
    for p in glob.glob(os.path.join(path, "*shard*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r["policy"] == "g:e":
                out[r["id"]] = r
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--mix", nargs="+", type=float, default=[0.9, 0.5], help="shares of clean images")
    parser.add_argument("--json")
    parser.add_argument("--t0", nargs=2, type=float, metavar=("LIST", "THUMB"), help="fixed time per call of each path, ms (cost_fit.dat)")
    parser.add_argument("--t-det-medium", nargs=2, type=float, metavar=("ED", "TD"), help="medium detector per image on the board, ms")
    args = parser.parse_args()
    if args.t0:
        T0.update(list=args.t0[0], thumb=args.t0[1])
    if args.t_det_medium:
        T_DET["medium"] = {"ed": args.t_det_medium[0], "td": args.t_det_medium[1]}
    items = {"clean": [json.loads(l) for sc in ("ed", "td") for l in open(os.path.join(args.bench, f"{sc}_val.jsonl"))],
             "dark2": [json.loads(l) for sc in ("ed", "td") for l in open(os.path.join(args.bench, "corrupt", f"{sc}_val@dark2.jsonl"))]}
    out = {}
    print("## 式 (12)：验证集上的 α 与板上预测延迟（2B；区间按图像重采样 2,000 次；延迟按 II-D 的延迟式逐题预测，ms）")
    print()
    print("| 配置 | 数据集 | 干净 α | 低光 α | 预测延迟中位（干净 / 低光） | 第 95 百分位（干净） | " + " | ".join(f"期望精度 {int(100 * m)}/{int(round(100 - 100 * m))}" for m in args.mix) + " |")
    print("|---|---|---:|---:|---:|---:|" + "---:|" * len(args.mix))
    for spec in args.config:
        name, rest = spec.split("=", 1)
        clean_res, dark_res, kind, *det = rest.split("|")
        det = det[0] if det else "nano"
        recs = {"clean": records(os.path.join(args.results, clean_res)), "dark2": records(os.path.join(args.results, dark_res))}
        for sc in ("ed", "td"):
            row = {}
            for z in ("clean", "dark2"):
                its = [i for i in items[z] if i["scenario"] == sc]
                assert all(i["id"] in recs[z] for i in its), (name, z, sc)
                v = [recs[z][i["id"]]["correct"] for i in its]
                lo, hi = dupeffect.clustered_ci(its, v)
                L = np.array([recs[z][i["id"]]["input_tokens"] for i in its], float)
                n = np.array([recs[z][i["id"]]["visual_tokens"] for i in its], float)
                T = (T_DET[det][sc] if T_DET[det] else 0.0) + GAMMA1 * n + GAMMA2 * n ** 2 + T0[kind] + BETA1 * L + BETA2 * L ** 2
                row[z] = {"acc": 100 * float(np.mean(v)), "lo": 100 * lo, "hi": 100 * hi, "n": len(its), "detector": det, "T_has_detector": bool(T_DET[det]),
                          "L_mean": float(L.mean()), "n_t": float(n.mean()),
                          "T_median": float(np.median(T)), "T_p95": float(np.percentile(T, 95)), "T_max": float(T.max())}
            row["expected"] = {str(m): m * row["clean"]["acc"] + (1 - m) * row["dark2"]["acc"] for m in args.mix}
            out.setdefault(name, {})[sc] = row
            c, d = row["clean"], row["dark2"]
            print(f"| {name} | {sc} | {c['acc']:.1f}（{c['lo']:.1f}–{c['hi']:.1f}） | {d['acc']:.1f}（{d['lo']:.1f}–{d['hi']:.1f}） | "
                  f"{c['T_median']:.0f} / {d['T_median']:.0f}{'' if T_DET[det] else ' + T_D'} | {c['T_p95']:.0f}{'' if T_DET[det] else ' + T_D'} | " + " | ".join(f"{row['expected'][str(m)]:.1f}" for m in args.mix) + " |")
    print()
    if args.json:
        json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
