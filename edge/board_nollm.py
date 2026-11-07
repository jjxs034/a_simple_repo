#!/usr/bin/env python3
"""On the board: the two pipelines without a language model (protocol P29, report item 3) answer the test questions from
the board's own detections, and the CPU time of their decision step is measured per question.

  rules       detector.rule_answer, thresholds tuned on the validation questions as in detector.py answer (on the
              server's detections of the validation images, since the board only holds the test images)
  aggregator  the small network of aggregator.py (64-64 MLP over candidate features), trained on the server on the
              out-of-fold nano candidates with the degraded training images (aggregator.py --save-weights), run with numpy

    python3 board_nollm.py --items ed_test.jsonl td_test.jsonl --tune-items ed_val.jsonl td_val.jsonl \
        --tune-detections ed_detections.jsonl td_detections.jsonl --detections det_ed_n.jsonl det_td_n.jsonl \
        --weights agg_nano_weights.json --out nollm.jsonl
"""
import argparse
import collections
import json
import math
import statistics
import time

import numpy as np

from board_compose import iou, select

ED_CLASSES = ["DaoXianYiWu", "DiaoChe", "ShiGongJiXie", "TaDiao", "YanHuo"]
TD_CLASSES = ["绑扎不规范", "并线线夹保护壳缺失", "耐张线夹保护壳缺失", "横杆腐蚀", "塔头损坏"]
CLASSES = {"ed": ED_CLASSES, "td": TD_CLASSES}
COUNT_OPTIONS = ["1 处", "2 处", "3 处", "4 处及以上"]
LETTERS = "ABCDEFGH"
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def truth(item):
    return item["options"][LETTERS.index(item["answer"])]


def rule_answer(item, dets, thr):  # detector.rule_answer
    if item["qtype"] == "cls":
        score = collections.Counter()
        for d in dets:
            score[d[5]] += d[4]
        ranked = [c for c, _ in score.most_common() if c in item["options"]]
        return ranked[0] if ranked else item["options"][0]
    hits = [d for d in dets if d[5] == item["target"] and d[4] >= thr]
    if item["qtype"] == "exist":
        return "是" if hits else "否"
    return COUNT_OPTIONS[min(max(len(hits), 1), 4) - 1]


def tune(items, dets):  # detector.py cmd_answer: per scenario and question type, threshold t/20 with the best tuning accuracy
    thr = {}
    for sc in ("ed", "td"):
        for q in ("exist", "count"):
            sub = [it for it in items if it["scenario"] == sc and it["qtype"] == q and it["image"] in dets]
            acc = lambda t: sum(rule_answer(it, dets[it["image"]], t) == truth(it) for it in sub) / max(1, len(sub))
            thr[(sc, q)] = max((acc(t / 20), t / 20) for t in range(1, 19))[1] if sub else 0.25
    return thr


def overlap(a, b):  # verify2.overlap
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return max(iou(a, b), inter / small if small > 0 else 0.0)


def suppressed(cands, tau=0.5):  # aggregator.suppressed
    kept = []
    for c in sorted(cands, key=lambda c: -c["conf"]):
        if all(overlap(c["box"], k["box"]) < tau for k in kept):
            kept.append(c)
    return kept


def features(item, cands):  # aggregator.features
    classes = CLASSES[item["scenario"]]
    area = lambda c: max(1e-7, (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) / (item["width"] * item["height"]))
    if item["qtype"] == "cls":
        out = []
        for k in classes:
            own = [c["conf"] for c in cands if c["det"] == k]
            out += [max(own, default=0.0), min(sum(own), 4.0) / 4, sum(v >= 0.25 for v in own) / 4, sum(v >= 0.5 for v in own) / 4]
        return out
    own = sorted((c for c in cands if c["det"] == item["target"]), key=lambda c: -c["conf"])
    rest = sorted((c["conf"] for c in cands if c["det"] != item["target"]), reverse=True)
    dedup = suppressed(own)
    out = [c["conf"] for c in own[:8]] + [0.0] * (8 - len(own[:8]))
    out += [sum(c["conf"] >= t for c in own) / 4 for t in THRESHOLDS]
    out += [sum(c["conf"] >= t for c in dedup) / 4 for t in THRESHOLDS]
    out += rest[:3] + [0.0] * (3 - len(rest[:3]))
    out += [math.log(area(c)) / 10 for c in own[:3]] + [0.0] * (3 - len(own[:3]))
    out += [float(item["target"] == k) for k in classes]
    return out


def label(item):  # aggregator.label
    t = truth(item)
    if item["qtype"] == "cls":
        return CLASSES[item["scenario"]].index(t)
    if item["qtype"] == "exist":
        return int(t == "是")
    return COUNT_OPTIONS.index(t)


def mlp(net, x):
    h = (np.asarray(x, np.float32) - net["mean"]) / net["std"]
    for k, (w, b) in enumerate(net["layers"]):
        h = w @ h + b
        if k < 2:
            h = np.maximum(h, 0.0)
    return int(h.argmax())


def load_dets(paths):
    out = {}
    for path in paths:
        for rec in map(json.loads, open(path)):
            out[rec["image"]] = rec["detections"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--tune-items", nargs="+", required=True)
    ap.add_argument("--tune-detections", nargs="+", required=True)
    ap.add_argument("--detections", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = [json.loads(l) for p in args.items for l in open(p)]
    thr = tune([json.loads(l) for p in args.tune_items for l in open(p)], load_dets(args.tune_detections))
    print("rule thresholds (tuned on the validation questions):", {f"{k[0]}/{k[1]}": v for k, v in thr.items()})
    nets = {k: {"mean": np.array(v["mean"], np.float32), "std": np.array(v["std"], np.float32),
                "layers": [(np.array(w, np.float32), np.array(b, np.float32)) for w, b in v["layers"]]} for k, v in json.load(open(args.weights)).items()}
    dets = load_dets(args.detections)
    t_rule, t_agg, rows = [], [], []
    for it in items:
        d = dets.get(it["image"], [])
        t0 = time.perf_counter()
        ok_rule = rule_answer(it, d, thr.get((it["scenario"], it["qtype"]), 0.25)) == truth(it)
        t1 = time.perf_counter()
        cands = select(d)
        ok_agg = mlp(nets[f"{it['scenario']}/{it['qtype']}"], features(it, cands)) == label(it)
        t2 = time.perf_counter()
        t_rule.append(t1 - t0); t_agg.append(t2 - t1)
        rows.append({"id": it["id"], "scenario": it["scenario"], "qtype": it["qtype"], "rule_correct": int(ok_rule), "agg_correct": int(ok_agg),
                     "rule_ms": round(1000 * (t1 - t0), 4), "agg_ms": round(1000 * (t2 - t1), 4)})
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    for sc in ("ed", "td", "all"):
        sub = [r for r in rows if sc == "all" or r["scenario"] == sc]
        print(f"{sc}: n {len(sub)}  rules {100 * sum(r['rule_correct'] for r in sub) / len(sub):.1f}%  aggregator {100 * sum(r['agg_correct'] for r in sub) / len(sub):.1f}%")
    print(f"decision step per question on the CPU: rules median {1000 * statistics.median(t_rule):.3f} ms (max {1000 * max(t_rule):.3f}); "
          f"aggregator incl. candidate selection median {1000 * statistics.median(t_agg):.3f} ms (max {1000 * max(t_agg):.3f})")


if __name__ == "__main__":
    main()
