"""Accuracy on the question forms beyond the three base forms (bench_x.py), for several systems, plus a hand-written-rule
baseline on the detector's output (one rule per form, reusing the thresholds tuned on the base forms).

    python xreport.py --bench <bench dir> --detections ed_detections.jsonl td_detections.jsonl [--files testx testx2] \\
        --row '大模型读缩略图+清单=<results>/sft_pack_text_x:g:e' --row '大模型读整图=<results>/sft_uniform_fit_x:u8192' [--pair A B]
        --heldout '多问法训练（该问法未参与训练）=<results of the model trained on half A>:g:e|<results of the model trained on half B>:g:e'
        --seen    '多问法训练（该问法参与了训练）=<same two>'
A --heldout row takes the questions of half B from the first model and those of half A from the second, so every answer
in it comes from a model that was not trained on that form; a --seen row does the opposite (reference).
"""
import argparse
import collections
import json
import os

import numpy as np

import bench_x
import core
import detector
import dupeffect

XTYPES = bench_x.FORMS["old"] + bench_x.FORMS["new"]
RULE = "人工规则读检测结果（每种问法各写一条）"


def tuned_thresholds(bench_dir, detections):
    """Presence and counting thresholds per scenario, tuned on the validation questions of the base forms."""
    out = {}
    for sc in ("ed", "td"):
        val = [json.loads(line) for line in open(os.path.join(bench_dir, f"{sc}_val.jsonl"))]
        for q in ("exist", "count"):
            pool = [it for it in val if it["qtype"] == q and it["image"] in detections]
            out[(sc, q)] = max((detector.accuracy(pool, detections, t / 20)[0], t / 20) for t in range(1, 19))[1] if pool else 0.25
    return out


def rule_answer(item, dets, thr_exist, thr_count):
    n = lambda c, t: sum(1 for d in dets if d[5] == c and d[4] >= t)
    yes = lambda flag: "是" if flag else "否"
    q, classes = item["qtype"], detector.CLASSES[item["scenario"]]
    kept = [d for d in dets if d[4] >= thr_exist]
    if q in ("x-more", "x-both", "x-either"):
        a, b = item["target"].split("|")
        if q == "x-more":
            return yes(n(a, thr_count) > n(b, thr_count))
        return yes(n(a, thr_exist) and n(b, thr_exist)) if q == "x-both" else yes(n(a, thr_exist) or n(b, thr_exist))
    if q == "x-most":
        return max(classes, key=lambda c: (n(c, thr_count), sum(d[4] for d in dets if d[5] == c)))
    if q == "x-total":
        return detector.COUNT_OPTIONS[min(max(sum(n(c, thr_count) for c in classes), 1), 4) - 1]
    if q == "x-left":
        return yes(any(d[5] == item["target"] and (d[0] + d[2]) / 2 < item["width"] / 2 for d in kept))
    if q == "x-upper":
        return yes(any(d[5] == item["target"] and (d[1] + d[3]) / 2 < item["height"] / 2 for d in kept))
    if q == "x-only":
        return yes({d[5] for d in kept} == {item["target"]})
    if q == "x-atleast":
        a, least = item["target"].split("|")
        return yes(n(a, thr_count) >= int(least))
    if q == "x-kinds":
        return bench_x.KIND_OPTIONS[min(max(len({d[5] for d in kept}), 1), 3) - 1]
    kept = kept or dets
    if not kept:
        return classes[0]
    if q == "x-leftmost":
        return min(kept, key=lambda d: (d[0] + d[2]) / 2)[5]
    assert q == "x-largest", q
    return max(kept, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))[5]


def load_multi(spec):
    """Like dupeffect.load_row, but the path may be several result folders joined by '+' (one per question file)."""
    name, src = spec.split("=", 1)
    paths, _, policy = src.partition(":")
    ok = {}
    for path in paths.split("+"):
        ok.update(dupeffect.load_row(f"x={path}{':' + policy if policy else ''}")[1])
    return name, ok


def split_row(spec, items, which):
    """A row assembled from two models, each trained on one half of the forms: `which` = "heldout" takes every form
    from the model that was NOT trained on it, "seen" from the one that was."""
    name, src = spec.split("=", 1)
    first, second = src.split("|")
    ok_a, ok_b = load_multi("a=" + first)[1], load_multi("b=" + second)[1]  # trained on half A / on half B
    half = {q: h for h, forms in bench_x.FOLDS.items() for q in forms}
    ok = {}
    for it in items:
        trained_on_form = ok_a if half[it["qtype"]] == "A" else ok_b
        other = ok_b if half[it["qtype"]] == "A" else ok_a
        source = other if which == "heldout" else trained_on_form
        if it["id"] in source:
            ok[it["id"]] = source[it["id"]]
    return name, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--detections", nargs="+", required=True)
    parser.add_argument("--files", nargs="+", default=["testx"])
    parser.add_argument("--row", action="append", default=[])
    parser.add_argument("--heldout", action="append", default=[])
    parser.add_argument("--seen", action="append", default=[])
    parser.add_argument("--pair", action="append", nargs=2, default=[])
    args = parser.parse_args()
    detections = {}
    for path in args.detections:
        for line in open(path):
            rec = json.loads(line)
            detections[rec["image"]] = rec["detections"]
    thr = tuned_thresholds(args.bench, detections)
    rows = collections.OrderedDict()
    rows[RULE] = {}
    items = {sc: [json.loads(line) for name in args.files for line in open(os.path.join(args.bench, f"{sc}_{name}.jsonl"))] for sc in ("ed", "td")}
    everything = items["ed"] + items["td"]
    for sc in ("ed", "td"):
        for it in items[sc]:
            truth = it["options"][core.LETTERS.index(it["answer"])]
            rows[RULE][it["id"]] = int(rule_answer(it, detections.get(it["image"], []), thr[(sc, "exist")], thr[(sc, "count")]) == truth)
    rows.update(load_multi(spec) for spec in args.row)
    rows.update(split_row(spec, everything, "heldout") for spec in args.heldout)
    rows.update(split_row(spec, everything, "seen") for spec in args.seen)
    for sc in ("ed", "td"):
        sub = items[sc]
        forms = [q for q in XTYPES if any(it["qtype"] == q for it in sub)]
        print(f"## {sc}：三类基础问法之外的 {len(forms)} 种问法上的准确率（%；共 {len(sub)} 题；答案由人工标注按固定规则导出）")
        print()
        print("| 方法 | 全部（95% 区间，按图像重采样） | " + " | ".join(f"{q[2:]}（{sum(it['qtype'] == q for it in sub)}）" for q in forms) + " |")
        print("|---|---|" + "---:|" * len(forms))
        for name, ok in rows.items():
            missing = sum(it["id"] not in ok for it in sub)
            if missing:
                print(f"| {name} | 缺 {missing} 题的输出 | " + " | ".join("—" for _ in forms) + " |")
                continue
            v = [ok[it["id"]] for it in sub]
            lo, hi = dupeffect.clustered_ci(sub, v)
            cells = [f"{100 * np.mean([ok[it['id']] for it in sub if it['qtype'] == q]):.1f}" for q in forms]
            print(f"| {name} | {100 * np.mean(v):.1f}（{100 * lo:.1f} 到 {100 * hi:.1f}） | " + " | ".join(cells) + " |")
        print()
        if args.pair:
            print(f"## {sc}：配对差值（百分点；95% 区间按图像重采样）")
            print()
            for a, b in args.pair:
                if any(it["id"] not in rows[a] or it["id"] not in rows[b] for it in sub):
                    continue
                d = [rows[a][it["id"]] - rows[b][it["id"]] for it in sub]
                lo, hi = dupeffect.clustered_ci(sub, d)
                print(f"- {a} − {b}：{100 * np.mean(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）")
            print()
    print("人工规则沿用的阈值（在基础问法的验证题上调得）：" + "；".join(f"{sc} 是否存在 {thr[(sc, 'exist')]:.2f}、计数 {thr[(sc, 'count')]:.2f}" for sc in ("ed", "td")) + "。")
    print("两半问法（事先写定）：" + "；".join(f"{h} = {'、'.join(q[2:] for q in forms)}" for h, forms in bench_x.FOLDS.items()) + "。")


if __name__ == "__main__":
    main()
