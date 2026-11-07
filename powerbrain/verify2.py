"""Second iteration of the aggregation step of find-then-verify, on stored per-candidate outputs (no model calls).

verify_v1 failed its preset criteria because a per-candidate verifier accepts duplicate boxes (a second box on an
object that is already counted): it sees one candidate at a time. Here the decision is split in two:
  is this a real object, and of which class?   -> the second stage's score (LLM verifier, small controls, or the detector)
  is this a second box on an object already kept? -> geometry and detector confidence, by one of
        nms    suppress when overlap with a kept same-class candidate is high (overlap = max(IoU, intersection / smaller box))
        dup    a small model on overlap features, trained on the out-of-fold training candidates, predicts P(duplicate)

  develop   everything is tuned and compared on VALIDATION questions only, two-fold (tune on one half of the
            validation images, score the other half). Test questions are not read.
  final     the chosen variants, tuned on all validation questions, evaluated once on the test questions.

    python verify2.py develop --cands cands_eval_v1.jsonl --fit-cands cands_fit_v1.jsonl --tune-items ed_val.jsonl td_val.jsonl \
        --mllm-results results/verifier_v1 --cls-probs resnet=... calib=... --out verify_v2_develop.md
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math

import core
import verify
from verify import NONE, iou

GRID = [round(0.02 * k, 2) for k in range(1, 50)]


def overlap(a, b):
    """max(IoU, intersection over the smaller box): a box inside another one counts as fully overlapping."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return max(iou(a, b), inter / small if small > 0 else 0.0)


def dup_features(cands, i):
    """Overlap of candidate i with the candidates the detector is more confident about (cands are sorted by confidence)."""
    c = cands[i]
    box, area = c["box"], max(1e-6, (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]))
    best = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # IoU, intersection/own area, intersection/other area, other's confidence, same class, log area ratio
    for j in range(i):
        o = cands[j]
        x0, y0, x1, y1 = max(box[0], o["box"][0]), max(box[1], o["box"][1]), min(box[2], o["box"][2]), min(box[3], o["box"][3])
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if inter <= 0:
            continue
        oa = max(1e-6, (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
        v = iou(box, o["box"])
        if v > best[0]:
            best = [v, inter / area, inter / oa, o["conf"], float(o["det"] == c["det"]), math.log(area / oa) / 5]
    return best + [c["conf"], float(i) / 12, sum(1 for j in range(i) if overlap(box, cands[j]["box"]) > 0.3) / 5]


def fit_dup_model(fit_cands_paths, seed=0):
    """P(duplicate | overlap features) from the out-of-fold training candidates that lie on an annotated object."""
    import torch

    torch.manual_seed(seed)
    xs, ys = [], []
    for path in fit_cands_paths:
        for line in open(path):
            rec = json.loads(line)
            for i, c in enumerate(rec["cands"]):
                if c["label"] != NONE:
                    xs.append(dup_features(rec["cands"], i))
                    ys.append(float(c["dup"]))
    x, y = torch.tensor(xs), torch.tensor(ys)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    for epoch in range(30):
        order = torch.randperm(len(x))
        for s in range(0, len(order), 512):
            idx = order[s : s + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x[idx]).squeeze(1), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = torch.sigmoid(net(x).squeeze(1))
        acc = ((p > 0.5).float() == y).float().mean().item()
    print(f"duplicate model: {len(y)} training candidates on objects, {100 * y.mean().item():.1f}% duplicates, training accuracy {100 * acc:.1f}%", flush=True)

    def predict(cands):
        if not cands:
            return []
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor([dup_features(cands, i) for i in range(len(cands))])).squeeze(1)).tolist()

    return predict


def answer(item, cands, theta, mode, tau):
    """cands: [{score, cls, box, conf, pdup}] of the image. Returns the option text.
    mode 'iou': v1 rule (kept by score, same-class IoU >= tau suppressed); 'nms': kept in order of detector confidence,
    same-class overlap >= tau suppressed; 'dup': candidates with P(duplicate) >= tau are dropped, no pairwise rule."""
    kept = []
    if mode == "dup":
        kept = [c for c in cands if c["score"] >= theta and c["pdup"] < tau]
    else:
        key = (lambda c: -c["score"]) if mode == "iou" else (lambda c: -c["conf"])
        measure = iou if mode == "iou" else overlap
        for c in sorted((c for c in cands if c["score"] >= theta), key=key):
            if all(not (c["cls"] == k["cls"] and measure(c["box"], k["box"]) >= tau) for k in kept):
                kept.append(c)
    if item["qtype"] == "cls":
        votes = collections.Counter()
        for c in kept:
            votes[c["cls"]] += c["score"]
        ranked = [k for k, _ in votes.most_common() if k in item["options"]]
        if ranked:
            return ranked[0]
        fallback = [c["cls"] for c in sorted(cands, key=lambda c: -c["score"]) if c["cls"] in item["options"]]
        return fallback[0] if fallback else item["options"][0]
    n = sum(c["cls"] == item["target"] for c in kept)
    if item["qtype"] == "exist":
        return "是" if n else "否"
    return verify.COUNT_OPTIONS[min(max(n, 1), 4) - 1]


VARIANTS = {  # name -> (aggregation mode, values of tau tried)
    "iou": ("iou", [0.3, 0.5]),
    "nms": ("nms", [0.3, 0.5, 0.7, 0.9]),
    "dup": ("dup", [0.3, 0.5, 0.7, 0.9]),
}


def build(args, images):
    """image -> scorer name -> list of candidate dicts."""
    llm, small = verify.load_scores(args)
    predict = fit_dup_model(args.fit_cands)
    table = {}
    for path in args.cands:
        for line in open(path):
            rec = json.loads(line)
            if rec["image"] not in images:
                continue
            pdup = predict(rec["cands"])
            per = collections.defaultdict(list)
            for i, c in enumerate(rec["cands"]):
                cid = verify.cand_id(rec["image"], i)
                base = {"box": c["box"], "conf": c["conf"], "pdup": pdup[i]}
                per["det"].append({**base, "score": c["conf"], "cls": c["det"]})
                per["ideal-object"].append({**base, "score": float(c["label"] != NONE), "cls": c["label"] if c["label"] != NONE else c["det"]})
                per["ideal"].append({**base, "score": float(c["label"] != NONE and not c["dup"]), "cls": c["label"] if c["label"] != NONE else c["det"]})
                for name, rows in llm.items():
                    r = rows[cid]
                    assert r["truth"] == c["label"], (cid, r["truth"], c["label"])
                    options = verify.options_of(rec["scenario"], cid)
                    m = max(r["logp"])
                    z = sum(math.exp(v - m) for v in r["logp"])
                    probs = {o: math.exp(v - m) / z for o, v in zip(options, r["logp"])}
                    s, k = 1.0 - probs[NONE], max((o for o in probs if o != NONE), key=lambda o: probs[o])
                    per[name].append({**base, "score": s, "cls": k})
                    for a in args.fuse:  # geometric mean with the detector's confidence: keeps the detector's low scores for duplicates
                        per[f"{name}*det^{a:g}"].append({**base, "score": s ** (1 - a) * c["conf"] ** a, "cls": k})
                for name, rows in small.items():
                    probs = rows[cid]
                    s, k = 1.0 - probs.get(NONE, 0.0), max((o for o in probs if o != NONE), key=lambda o: probs[o])
                    per[name].append({**base, "score": s, "cls": k})
                    for a in args.fuse:  # the same fusion for the small second stages, so that it is not an advantage of the LLM
                        per[f"{name}*det^{a:g}"].append({**base, "score": s ** (1 - a) * c["conf"] ** a, "cls": k})
            table[rec["image"]] = per
    return table


def tune(items, table, scorer, variant, truth):
    mode, taus = VARIANTS[variant]
    best = (-1.0, 0.5, taus[0])
    for tau in taus:
        for theta in GRID:
            acc = sum(answer(it, table[it["image"]][scorer] if it["image"] in table else [], theta, mode, tau) == truth(it) for it in items) / max(1, len(items))
            if (acc, -abs(theta - 0.5)) > (best[0], -abs(best[1] - 0.5)):
                best = (acc, theta, tau)
    return best


def evaluate(items, table, scorer, variant, theta, tau, truth):
    mode, _ = VARIANTS[variant]
    return {it["id"]: int(answer(it, table[it["image"]][scorer] if it["image"] in table else [], theta, mode, tau) == truth(it)) for it in items}


def cells(items):
    return [(sc, q, [it for it in items if it["scenario"] == sc and it["qtype"] == q]) for sc in ("ed", "td") for q in ("cls", "exist", "count")]


def table_lines(title, correct, items, ref=None):
    lines = [title, "", "| 打分者 · 去重方式 | 外力破坏 | cls | exist | count | 塔线缺陷 | cls | exist | count |" + (" 相对参照（外力破坏 / 塔线缺陷，百分点） |" if ref else ""),
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---|" if ref else "")]
    acc = lambda ok, pool: f"{100 * sum(ok[it['id']] for it in pool) / len(pool):.1f}" if pool else "—"
    for name, ok in correct.items():
        row = []
        for sc in ("ed", "td"):
            sub = [it for it in items if it["scenario"] == sc]
            row += [acc(ok, sub)] + [acc(ok, [it for it in sub if it["qtype"] == q]) for q in ("cls", "exist", "count")]
        extra = ""
        if ref:
            parts = []
            for sc in ("ed", "td"):
                sub = [it for it in items if it["scenario"] == sc]
                d = [ok[it["id"]] - correct[ref][it["id"]] for it in sub]
                lo, hi = verify.clustered_ci(sub, d)
                parts.append(f"{100 * sum(d) / len(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）")
            extra = " " + " / ".join(parts) + " |"
        lines.append(f"| {name} | " + " | ".join(row) + " |" + extra)
    return lines


def cmd_develop(args):
    items = [json.loads(line) for path in args.tune_items for line in open(path)]
    truth = lambda it: it["options"][core.LETTERS.index(it["answer"])]
    table = build(args, {it["image"] for it in items})
    half = lambda it: int(hashlib.md5(("v2:" + it["image"]).encode()).hexdigest(), 16) % 2
    scorers = sorted({s for per in table.values() for s in per})
    correct = collections.OrderedDict()
    for scorer in scorers:
        for variant in VARIANTS:
            if scorer == "ideal" and variant != "dup":
                continue
            ok = {}
            for sc, q, sub in cells(items):
                for h in (0, 1):  # tune on one half of the validation images, score the other half
                    _, theta, tau = tune([it for it in sub if half(it) != h], table, scorer, variant, truth)
                    if scorer == "ideal":
                        theta, tau = 0.5, 1.01
                    ok.update(evaluate([it for it in sub if half(it) == h], table, scorer, variant, theta, tau, truth))
            correct[f"{scorer} · {variant}"] = ok
    lines = table_lines("## 验证集上的两折结果（%）：在一半验证图像上选阈值，在另一半上计分；不涉及测试集", correct, items, ref="det · iou")
    lines += ["", "去重方式：iou = 第一版规则（按得分保留，同类 IoU 过阈值者抑制）；nms = 按检测器置信度保留，同类重叠（IoU 与「交集占较小框比例」取大）过阈值者抑制；dup = 用折外训练候选学到的重复框概率过阈值者剔除。",
              "ideal-object = 假设「是不是目标、是哪一类」全对，重复框仍靠去重方式处理；ideal = 两者都对。"]
    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


def cmd_final(args):
    tune_items = [json.loads(line) for path in args.tune_items for line in open(path)]
    test = [json.loads(line) for path in args.items for line in open(path)]
    truth = lambda it: it["options"][core.LETTERS.index(it["answer"])]
    table = build(args, {it["image"] for it in tune_items + test})
    correct, notes = collections.OrderedDict(), []
    for spec in args.variants:
        scorer, variant = [s.strip() for s in spec.split("·")]
        ok = {}
        for sc, q, sub_test in cells(test):
            acc, theta, tau = tune([it for it in tune_items if it["scenario"] == sc and it["qtype"] == q], table, scorer, variant, truth)
            if scorer == "ideal":
                theta, tau = 0.5, 1.01
            ok.update(evaluate(sub_test, table, scorer, variant, theta, tau, truth))
            notes.append(f"{spec} {sc} {q}: threshold {theta:.2f}, tau {tau:g} (validation accuracy {100 * acc:.1f}%)")
        correct[spec] = ok
    lines = table_lines("## 测试集（%）：阈值与去重参数在全部验证题上选定", correct, test, ref=args.variants[0])
    lines += [""] + [f"- {n}" for n in notes]
    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        with open(args.out.replace(".md", ".jsonl"), "w") as f:
            for name, per in correct.items():
                for i, v in per.items():
                    f.write(json.dumps({"scorer": name, "id": i, "correct": v}, ensure_ascii=False) + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("develop", cmd_develop), ("final", cmd_final)):
        p = sub.add_parser(name)
        p.add_argument("--cands", nargs="+", required=True)
        p.add_argument("--fit-cands", nargs="+", required=True)
        p.add_argument("--tune-items", nargs="+", required=True)
        p.add_argument("--mllm-results", nargs="*")
        p.add_argument("--cls-probs", nargs="*")
        p.add_argument("--fuse", type=float, nargs="*", default=[0.25, 0.5])
        p.add_argument("--out")
        if name == "final":
            p.add_argument("--items", nargs="+", required=True)
            p.add_argument("--variants", nargs="+", required=True, help="'scorer · variant' rows; the first is the reference for paired differences")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
