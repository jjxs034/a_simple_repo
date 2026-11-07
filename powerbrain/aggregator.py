"""A learned aggregator without any LLM: a small MLP that reads the detector branch's candidates of an image (as numbers)
and answers the benchmark question. It gets the same information as the LLM that reads the candidate list as text, the
same training questions and the same out-of-fold candidates, and no image. It answers the question "is the LLM doing
anything beyond a learned replacement of the hand-written rules?".

    python aggregator.py --fit-cands cands_fit_v1.jsonl --eval-cands cands_eval_v1.jsonl --bench <bench dir> --out aggregator_v1.jsonl
"""
import argparse
import collections
import copy
import json
import math
import os

import torch

import core
import verify
import verify2

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QTYPES = ("cls", "exist", "count")


def suppressed(cands, tau=0.5):
    kept = []
    for c in sorted(cands, key=lambda c: -c["conf"]):
        if all(verify2.overlap(c["box"], k["box"]) < tau for k in kept):
            kept.append(c)
    return kept


def features(item, rec):
    cands = rec["cands"] if rec else []
    classes = verify.CLASSES[item["scenario"]]
    area = lambda c: max(1e-7, (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) / (rec["width"] * rec["height"]))
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


def label(item):
    truth = item["options"][core.LETTERS.index(item["answer"])]
    if item["qtype"] == "cls":
        return verify.CLASSES[item["scenario"]].index(truth)
    if item["qtype"] == "exist":
        return int(truth == "是")  # the option order is shuffled per question: the label is the answer itself, not its position
    return verify.COUNT_OPTIONS.index(truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-cands", nargs="+", required=True)
    parser.add_argument("--eval-cands", nargs="+", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--suffix", nargs=3, default=["fit", "val", "test"], help="question files <scenario>_<suffix>.jsonl used for training, model selection and testing")
    parser.add_argument("--scenarios", nargs="+", default=["ed", "td"])
    parser.add_argument("--drop-class", default="", help="the detector branch has never been taught this class: its candidates of the class are removed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-weights", help="write the selected networks with their input normalisation as json (to run them on the board with numpy)")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    saved = {}
    recs = {}
    for path in args.fit_cands + args.eval_cands:
        for line in open(path):
            rec = json.loads(line)
            rec["cands"] = [c for c in rec["cands"] if c["det"] != args.drop_class]
            recs[rec["image"]] = rec
    read = lambda name: [json.loads(line) for line in open(os.path.join(args.bench, name))]
    rows = []
    for sc in args.scenarios:
        fit, val, test = (read(f"{sc}_{suffix}.jsonl") for suffix in args.suffix)
        for q in QTYPES:
            sub = {name: [it for it in items if it["qtype"] == q] for name, items in (("fit", fit), ("val", val), ("test", test))}
            if not sub["test"]:
                continue
            x = {n: torch.tensor([features(it, recs.get(it["image"])) for it in s], dtype=torch.float32) for n, s in sub.items()}
            y = {n: torch.tensor([label(it) for it in s]) for n, s in sub.items()}
            n_out = {"cls": len(verify.CLASSES[sc]), "exist": 2, "count": len(verify.COUNT_OPTIONS)}[q]
            if not len(y["fit"]) or not len(y["val"]):
                continue
            mean, std = x["fit"].mean(0, keepdim=True), x["fit"].std(0, keepdim=True) + 1e-6
            net = torch.nn.Sequential(torch.nn.Linear(x["fit"].shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, n_out))
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
            best = (-1.0, None)
            xs, ys = (x["fit"] - mean) / std, y["fit"]
            for epoch in range(args.epochs):
                net.train()
                order = torch.randperm(len(xs))
                for s in range(0, len(order), 256):
                    idx = order[s : s + 256]
                    loss = torch.nn.functional.cross_entropy(net(xs[idx]), ys[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    acc = (net((x["val"] - mean) / std).argmax(1) == y["val"]).float().mean().item()
                if acc > best[0]:
                    best = (acc, copy.deepcopy(net.state_dict()))
            net.load_state_dict(best[1])
            net.eval()
            saved[f"{sc}/{q}"] = {"mean": mean[0].tolist(), "std": std[0].tolist(),
                                  "layers": [[net[k].weight.tolist(), net[k].bias.tolist()] for k in (0, 2, 4)]}
            with torch.no_grad():
                pred = net((x["test"] - mean) / std).argmax(1)
            ok = (pred == y["test"]).float()
            print(f"{sc} {q}: train {len(ys)} questions, best validation accuracy {100 * best[0]:.1f}%, test {100 * ok.mean().item():.1f}% (n={len(ok)})", flush=True)
            rows += [{"id": it["id"], "policy": "aggregator", "scenario": sc, "qtype": q, "correct": int(o), "bin": it["bin"]} for it, o in zip(sub["test"], ok.tolist())]
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.save_weights:
        json.dump(saved, open(args.save_weights, "w"))
    for sc in args.scenarios:
        v = [r["correct"] for r in rows if r["scenario"] == sc]
        if v:
            print(f"{sc} overall: {100 * sum(v) / len(v):.1f}% (n={len(v)})")


if __name__ == "__main__":
    main()
