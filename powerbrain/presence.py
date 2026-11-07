"""Presence questions ("is there a ...?") broken down two ways, from stored per-question outputs only.

1. By reference answer, on the clean test set: accuracy of every system on the questions whose answer is yes and on those
   whose answer is no (misses and false alarms; the only sense in which the benchmark measures hallucination).
2. By whether the evidence list still holds the asked class (confidence >= --min-conf), for the questions whose answer is yes,
   on clean images and under each corruption: how often the list keeps the object, and each system's accuracy in both cases.

    python presence.py --bench <bench dir> --results <results dir> --runs fog:2 dark:2 ... \
        --evidence '<clean evidence file>|<evidence under a corruption, {C} for its name>' \
        --system '名称=<干净结果>|<退化结果，{C} 处填退化名>' ...

Result specs are as in robustreport.py (paths relative to --results; a directory with a policy, or a jsonl file).
"""
import argparse
import json
import os

import numpy as np

import bench
import dupeffect

LETTERS = "ABCDEFGH"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--evidence", required=True, help="'<clean>|<corrupted with {C}>': evidence files of the model that reads the list")
    parser.add_argument("--system", action="append", default=[], required=True)
    parser.add_argument("--min-conf", type=float, default=0.25)
    args = parser.parse_args()
    tags = [r.replace(":", "") for r in args.runs]
    systems = [(spec.split("=", 1)[0], *spec.split("=", 1)[1].split("|")) for spec in args.system]
    ev_clean, ev_dirty = args.evidence.split("|")

    def load(spec):
        try:
            return dupeffect.load_row("x=" + os.path.join(args.results, spec))[1]
        except (FileNotFoundError, OSError):
            return {}

    def evidence(path):
        try:
            return {r["id"]: r["dets"] for r in map(json.loads, open(path))}
        except (FileNotFoundError, OSError):
            return {}

    def cell(rows, ok):
        if not rows:
            return "—"
        if any(it["id"] not in ok for it in rows):
            return "缺"
        return f"{100 * np.mean([ok[it['id']] for it in rows]):.1f}"

    truth = lambda it: it["options"][LETTERS.index(it["answer"])]
    shown = lambda c: bench.ED_NAMES.get(c, c)
    clean = {name: load(c) for name, c, _ in systems}
    for sc in ("ed", "td"):
        items = [json.loads(line) for line in open(os.path.join(args.bench, f"{sc}_test.jsonl"))]
        exist = [it for it in items if it["qtype"] == "exist"]
        yes = [it for it in exist if truth(it) == "是"]
        no = [it for it in exist if truth(it) == "否"]
        print(f"## {sc}：「是否存在」题按参考答案分开的准确率（干净测试集，%；答案为是 {len(yes)} 题，为否 {len(no)} 题）")
        print()
        print("| 系统 | 答案为是（答错即漏报） | 答案为否（答错即误报） |")
        print("|---|---:|---:|")
        for name, _, _ in systems:
            print(f"| {name} | {cell(yes, clean[name])} | {cell(no, clean[name])} |")
        print()
        print(f"## {sc}：答案为「是」的题，清单里还有没有所问类别（置信度 {args.min_conf:g} 以上）时各系统的准确率（%）")
        print()
        print("| 条件 | 清单里有该类的比例 | " + " | ".join(f"{name}：清单里有 / 没有" for name, _, _ in systems) + " |")
        print("|---|---:|" + "---|" * len(systems))
        for tag in [""] + tags:
            dets = evidence(ev_dirty.format(C=tag) if tag else ev_clean)
            if any(it["id"] not in dets for it in yes):
                print(f"| {tag or '干净'} | 缺证据文件 |" + " |" * len(systems))
                continue
            has = lambda it: any((d.get("cls") == it["target"] or d.get("name") == shown(it["target"])) and d["conf"] >= args.min_conf for d in dets[it["id"]])
            kept = [it for it in yes if has(it)]
            lost = [it for it in yes if not has(it)]
            cells = []
            for name, c, d in systems:
                ok = clean[name] if not tag else load(d.format(C=tag))
                cells.append(f"{cell(kept, ok)} / {cell(lost, ok)}")
            print(f"| {tag or '干净'} | {100 * len(kept) / len(yes):.0f}%（{len(kept)} / {len(yes)}） | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
