"""Accuracy on the questions about a held-out class, split by question type and reference answer.

    python locoreport.py --items ed_test_onlyTaDiao.jsonl --row 'name=<results dir or jsonl>[:policy]' ...
"""
import argparse
import collections
import json

import numpy as np

import core
import dupeffect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--row", action="append", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--all-classes", action="store_true", help="the questions are not about one held-out class: plain group names")
    args = parser.parse_args()
    items = [json.loads(line) for path in args.items for line in open(path)]
    truth = lambda it: it["options"][core.LETTERS.index(it["answer"])]
    groups = collections.OrderedDict()
    groups["全部"] = items
    groups["类别题" if args.all_classes else "类别题（答案是该类）"] = [it for it in items if it["qtype"] == "cls"]
    groups["是否存在：答案为是" if args.all_classes else "是否存在：图中有该类"] = [it for it in items if it["qtype"] == "exist" and truth(it) == "是"]
    groups["是否存在：答案为否" if args.all_classes else "是否存在：图中没有该类"] = [it for it in items if it["qtype"] == "exist" and truth(it) == "否"]
    groups["计数题"] = [it for it in items if it["qtype"] == "count"]
    print(f"## {args.title or ' '.join(args.items)}（%）")
    print()
    print("| 方法 | " + " | ".join(f"{k}（{len(v)}）" for k, v in groups.items()) + " |")
    print("|---|" + "---:|" * len(groups))
    for spec in args.row:
        name, ok = dupeffect.load_row(spec)
        if any(it["id"] not in ok for it in items):
            print(f"| {name} | 缺 {sum(it['id'] not in ok for it in items)} 题的输出 |" + " |" * (len(groups) - 1))
            continue
        cells = []
        for k, sub in groups.items():
            if not sub:
                cells.append("—")
                continue
            v = [ok[it["id"]] for it in sub]
            if k == "全部":
                lo, hi = dupeffect.clustered_ci(sub, v)
                cells.append(f"{100 * np.mean(v):.1f}（{100 * lo:.1f} 到 {100 * hi:.1f}）")
            else:
                cells.append(f"{100 * np.mean(v):.1f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    main()
