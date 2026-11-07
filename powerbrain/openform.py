"""Open-ended (option-free) versions of the benchmark questions and their rule-based scoring (P19).

The reference answers come from the same human boxes and fixed rules as the multiple-choice benchmark; no model is
involved in writing or scoring them.

  the three base forms without options     which class / is class X present / how many X      -> a name, 是 or 否, a number
  one inventory question per image         "which hazards are in the image and how many of each" -> 类别 n 处；类别 n 处 (or 无)

    python openform.py build --bench <dir> --splits fit val test       -> <scenario>_<split>_open.jsonl
    python openform.py selftest

Scoring (score(item, text) -> bool), fixed before any output was read:
  class      the text names the right class and no other class of the scenario
  presence   the text starts with 是 / 有 / 存在 (yes) or 否 / 不 / 没 / 无 (no)
  count      the first number in the text; 1, 2 and 3 must match exactly, 4 or more counts as one answer (as in the
             multiple-choice options)
  inventory  the classes named and their counts (same rule for 4 or more) equal the reference exactly; a class named
             without a number counts once
"""
import argparse
import collections
import json
import os
import re

import bench
import core

INSTRUCTION = {"cls": "请直接回答类别名称。", "exist": "请直接回答「是」或「否」。", "count": "请直接回答数字。"}
INVENTORY = {
    "ed": "这张输电线路通道巡检图像中有哪些外力破坏隐患、各有几处？请逐类回答，格式为「类别 数量 处」，用分号分隔；没有则回答「无」。",
    "td": "这张输电线路无人机巡检图像中有哪些缺陷、各有几处？请逐类回答，格式为「类别 数量 处」，用分号分隔；没有则回答「无」。",
}
NUMERALS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def classes(scenario):
    return list(bench.ED_NAMES) if scenario == "ed" else list(bench.TD_CLASSES)


def short(scenario, cls):
    """The class name as it is to be written in an answer (the display name without its parenthesis)."""
    return (bench.ED_NAMES.get(cls, cls) if scenario == "ed" else cls).split("（")[0]


def cap(n):
    return min(int(n), 4)


def counts_of(item):
    return collections.Counter(b[4] for b in item["boxes"])


def open_item(item):
    """A base-form question without its options."""
    truth = item["options"][core.LETTERS.index(item["answer"])]
    if item["qtype"] == "cls":
        text = short(item["scenario"], truth)
    elif item["qtype"] == "exist":
        text = truth
    else:
        n = counts_of(item)[item["target"]]
        assert bench.COUNT_OPTIONS[cap(n) - 1] == truth, (item["id"], n, truth)
        text = str(n)
    return {**item, "id": item["id"] + "-open", "question": item["question"].split("\n")[0] + "\n" + INSTRUCTION[item["qtype"]], "answer_text": text}


def inventory_item(item):
    """One inventory question for the image of `item` (any question on that image)."""
    sc = item["scenario"]
    have = counts_of(item)
    text = "；".join(f"{short(sc, c)} {have[c]} 处" for c in classes(sc) if have[c]) or "无"
    key = os.path.splitext(os.path.basename(item["image"]))[0]
    return {**item, "id": f"{sc}-{key}-inventory-open", "qtype": "inventory", "target": "", "question": INVENTORY[sc], "answer_text": text,
            "evidence": [b[:4] for b in item["boxes"]] or item["evidence"]}


def first_number(text):
    m = re.search(r"\d+|[一二两三四五六七八九十]", text)
    if not m:
        return None
    return int(m.group()) if m.group().isdigit() else NUMERALS[m.group()]


def parse_inventory(scenario, text):
    """{class: count} named in the text. The count is the first number after the name, before the next class name or
    separator; a name without a number counts once."""
    names = {short(scenario, c): c for c in classes(scenario)}
    found = sorted((m.start(), name) for name in names for m in re.finditer(re.escape(name), text))
    out = {}
    for k, (pos, name) in enumerate(found):
        end = found[k + 1][0] if k + 1 < len(found) else len(text)
        segment = re.split(r"[；;。\n]", text[pos + len(name) : end])[0]
        n = first_number(segment)
        out[names[name]] = out.get(names[name], 0) + (n if n is not None else 1)
    return out


def score(item, text):
    sc, kind, text = item["scenario"], item["qtype"], text.strip()
    if kind == "cls":
        truth = item["options"][core.LETTERS.index(item["answer"])]
        return [c for c in classes(sc) if short(sc, c) in text] == [truth]
    if kind == "exist":
        truth = item["options"][core.LETTERS.index(item["answer"])]
        if text.startswith(("否", "不", "没", "无")):
            return truth == "否"
        if text.startswith(("是", "有", "存在")):
            return truth == "是"
        return False
    if kind == "count":
        n = first_number(text)
        return n is not None and n >= 1 and cap(n) == cap(counts_of(item)[item["target"]])
    if kind == "inventory":
        truth = {c: cap(n) for c, n in counts_of(item).items() if c in classes(sc)}
        return {c: cap(n) for c, n in parse_inventory(sc, text).items() if n >= 1} == truth
    raise ValueError(kind)


def cmd_build(args):
    for sc in ("ed", "td"):
        for split in args.splits:
            path = os.path.join(args.bench, f"{sc}_{split}.jsonl")
            items = [json.loads(line) for line in open(path)]
            out = [open_item(it) for it in items]
            seen = set()
            for it in items:  # one inventory question per image, in the order the images first appear
                if it["image"] not in seen:
                    seen.add(it["image"])
                    out.append(inventory_item(it))
            assert len({it["id"] for it in out}) == len(out), "duplicate ids"
            assert all(score(it, it["answer_text"]) for it in out), "a reference answer does not pass its own scoring rule"
            target = os.path.join(args.bench, f"{sc}_{split}_open.jsonl")
            with open(target, "w") as f:
                f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in out)
            kinds = collections.Counter(it["qtype"] for it in out)
            print(f"{target}: {len(out)} questions {dict(kinds)} on {len(seen)} images")


def cmd_selftest(args):
    item = {"scenario": "ed", "qtype": "inventory", "boxes": [[0, 0, 1, 1, "TaDiao"], [0, 0, 1, 1, "TaDiao"], [0, 0, 1, 1, "DiaoChe"]], "options": ["-"], "answer": "A", "target": ""}
    assert parse_inventory("ed", "塔吊 2 处；吊车 1 处") == {"TaDiao": 2, "DiaoChe": 1}
    assert score(item, "吊车 1 处；塔吊 2 处") and score(item, "塔吊两处，吊车") and not score(item, "塔吊 1 处；吊车 1 处") and not score(item, "塔吊 2 处") and not score(item, "无")
    assert not score(item, "塔吊 2 处；吊车 1 处；烟火 1 处")
    many = {**item, "boxes": [[0, 0, 1, 1, "TaDiao"]] * 6}
    assert score(many, "塔吊 5 处") and score(many, "塔吊 4 处") and not score(many, "塔吊 3 处")
    none = {**item, "boxes": []}
    assert score(none, "无") and not score(none, "塔吊 1 处")
    count = {"scenario": "ed", "qtype": "count", "target": "TaDiao", "boxes": [[0, 0, 1, 1, "TaDiao"]] * 2, "options": ["1 处", "2 处", "3 处", "4 处及以上"], "answer": "B"}
    assert score(count, "2") and score(count, "两处") and score(count, "共有 2 处塔吊") and not score(count, "3") and not score(count, "不清楚")
    yes = {"scenario": "ed", "qtype": "exist", "target": "TaDiao", "boxes": [], "options": ["是", "否"], "answer": "A"}
    assert score(yes, "是") and score(yes, "存在") and not score(yes, "否") and not score(yes, "不是") and not score(yes, "不存在") and not score(yes, "可能")
    cls = {"scenario": "ed", "qtype": "cls", "target": "", "boxes": [], "options": ["DaoXianYiWu", "ShiGongJiXie", "TaDiao"], "answer": "B"}
    assert score(cls, "施工机械") and score(cls, "施工机械（挖掘机）") and not score(cls, "塔吊") and not score(cls, "施工机械和塔吊") and not score(cls, "机械")
    td = {"scenario": "td", "qtype": "cls", "target": "", "boxes": [], "options": list(bench.TD_CLASSES), "answer": "A"}
    assert score(td, bench.TD_CLASSES[0]) and not score(td, bench.TD_CLASSES[1])
    print("OPENFORM_SELFTEST_OK")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--bench", required=True)
    p.add_argument("--splits", nargs="+", default=["fit", "val", "test"])
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
