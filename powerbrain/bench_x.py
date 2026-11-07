"""Question forms beyond the three base forms (class / existence / counting), with answers derived by fixed rules from
the human boxes, like the base forms. They test whether a system can answer a NEW kind of question about the same images
without new training and without a new hand-written rule.

First set (bench_v1x, written 2026-10-08; files <scenario>_testx.jsonl):
  x-more     is there more A than B?                       (yes / no)
  x-both     are A and B both present?                     (yes / no)
  x-most     which class has the most objects?             (classes; images with at least two classes)
  x-total    how many objects of all classes together?     (1 / 2 / 3 / 4 or more)
  x-left     is there an A in the left half of the image?  (yes / no)
  x-largest  which class does the largest object belong to? (classes; images with at least two classes)
Second set (files <scenario>_testx2.jsonl):
  x-either   is A or B present (at least one of them)?     (yes / no)
  x-only     is A the only class present?                  (yes / no; "no" either because another class is there too or because A is absent)
  x-atleast  are there at least n objects of A?            (yes / no; n = 2 or 3; images that contain A)
  x-kinds    how many different classes are present?       (1 / 2 / 3 or more)
  x-upper    is there an A in the upper half of the image? (yes / no)
  x-leftmost which class does the leftmost object belong to? (classes; images with at least two classes)

    python bench_x.py --out-dir <bench dir> [--per-type 400]                       ->  ed_testx.jsonl, td_testx.jsonl
    python bench_x.py --out-dir <bench dir> --forms new                            ->  ed_testx2.jsonl, td_testx2.jsonl
    python bench_x.py --out-dir <bench dir> --split fit --forms old new --per-type 600   ->  ed_fitx.jsonl, td_fitx.jsonl (both sets)
The fit / validation questions are for training on SOME forms and testing on the others (held-out forms).
"""
import argparse
import collections
import json
import os
import random

import bench

YESNO = ["是", "否"]


def item(scenario, img, qtype, target, options, answer_text, question, rng, names):
    options = list(options)
    if qtype not in ("x-total", "x-kinds"):  # counts stay in natural order
        rng.shuffle(options)
    letters = "ABCDE"[: len(options)]
    area = img["width"] * img["height"]
    ratios = sorted(b[2] * b[3] / area for b in img["boxes"])
    return {
        "id": f"{img['key']}-{qtype}-{target or 'x'}", "scenario": scenario, "image": img["image"], "width": img["width"], "height": img["height"],
        "qtype": qtype, "target": target, "options": options, "answer": letters[options.index(answer_text)],
        "question": question + "\n" + "\n".join(f"{l}. {names.get(o, o)}" for l, o in zip(letters, options)) + "\n请只回答选项字母。",
        "evidence": [b[:4] for b in img["boxes"]], "boxes": img["boxes"], "area_ratio": ratios[len(ratios) // 2], "bin": bench.area_bin(ratios[len(ratios) // 2]),
    }


def pools(scenario, images, rng):
    classes = list(bench.ED_NAMES) if scenario == "ed" else bench.TD_CLASSES
    names = bench.ED_NAMES if scenario == "ed" else {}
    show = lambda c: names.get(c, c)
    kind = "隐患目标" if scenario == "ed" else "缺陷"
    scene = bench.SCENE[scenario]
    pool = collections.defaultdict(list)
    for img in images:
        if not img["boxes"]:
            continue
        present = collections.Counter(b[4] for b in img["boxes"])
        width = img["width"]
        mk = lambda qtype, target, options, answer, question: item(scenario, img, qtype, target, options, answer, question, rng, names)
        for a in classes:
            for b in classes:
                if a == b or (present[a] == 0 and present[b] == 0) or present[a] == present[b]:
                    continue
                more = "是" if present[a] > present[b] else "否"
                pool[("x-more", more)].append(mk("x-more", f"{a}|{b}", YESNO, more, f"{scene}{show(a)}的数量是否多于{show(b)}？"))
                if a < b:
                    both = "是" if present[a] and present[b] else "否"
                    pool[("x-both", both)].append(mk("x-both", f"{a}|{b}", YESNO, both, f"{scene}是否同时存在{show(a)}和{show(b)}？"))
        for a in present:  # two classes with the same count are covered by x-both "是" candidates below
            for b in present:
                if a < b and present[a] == present[b]:
                    pool[("x-both", "是")].append(mk("x-both", f"{a}|{b}", YESNO, "是", f"{scene}是否同时存在{show(a)}和{show(b)}？"))
        n = min(len(img["boxes"]), 4)
        pool[("x-total", n)].append(mk("x-total", None, bench.COUNT_OPTIONS, bench.COUNT_OPTIONS[n - 1], f"{scene}各类{kind}一共有几处？"))
        if len(present) >= 2:
            ranked = present.most_common()
            if ranked[0][1] > ranked[1][1]:
                pool[("x-most", ranked[0][0])].append(mk("x-most", None, classes, ranked[0][0], f"{scene}数量最多的是哪一类{kind}？"))
            by_class = {c: max(b[2] * b[3] for b in img["boxes"] if b[4] == c) for c in present}
            order = sorted(by_class, key=lambda c: -by_class[c])
            if by_class[order[0]] >= 1.3 * by_class[order[1]]:
                pool[("x-largest", order[0])].append(mk("x-largest", None, classes, order[0], f"{scene}面积最大的{kind}属于哪一类？"))
        for a in present:
            centres = [(b[0] + b[2] / 2) / width for b in img["boxes"] if b[4] == a]
            if any(c < 0.45 for c in centres):
                left = "是"
            elif all(c > 0.55 for c in centres):
                left = "否"
            else:
                continue  # an object on the midline: not asked
            pool[("x-left", left)].append(mk("x-left", a, YESNO, left, f"这张{'输电线路通道巡检' if scenario == 'ed' else '输电线路无人机巡检'}图像的左半部分是否存在{show(a)}？"))
    return pool


KIND_OPTIONS = ["1 类", "2 类", "3 类及以上"]


def pools2(scenario, images, rng):
    """The second set of forms. Kept apart from pools() so that the first set stays reproducible bit for bit."""
    classes = list(bench.ED_NAMES) if scenario == "ed" else bench.TD_CLASSES
    names = bench.ED_NAMES if scenario == "ed" else {}
    countable = bench.ED_COUNTABLE if scenario == "ed" else set(bench.TD_CLASSES)
    show = lambda c: names.get(c, c)
    kind = "隐患目标" if scenario == "ed" else "缺陷"
    scene = bench.SCENE[scenario]
    view = "输电线路通道巡检" if scenario == "ed" else "输电线路无人机巡检"
    pool = collections.defaultdict(list)
    for img in images:
        if not img["boxes"]:
            continue
        present = collections.Counter(b[4] for b in img["boxes"])
        width, height = img["width"], img["height"]
        mk = lambda qtype, target, options, answer, question: item(scenario, img, qtype, target, options, answer, question, rng, names)
        for a in classes:
            for b in classes:
                if a < b:
                    either = "是" if present[a] or present[b] else "否"
                    pool[("x-either", either)].append(mk("x-either", f"{a}|{b}", YESNO, either, f"{scene}是否存在{show(a)}或{show(b)}（至少一类）？"))
            if present[a]:
                stratum = "是" if len(present) == 1 else "否：还有别的类"
            else:
                stratum = "否：没有这一类"
            pool[("x-only", stratum)].append(mk("x-only", a, YESNO, stratum[0], f"{scene}的{kind}是否只有{show(a)}这一类？"))
            if present[a] and a in countable:
                for n in (2, 3):
                    least = "是" if present[a] >= n else "否"
                    pool[("x-atleast", f"{n}{least}")].append(mk("x-atleast", f"{a}|{n}", YESNO, least, f"{scene}{show(a)}是否不少于 {n} 处？"))
        k = min(len(present), 3)
        pool[("x-kinds", k)].append(mk("x-kinds", None, KIND_OPTIONS, KIND_OPTIONS[k - 1], f"{scene}共有几类不同的{kind}？"))
        for a in present:
            centres = [(b[1] + b[3] / 2) / height for b in img["boxes"] if b[4] == a]
            if any(c < 0.45 for c in centres):
                upper = "是"
            elif all(c > 0.55 for c in centres):
                upper = "否"
            else:
                continue  # an object on the midline: not asked
            pool[("x-upper", upper)].append(mk("x-upper", a, YESNO, upper, f"这张{view}图像的上半部分是否存在{show(a)}？"))
        if len(present) >= 2:
            ordered = sorted(img["boxes"], key=lambda b: b[0] + b[2] / 2)
            first = ordered[0]
            other = next(b for b in ordered if b[4] != first[4])
            if ((other[0] + other[2] / 2) - (first[0] + first[2] / 2)) / width >= 0.05:  # a clear margin to the nearest object of another class
                pool[("x-leftmost", first[4])].append(mk("x-leftmost", None, classes, first[4], f"{scene}最靠左的{kind}属于哪一类？"))
    return pool


FORMS = {"old": ("x-more", "x-both", "x-most", "x-total", "x-left", "x-largest"), "new": ("x-either", "x-only", "x-atleast", "x-kinds", "x-upper", "x-leftmost")}
# Two halves for "train on some forms, test on the others". Every form has a relative in the other half that uses the
# same primitive (presence logic, counting, position or size) in a different composition - written before any run.
FOLDS = {"A": ("x-more", "x-total", "x-left", "x-either", "x-kinds", "x-leftmost"), "B": ("x-both", "x-most", "x-largest", "x-only", "x-atleast", "x-upper")}
SHARE = {"x-only": {"是": 0.5, "否：还有别的类": 0.25, "否：没有这一类": 0.25}}  # answer prior 50/50; other forms: the same number from every stratum


def choose(pool, forms, per_type, rng, scenario):
    chosen, seen = [], set()
    for qtype in forms:
        strata = sorted((k for k in pool if k[0] == qtype), key=str)
        got = 0
        for k in strata:  # one question of a form per image
            per = max(1, int(per_type * SHARE[qtype][k[1]])) if qtype in SHARE else max(1, per_type // max(1, len(strata)))
            rng.shuffle(pool[k])
            taken = 0
            for it in pool[k]:
                if taken >= per:
                    break
                if (it["image"], qtype) in seen:
                    continue
                seen.add((it["image"], qtype))
                chosen.append(it)
                taken += 1
            got += taken
        print(f"{scenario} {qtype}: {got} questions from {len(strata)} answer strata ({', '.join(str(k[1]) + ':' + str(len(pool[k])) for k in strata)})")
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-type", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["test", "fit", "val"], default="test")
    parser.add_argument("--forms", nargs="+", choices=["old", "new"], default=["old"])
    args = parser.parse_args()
    td = bench.load_td()
    for scenario in ("ed", "td"):
        images = (bench.load_ed("test") if scenario == "ed" else td["test"]) if args.split == "test" else \
            [img for img in (bench.load_ed("train") if scenario == "ed" else td["train"]) if bench.is_val(img["image"]) == (args.split == "val")]
        chosen = []
        for forms in args.forms:
            # the first set on the test images keeps the seed string it was first written with
            rng = random.Random(f"{args.seed}-{scenario}-x" if (args.split, forms) == ("test", "old") else f"{args.seed}-{scenario}-{args.split}-{forms}")
            chosen += choose((pools if forms == "old" else pools2)(scenario, images, rng), FORMS[forms], args.per_type, rng, scenario)
        suffix = {("old",): "x", ("new",): "x2"}.get(tuple(args.forms), "x") if args.split == "test" else "x"
        assert args.split != "test" or len(args.forms) == 1, "test questions: one set per file (testx = first set, testx2 = second set)"
        path = os.path.join(args.out_dir, f"{scenario}_{args.split}{suffix}.jsonl")
        with open(path, "w") as f:
            for it in chosen:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{path}: {len(chosen)} questions on {len({it['image'] for it in chosen})} images")


if __name__ == "__main__":
    main()
