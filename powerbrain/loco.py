"""Leave one class out: question files for "a hazard type that has never been labelled".

  <scenario>_fit_no<X>.jsonl / <scenario>_val_no<X>.jsonl   training / validation questions that do not ask about X
                                                           (images that contain X stay: the objects are there, only unlabelled)
  <scenario>_test_only<X>.jsonl                            test questions about X (class questions whose answer is X,
                                                           existence and counting questions whose target is X)
    python loco.py --bench <bench dir> --scenario ed --class TaDiao [--strict]

--strict writes <scenario>_{fit,val}_no<X>_strict.jsonl, in which the which-class questions do not list X as an option
either: before a hazard type is defined nobody writes it into a question. (Without it, X stays in every option list as
an always-wrong choice, which by itself teaches a model never to pick it.)
"""
import argparse
import json
import os

import bench
import core


def without_option(item, cls):
    """A which-class question as it was written before class `cls` existed: the option is not there."""
    if item["qtype"] != "cls" or cls not in item["options"]:
        return item
    options = [o for o in item["options"] if o != cls]
    truth = item["options"][core.LETTERS.index(item["answer"])]
    letters = core.LETTERS[: len(options)]
    names = bench.ED_NAMES if item["scenario"] == "ed" else {}
    question = item["question"].split("\n")[0] + "\n" + "\n".join(f"{l}. {names.get(o, o)}" for l, o in zip(letters, options)) + "\n请只回答选项字母。"
    return {**item, "options": options, "answer": letters[options.index(truth)], "question": question}


def about(item, cls):
    return item["target"] == cls or (item["qtype"] == "cls" and item["options"][core.LETTERS.index(item["answer"])] == cls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--class", dest="cls", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    read = lambda name: [json.loads(line) for line in open(os.path.join(args.bench, name))]
    for split in ("fit", "val"):
        items = [it for it in read(f"{args.scenario}_{split}.jsonl") if not about(it, args.cls)]
        if args.strict:
            items = [without_option(it, args.cls) for it in items]
        path = os.path.join(args.bench, f"{args.scenario}_{split}_no{args.cls}{'_strict' if args.strict else ''}.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
        print(f"{path}: {len(items)} questions")
    items = [it for it in read(f"{args.scenario}_test.jsonl") if about(it, args.cls)]
    path = os.path.join(args.bench, f"{args.scenario}_test_only{args.cls}.jsonl")
    with open(path, "w") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    by = {}
    for it in items:
        key = (it["qtype"], it["options"][core.LETTERS.index(it["answer"])])
        by[key] = by.get(key, 0) + 1
    print(f"{path}: {len(items)} questions | " + ", ".join(f"{q} {a}: {n}" for (q, a), n in sorted(by.items())))


if __name__ == "__main__":
    main()
