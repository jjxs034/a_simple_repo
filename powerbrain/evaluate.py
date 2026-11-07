"""Evaluate input policies on benchmark items. One process per GPU; resumable by (id, policy).

    python evaluate.py --items a.jsonl b.jsonl --out-dir <dir> --model <path> [--lora <dir>] \
        --policies u256 u576 o1 ol r1 --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import core
import look
import openform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="", help="prefix of the output file, so several runs can share one results directory")
    parser.add_argument("--regions", nargs="*", default=[], help="tag=path of a selector's output, used by policy g:tag")
    parser.add_argument("--projector", help="evidence-token projector of the adapter, for policies named t:<tag> (evtok.py)")
    parser.add_argument("--fastv", help="LAYER:KEEP (KEEP a count, or a fraction such as 0.5) - prune visual tokens inside the language model (fastv.py); give the run its own --out-dir")
    args = parser.parse_args()

    items = [json.loads(line) for path in args.items for line in open(path)]
    if args.limit:
        items = items[: args.limit]
    items = items[args.shard :: args.num_shards]
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}shard{args.shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            row = json.loads(line)
            done.add((row["id"], row["policy"]))
    given = {}
    for spec in args.regions:
        tag, path = spec.split("=", 1)
        given[tag] = {rec["id"]: rec for rec in map(json.loads, open(path))}
    model = core.MLLM(args.model, lora=args.lora)
    if args.projector:
        import evtok

        model.load_projector(args.projector)
    if args.lora and any(os.path.exists(os.path.join(args.lora, f)) for f in ("state_projector.pt", "evidence_tokenizer.pt")):
        model.load_look_modules(args.lora)  # the learned input tokens of a look-on-doubt adapter (look.py)
    pruning = None
    if args.fastv:
        import fastv

        layer, keep = args.fastv.split(":")
        layer, keep = int(layer), float(keep) if "." in keep else int(keep)
        pruning = fastv.install(model, layer=layer, keep=keep)

    with open(out_path, "a") as out:
        for n, item in enumerate(items):
            todo = [p for p in args.policies if (item["id"], p) not in done]
            if todo:
                image = core.open_image(item) if any(not p.startswith(("t:", "look1:")) for p in todo) else None  # evidence tokens and the evidence-only step need no pixels
                letters = core.LETTERS[: len(item["options"])]
                answer = letters.index(item["answer"])
                question = {"type": "text", "text": item["question"]}
            for policy in todo:
                cot = policy.startswith("cot:")  # list-then-answer on a whole-image policy, e.g. cot:u8192
                if policy.startswith("look1:"):  # the evidence-only step alone (no image is opened): an ordinary result row
                    one = model.answer_look(item, None, given[policy[6:]][item["id"]], steps=(1,))[1]
                    pred = max(range(len(letters)), key=lambda i: one["logp"][i])
                    out.write(json.dumps({
                        "id": item["id"], "policy": policy, "scenario": item["scenario"], "qtype": item["qtype"], "target": item["target"], "bin": item["bin"],
                        "truth": item["options"][answer], "pred": item["options"][pred], "correct": int(pred == answer), "p_look": round(one["p_look"], 5),
                        "logp": [round(v, 4) for v in one["logp"]], "visual_tokens": 0, "input_tokens": one["input_tokens"], "seconds": round(one["seconds"], 4),
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    continue
                if policy.startswith("look:"):  # look-on-doubt (look.py): step 1 without the image, step 2 with a thumbnail; both are stored
                    rec = given[policy[5:]][item["id"]]
                    if "answer_text" in item:  # open-ended question: both steps are written out and scored by the rules of openform.py
                        res = model.answer_look_text(item, image, rec)
                        one, two = res[1], res[2]
                        asks = look.asks_to_look(one["text"])  # a request for the image is not an answer
                        out.write(json.dumps({
                            "id": item["id"], "policy": policy, "scenario": item["scenario"], "qtype": item["qtype"], "target": item["target"], "bin": item["bin"],
                            "truth": item["answer_text"], "pred": two["text"], "correct": int(openform.score(item, two["text"])), "text": two["text"],
                            "pred1": one["text"], "correct1": int(not asks and openform.score(item, one["text"])), "look1": int(asks), "p_look": round(one["p_look"], 5),
                            "generated_tokens": two["generated_tokens"], "generated_tokens1": one["generated_tokens"],
                            "visual_tokens": two["visual_tokens"], "input_tokens": two["input_tokens"], "input_tokens1": one["input_tokens"],
                            "seconds": round(two["seconds"], 4), "seconds1": round(one["seconds"], 4),
                        }, ensure_ascii=False) + "\n")
                        out.flush()
                        continue
                    res = model.answer_look(item, image, rec)
                    one, two = res[1], res[2]
                    pred1 = max(range(len(letters)), key=lambda i: one["logp"][i])
                    pred2 = max(range(len(letters)), key=lambda i: two["logp"][i])
                    out.write(json.dumps({
                        "id": item["id"], "policy": policy, "scenario": item["scenario"], "qtype": item["qtype"], "target": item["target"], "bin": item["bin"],
                        "truth": item["options"][answer], "pred": item["options"][pred2], "correct": int(pred2 == answer),
                        "pred1": item["options"][pred1], "correct1": int(pred1 == answer), "p_look": round(one["p_look"], 5),
                        "logp1": [round(v, 4) for v in one["logp"]], "logp": [round(v, 4) for v in two["logp"]],
                        "visual_tokens": two["visual_tokens"], "input_tokens": two["input_tokens"], "input_tokens1": one["input_tokens"],
                        "seconds": round(two["seconds"], 4), "seconds1": round(one["seconds"], 4),
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    continue
                if policy.startswith("t:"):  # evidence tokens: no image, one learned token per candidate
                    rec = given[policy[2:]][item["id"]]
                    cls, feat = evtok.features(rec, item["width"], item["height"])
                    res = model.answer_tokens(core.evidence_token_content(rec) + [question], cls, feat, len(letters))
                    pred = max(range(len(letters)), key=lambda i: res["logp"][i])
                    out.write(json.dumps({
                        "id": item["id"], "policy": policy, "scenario": item["scenario"], "qtype": item["qtype"], "target": item["target"], "bin": item["bin"],
                        "truth": item["options"][answer], "pred": item["options"][pred], "correct": int(pred == answer), "logp": [round(v, 4) for v in res["logp"]],
                        "logp_correct": res["logp"][answer], "visual_tokens": 0, "input_tokens": res["input_tokens"], "seconds": round(res["seconds"], 4),
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    continue
                content = core.build_content(item, policy[4:] if cot else policy, image, given[policy[2:]].get(item["id"]) if policy.startswith("g:") else None)
                if content is None:
                    continue
                if "answer_text" in item:  # open-ended question: a written answer, scored by the fixed rules of openform.py
                    try:
                        res = model.answer_text(content + [question])
                    except RuntimeError as err:
                        if "out of memory" not in str(err).lower():
                            raise
                        torch.cuda.empty_cache()
                        print(f"[shard {args.shard}] OOM skipped: {item['id']} {policy}", flush=True)
                        continue
                    out.write(json.dumps({
                        "id": item["id"], "policy": policy, "scenario": item["scenario"], "qtype": item["qtype"], "target": item["target"], "bin": item["bin"],
                        "truth": item["answer_text"], "pred": res["text"], "correct": int(openform.score(item, res["text"])), "text": res["text"],
                        "generated_tokens": res["generated_tokens"], "visual_tokens": res["visual_tokens"], "input_tokens": res["input_tokens"], "seconds": round(res["seconds"], 4),
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    continue
                try:
                    if cot:
                        res = model.answer_cot(content + [{"type": "text", "text": core.cot_question(item)}], len(letters))
                    else:
                        res = model.answer(content + [question], len(letters))
                except RuntimeError as err:  # a very large input can exceed one GPU; log it and move on
                    if "out of memory" not in str(err).lower():
                        raise
                    torch.cuda.empty_cache()
                    print(f"[shard {args.shard}] OOM skipped: {item['id']} {policy}", flush=True)
                    continue
                pred = max(range(len(letters)), key=lambda i: res["logp"][i])
                valid = res.get("parsed", True)  # a list-then-answer output without an answer line counts as wrong
                out.write(json.dumps({
                    "id": item["id"], "policy": policy, "scenario": item["scenario"], "qtype": item["qtype"],
                    "target": item["target"], "bin": item["bin"], "truth": item["options"][answer],
                    "pred": item["options"][pred] if valid else "", "correct": int(valid and pred == answer), "logp": [round(v, 4) for v in res["logp"]],
                    "logp_correct": res["logp"][answer], "visual_tokens": res["visual_tokens"],
                    "input_tokens": res["input_tokens"], "seconds": round(res["seconds"], 4),
                    **({"kept_visual_tokens": pruning["kept"]} if pruning else {}),
                    **({"text": res["text"], "generated_tokens": res["generated_tokens"]} if cot else {}),
                }, ensure_ascii=False) + "\n")
                out.flush()
            if (n + 1) % 50 == 0:
                print(f"[shard {args.shard}] {n + 1}/{len(items)}", flush=True)
    print(f"[shard {args.shard}] EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
