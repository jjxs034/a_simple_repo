"""Write the exact text-only inputs of the evidence-first configuration (no image: the language model reads the
detector branch's list and the question), so that the quantised model is simulated on the very token sequences the
server evaluation used.

    python dump_prompts.py --model <dir> --items test.jsonl ... --evidence evid_eval.jsonl --out prompts.jsonl
    python dump_prompts.py --model <dir> --items fit.jsonl ... --evidence evid_fit.jsonl --calib calib.json --n 64
"""
import argparse
import json
import os
import random
import sys
import types

from transformers import AutoProcessor

sys.path.insert(0, os.environ.get("PB_CODE", "../powerbrain"))
import core  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out")
    parser.add_argument("--calib")
    parser.add_argument("--n", type=int, default=64)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    given = {rec["id"]: rec for rec in map(json.loads, open(args.evidence))}
    items = [json.loads(line) for path in args.items for line in open(path)]
    if args.calib:
        random.Random(0).shuffle(items)
        items = items[: args.n]
    rows, calib = [], []
    for item in items:
        rec = given[item["id"]]
        assert rec.get("thumb_tokens", 1) == 0, "the evidence-first configuration has no image"
        stub = types.SimpleNamespace(size=(item["width"], item["height"]))  # positions are relative to the image size
        content = core.evidence_content(stub, rec) + [{"type": "text", "text": item["question"]}]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
        rows.append({"id": item["id"], "scenario": item["scenario"], "qtype": item["qtype"], "n_options": len(item["options"]), "answer": item["answer"], "input_ids": ids,
                     "user_text": "".join(c["text"] for c in content)})  # the user turn without the chat template (for runtimes that template themselves)
        calib.append({"input": text, "target": item["answer"]})
    if args.calib:
        json.dump(calib, open(args.calib, "w"), ensure_ascii=False, indent=1)
        print(f"{args.calib}: {len(calib)} calibration prompts, {sum(len(r['input_ids']) for r in rows) / len(rows):.0f} tokens on average")
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"{args.out}: {len(rows)} prompts, {sum(len(r['input_ids']) for r in rows) / len(rows):.0f} tokens on average, longest {max(len(r['input_ids']) for r in rows)}")


if __name__ == "__main__":
    main()
