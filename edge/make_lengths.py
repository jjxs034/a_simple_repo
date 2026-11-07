#!/usr/bin/env python3
"""Prompts of fixed lengths for timing the language model's prefill on the board against the input length (protocol P29,
supplement): real list-only prompts whose user turn is repeated until the sequence has the wanted number of tokens,
inside the same chat template. Only the time is used; the answers mean nothing.

    python3 make_lengths.py --prompts test_prompts.jsonl --template template.json --tokenizer tokenizer.json \
        --lengths 64 128 256 384 512 640 768 896 --per 10 --out prompts_lengths.jsonl
"""
import argparse
import json

from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tpl = json.load(open(args.template))
    tok = Tokenizer.from_file(args.tokenizer)
    pre = tok.encode(tpl["prefix"], add_special_tokens=False).ids
    suf = tok.encode(tpl["suffix"], add_special_tokens=False).ids
    rows = [json.loads(line) for line in open(args.prompts)]
    rows = rows[:: max(1, len(rows) // args.per)][: args.per]
    with open(args.out, "w") as out:
        for n in args.lengths:
            for k, r in enumerate(rows):
                body = tok.encode(r["user_text"], add_special_tokens=False).ids
                need = n - len(pre) - len(suf)
                filler = (body * (need // len(body) + 1))[:need]
                ids = pre + filler + suf
                assert len(ids) == n
                out.write(json.dumps({"id": f"len{n}-{k}", "scenario": f"len{n}", "qtype": "timing", "n_options": r["n_options"], "answer": r["answer"],
                                      "input_ids": ids, "user_text": ""}) + "\n")
    print(f"{args.out}: {len(args.lengths) * len(rows)} prompts, lengths {args.lengths}")


if __name__ == "__main__":
    main()
