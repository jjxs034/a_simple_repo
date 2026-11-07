"""Write what the board needs to rebuild the list-only prompts without transformers (protocol P29, list on the board):
the chat template around the user turn, the token ids of the answer letters and the display names of the classes.
Run on the server in the experiment environment; board_compose.py reads the output together with tokenizer.json.

    python make_template.py --model <merged dir> --out template.json
"""
import argparse
import json
import os
import sys

from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, os.environ.get("PB_CODE", "../powerbrain"))
import bench  # noqa: E402

MARK = "⁣USER⁣"  # a placeholder that cannot occur in a prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    render = lambda content: processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    text = render([{"type": "text", "text": MARK}])
    assert text.count(MARK) == 1
    prefix, suffix = text.split(MARK)
    # dump_prompts.py passes the list and the question as two text items; the template must simply concatenate them
    assert render([{"type": "text", "text": "AAA"}, {"type": "text", "text": "BBB"}]) == prefix + "AAABBB" + suffix
    letters = "ABCDEFGH"
    out = {"prefix": prefix, "suffix": suffix, "letters": letters,
           "letter_ids": [tok.encode(c, add_special_tokens=False)[0] for c in letters],
           "names": dict(bench.ED_NAMES), "model": args.model}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"{args.out}: prefix {prefix!r}, suffix {suffix!r}, letter ids {out['letter_ids']}")


if __name__ == "__main__":
    main()
