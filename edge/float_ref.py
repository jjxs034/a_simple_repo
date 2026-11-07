"""Answer letters of the float model on prepared token sequences (the reference the quantised simulation is compared with).

    python float_ref.py --model <merged dir> --prompts prompts.jsonl --out float.jsonl [--limit 200]
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

LETTERS = "ABCDEFGH"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
    rows = [json.loads(line) for line in open(args.prompts)]
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    correct = 0
    with open(args.out, "w") as f:
        for row in rows:
            ids = torch.tensor([row["input_ids"]], device="cuda:0")
            scores = model(input_ids=ids).logits[0, -1].float()[letter_ids[: row["n_options"]]].cpu()
            pred = LETTERS[int(scores.argmax())]
            correct += pred == row["answer"]
            f.write(json.dumps({"id": row["id"], "scenario": row["scenario"], "qtype": row["qtype"], "pred": pred, "answer": row["answer"],
                                "correct": int(pred == row["answer"]), "logp": [round(v, 4) for v in torch.log_softmax(scores, -1).tolist()]}) + "\n")
    print(f"FLOAT_DONE {len(rows)} prompts, accuracy {100 * correct / len(rows):.1f}%", flush=True)


if __name__ == "__main__":
    main()
