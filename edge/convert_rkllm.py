"""Quantise the language model of a merged Qwen3-VL checkpoint for a Rockchip NPU with the vendor toolkit, export it,
and use the toolkit's PC simulation (get_logits) to read the answer letter of the quantised model on prepared prompts.

    python convert_rkllm.py --model <merged dir> --calib calib.json --prompts prompts.jsonl --out model.rkllm \
        --dtype w8a8 --opt 0 --sim-out sim.jsonl [--limit 200]
Run inside the rkllm-toolkit environment. Timing cannot be simulated; it has to be measured on the board.
"""
import argparse
import json
import time

import torch
from rkllm.api import RKLLM
from transformers import AutoTokenizer

LETTERS = "ABCDEFGH"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="w8a8")
    parser.add_argument("--opt", type=int, default=0, help="0 = fastest runtime (the setting of the vendor benchmark), 1 = precision optimisation")
    parser.add_argument("--platform", default="rk3588")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--sim-out")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    llm = RKLLM()
    assert llm.load_huggingface(model=args.model, model_lora=None, device=args.device, dtype="float32", custom_config=None, load_weight=True) == 0, "load failed"
    start = time.time()
    assert llm.build(do_quantization=True, optimization_level=args.opt, quantized_dtype=args.dtype, quantized_algorithm="normal",
                     target_platform=args.platform, num_npu_core=3, extra_qparams=None, dataset=args.calib, hybrid_rate=0, max_context=args.max_context) == 0, "build failed"
    print(f"BUILD_DONE {time.time() - start:.0f}s", flush=True)
    assert llm.export_rkllm(args.out) == 0, "export failed"
    print("EXPORT_DONE", args.out, flush=True)
    if not args.prompts:
        return
    tok = AutoTokenizer.from_pretrained(args.model)
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    rows = [json.loads(line) for line in open(args.prompts)]
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    correct = 0
    with open(args.sim_out, "w") as f:
        for n, row in enumerate(rows):
            logits = llm.get_logits({"input_ids": torch.tensor([row["input_ids"]])})
            last = logits[0, -1].float().cpu()
            scores = last[letter_ids[: row["n_options"]]]
            pred = LETTERS[int(scores.argmax())]
            correct += pred == row["answer"]
            f.write(json.dumps({"id": row["id"], "scenario": row["scenario"], "qtype": row["qtype"], "pred": pred, "answer": row["answer"],
                                "correct": int(pred == row["answer"]), "logp": [round(v, 4) for v in torch.log_softmax(scores, -1).tolist()]}) + "\n")
            if (n + 1) % 200 == 0:
                print(f"{n + 1}/{len(rows)} simulated, accuracy so far {100 * correct / (n + 1):.1f}%", flush=True)
    print(f"SIM_DONE {len(rows)} prompts, accuracy {100 * correct / len(rows):.1f}%", flush=True)


if __name__ == "__main__":
    main()
