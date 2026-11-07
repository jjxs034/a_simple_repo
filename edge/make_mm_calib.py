"""Calibration samples for quantising the thumbnail tier (list + 256-token thumbnail) with rkllm-toolkit, following the
vendor's make_input_embeds_for_quantize.py: the inputs of the language model (embeddings with the image features, mrope
positions, DeepStack features) are captured with a forward pre-hook and pickled. The 64 training questions are chosen as
dump_prompts.py chooses them for the list-only tier (training files with degraded images, shuffled with seed 0).

    python make_mm_calib.py --model <merged dir> --items fit.jsonl ... --evidence evid_fit_n256.jsonl --out <dir> [--n 64]
writes <dir>/data/inputs.json and <dir>/data/llm_inputs/sample_<k> (the vendor's layout; build with dataset='data/inputs.json' from <dir>).
"""
import argparse
import json
import os
import pickle
import random
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.environ.get("PB_CODE", "../powerbrain"))
import core  # noqa: E402


class Stop(Exception):
    pass


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32, device_map="cpu").eval()
    processor = AutoProcessor.from_pretrained(args.model)
    given = {r["id"]: r for r in map(json.loads, open(args.evidence))}
    items = [json.loads(line) for path in args.items for line in open(path)]
    random.Random(0).shuffle(items)
    items = items[: args.n]
    captured = []

    def hook(module, a, kw):
        captured.append(kw)
        raise Stop

    handle = model.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
    os.makedirs(f"{args.out}/data/llm_inputs", exist_ok=True)
    info = []
    for k, item in enumerate(items):
        rec = given[item["id"]]
        assert rec.get("thumb_tokens") == 256, rec.get("thumb_tokens")
        image = core.open_image(item)  # as in training: EXIF orientation and the item's synthetic degradation, if any
        content = core.evidence_content(image, rec) + [{"type": "text", "text": item["question"]}]
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        try:
            model(**inputs)
        except Stop:
            pass
        kw = captured.pop()
        name = f"llm_inputs/sample_{k}"
        with open(f"{args.out}/data/{name}", "wb") as f:
            pickle.dump(kw, f)
        info.append({"sample": name, "token_nums": int(kw["inputs_embeds"].shape[1])})
    handle.remove()
    for fname in ("inputs.json", "llm_inputs.json"):
        json.dump(info, open(f"{args.out}/data/{fname}", "w"), indent=1)
    print(f"{args.out}: {len(info)} samples, {sum(i['token_nums'] for i in info) / len(info):.0f} tokens on average; keys {sorted(kw)}")


if __name__ == "__main__":
    main()
