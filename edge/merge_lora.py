"""Merge a LoRA adapter into its Qwen3-VL base and save a plain float32 checkpoint (input of the edge conversion).

    python merge_lora.py <base dir> <adapter dir> <out dir>
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

base, adapter, out = sys.argv[1:4]
model = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.float32)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoProcessor.from_pretrained(base).save_pretrained(out)
print("MERGE_DONE", out)
