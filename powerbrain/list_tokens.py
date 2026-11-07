# Token counts of the compact candidate list (fmt 3, core.evidence_content) with the tokenizer of Qwen3-VL-4B:
# the lead-in line and the longest possible candidate line of each scenario (l_0 and l of bound (9)).
#   cd $PB_CODE && nice -n 19 python - < list_tokens.py
# The longest possible line takes the longest class name, a two-digit confidence and four three-digit coordinates.
import os, sys
sys.path.insert(0, os.environ.get("PB_CODE", "."))
from transformers import AutoTokenizer
import verify
tok = AutoTokenizer.from_pretrained(os.environ.get("PB_M4", "Qwen/Qwen3-VL-4B-Instruct"))
n = lambda s: len(tok.encode(s, add_special_tokens=False))
head = "检测候选（类别 置信度 左,上,右,下%；可能含误检和重复）："
print("lead-in line incl. newline", n(head + "\n"))
for sc in ("ed", "td"):
    names = sorted({verify.NAMES.get(c, c).split("（")[0] for c in verify.CLASSES[sc]})
    print(sc, "worst-case line incl. newline", max((n(f"{nm} 0.99 100,100,100,100\n"), nm) for nm in names))
