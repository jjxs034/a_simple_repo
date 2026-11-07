#!/usr/bin/env python3
"""p29_mm_summary.py with the float reference as an argument (2026-10-29): the thumbnail-tier runs on the board's own
candidates (mm_boardlists_all, mm_cdark2_boardlists) are compared with the server float results of the same condition.
Input lengths now differ where the board's lists differ from the server's; the count of equal lengths is reported.

    python3 tools/p29_mm_summary_f.py runs/p29/mm_boardlists_all.jsonl runs/p29/mm_summary_all_boardlists.json \
        board_p29m/float/f_onen2b_s0_shard0.jsonl,board_p29m/float/f_onen2b_s0_shard1.jsonl
"""
import json
import statistics as st
import sys

src, dst, ref = sys.argv[1], sys.argv[2], sys.argv[3]
b = {r["id"]: r for r in map(json.loads, open(src))}
f = {}
for path in ref.split(","):
    for line in open(path):
        r = json.loads(line)
        f[r["id"]] = r
ids = [q for q in b if q in f]
same_len = sum(b[q]["prefill_tokens"] == f[q]["input_tokens"] for q in ids)
out = {"source": src, "float": ref, "n": len(ids), "same_input_length": same_len, "scenarios": {}}
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
for sc in ("ed", "td"):
    s = [b[x] for x in ids if b[x]["scenario"] == sc]
    if not s:
        continue
    v = [x["vision_ms"] for x in s if x["vision_ms"] > 0]  # vision_ms is set on the first question of each image only
    w = [x["llm_wall_ms"] for x in s]
    t = [x["prefill_tokens"] for x in s]
    out["scenarios"][sc] = {"questions": len(s), "images": len(v), "vision_ms_median": round(st.median(v), 1), "vision_ms_p95": round(q(v, 0.95), 1),
                            "llm_ms_median": round(st.median(w), 1), "llm_ms_p5": round(q(w, 0.05), 1), "llm_ms_p95": round(q(w, 0.95), 1),
                            "tokens_median": st.median(t), "tokens_max": max(t), "memory_mb_max": round(max(x["memory_mb"] for x in s)),
                            "accuracy_board": round(100 * sum(x["correct"] for x in s) / len(s), 4),
                            "accuracy_float_same_questions": round(100 * sum(f[x["id"]]["correct"] for x in s) / len(s), 4),
                            "greedy_equals_letter": round(100 * sum(bool(x["greedy_is_pred"]) for x in s) / len(s), 2), "errors": sum(x["error"] for x in s)}
json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "scenarios"}, ensure_ascii=False))
