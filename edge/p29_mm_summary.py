#!/usr/bin/env python3
"""Aggregates of a thumbnail-tier board run (board_mm.py output) against the server float results of f_onen2b_s0:
input-length agreement, stage times, memory and accuracy per scenario. Run from ~/pb_edge on the board.

    python3 tools/p29_mm_summary.py runs/p29/mm_serverlists_all.jsonl runs/p29/mm_summary_all.json
"""
import json
import statistics as st
import sys

src, dst = sys.argv[1], sys.argv[2]
b = {r["id"]: r for r in map(json.loads, open(src))}
f = {}
for s in ("shard0", "shard1"):
    for line in open(f"board_p29m/float/f_onen2b_s0_{s}.jsonl"):
        r = json.loads(line)
        f[r["id"]] = r
ids = [q for q in b if q in f]
same_len = sum(b[q]["prefill_tokens"] == f[q]["input_tokens"] for q in ids)
diff = sorted(set(b[q]["prefill_tokens"] - f[q]["input_tokens"] for q in ids))
out = {"source": src, "n": len(ids), "same_input_length": same_len, "length_differences": diff[:10], "scenarios": {}}
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
print(json.dumps(out, ensure_ascii=False))
