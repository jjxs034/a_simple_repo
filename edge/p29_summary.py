#!/usr/bin/env python3
"""On the board: summarise the P29 runs in runs/p29 into one json and a markdown table (stage times with median, 5th and
95th percentiles, end-to-end time per question, memory, accuracy, and the measurement conditions of every run).

    python3 p29_summary.py --dir runs/p29 --out runs/p29/summary
"""
import argparse
import glob
import json
import os


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stats(values):
    return {"n": len(values), "median": round(pct(values, 0.5), 2), "p5": round(pct(values, 0.05), 2), "p95": round(pct(values, 0.95), 2),
            "mean": round(sum(values) / len(values), 2)} if values else None


def rows(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = args.dir
    out = {"conditions": {}, "detector": {}, "compose": {}, "llm": {}, "end_to_end": {}, "nollm": {}}
    for path in sorted(glob.glob(f"{D}/watch_*.json")):
        w = json.load(open(path))
        soc = w["temps"].get("soc-thermal", {})
        out["conditions"][w["tag"]] = {"valid": w["valid"], "returncode": w["returncode"], "seconds": w["seconds"], "start_utc": w["start_utc"],
                                       "soc_start": soc.get("start"), "soc_max": soc.get("max"), "hottest_max": max(t["max"] for t in w["temps"].values()),
                                       "clocks_min_share_at_fixed": min(c["at_fixed"] for c in w["clocks"].values()),
                                       "cooling_max": {k: v for k, v in w["cooling_max"].items() if v}, "reasons": w["reasons"], "warnings": w["warnings"],
                                       "memory": w.get("memory")}
    det_whole = {}
    for sc in ("ed", "td"):
        det = rows(f"{D}/det_{sc}_n.jsonl")
        if det:
            names = ["read and decode", "letterbox", "NPU inference", "decode and NMS"]
            out["detector"][sc] = {name: stats([1000 * r["seconds"][k] for r in det]) for k, name in enumerate(names)}
            out["detector"][sc]["whole image"] = stats([1000 * sum(r["seconds"]) for r in det])
            det_whole.update({r["image"]: 1000 * sum(r["seconds"]) for r in det})
    comp = {r["id"]: r for r in rows(f"{D}/prompts_boarddet.jsonl")}
    if comp:
        out["compose"] = {sc: stats([r["asm_ms"] for r in comp.values() if r["scenario"] == sc]) for sc in ("ed", "td")}
    for path in sorted(glob.glob(f"{D}/llm_*.jsonl")):
        tag = os.path.basename(path)[:-6]
        rs = rows(path)
        if not rs:
            continue
        if rs[0]["scenario"].startswith("len"):  # make_lengths.py: prefill time against the input length
            by_len = {}
            for r in rs:
                by_len.setdefault(int(r["scenario"][3:]), []).append(r)
            out["llm"][tag] = {str(n): {"prefill_ms": stats([r["prefill_ms"] for r in v]), "wall_ms": stats([r["wall_ms"] for r in v])} for n, v in sorted(by_len.items())}
            continue
        entry = {}
        for sc in ("ed", "td", "all"):
            sub = [r for r in rs if sc == "all" or r["scenario"] == sc]
            if not sub:
                continue
            entry[sc] = {"n": len(sub), "accuracy": round(100 * sum(r["correct"] for r in sub) / len(sub), 2),
                         "wall_ms": stats([r["wall_ms"] for r in sub]), "prefill_ms": stats([r["prefill_ms"] for r in sub if r.get("prefill_ms") is not None]),
                         "input_tokens": stats([r["n_input"] for r in sub if r.get("n_input")]),
                         "memory_mb_max": max((r.get("memory_mb") or 0) for r in sub)}
            gen = [r for r in sub if (r.get("generate_tokens") or 0) > 1]  # decoding is only measured by the --max-new runs
            if gen:
                entry[sc]["generate_ms"] = stats([r["generate_ms"] for r in gen])
                entry[sc]["tokens_per_s"] = round(sum(r["generate_tokens"] for r in gen) / (sum(r["generate_ms"] for r in gen) / 1000), 2)
            if any("greedy_is_pred" in r for r in sub):
                entry[sc]["greedy_equals_letter_rule"] = round(100 * sum(bool(r.get("greedy_is_pred")) for r in sub) / len(sub), 2)
        out["llm"][tag] = entry
    board = {r["id"]: r for r in rows(f"{D}/llm_boardlists.jsonl")}
    if board and comp and det_whole:
        for sc in ("ed", "td", "all"):
            ids = [q for q, r in board.items() if (sc == "all" or r["scenario"] == sc) and q in comp and comp[q]["image"] in det_whole]
            out["end_to_end"][sc] = {"detector_ms": stats([det_whole[comp[q]["image"]] for q in ids]), "compose_ms": stats([comp[q]["asm_ms"] for q in ids]),
                                     "llm_wall_ms": stats([board[q]["wall_ms"] for q in ids]),
                                     "total_ms": stats([det_whole[comp[q]["image"]] + comp[q]["asm_ms"] + board[q]["wall_ms"] for q in ids])}
    nl = rows(f"{D}/nollm.jsonl")
    if nl and comp and det_whole:
        for sc in ("ed", "td", "all"):
            sub = [r for r in nl if (sc == "all" or r["scenario"] == sc) and r["id"] in comp]
            out["nollm"][sc] = {"rules_accuracy": round(100 * sum(r["rule_correct"] for r in sub) / len(sub), 2),
                                "aggregator_accuracy": round(100 * sum(r["agg_correct"] for r in sub) / len(sub), 2),
                                "rules_total_ms": stats([det_whole[comp[r["id"]]["image"]] + r["rule_ms"] for r in sub]),
                                "aggregator_total_ms": stats([det_whole[comp[r["id"]]["image"]] + r["agg_ms"] for r in sub])}
    json.dump(out, open(args.out + ".json", "w"), ensure_ascii=False, indent=1)
    lines = ["# P29 board summary", ""]
    for tag, c in out["conditions"].items():
        lines.append(f"- run `{tag}`: {'valid' if c['valid'] else 'INVALID'}, {c['seconds']} s, soc {c['soc_start']:.1f} -> {c['soc_max']:.1f} C, "
                     f"clocks at fixed values in >= {100 * c['clocks_min_share_at_fixed']:.1f}% of samples {('; ' + '; '.join(c['reasons'] + c['warnings'])) if c['reasons'] or c['warnings'] else ''}")
    lines.append("")
    fmt = lambda s: f"{s['median']:.1f} ({s['p5']:.1f}-{s['p95']:.1f})" if s else "-"
    for sc, d in out["detector"].items():
        lines.append(f"detector {sc}: " + "; ".join(f"{k} {fmt(v)} ms" for k, v in d.items()))
    for sc, s in out["compose"].items():
        lines.append(f"list and tokenisation {sc}: {fmt(s)} ms")
    for tag, e in out["llm"].items():
        if all(k.isdigit() for k in e):
            lines.append(f"{tag}: prefill against input length: " + "; ".join(f"{n} tokens {fmt(v['prefill_ms'])} ms" for n, v in e.items()))
            continue
        for sc, v in e.items():
            lines.append(f"{tag} {sc}: n {v['n']}, accuracy {v['accuracy']:.1f}%, wall {fmt(v['wall_ms'])} ms, prefill {fmt(v['prefill_ms'])} ms, "
                         f"input tokens {fmt(v['input_tokens'])}, runtime memory max {v['memory_mb_max']:.0f} MB"
                         + (f", decoding {v['tokens_per_s']} tokens/s" if "tokens_per_s" in v else ""))
    for sc, v in out["end_to_end"].items():
        lines.append(f"end to end per question {sc}: total {fmt(v['total_ms'])} ms = detector {fmt(v['detector_ms'])} + list {fmt(v['compose_ms'])} + LLM {fmt(v['llm_wall_ms'])}")
    for sc, v in out["nollm"].items():
        lines.append(f"no LLM {sc}: rules {v['rules_accuracy']:.1f}% in {fmt(v['rules_total_ms'])} ms; aggregator {v['aggregator_accuracy']:.1f}% in {fmt(v['aggregator_total_ms'])} ms")
    open(args.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
