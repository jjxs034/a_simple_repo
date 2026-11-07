#!/usr/bin/env python3
"""On the board: write the list-only prompts (the compact candidate list and the question as token ids) from detections,
without transformers, so that the board can answer from the lists of its own detector (protocol P29, report item 1).

The steps repeat the server code exactly: verify.py cands (select), verify.py evidence (min confidence, K), the compact
list of core.evidence_content (fmt 3, no image) and the chat template of the processor, whose text around the user turn
make_template.py wrote on the server. Tokenisation uses tokenizer.json through the tokenizers package.

    python3 board_compose.py --template template.json --tokenizer tokenizer.json --items ed_test.jsonl td_test.jsonl \
        (--detections det_ed.jsonl det_td.jsonl | --cands cands_eval_nano.jsonl) --out prompts.jsonl [--verify server_prompts.jsonl]

With --cands (the server's candidates) and --verify (the server's prompts), every prompt must come out token for token
identical; that check is what makes the prompts built from the board's own detections trustworthy.
"""
import argparse
import json
import statistics
import time

from tokenizers import Tokenizer

NONE_TEXT = "以上都不是（背景或误检）"


def iou(a, b):  # verify.iou
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select(detections, min_conf=0.02, nms=0.6, k=12):  # verify.select with the defaults of verify.py cands
    kept = []
    for d in sorted((d for d in detections if d[4] >= min_conf), key=lambda d: -d[4]):
        if all(iou(d[:4], other[:4]) < nms for other in kept):
            kept.append(d)
        if len(kept) >= k:
            break
    return [{"box": [round(v, 1) for v in d[:4]], "conf": round(d[4], 4), "det": d[5]} for d in kept]


def user_text(cands, item, names, min_conf=0.05, k=8):
    """verify.py evidence (--min-conf 0.05 --k 8) and core.evidence_content (fmt 3, thumb_tokens 0), then the question."""
    dets = [{"box": c["box"], "conf": c["conf"], "name": names.get(c["det"], c["det"])} for c in cands if c["conf"] >= min_conf][:k]
    width, height = item["width"], item["height"]
    pct = lambda v, total: int(round(100 * v / total))
    if not dets:
        text = "检测候选：无\n"
    else:
        lines = [f"{d['name'].split('（')[0]} {d['conf']:.2f} {pct(d['box'][0], width)},{pct(d['box'][1], height)},{pct(d['box'][2], width)},{pct(d['box'][3], height)}" for d in dets]
        text = "检测候选（类别 置信度 左,上,右,下%；可能含误检和重复）：" + "\n" + "\n".join(lines) + "\n"
    return text + item["question"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--detections", nargs="*", default=[])
    parser.add_argument("--cands", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify")
    args = parser.parse_args()
    assert bool(args.detections) != bool(args.cands), "give either --detections or --cands"
    tpl = json.load(open(args.template))
    names = {**tpl["names"], "none": NONE_TEXT}
    tok = Tokenizer.from_file(args.tokenizer)
    by_image, select_s = {}, []
    for path in args.detections:
        for rec in map(json.loads, open(path)):
            t0 = time.perf_counter()
            by_image[rec["image"]] = select(rec["detections"])
            select_s.append(time.perf_counter() - t0)
    for path in args.cands:
        for rec in map(json.loads, open(path)):
            by_image[rec["image"]] = rec["cands"]
    ref = {r["id"]: r for r in map(json.loads, open(args.verify))} if args.verify else {}
    same = differ = missing = 0
    asm_s = []
    with open(args.out, "w") as out:
        for path in args.items:
            for item in map(json.loads, open(path)):
                if item["image"] not in by_image:
                    missing += 1
                    continue
                t0 = time.perf_counter()
                text = user_text(by_image[item["image"]], item, names)
                ids = tok.encode(tpl["prefix"] + text + tpl["suffix"], add_special_tokens=False).ids
                asm_s.append(time.perf_counter() - t0)
                row = {"id": item["id"], "scenario": item["scenario"], "qtype": item["qtype"], "n_options": len(item["options"]), "answer": item["answer"],
                       "input_ids": ids, "user_text": text, "image": item["image"], "asm_ms": round(1000 * asm_s[-1], 4)}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if ref:
                    r = ref.get(item["id"])
                    if r is None:
                        missing += 1
                    elif r["input_ids"] == ids:
                        same += 1
                    else:
                        differ += 1
    n = len(asm_s)
    print(f"{args.out}: {n} prompts, {missing} without detections or reference; "
          f"list and tokenisation per question: median {1000 * statistics.median(asm_s):.2f} ms, max {1000 * max(asm_s):.2f} ms"
          + (f"; candidate selection per image: median {1000 * statistics.median(select_s):.3f} ms" if select_s else ""))
    if ref:
        print(f"VERIFY {same} identical to the reference prompts, {differ} different")


if __name__ == "__main__":
    main()
