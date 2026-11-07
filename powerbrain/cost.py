"""End-to-end cost of the input policies on one GPU: latency, peak memory, visual tokens.

Run on an otherwise idle GPU. For every question the whole pipeline is executed online:
  uniform   resize -> processor -> full model
  ours      1,024-token overview -> vision encoder only -> allocation head -> budgeted plan
            -> thumbnail + regions -> processor -> full model
Image decoding from disk is excluded from the timings (it is identical for all policies).
The regions planned online are checked against the ones used in the accuracy evaluation.

    python cost.py --items a.jsonl b.jsonl --model <path> --lora <dir> --head <pt> --regions <evaluated regions file> --n 200
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time

import torch

import core
import head as head_lib


def sync():
    torch.cuda.synchronize()


class Ours:
    def __init__(self, model, head_path, budget, overview=1024, rel_threshold=0.25, k=6, min_peak=0.08, min_scale=0.5, pad=0.5):
        self.m = model
        ckpt = torch.load(head_path)
        self.head = head_lib.Head(ckpt["in_dim"]).to(model.device)
        self.head.load_state_dict(ckpt["state"])
        self.head.eval()
        self.cfg = dict(budget=budget, overview=overview, rel_threshold=rel_threshold, k=k, min_peak=min_peak, min_scale=min_scale, pad=pad)
        self.table = model.model.get_input_embeddings()
        self.stem_cache = {}

    @torch.inference_mode()
    def plan(self, item, image):
        """Returns (regions, scale, timings). Mirrors head.py predict + allocate.py plan --mode coverage."""
        cfg, m = self.cfg, self.m
        t0 = time.perf_counter()
        overview = core.resize_to_token_cap(image, cfg["overview"])
        inputs = m.processor.image_processor(images=[overview], return_tensors="pt").to(m.device)
        feats = m.model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])[0]
        feats = (feats[0] if isinstance(feats, (tuple, list)) else feats).float()
        gh, gw = overview.size[1] // core.CELL, overview.size[0] // core.CELL
        stem = head_lib.stem_of(item)
        if stem not in self.stem_cache:
            ids = torch.tensor(m.processor.tokenizer.encode(stem, add_special_tokens=False), device=m.device)
            self.stem_cache[stem] = self.table(ids).float().mean(0)
        sync()
        t1 = time.perf_counter()
        mask = torch.ones(1, gh * gw, dtype=torch.bool, device=m.device)
        rel, size = self.head(feats.half().float()[None], head_lib.positions(gh, gw)[None].to(m.device), self.stem_cache[stem].half().float()[None], mask)
        prob = torch.sigmoid(rel[0]).view(gh, gw).cpu()
        sync()
        t2 = time.perf_counter()
        width, height = image.size
        cw, ch = width / gw, height / gh
        comps = head_lib.components(prob, max(0.1, cfg["rel_threshold"] * float(prob.max())))
        if not comps:
            r, c = divmod(int(prob.argmax()), gw)
            comps = [(float(prob[r, c]), r, c, r, c)]
        cands = [{"box": [c0 * cw, r0 * ch, (c1 + 1) * cw, (r1 + 1) * ch], "score": s, "peak": float(prob[r0 : r1 + 1, c0 : c1 + 1].max())} for s, r0, c0, r1, c1 in comps[:8]]
        picked = [c for n, c in enumerate(sorted(cands, key=lambda c: -c["score"])[: cfg["k"]]) if n == 0 or c["peak"] >= cfg["min_peak"]]
        scale0 = min(1.0, math.sqrt(core.THUMB_TOKENS * core.CELL * core.CELL / (width * height)))
        pad_px = cfg["pad"] * core.CELL / scale0
        boxes = []
        for c in picked:
            x0, y0, x1, y1 = c["box"]
            x0, y0, x1, y1 = x0 - pad_px, y0 - pad_px, x1 + pad_px, y1 + pad_px
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 256), max(y1 - y0, 256)
            boxes.append([max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(width), cx + w / 2), min(float(height), cy + h / 2)])
        room = cfg["budget"] - core.THUMB_TOKENS
        cost = lambda bs, sc: sum(core.tokens_of(*core.render_size(b, sc)) for b in bs)
        scale = 1.0
        while boxes:
            scale = 1.0
            while cost(boxes, scale) > room and scale > cfg["min_scale"]:
                scale = max(cfg["min_scale"], scale * 0.9)
            if cost(boxes, scale) <= room or len(boxes) == 1:
                break
            boxes = boxes[:-1]
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes, round(scale, 3), {"vision_overview": t1 - t0, "head": t2 - t1, "plan": time.perf_counter() - t2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--head", required=True)
    parser.add_argument("--regions", required=True, help="regions file used in the accuracy evaluation at this budget (consistency check)")
    parser.add_argument("--budget", type=int, default=768)
    parser.add_argument("--overview", type=int, default=1024, help="visual tokens of the image the head reads")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--uniform", nargs="*", default=["u256", "u576", "u1280", "u4096", "u8192"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = core.MLLM(args.model, lora=args.lora)
    ours = Ours(model, args.head, args.budget, overview=args.overview)
    evaluated = {rec["id"]: rec for rec in map(json.loads, open(args.regions))}
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    mismatch = 0
    for path in args.items:
        items = [json.loads(line) for line in open(path)]
        random.Random(0).shuffle(items)
        for n, item in enumerate(items[: args.n + 5]):
            image = core.open_image(item)
            question = {"type": "text", "text": item["question"]}
            for policy in args.uniform + ["ours"]:
                torch.cuda.reset_peak_memory_stats()
                sync()
                start = time.perf_counter()
                extra = {}
                if policy == "ours":
                    regions, scale, extra = ours.plan(item, image)
                    content = core.region_content(image, regions, scale=scale)
                else:
                    content = core.build_content(item, policy, image)
                res = model.answer(content + [question], len(item["options"]))
                sync()
                total = time.perf_counter() - start
                if n < 5:  # warm-up
                    continue
                key = (item["scenario"], policy)
                stats[key]["total"].append(total)
                stats[key]["llm_forward"].append(res["seconds"])
                stats[key]["tokens"].append(res["visual_tokens"])
                stats[key]["peak_gib"].append(torch.cuda.max_memory_allocated() / 2**30)
                for k, v in extra.items():
                    stats[key][k].append(v)
                if policy == "ours":
                    ref = evaluated[item["id"]]
                    same = len(ref["regions"]) == len(regions) and all(abs(a - b) < 2 for r1, r2 in zip(ref["regions"], regions) for a, b in zip(r1, r2))
                    mismatch += not same
    med = lambda v: sorted(v)[len(v) // 2]
    lines = ["| 场景 | 输入 | 视觉令牌 | 端到端中位耗时 ms | 其中：概览视觉编码 | 分配头 | 规划 | 语言模型前向 | 吞吐 题/秒 | 峰值显存 GiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (scenario, policy), s in stats.items():
        part = lambda k: f"{1000 * med(s[k]):.1f}" if s.get(k) else "—"
        lines.append(f"| {scenario} | {policy} | {sum(s['tokens']) / len(s['tokens']):.0f} | {1000 * med(s['total']):.0f} | {part('vision_overview')} | {part('head')} | {part('plan')} | "
                     f"{1000 * med(s['llm_forward']):.0f} | {len(s['total']) / sum(s['total']):.2f} | {max(s['peak_gib']):.2f} |")
    n_ours = sum(len(s["total"]) for (sc, p), s in stats.items() if p == "ours")
    lines += ["", f"在线规划的区域与精度评测所用区域不一致的题目：{mismatch} / {n_ours}（容差 2 像素）。", f"每个场景 {args.n} 题，另有 5 题预热不计。预算 {args.budget}。"]
    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
