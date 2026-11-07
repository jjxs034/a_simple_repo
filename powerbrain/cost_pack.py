"""End-to-end cost of the single-pass evidence pipeline on one GPU, executed online for every question:

  ours      image -> detector branch (resize to its input size, forward, suppression) -> candidates
            -> thumbnail + text list -> processor -> vision encoder + language model -> answer letter
  u<cap>    image -> resize to the token cap -> processor -> vision encoder + language model -> answer letter
  look      look-on-doubt (look.py): image -> detector branch (candidates and branch state) -> step 1 on the evidence
            alone -> only if step 1 answers X: thumbnail -> processor -> vision encoder + continuation of the same
            sequence on the cache of step 1. Both steps are run and timed for every question; the cost of the cascade
            counts step 2 only for the questions on which the model itself asks for the image.

Run on an otherwise idle GPU. Image decoding from disk is excluded (it is the same for every policy). The answers
produced online are compared with the ones recorded in the accuracy evaluation; each run appends its rows to a
jsonl file, and `table` prints them side by side.

    python cost_pack.py run --items ed_test.jsonl td_test.jsonl --model <path> --lora <dir> --policies ours \\
        --weights ed=<best.pt> td=<best.pt> --evaluated <results dir>:g:e --name "4B + medium branch" --rows cost_rows.jsonl
    python cost_pack.py run ... --lora <whole-image adapter> --policies u576 u8192 --evaluated <results dir> --name "4B whole image"
    python cost_pack.py table --rows cost_rows.jsonl --out cost.md
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import time

import torch

import core
import verify


def recorded(spec):
    """{(id, policy): predicted option} from the shard files of an accuracy run; `spec` = dir[:policy]."""
    path, _, policy = spec.partition(":")
    out = {}
    for name in glob.glob(os.path.join(path, "*shard*.jsonl")):
        for line in open(name):
            r = json.loads(line)
            if not policy or r["policy"] == policy:
                out[(r["id"], r["policy"])] = r["pred"]
    return out


def cmd_run(args):
    model = core.MLLM(args.model, lora=args.lora)
    if args.fastv:  # token pruning inside the language model, for the whole-image policies
        import fastv

        layer, keep = args.fastv.split(":")
        layer, keep = int(layer), float(keep) if "." in keep else int(keep)
        fastv.install(model, layer=layer, keep=keep)
    branch, probe = {}, {}
    if "ours" in args.policies or "look" in args.policies:
        from ultralytics import YOLO

        for spec in args.weights:
            scenario, path = spec.split("=", 1)
            branch[scenario] = YOLO(path)
        for spec in args.stats:  # look: the branch-state token needs the feature statistics of the detector (branchstate.py stats)
            import branchstate

            scenario, path = spec.split("=", 1)
            probe[scenario] = branchstate.Probe(branch[scenario], path)
    if args.lora and any(os.path.exists(os.path.join(args.lora, f)) for f in ("state_projector.pt", "evidence_tokenizer.pt")):
        model.load_look_modules(args.lora)
    looked = {}  # raw two-step rows of the accuracy run, for the consistency check of the look policy
    if args.evaluated and "look" in args.policies:
        path, _, pol = args.evaluated.partition(":")
        for name in glob.glob(os.path.join(path, "*shard*.jsonl")):
            for r in map(json.loads, open(name)):
                if r["policy"] == pol:
                    looked[r["id"]] = r
    ref = recorded(args.evaluated) if args.evaluated else {}
    ref_policy = args.evaluated.partition(":")[2] if args.evaluated else ""
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    agree = collections.Counter()
    sync = torch.cuda.synchronize
    for path in args.items:
        items = [json.loads(line) for line in open(path)]
        random.Random(0).shuffle(items)
        for n, item in enumerate(items[: args.n + args.warmup]):
            image = core.open_image(item)
            image.load()
            question = {"type": "text", "text": item["question"]}
            for policy in args.policies:
                torch.cuda.reset_peak_memory_stats()
                sync()
                start = time.perf_counter()
                parts = {}
                if policy == "look":
                    sc = item["scenario"]
                    res = branch[sc].predict(image, imgsz=args.imgsz, conf=0.01, device=0, verbose=False)[0]
                    state = probe[sc].state() if sc in probe else None
                    sync()
                    t_branch = time.perf_counter() - start
                    names = verify.CLASSES[sc]
                    dets = [[*map(float, box), float(conf), names[int(cls)]] for box, conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]
                    kept = [d for d in verify.select(dets) if round(d[4], 4) >= args.min_conf][: args.k]
                    given = {"dets": [{"box": [round(v, 1) for v in d[:4]], "conf": round(d[4], 4), "name": verify.NAMES.get(d[5], d[5]), "cls": d[5]} for d in kept],
                             "mode": "text", "thumb_tokens": args.thumb_tokens, "fmt": args.fmt, **({"state": state} if state is not None else {})}
                    t_pack = time.perf_counter() - start - t_branch
                    out = model.answer_look_cached(item, image, given)
                    if n < args.warmup:
                        continue
                    s = stats[(sc, policy)]
                    never = t_branch + t_pack + out["prep1"] + out["run1"]
                    extra = out["prep2"] + out["run2"]
                    for k, v in (("branch", t_branch), ("pack", t_pack), ("prep1", out["prep1"]), ("run1", out["run1"]), ("prep2", out["prep2"]), ("run2", out["run2"]),
                                 ("never", never), ("always", never + extra), ("total", never + (extra if out["looks"] else 0.0)), ("looks", float(out["looks"])),
                                 ("visual_tokens", out["visual_tokens"] if out["looks"] else 0), ("input_tokens", out["input_tokens"] if out["looks"] else out["input_tokens1"]),
                                 ("input_tokens1", out["input_tokens1"]), ("peak_gib", torch.cuda.max_memory_allocated() / 2**30)):
                        s[k].append(v)
                    want = looked.get(item["id"])
                    if want is not None:  # the online pipeline against the stored accuracy run: step 1, step 2 (continuation on the cache), and the decision
                        best = lambda logp: item["options"][max(range(len(item["options"])), key=lambda i: logp[i])]
                        agree[((sc, policy), "step1", best(out["logp1"]) == want["pred1"])] += 1
                        agree[((sc, policy), "step2", best(out["logp"]) == want["pred"])] += 1
                        agree[((sc, policy), "decision", out["looks"] == (want["p_look"] > max(math.exp(v) for v in want["logp1"])))] += 1
                    continue
                if policy == "ours":
                    res = branch[item["scenario"]].predict(image, imgsz=args.imgsz, conf=0.01, device=0, verbose=False)[0]
                    sync()
                    parts["branch"] = time.perf_counter() - start
                    names = verify.CLASSES[item["scenario"]]
                    dets = [[*map(float, box), float(conf), names[int(cls)]] for box, conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]
                    kept = [d for d in verify.select(dets) if round(d[4], 4) >= args.min_conf][: args.k]  # as verify.py cands + evidence
                    given = {"dets": [{"box": [round(v, 1) for v in d[:4]], "conf": round(d[4], 4), "name": verify.NAMES.get(d[5], d[5])} for d in kept],
                             "mode": "text", "thumb_tokens": args.thumb_tokens, **({"fmt": args.fmt} if args.fmt != 1 else {})}
                    content = core.evidence_content(image, given)
                else:
                    content = core.build_content(item, policy, image)
                t1 = time.perf_counter()
                out = model.answer(content + [question], len(item["options"]))
                sync()
                total = time.perf_counter() - start
                if n < args.warmup:
                    continue
                key = (item["scenario"], policy)
                s = stats[key]
                s["total"].append(total)
                s["branch"].append(parts.get("branch", 0.0))
                s["pack"].append(t1 - start - parts.get("branch", 0.0))  # candidate selection / resizing and building the content
                s["processor"].append(total - (t1 - start) - out["seconds"])  # tokenisation and image preprocessing on the CPU
                s["llm"].append(out["seconds"])
                s["visual_tokens"].append(out["visual_tokens"])
                s["input_tokens"].append(out["input_tokens"])
                s["peak_gib"].append(torch.cuda.max_memory_allocated() / 2**30)
                if args.gen_tokens and n % args.gen_every == 0:  # a longer reply: prefill once, then decode token by token
                    inputs = model.encode(content + [question]).to(model.device)
                    sync()
                    g0 = time.perf_counter()
                    model.model.generate(**inputs, max_new_tokens=args.gen_tokens, min_new_tokens=args.gen_tokens, do_sample=False)
                    sync()
                    s["gen"].append(time.perf_counter() - g0)
                pred = item["options"][max(range(len(item["options"])), key=lambda i: out["logp"][i])]
                want = ref.get((item["id"], ref_policy if policy == "ours" else policy))
                if want is not None:
                    agree[(key, pred == want)] += 1
    med = lambda v: sorted(v)[len(v) // 2]
    mean = lambda v: sum(v) / len(v)
    with open(args.rows, "a") as f:
        for (scenario, policy), s in stats.items():
            key = (scenario, policy)
            if policy == "look":  # the cascade: means, because the cost of a question depends on whether the model looks
                same = lambda what: f"{agree[(key, what, True)]}/{agree[(key, what, True)] + agree[(key, what, False)]}"
                row = {"name": args.name, "scenario": scenario, "policy": policy, "n": len(s["total"]), "look_share": 100 * mean(s["looks"]),
                       "visual_tokens": mean(s["visual_tokens"]), "input_tokens": mean(s["input_tokens"]), "input_tokens1": mean(s["input_tokens1"]), "peak_gib": max(s["peak_gib"]),
                       "throughput": len(s["total"]) / sum(s["total"]), "p90_total_ms": 1000 * sorted(s["total"])[int(0.9 * len(s["total"]))], "total_mean_ms": 1000 * mean(s["total"]),
                       **{f"{k}_ms": 1000 * med(s[k]) for k in ("never", "always", "branch", "pack", "prep1", "run1", "prep2", "run2")},
                       "same_step1": same("step1"), "same_step2": same("step2"), "same_decision": same("decision")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(row, flush=True)
                continue
            row = {"name": args.name, "scenario": scenario, "policy": policy, "n": len(s["total"]), "visual_tokens": sum(s["visual_tokens"]) / len(s["visual_tokens"]),
                   "input_tokens": sum(s["input_tokens"]) / len(s["input_tokens"]), "peak_gib": max(s["peak_gib"]),
                   "throughput": len(s["total"]) / sum(s["total"]), "p90_total_ms": 1000 * sorted(s["total"])[int(0.9 * len(s["total"]))],
                   "same_answer": agree[(key, True)], "compared": agree[(key, True)] + agree[(key, False)],
                   **{f"{k}_ms": 1000 * med(s[k]) for k in ("total", "branch", "pack", "processor", "llm")},
                   **({"gen_tokens": args.gen_tokens, "gen_ms": 1000 * med(s["gen"])} if s.get("gen") else {})}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row, flush=True)
    print("COST_RUN_DONE", flush=True)


def cmd_table(args):
    rows = [json.loads(line) for line in open(args.rows)]
    lines = []
    for scenario in ("ed", "td"):
        lines += [f"## {scenario}：单卡在线端到端代价（每题一次完整流程；中位数；图像解码不计）", "",
                  "| 系统 | 输入 | 视觉令牌 | 输入令牌 | 端到端 ms | 其中：检测分支 | 组装输入 | 预处理 | 大模型前向 | 90 分位 ms | 吞吐 题/秒 | 峰值显存 GiB | 与精度评测答案一致 | 生成较长回答 ms |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["scenario"] != scenario or r["policy"] == "look":
                continue
            same = f"{r['same_answer']}/{r['compared']}" if r["compared"] else "—"
            lines.append(f"| {r['name']} | {r['policy']} | {r['visual_tokens']:.0f} | {r['input_tokens']:.0f} | {r['total_ms']:.0f} | {r['branch_ms']:.1f} | {r['pack_ms']:.1f} | {r['processor_ms']:.1f} | "
                         f"{r['llm_ms']:.0f} | {r['p90_total_ms']:.0f} | {r['throughput']:.2f} | {r['peak_gib']:.2f} | {same} | "
                         + (f"{r['gen_ms']:.0f}（{r['gen_tokens']} 个令牌）" if r.get("gen_ms") else "—") + " |")
        lines.append("")
        cascade = [r for r in rows if r["scenario"] == scenario and r["policy"] == "look"]
        if cascade:
            lines += [f"## {scenario}：存疑再看的在线代价（每题两步都实际执行并计时；第二步在第一步的缓存上续算；「按需」只对模型自己要求看图的题计入第二步）", "",
                      "| 系统 | 看图比例 | 第一步输入令牌 | 按需：平均视觉令牌 | 按需：端到端平均 ms | 90 分位 ms | 吞吐 题/秒 | 不看图：端到端 ms（中位） | 看图：端到端 ms（中位） | "
                      "检测分支与状态 | 组装 | 第一步：准备 / 前向 | 第二步：准备 / 续算 | 峰值显存 GiB | 与精度评测一致：第一步 / 第二步 / 是否看图 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|"]
            for r in cascade:
                lines.append(f"| {r['name']} | {r['look_share']:.1f}% | {r['input_tokens1']:.0f} | {r['visual_tokens']:.0f} | {r['total_mean_ms']:.0f} | {r['p90_total_ms']:.0f} | {r['throughput']:.2f} | "
                             f"{r['never_ms']:.0f} | {r['always_ms']:.0f} | {r['branch_ms']:.1f} | {r['pack_ms']:.1f} | {r['prep1_ms']:.1f} / {r['run1_ms']:.0f} | {r['prep2_ms']:.1f} / {r['run2_ms']:.0f} | "
                             f"{r['peak_gib']:.2f} | {r['same_step1']} / {r['same_step2']} / {r['same_decision']} |")
            lines.append("")
    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--items", nargs="+", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--lora")
    r.add_argument("--policies", nargs="+", required=True, help="'ours', 'look' and/or whole-image policies such as u576 u8192")
    r.add_argument("--weights", nargs="*", default=[], help="scenario=path of the detector branch")
    r.add_argument("--imgsz", type=int, default=1280)
    r.add_argument("--k", type=int, default=8)
    r.add_argument("--min-conf", type=float, default=0.05)
    r.add_argument("--thumb-tokens", type=int, default=core.THUMB_TOKENS)
    r.add_argument("--fmt", type=int, default=1, choices=[1, 2, 3, 4], help="wording of the evidence list (as verify.py evidence --fmt)")
    r.add_argument("--stats", nargs="*", default=[], help="look: scenario=path of the detector's feature statistics, for the branch-state token")
    r.add_argument("--fastv", help="LAYER:KEEP (KEEP a count, or a fraction such as 0.5) - prune visual tokens inside the language model (fastv.py)")
    r.add_argument("--gen-tokens", type=int, default=0, help="also time the generation of a reply of this many tokens (prefill + decoding), on every --gen-every-th question")
    r.add_argument("--gen-every", type=int, default=5)
    r.add_argument("--evaluated", help="results dir[:policy] of the accuracy run, for the consistency check")
    r.add_argument("--name", required=True)
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--warmup", type=int, default=8)
    r.add_argument("--rows", required=True)
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("table")
    t.add_argument("--rows", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_table)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
