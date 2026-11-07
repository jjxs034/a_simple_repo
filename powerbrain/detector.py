"""Expert small-model baseline: a detector trained on the same boxes.

  prep     write a YOLO-format dataset (symlinked images, label files, yaml) for one scenario
  train    fine-tune YOLO11m
  detect   run the detector on the images of benchmark items -> detections.jsonl
  answer   answer the benchmark questions from detections by fixed rules (thresholds tuned on held-out training images)
  regions  detector + crop: regions file for the MLLM (policy g:<tag>)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil

import bench
import core

CLASSES = {"ed": list(bench.ED_NAMES), "td": bench.TD_CLASSES, "pl": list(bench.PL_NAMES)}
COUNT_OPTIONS = bench.COUNT_OPTIONS


def cmd_prep(args):
    images = {"ed": lambda: bench.load_ed("train"), "td": lambda: bench.load_td()["train"], "pl": lambda: bench.load_pl("train")}[args.scenario]()
    val = [img for img in images if bench.is_val(img["image"])]
    fit = [img for img in images if not bench.is_val(img["image"])]
    if args.holdout_fold >= 0:  # cross-fitting: this detector never sees one half of the fit images
        import head as head_lib

        fit = [img for img in fit if head_lib.fold_of(img["image"], 2) != args.holdout_fold]
    if args.exclude_class:  # a hazard type that has never been labelled: no image containing it is available
        has = lambda img: any(b[4] == args.exclude_class for b in img["boxes"])
        new = sorted((img for img in fit if has(img)), key=lambda img: hashlib.md5(img["image"].encode()).hexdigest())[: args.add_k]
        fit = [img for img in fit if not has(img)] + new  # plus the first k labelled examples of the new type
        val = [img for img in val if not has(img)] + sorted((img for img in val if has(img)), key=lambda img: img["image"])[: max(0, min(args.add_k // 5, 40))]
    if args.percent < 100:  # label budget: the same fraction of fit and of validation images
        fit = [img for img in fit if bench.in_budget(img["image"], args.percent, args.seed)]
        small = [img for img in val if bench.in_budget(img["image"], args.percent, args.seed)]
        val = small if len(small) >= 30 else sorted(val, key=lambda img: hashlib.md5(f"{args.seed}:{img['image']}".encode()).hexdigest())[:30]
    n_val = len(val)
    root = os.path.join(args.out_dir, args.name or (args.scenario + (f"_cf{args.holdout_fold}" if args.holdout_fold >= 0 else "")))
    for part in ("images", "labels"):  # the detector trains on what is in these folders: nothing from an earlier call may remain
        shutil.rmtree(os.path.join(root, part), ignore_errors=True)
    for split, subset in (("val", val), ("train", fit)):
        os.makedirs(f"{root}/images/{split}", exist_ok=True)
        os.makedirs(f"{root}/labels/{split}", exist_ok=True)
        for img in subset:
            name = img["key"]
            link = f"{root}/images/{split}/{name}.jpg"
            if not os.path.lexists(link):
                os.symlink(img["image"], link)
            with open(f"{root}/labels/{split}/{name}.txt", "w") as f:
                for x, y, w, h, c in img["boxes"]:
                    f.write(f"{CLASSES[args.scenario].index(c)} {(x + w / 2) / img['width']:.6f} {(y + h / 2) / img['height']:.6f} {w / img['width']:.6f} {h / img['height']:.6f}\n")
    with open(f"{root}/images.json", "w") as f:
        json.dump({"train": [img["image"] for img in fit], "val": [img["image"] for img in val]}, f)
    with open(f"{root}/data.yaml", "w") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: c{i}\n" for i in range(len(CLASSES[args.scenario]))))
    print(f"{args.scenario}: train {len(fit)} val {n_val} -> {root}")


def cmd_train(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=os.path.join(args.data_dir, args.name or args.scenario, "data.yaml"), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=[int(d) for d in args.devices.split(",")], project=args.project, name=args.name or args.scenario, exist_ok=True, workers=8, plots=False,
    )
    print("DET_TRAIN_DONE", flush=True)


def cmd_detect(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    if args.image_list:
        paths = [line.strip() for line in open(args.image_list) if line.strip()]
    else:
        items = [json.loads(line) for path in args.items for line in open(path)]
        paths = sorted({it["image"] for it in items if it["scenario"] == args.scenario})
    names = CLASSES[args.scenario]
    load = lambda batch: [core.local_path(p) for p in batch]
    if args.corrupt_plan:  # degradation-aware alignment (P20): each listed image with its own corruption; other images are skipped
        import corrupt
        from PIL import Image

        plan = {r["image"]: (r["kind"], r["severity"]) for r in map(json.loads, open(args.corrupt_plan))}
        paths = [p for p in paths if p in plan]
        load = lambda batch: [corrupt.apply(Image.open(core.local_path(p)).convert("RGB"), plan[p][0], plan[p][1], p) for p in batch]
    if args.corrupt:  # robustness test: the branch sees the corrupted image (corrupt.py), like every other system
        import corrupt
        from PIL import Image

        kind, severity = args.corrupt.split(":")
        load = lambda batch: [corrupt.apply(Image.open(core.local_path(p)).convert("RGB"), kind, int(severity), p) for p in batch]
    if args.enhance:  # control: a standard enhancement in front of the branch (after the corruption, if there is one)
        import corrupt
        from PIL import Image

        plain = load
        load = lambda batch: [corrupt.enhance(x if not isinstance(x, str) else Image.open(x).convert("RGB")) for x in plain(batch)]
    with open(args.out, "w") as out:
        for start in range(0, len(paths), 16):
            batch = paths[start : start + 16]
            for path, res in zip(batch, model.predict(load(batch), imgsz=args.imgsz, conf=0.01, device=int(args.devices.split(",")[0]), verbose=False)):
                dets = [[*map(float, box), float(conf), names[int(cls)]] for box, conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]
                out.write(json.dumps({"image": path, "detections": dets}, ensure_ascii=False) + "\n")
    print("DET_DETECT_DONE", flush=True)


def rule_answer(item, dets, thr):
    """dets: [x0, y0, x1, y1, conf, class]. Returns the option text the detector implies."""
    if item["qtype"] == "cls":
        score = collections.Counter()
        for d in dets:
            score[d[5]] += d[4]
        ranked = [c for c, _ in score.most_common() if c in item["options"]]
        return ranked[0] if ranked else item["options"][0]
    hits = [d for d in dets if d[5] == item["target"] and d[4] >= thr]
    if item["qtype"] == "exist":
        return "是" if hits else "否"
    return COUNT_OPTIONS[min(max(len(hits), 1), 4) - 1]


def accuracy(items, detections, thr):
    ok = [rule_answer(it, detections.get(it["image"], []), thr) == it["options"][core.LETTERS.index(it["answer"])] for it in items]
    return sum(ok) / max(1, len(ok)), ok


def cmd_answer(args):
    detections = {}
    for path in args.detections:
        for line in open(path):
            rec = json.loads(line)
            detections[rec["image"]] = rec["detections"]
    tune = [json.loads(line) for path in args.tune_items for line in open(path)]
    test = [json.loads(line) for path in args.items for line in open(path)]
    rows = []
    scenarios = sorted({it["scenario"] for it in test})  # ed, td as before; pl for the public replication (P30)
    for scenario in scenarios:
        for qtype in ("cls", "exist", "count"):
            sub_tune = [it for it in tune if it["scenario"] == scenario and it["qtype"] == qtype and it["image"] in detections]
            sub_test = [it for it in test if it["scenario"] == scenario and it["qtype"] == qtype]
            if not sub_test:
                continue
            best = max((accuracy(sub_tune, detections, t / 20)[0], t / 20) for t in range(1, 19)) if sub_tune and qtype != "cls" else (float("nan"), 0.25)
            acc, ok = accuracy(sub_test, detections, best[1])
            print(f"{scenario} {qtype}: threshold {best[1]:.2f} (tuned on {len(sub_tune)} held-out training questions, acc {100 * best[0]:.1f}%) -> test {100 * acc:.1f}% (n={len(sub_test)})")
            rows += [{"id": it["id"], "policy": "detector", "scenario": scenario, "qtype": qtype, "correct": int(o), "bin": it["bin"]} for it, o in zip(sub_test, ok)]
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for scenario in scenarios:
        sub = [r["correct"] for r in rows if r["scenario"] == scenario]
        if sub:
            print(f"{scenario} overall: {100 * sum(sub) / len(sub):.1f}% (n={len(sub)})")


DISPLAY = {**bench.ED_NAMES}


def cmd_regions(args):
    """Detector proposals as evidence regions.

    Default (conf 0.25, k 3, merged): the plain detector-plus-crop baseline.
    High-recall mode (--conf 0.05 --k 8 --no-merge): many low-confidence candidates for the LLM to verify.
    --size-aware T: each region is rendered so that the detected box spans about T visual tokens
    (tiny boxes are magnified, large ones shrunk), within --budget total visual tokens.
    --notes: tell the LLM what the detector said about each region (class and confidence).
    """
    import math

    detections = {}
    for path in args.detections:
        for line in open(path):
            rec = json.loads(line)
            detections[rec["image"]] = rec["detections"]
    items = [json.loads(line) for path in args.items for line in open(path)]
    if args.only_fold >= 0:
        import head as head_lib

        items = [it for it in items if head_lib.fold_of(it["image"], 2) == args.only_fold]
    used, kept = [], []
    with open(args.out, "w") as out:
        for it in items:
            dets = [d for d in detections.get(it["image"], []) if d[4] >= args.conf]
            if it["qtype"] != "cls" and any(d[5] == it["target"] for d in dets):
                dets = [d for d in dets if d[5] == it["target"]]  # the question names a class: look at its detections first
            if args.merge:
                dets = sorted(dets, key=lambda d: -d[4])[: 2 * args.k]
                regions = core.merge_regions([core.expand([d[0], d[1], d[2] - d[0], d[3] - d[1]], it["width"], it["height"]) for d in dets], args.k) if dets else []
                out.write(json.dumps({"id": it["id"], "regions": regions, "scale": 1.0}) + "\n")
                continue
            dets = sorted(dets, key=lambda d: -d[4])[: args.k]
            picked = []
            for d in dets:
                box = core.expand([d[0], d[1], d[2] - d[0], d[3] - d[1]], it["width"], it["height"])
                scale = 1.0
                if args.size_aware > 0:
                    tokens_on_box = max((d[2] - d[0]) * (d[3] - d[1]) / (core.CELL * core.CELL), 0.05)
                    scale = min(args.max_scale, max(args.min_scale, math.sqrt(args.size_aware / tokens_on_box)))
                note = f"（检测器判断：{DISPLAY.get(d[5], d[5])}，置信度 {d[4]:.2f}）" if args.notes else ""
                picked.append((box, round(scale, 3), note, core.tokens_of(*core.render_size(box, scale))))
            total = core.THUMB_TOKENS + sum(p[3] for p in picked)
            while len(picked) > 1 and total > args.budget:  # drop the least confident candidate first
                total -= picked.pop()[3]
            picked.sort(key=lambda p: (p[0][1], p[0][0]))
            out.write(json.dumps({"id": it["id"], "regions": [p[0] for p in picked], "scale": [p[1] for p in picked],
                                  "notes": [p[2] for p in picked] if args.notes else None, "planned_tokens": total}, ensure_ascii=False) + "\n")
            used.append(total)
            kept.append(len(picked))
    if used:
        print(f"{args.out}: mean planned tokens {sum(used) / len(used):.0f} | mean regions {sum(kept) / len(kept):.2f} | items {len(used)}")
    print("DET_REGIONS_DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prep"); p.add_argument("--scenario", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--holdout-fold", type=int, default=-1); p.add_argument("--percent", type=float, default=100.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-class", default=""); p.add_argument("--add-k", type=int, default=0); p.add_argument("--name", default=""); p.set_defaults(func=cmd_prep)
    t = sub.add_parser("train")
    for name, default in (("--scenario", None), ("--data-dir", None), ("--weights", None), ("--project", None), ("--devices", "0")):
        t.add_argument(name, default=default, required=default is None)
    t.add_argument("--imgsz", type=int, default=1280); t.add_argument("--epochs", type=int, default=50); t.add_argument("--batch", type=int, default=16)
    t.add_argument("--name", default="", help="data set / run name when it differs from the scenario (cross-fitting folds)")
    t.set_defaults(func=cmd_train)
    d = sub.add_parser("detect")
    d.add_argument("--scenario", required=True); d.add_argument("--items", nargs="*", default=[]); d.add_argument("--image-list"); d.add_argument("--weights", required=True)
    d.add_argument("--out", required=True); d.add_argument("--imgsz", type=int, default=1280); d.add_argument("--devices", default="0")
    d.add_argument("--corrupt", help="KIND:SEVERITY - run the branch on corrupted images (corrupt.py)")
    d.add_argument("--corrupt-plan", help="jsonl of {image, kind, severity} written by `corrupt.py plan`: only these images are run, each with its own corruption")
    d.add_argument("--enhance", action="store_true", help="apply corrupt.enhance (gamma correction and CLAHE) to every image before the branch sees it")
    d.set_defaults(func=cmd_detect)
    a = sub.add_parser("answer")
    a.add_argument("--detections", nargs="+", required=True); a.add_argument("--items", nargs="+", required=True)
    a.add_argument("--tune-items", nargs="+", required=True); a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_answer)
    r = sub.add_parser("regions")
    r.add_argument("--detections", nargs="+", required=True); r.add_argument("--items", nargs="+", required=True); r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=3); r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--no-merge", dest="merge", action="store_false"); r.add_argument("--size-aware", type=float, default=0.0)
    r.add_argument("--min-scale", type=float, default=0.5); r.add_argument("--max-scale", type=float, default=2.5)
    r.add_argument("--budget", type=int, default=1024); r.add_argument("--notes", action="store_true"); r.add_argument("--only-fold", type=int, default=-1)
    r.set_defaults(func=cmd_regions)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
