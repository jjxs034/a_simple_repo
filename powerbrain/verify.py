"""Find, then verify: the same candidates and the same aggregation rule, with different per-candidate scorers.

  cands      detections + human boxes -> de-duplicated candidates per image, each labelled by matching to the boxes
  items      one verification question per candidate for the multimodal LLM (+ the regions file it reads)
  evidence   evidence records for benchmark questions: single-pass packing of the detector branch's candidates
  variant    the same verification questions with parts of the LLM's input removed (input ablations)
  gtcands    candidates made from the human boxes instead of detector output (control for training on real candidates)
  crops      three views of every candidate plus the detector's output, for the small second stages (controls)
  aggregate  answer the benchmark questions by fixed rules from per-candidate scores of a chosen scorer

Scorers compared by `aggregate` (thresholds for every scorer are tuned on validation questions only):
  det    the detector's own class and confidence                       (first stage alone)
  cls    a small crop classifier trained on the same candidates          (generic second stage)
  mllm   the multimodal LLM verifier trained on the same candidates      (proposed second stage; further verifiers by name)
  gate   detector decides confident candidates, the LLM only the uncertain band (edge-cloud operating points)
  ideal  every candidate labelled correctly                              (ceiling for this proposer)
Reference rows: the detector's own rule on all of its boxes (`--detections`), and both first-stage rows again with
thresholds chosen on the test questions (an upper bound of the first stage, so that no gain is owed to its tuning).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random

import bench
import core

NONE = "none"
NONE_TEXT = "以上都不是（背景或误检）"
NAMES = {**bench.ED_NAMES, **bench.PL_NAMES, NONE: NONE_TEXT}
CLASSES = {"ed": list(bench.ED_NAMES), "td": bench.TD_CLASSES, "pl": list(bench.PL_NAMES)}
COUNT_OPTIONS = bench.COUNT_OPTIONS


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def all_images():
    """image path -> {scenario, width, height, boxes [[x, y, w, h, class], ...]} for every annotated image."""
    out = {}
    wanted = os.environ.get("PB_SCENARIOS", "ed td").split()  # pl: the public replication (P30), on a machine without the grid data
    if "ed" in wanted:
        for split in ("train", "test"):
            for img in bench.load_ed(split):
                out[img["image"]] = {**img, "scenario": "ed"}
    if "td" in wanted:
        for imgs in bench.load_td().values():
            for img in imgs:
                out[img["image"]] = {**img, "scenario": "td"}
    if "pl" in wanted:
        for split in ("train", "test"):
            for img in bench.load_pl(split):
                out[img["image"]] = {**img, "scenario": "pl"}
    return out


# --------------------------------------------------------------------------- cands
def select(detections, min_conf=0.02, nms=0.6, k=12):
    """The branch's candidates for one image: its detections [x0, y0, x1, y1, conf, class] by falling confidence, after
    class-agnostic suppression of near-identical boxes (one region is one candidate, whatever class it was given)."""
    kept = []
    for d in sorted((d for d in detections if d[4] >= min_conf), key=lambda d: -d[4]):
        if all(iou(d[:4], other[:4]) < nms for other in kept):
            kept.append(d)
        if len(kept) >= k:
            break
    return kept


def cmd_cands(args):
    images = all_images()
    n_img = n_cand = n_true = 0
    recall = collections.defaultdict(list)
    seen = {}
    with open(args.out, "w") as out:
        for path in args.detections:
            for line in open(path):
                rec = json.loads(line)
                stem = os.path.splitext(os.path.basename(rec["image"]))[0]
                assert stem not in seen, f"candidate ids would collide: {rec['image']} and {seen[stem]}"
                seen[stem] = rec["image"]
                info = images[rec["image"]]
                kept = select(rec["detections"], args.min_conf, args.nms, args.k)
                gts = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in info["boxes"]]
                owner = {}
                cands = []
                for d in kept:  # greedy by confidence, as in recoverable.match: a candidate takes the best object that has no owner yet
                    free, free_iou, shown, shown_iou = None, args.match, None, args.match
                    for j, g in enumerate(gts):
                        v = iou(d[:4], g)
                        if v >= shown_iou:
                            shown, shown_iou = j, v
                        if j not in owner and v >= free_iou:
                            free, free_iou = j, v
                    dup = free is None and shown is not None  # it only overlaps objects that already have an owner
                    j = free if free is not None else shown
                    if free is not None:
                        owner[free] = len(cands)
                    cands.append({"box": [round(v, 1) for v in d[:4]], "conf": round(d[4], 4), "det": d[5],
                                  "label": info["boxes"][j][4] if j is not None else NONE, "gt": j, "dup": dup})
                out.write(json.dumps({"image": rec["image"], "scenario": info["scenario"], "width": info["width"], "height": info["height"],
                                      "n_gt": len(gts), "cands": cands}, ensure_ascii=False) + "\n")
                n_img += 1
                n_cand += len(cands)
                n_true += sum(c["label"] != NONE for c in cands)
                if gts:
                    recall[info["scenario"]].append(len(owner) / len(gts))
    print(f"{args.out}: {n_img} images, {n_cand} candidates ({n_cand / max(1, n_img):.1f} per image), {100 * n_true / max(1, n_cand):.0f}% on an annotated object | "
          + " | ".join(f"{s} box recall {100 * sum(v) / len(v):.1f}%" for s, v in recall.items()))


# --------------------------------------------------------------------------- items
def cand_id(image, index):
    return f"{os.path.splitext(os.path.basename(image))[0]}#{index}"


def render_scale(box, target_tokens, max_scale):
    tokens = max((box[2] - box[0]) * (box[3] - box[1]) / (core.CELL * core.CELL), 0.05)
    return round(min(max_scale, max(1.0, math.sqrt(target_tokens / tokens))), 3)


def read_rows(paths, keep_images=None):
    rows = []
    for path in paths:
        for line in open(path):
            rec = json.loads(line)
            if keep_images is not None and rec["image"] not in keep_images:
                continue
            for i, c in enumerate(rec["cands"]):
                rows.append((rec, i, c))
    return rows


def select_training(rows, neg_ratio, limit, per_scenario=0):
    """The training candidates of every second stage (LLM verifier and small controls alike): all candidates on
    objects, at most neg_ratio times as many false alarms, in a fixed random order; then the first `limit` of them,
    or (label-budget experiments) the first `per_scenario` of each scenario, so that smaller budgets are nested in larger ones."""
    rng = random.Random(0)
    pos = [r for r in rows if r[2]["label"] != NONE]
    neg = [r for r in rows if r[2]["label"] == NONE]
    rng.shuffle(neg)
    rows = pos + neg[: int(len(pos) * neg_ratio)]
    rng.shuffle(rows)
    if per_scenario:
        seen = collections.Counter()
        kept = []
        for r in rows:
            if seen[r[0]["scenario"]] < per_scenario:
                seen[r[0]["scenario"]] += 1
                kept.append(r)
        return kept
    return rows[:limit] if limit else rows


def cmd_items(args):
    keep_images = None
    if args.only_images_of:
        keep_images = {json.loads(line)["image"] for path in args.only_images_of for line in open(path)}
    rows = read_rows(args.cands, keep_images)
    if args.balance:
        rows = select_training(rows, args.neg_ratio, args.limit, args.per_scenario)
    elif args.sample_per_scenario:  # a fixed random sample of candidates per scenario (candidate-level evaluation sets)
        rng = random.Random(0)
        rng.shuffle(rows)
        seen = collections.Counter()
        kept = []
        for r in rows:
            if seen[r[0]["scenario"]] < args.sample_per_scenario:
                seen[r[0]["scenario"]] += 1
                kept.append(r)
        rows = kept
    with open(args.out, "w") as items, open(args.out.replace(".jsonl", ".regions.jsonl"), "w") as regions:
        for rec, i, c in rows:
            cid = cand_id(rec["image"], i)
            options = CLASSES[rec["scenario"]] + [NONE]
            random.Random(cid).shuffle(options)
            letters = core.LETTERS[: len(options)]
            box = c["box"]
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            question = ("缩略图之后给出的是检测器提出的一个候选区域。请判断这个候选区域内的主要目标实际属于哪一类。\n"
                        + "\n".join(f"{l}. {NAMES.get(o, o)}" for l, o in zip(letters, options)) + "\n请只回答选项字母。")
            area = w * h / (rec["width"] * rec["height"])
            items.write(json.dumps({
                "id": cid, "scenario": rec["scenario"], "image": rec["image"], "width": rec["width"], "height": rec["height"],
                "qtype": "verify", "target": c["det"], "options": options, "answer": letters[options.index(c["label"])], "question": question,
                "evidence": [[x, y, w, h]], "boxes": [], "area_ratio": area, "bin": bench.area_bin(area), "conf": c["conf"],
            }, ensure_ascii=False) + "\n")
            note = f"（检测器判断：{NAMES.get(c['det'], c['det'])}，置信度 {c['conf']:.2f}）" if args.notes else ""
            regions.write(json.dumps({"id": cid, "regions": [core.expand([x, y, w, h], rec["width"], rec["height"])],
                                      "scale": [render_scale(box, args.target_tokens, args.max_scale)], "notes": [note]}, ensure_ascii=False) + "\n")
    labels = collections.Counter(r[2]["label"] for r in rows)
    print(f"{args.out}: {len(rows)} verification questions | labels {dict(labels)}")


def cmd_variant(args):
    """Input ablations of the LLM verifier: the same candidates and questions, with parts of the input removed."""
    n = 0
    with open(args.out, "w") as out:
        for line in open(args.regions):
            rec = json.loads(line)
            if args.notes == "none":
                rec["notes"] = ["" for _ in rec["notes"]]
            elif args.notes == "class":  # keep the detector's class, drop its confidence
                rec["notes"] = [note.split("，置信度")[0] + "）" if note else "" for note in rec["notes"]]
            if args.no_thumbnail:
                rec["thumbnail"] = False
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"{args.out}: {n} region records (notes: {args.notes}, thumbnail: {not args.no_thumbnail})")


def cmd_gtcands(args):
    """Candidates made from the human boxes instead of detector output, in the format of `cands`: the usual way to
    train a verifier, used as the control for training on the detector's own candidates. Every annotated box of a
    fit image with a random jitter, plus boxes placed away from all annotations as false alarms."""
    rng = random.Random(args.seed)
    fit = [({**img, "scenario": "ed"}) for img in bench.load_ed("train")] + [({**img, "scenario": "td"}) for img in bench.load_td()["train"]]
    fit = [img for img in fit if not bench.is_val(img["image"])]
    n_pos = n_neg = 0
    with open(args.out, "w") as out:
        for img in fit:
            w, h = img["width"], img["height"]
            gts = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in img["boxes"]]
            cands = []
            for j, (g, b) in enumerate(zip(gts, img["boxes"])):
                bw, bh = g[2] - g[0], g[3] - g[1]
                for _ in range(20):  # jitter that still clearly shows the object
                    sx, sy = rng.uniform(0.85, 1.25), rng.uniform(0.85, 1.25)
                    cx, cy = (g[0] + g[2]) / 2 + rng.uniform(-0.1, 0.1) * bw, (g[1] + g[3]) / 2 + rng.uniform(-0.1, 0.1) * bh
                    box = [max(0.0, cx - sx * bw / 2), max(0.0, cy - sy * bh / 2), min(float(w), cx + sx * bw / 2), min(float(h), cy + sy * bh / 2)]
                    if iou(box, g) >= 0.6:
                        break
                else:
                    box = g
                cands.append({"box": [round(v, 1) for v in box], "conf": 0.0, "det": b[4], "label": b[4], "gt": j, "dup": False})
            for _ in range(max(1, round(args.neg_per_box * len(gts)))):  # false alarms: boxes of annotated sizes away from every annotation
                bw, bh = (lambda g: (g[2] - g[0], g[3] - g[1]))(rng.choice(gts)) if gts else (0.1 * w, 0.1 * h)
                for _ in range(30):
                    x0, y0 = rng.uniform(0, max(1.0, w - bw)), rng.uniform(0, max(1.0, h - bh))
                    box = [x0, y0, min(float(w), x0 + bw), min(float(h), y0 + bh)]
                    if all(iou(box, g) < 0.05 for g in gts):
                        cands.append({"box": [round(v, 1) for v in box], "conf": 0.0, "det": rng.choice(CLASSES[img["scenario"]]), "label": NONE, "gt": None, "dup": False})
                        break
            n_pos += sum(c["label"] != NONE for c in cands)
            n_neg += sum(c["label"] == NONE for c in cands)
            out.write(json.dumps({"image": img["image"], "scenario": img["scenario"], "width": w, "height": h, "n_gt": len(gts), "cands": cands}, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(fit)} fit images, {n_pos} jittered annotated boxes, {n_neg} background boxes")


def cmd_evidence(args):
    """Evidence records for benchmark questions (single-pass evidence packing, policy g:<tag>): for every question,
    the candidates of its image from a candidates file. Training questions take out-of-fold candidates, so that the
    LLM sees the detector branch as it behaves on images it was not trained on."""
    by_image = {}
    for path in args.cands:
        for line in open(path):
            rec = json.loads(line)
            by_image[rec["image"]] = rec
    n = missing = total = degraded = 0
    with open(args.out, "w") as out:
        for path in args.items:
            for line in open(path):
                item = json.loads(line)
                rec = by_image.get(item["image"])
                if rec is None and not args.from_gt:
                    missing += 1
                if args.from_gt:  # control for the training recipe: a clean list made from the human boxes, as if the branch never erred
                    dets = [{"box": [b[0], b[1], b[0] + b[2], b[1] + b[3]], "conf": 0.9, "name": NAMES.get(b[4], b[4]), "cls": b[4]}
                            for b in sorted(item["boxes"], key=lambda b: (b[0], b[1])) if b[4] != args.drop_class][: args.k]
                else:
                    dets = [{"box": c["box"], "conf": c["conf"], "name": NAMES.get(c["det"], c["det"]), "cls": c["det"]}
                            for c in (rec["cands"] if rec else []) if c["conf"] >= args.min_conf and c["det"] != args.drop_class][: args.k]
                if args.list_dropout:  # training only: the list is sometimes unreliable, so the model has to check it against the image
                    rng = random.Random(f"{args.seed}-{item['id']}")
                    if rng.random() < args.list_dropout:
                        rate = rng.random()
                        dets = [d for d in dets if rng.random() >= rate]
                        if rng.random() < 0.5:
                            scale = rng.uniform(0.5, 1.0)
                            dets = [{**d, "conf": d["conf"] * scale} for d in dets]
                        degraded += 1
                record = {"id": item["id"], "dets": dets, "mode": args.mode, "band": args.band, "max_crops": args.max_crops,
                          "thumb_tokens": args.thumb_tokens, "mark": args.mark, "positions": not args.no_positions}
                if args.fmt != 1:
                    record["fmt"] = args.fmt
                if args.no_conf:
                    record["conf_text"] = False
                if args.vocab:  # tell the LLM which classes the branch can output; the classes a question depends on that are outside it are "uncovered"
                    vocab = [c for c in CLASSES[item["scenario"]] if c != args.drop_class]
                    record["vocab"] = [[c, NAMES.get(c, c)] for c in vocab]
                    record["uncovered"] = [NAMES.get(c, c) for c in core.asked_classes(item) if c not in vocab]
                    record["uncovered_tokens"] = args.uncovered_tokens
                    if args.mask_prob:
                        record["mask_prob"] = args.mask_prob
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
                total += len(dets)
    print(f"{args.out}: {n} questions, {total / max(1, n):.1f} candidates per question, {missing} questions whose image has no candidates record"
          + (f", {degraded} lists degraded" if args.list_dropout else ""))
    assert not (args.require_cands and missing), f"{missing} questions on images the detector branch was never run on"


# --------------------------------------------------------------------------- small second stages (controls)
VIEWS = ("tight", "context", "whole")


def window(image, box, scale, min_side, size=224):
    """Square window centred on the box, `scale` times its longer side (at least min_side pixels), resized to size x size.
    The part of the window that falls outside the image stays black, so the aspect ratio is never distorted."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = int(round(max(min_side, scale * max(box[2] - box[0], box[3] - box[1]))))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    return image.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), core.Image.BICUBIC)


def whole_view(image):
    return window(image, [0, 0, image.size[0], image.size[1]], 1.0, 1)


def views_of(image, box, whole=None):
    """What a small second stage sees of one candidate: the object, its surroundings, and the whole picture
    (the LLM verifier sees a thumbnail of the whole picture and the region around the candidate)."""
    return {"tight": window(image, box, 1.5, 64), "context": window(image, box, 4.0, 384), "whole": whole if whole is not None else whole_view(image)}


def det_features(scenario, c, width, height):
    """The detector's output for a candidate, as given to the small second stages (the LLM verifier reads it as text):
    one-hot detector class, confidence and its logit, box centre, size and log area relative to the image."""
    box, conf = c["box"], min(max(c["conf"], 1e-4), 1 - 1e-4)
    bw, bh = (box[2] - box[0]) / width, (box[3] - box[1]) / height
    onehot = [float(c["det"] == k) for k in CLASSES[scenario]]
    return onehot + [conf, math.log(conf / (1 - conf)) / 5, (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height, bw, bh, math.log(max(bw * bh, 1e-7)) / 10]


def cmd_crops(args):
    """Data for the small second stages: <out>/<scenario>/<split>/<view>/<name>.jpg and <out>/<scenario>/<split>/meta.jsonl.
    train = exactly the candidates the LLM verifier is trained on; val = every candidate of the validation images."""
    val_images = {json.loads(line)["image"] for path in args.val_items for line in open(path)}
    train = select_training(read_rows(args.cands), args.neg_ratio, args.limit, args.per_scenario)
    assert not any(rec["image"] in val_images for rec, _, _ in train), "training candidates must not come from validation images"
    val = read_rows(args.val_cands, val_images)  # the evaluation candidates file also holds test images: they are not exported
    count = collections.Counter()
    for split, rows in (("train", train), ("val", val)):
        by_image = collections.defaultdict(list)
        for rec, i, c in rows:
            by_image[rec["image"]].append((rec, i, c))
        metas = collections.defaultdict(list)
        for group in by_image.values():
            rec = group[0][0]
            image = core.open_image(rec)
            whole = whole_view(image)
            for _, i, c in group:
                name = cand_id(rec["image"], i).replace("#", "_")
                for view, crop in views_of(image, c["box"], whole).items():
                    d = os.path.join(args.out, rec["scenario"], split, view)
                    os.makedirs(d, exist_ok=True)
                    crop.save(os.path.join(d, name + ".jpg"), quality=90)
                metas[rec["scenario"]].append({"name": name, "label": c["label"], "feat": det_features(rec["scenario"], c, rec["width"], rec["height"])})
                count[(rec["scenario"], split, c["label"])] += 1
        for sc, rows_meta in metas.items():
            with open(os.path.join(args.out, sc, split, "meta.jsonl"), "w") as f:
                for m in rows_meta:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(dict(sorted(count.items())))


# --------------------------------------------------------------------------- aggregate
def load_scores(args):
    """name -> candidate id -> evaluate.py row for each LLM verifier; name -> candidate id -> {label: probability} for each
    small second stage. `--mllm-results` takes results directories, or name=directory to compare several verifiers
    (a plain directory is "mllm"); `--cls-probs` takes files, or name=file (plain files belong to "cls")."""
    llm, small = collections.defaultdict(dict), collections.defaultdict(dict)
    for spec in args.mllm_results or []:
        name, d = spec.split("=", 1) if "=" in spec else ("mllm", spec)
        for p in glob.glob(os.path.join(d, "*shard*.jsonl")):
            for line in open(p):
                r = json.loads(line)
                if r["policy"] == "g:v":
                    llm[name][r["id"]] = r
    for spec in args.cls_probs or []:
        name, path = spec.split("=", 1) if "=" in spec else ("cls", spec)
        for line in open(path):
            r = json.loads(line)
            small[name][r["id"]] = r["probs"]
    assert not set(llm) & set(small), "an LLM verifier and a small second stage share a name"
    return dict(llm), dict(small)


def options_of(scenario, cid):
    options = CLASSES[scenario] + [NONE]
    random.Random(cid).shuffle(options)
    return options


def score(scorer, rec, i, c, llm, small, band):
    """Returns (acceptance score in [0, 1], class) for one candidate under a scorer."""
    cid = cand_id(rec["image"], i)
    if scorer == "det":
        return c["conf"], c["det"]
    if scorer == "ideal":
        return (1.0 if c["label"] != NONE and not c["dup"] else 0.0), (c["label"] if c["label"] != NONE else c["det"])

    def from_probs(probs):
        best = max((k for k in probs if k != NONE), key=lambda k: probs[k])
        return 1.0 - probs.get(NONE, 0.0), best

    if scorer in llm or scorer == "gate":
        if scorer == "gate" and not (band[0] <= c["conf"] < band[1]):
            return (1.0 if c["conf"] >= band[1] else 0.0), c["det"]  # confident either way: the detector decides on the device
        r = llm["mllm" if scorer == "gate" else scorer][cid]
        options = options_of(rec["scenario"], cid)  # evaluate.py stores log-probabilities in option order
        m = max(r["logp"])
        z = sum(math.exp(v - m) for v in r["logp"])
        return from_probs({o: math.exp(v - m) / z for o, v in zip(options, r["logp"])})
    return from_probs(small[scorer][cid])


def check_coverage(by_image, images, llm, small, scorers):
    """Every candidate of every evaluated image must have an output from every scorer, and the LLM rows must belong
    to these candidates (same label, same number of options). Stops instead of reporting numbers on partial outputs."""
    problems = []
    for name in scorers:
        if name in ("det", "ideal"):
            continue
        is_llm = name in llm or name == "gate"
        table = llm.get("mllm" if name == "gate" else name, {}) if is_llm else small.get(name, {})
        total = missing = wrong = 0
        for im in images:
            rec = by_image.get(im)
            for i, c in enumerate(rec["cands"] if rec else []):
                total += 1
                r = table.get(cand_id(im, i))
                if r is None:
                    missing += 1
                elif is_llm and (r.get("truth") != c["label"] or len(r["logp"]) != len(CLASSES[rec["scenario"]]) + 1):
                    wrong += 1
        print(f"coverage {name}: {total - missing}/{total} candidates scored, {wrong} rows that do not match their candidate")
        if missing or wrong:
            problems.append(name)
    if problems:
        raise SystemExit(f"incomplete or mismatched per-candidate outputs for: {', '.join(problems)}")


def answer(item, scored, theta, dedupe):
    """scored: [(score, class, box)] for the image. Fixed rules, identical for every scorer."""
    kept = []
    for s, c, box in sorted((x for x in scored if x[0] >= theta), key=lambda x: -x[0]):
        if all(not (c == k[1] and iou(box, k[2]) >= dedupe) for k in kept):
            kept.append((s, c, box))
    if item["qtype"] == "cls":
        votes = collections.Counter()
        for s, c, _ in kept:
            votes[c] += s
        ranked = [c for c, _ in votes.most_common() if c in item["options"]]
        if ranked:
            return ranked[0]
        fallback = [c for _, c, _ in sorted(scored, key=lambda x: -x[0]) if c in item["options"]]  # nothing accepted: best-scored candidate
        return fallback[0] if fallback else item["options"][0]
    n = sum(c == item["target"] for _, c, _ in kept)
    if item["qtype"] == "exist":
        return "是" if n else "否"
    return COUNT_OPTIONS[min(max(n, 1), 4) - 1]


def clustered_ci(items, values, reps=2000, seed=0):
    """95% interval of the mean of `values` (one per question), resampling images: questions on one image share its candidates."""
    import numpy as np

    by_image = collections.defaultdict(list)
    for it, v in zip(items, values):
        by_image[it["image"]].append(v)
    if not by_image:
        return float("nan"), float("nan")
    sums = np.array([sum(v) for v in by_image.values()], dtype=float)
    counts = np.array([len(v) for v in by_image.values()], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(reps, len(sums)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


RAW = "检测器直接答题（全部检测框）"


def cmd_aggregate(args):
    import detector

    by_image = {}
    for path in args.cands:
        for line in open(path):
            rec = json.loads(line)
            by_image[rec["image"]] = rec
    llm, small = load_scores(args)
    test = [json.loads(line) for path in args.items for line in open(path)]
    tune = [json.loads(line) for path in args.tune_items for line in open(path)]
    assert not {it["image"] for it in test} & {it["image"] for it in tune}, "tuning and test questions share images"
    scorers = [s for s in args.scorers if s != "cls" or "cls" in small] + [n for n in list(llm) + list(small) if n not in args.scorers]
    check_coverage(by_image, {it["image"] for it in test + tune}, llm, small, scorers)
    truth = lambda it: it["options"][core.LETTERS.index(it["answer"])]
    group = lambda it: 0 if it["bin"] == 0 else (1 if it["bin"] == 1 else 2)
    grid = [round(0.02 * k, 2) for k in range(1, 50)]
    correct = collections.defaultdict(dict)   # row name -> question id -> 0/1
    usage, notes = {}, []
    raw = {}
    for path in args.detections or []:
        for line in open(path):
            rec = json.loads(line)
            raw[rec["image"]] = rec["detections"]

    def scored_for(scorer, it, band):
        rec = by_image.get(it["image"])
        return [(*score(scorer, rec, i, c, llm, small, band), c["box"]) for i, c in enumerate(rec["cands"])] if rec else []

    variants = []
    for s in scorers:
        variants += [("gate", band) for band in args.bands] if s == "gate" else [(s, None)]
    cells_of = [(sc, q) for sc in ("ed", "td") for q in ("cls", "exist", "count")]
    for sc, q in cells_of:
        sub_tune = [it for it in tune if it["scenario"] == sc and it["qtype"] == q]
        sub_test = [it for it in test if it["scenario"] == sc and it["qtype"] == q]
        if not sub_test:
            continue
        if raw:  # the detector's own rule on all of its boxes (detector.py answer), as the first-stage-alone reference
            for name, pool in ((RAW, [it for it in sub_tune if it["image"] in raw]), (RAW + "·阈值取测试集最优", sub_test)):
                thr = max((detector.accuracy(pool, raw, t / 20)[0], t / 20) for t in range(1, 19))[1] if pool and q != "cls" else 0.25
                for it, ok in zip(sub_test, detector.accuracy(sub_test, raw, thr)[1]):
                    correct[name][it["id"]] = int(ok)
        for scorer, band in variants:
            name = scorer if band is None else f"gate[{band[0]:g},{band[1]:g})"
            pick = lambda pool, cache, dedupes: max(((sum(answer(it, cache[it["id"]], th, dd) == truth(it) for it in pool) / max(1, len(pool)), th, dd)
                                                     for th in grid for dd in dedupes), key=lambda t: (t[0], -abs(t[1] - 0.5)))
            tune_cache = {it["id"]: scored_for(scorer, it, band) for it in sub_tune}
            test_cache = {it["id"]: scored_for(scorer, it, band) for it in sub_test}
            # ideal: duplicates are already rejected by their flag, so accepted candidates are never merged (dedupe IoU above 1)
            dedupes = [1.01] if scorer == "ideal" else args.dedupe_grid
            best = pick(sub_tune, tune_cache, dedupes)
            for it in sub_test:
                correct[name][it["id"]] = int(answer(it, test_cache[it["id"]], best[1], best[2]) == truth(it))
            notes.append(f"{name} {sc} {q}: threshold {best[1]:.2f}, dedupe IoU {best[2]:.1f} (validation accuracy {100 * best[0]:.1f}%)")
            if scorer == "det" and q != "cls":  # the cheap fix: one threshold per target class instead of one per question type
                for target in sorted({it["target"] for it in sub_test}):
                    pool = [it for it in sub_tune if it["target"] == target]
                    per = pick(pool, tune_cache, dedupes) if len(pool) >= 20 else best
                    for it in sub_test:
                        if it["target"] == target:
                            correct["det·逐类调阈值"][it["id"]] = int(answer(it, test_cache[it["id"]], per[1], per[2]) == truth(it))
            elif scorer == "det":
                for it in sub_test:
                    correct["det·逐类调阈值"][it["id"]] = correct[name][it["id"]]
            if scorer == "det" or scorer in small:  # upper bound of this scorer: thresholds chosen on the test questions themselves
                top = pick(sub_test, test_cache, dedupes)
                for it in sub_test:
                    correct[name + "·阈值取测试集最优"][it["id"]] = int(answer(it, test_cache[it["id"]], top[1], top[2]) == truth(it))
    for scorer, band in variants:
        if scorer in llm or scorer == "gate":
            name = scorer if band is None else f"gate[{band[0]:g},{band[1]:g})"
            for sc in ("ed", "td"):
                images = {it["image"] for it in test if it["scenario"] == sc}
                calls = [sum(1 for c in by_image[im]["cands"] if band is None or band[0] <= c["conf"] < band[1]) if im in by_image else 0 for im in images]
                usage[(name, sc)] = sum(calls) / max(1, len(calls))

    sub = {sc: [it for it in test if it["scenario"] == sc] for sc in ("ed", "td")}
    lines = ["## 同一批候选、同一套汇总规则，不同的逐候选打分者（测试集准确率 %）", "",
             "| 打分者 | 外力破坏 | cls | exist | count | <0.2% | ≥0.2% | 每图大模型调用 | 塔线缺陷 | cls | exist | count | <0.2% | ≥0.2% | 每图大模型调用 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    acc = lambda name, pool: f"{100 * sum(correct[name][it['id']] for it in pool) / len(pool):.1f}" if pool else "—"
    for name in correct:
        cells = []
        for sc in ("ed", "td"):
            cells.append(acc(name, sub[sc]))
            cells += [acc(name, [it for it in sub[sc] if it["qtype"] == q]) for q in ("cls", "exist", "count")]
            cells += [acc(name, [it for it in sub[sc] if group(it) == 0]), acc(name, [it for it in sub[sc] if group(it) > 0])]
            cells.append(f"{usage[(name, sc)]:.1f}" if (name, sc) in usage else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    def diff(a, b, pool):
        d = [correct[a][it["id"]] - correct[b][it["id"]] for it in pool]
        lo, hi = clustered_ci(pool, d)
        return f"{100 * sum(d) / len(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）" if d else "—"

    references = [r for r in (RAW, RAW + "·阈值取测试集最优", "det", "det·逐类调阈值", "det·阈值取测试集最优") if r in correct]
    references += [r for n in small for r in (n, n + "·阈值取测试集最优") if r in correct] + [n for n in llm if n != "mllm" and n in correct]
    for ref in references:
        lines += ["", f"## 相对「{ref}」的配对差值（百分点；95% 区间按图像重采样）", "",
                  "| 打分者 | 外力破坏 | 外力破坏计数题 | 外力破坏 <0.2% | 塔线缺陷 | 塔线缺陷计数题 | 塔线缺陷 <0.2% |", "|---|---|---|---|---|---|---|"]
        for name in correct:
            if name == ref or name.startswith(RAW) or name.startswith("det") or name.endswith("·阈值取测试集最优"):
                continue
            cells = []
            for sc in ("ed", "td"):
                cells += [diff(name, ref, sub[sc]), diff(name, ref, [it for it in sub[sc] if it["qtype"] == "count"]), diff(name, ref, [it for it in sub[sc] if group(it) == 0])]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "各打分者的阈值与去重 IoU 均只在验证题上选定（标「阈值取测试集最优」的各行除外，它们是相应打分者的上界，只作参照）；区间以选定的阈值为条件，未计入选阈值的波动：", ""] + [f"- {n}" for n in notes]
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        with open(os.path.splitext(args.out)[0] + ".jsonl", "w") as f:
            for name, per in correct.items():
                for i, v in per.items():
                    f.write(json.dumps({"scorer": name, "id": i, "correct": v}, ensure_ascii=False) + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cands")
    c.add_argument("--detections", nargs="+", required=True); c.add_argument("--out", required=True)
    c.add_argument("--min-conf", type=float, default=0.02); c.add_argument("--k", type=int, default=12)
    c.add_argument("--nms", type=float, default=0.6); c.add_argument("--match", type=float, default=0.3)
    c.set_defaults(func=cmd_cands)
    i = sub.add_parser("items")
    i.add_argument("--cands", nargs="+", required=True); i.add_argument("--out", required=True)
    i.add_argument("--only-images-of", nargs="*", default=[]); i.add_argument("--balance", action="store_true")
    i.add_argument("--neg-ratio", type=float, default=1.0); i.add_argument("--limit", type=int, default=0)
    i.add_argument("--per-scenario", type=int, default=0); i.add_argument("--sample-per-scenario", type=int, default=0)
    i.add_argument("--notes", action="store_true"); i.add_argument("--target-tokens", type=float, default=16.0); i.add_argument("--max-scale", type=float, default=2.5)
    i.set_defaults(func=cmd_items)
    v = sub.add_parser("variant")
    v.add_argument("--regions", required=True); v.add_argument("--out", required=True)
    v.add_argument("--notes", choices=["keep", "class", "none"], default="keep"); v.add_argument("--no-thumbnail", action="store_true")
    v.set_defaults(func=cmd_variant)
    g = sub.add_parser("gtcands")
    g.add_argument("--out", required=True); g.add_argument("--neg-per-box", type=float, default=0.5); g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gtcands)
    e = sub.add_parser("evidence")
    e.add_argument("--cands", nargs="*", default=[]); e.add_argument("--items", nargs="+", required=True); e.add_argument("--out", required=True)
    e.add_argument("--k", type=int, default=8); e.add_argument("--min-conf", type=float, default=0.05)
    e.add_argument("--mode", choices=["full", "text", "crops"], default="full"); e.add_argument("--band", type=float, nargs=2, default=[0.05, 0.6])
    e.add_argument("--max-crops", type=int, default=4); e.add_argument("--thumb-tokens", type=int, default=core.THUMB_TOKENS)
    e.add_argument("--no-conf", action="store_true", help="list the detections without their confidence (the interface of earlier work; use with a higher --min-conf)")
    e.add_argument("--vocab", action="store_true", help="state the branch's class vocabulary; questions about a class outside it get a larger whole-image view")
    e.add_argument("--uncovered-tokens", type=int, default=1280)
    e.add_argument("--mask-prob", type=float, default=0.0, help="training records only: probability of masking the asked class out of the branch (evidence masking)")
    e.add_argument("--from-gt", action="store_true", help="build the list from the human boxes of the image instead of detector candidates (training-recipe control)")
    e.add_argument("--list-dropout", type=float, default=0.0, help="training records only: probability that the list of a question is degraded (candidates removed at a random rate, confidences sometimes scaled down)")
    e.add_argument("--seed", type=int, default=0, help="seed of --list-dropout")
    e.add_argument("--fmt", type=int, default=1, choices=[1, 2, 3, 4], help="wording of the list (core.evidence_content): 1 runs up to 2026-10-09, 2 final round, 3 compact lines for the edge, 4 learned tokens (look.py)")
    e.add_argument("--mark", action="store_true", help="draw the candidates, numbered, on the thumbnail")
    e.add_argument("--no-positions", action="store_true", help="leave the positions out of the text list (they are then only shown by --mark)")
    e.add_argument("--drop-class", default="", help="simulate a detector branch that has never been taught this class: its candidates of the class are removed")
    e.add_argument("--require-cands", action="store_true", help="fail when the image of a question has no candidates record")
    e.set_defaults(func=cmd_evidence)
    r = sub.add_parser("crops")
    r.add_argument("--cands", nargs="+", required=True); r.add_argument("--val-cands", nargs="+", required=True)
    r.add_argument("--val-items", nargs="+", required=True); r.add_argument("--out", required=True)
    r.add_argument("--neg-ratio", type=float, default=1.0); r.add_argument("--limit", type=int, default=24000); r.add_argument("--per-scenario", type=int, default=0)
    r.set_defaults(func=cmd_crops)
    a = sub.add_parser("aggregate")
    a.add_argument("--cands", nargs="+", required=True); a.add_argument("--items", nargs="+", required=True); a.add_argument("--tune-items", nargs="+", required=True)
    a.add_argument("--scorers", nargs="+", default=["det", "cls", "mllm", "gate", "ideal"])
    a.add_argument("--mllm-results", nargs="*"); a.add_argument("--cls-probs", nargs="*"); a.add_argument("--detections", nargs="*")
    a.add_argument("--dedupe-grid", type=float, nargs="*", default=[0.3, 0.5])
    a.add_argument("--bands", type=lambda s: tuple(float(v) for v in s.split(",")), nargs="*", default=[(0.05, 0.5), (0.1, 0.4), (0.02, 0.7)])
    a.add_argument("--out")
    a.set_defaults(func=cmd_aggregate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
