"""Synthetic image corruptions for the robustness test (unseen weather, light and transmission conditions).

Applied on the fly to the source image, deterministically (seeded by the image path), before any system sees it:
the detector branch, the thumbnail and the whole-image inputs all get the same corrupted image.

  fog     uniform haze plus low-frequency density variation
  dark    low light: gain and gamma, plus sensor noise
  blur    defocus blur, radius proportional to the image size
  noise   Gaussian sensor noise
  jpeg    strong JPEG compression (bad transmission link)
  rain    slanted rain streaks and a slight loss of contrast

Held out (HELDOUT): never used while the method was designed; applied once, in the final round.
  motion  horizontal motion blur (shake of the camera or the platform)
  bright  overexposure: gain with clipping
  lowres  the frame transmitted at a fraction of its resolution and scaled back up

    python corrupt.py --items ed_test.jsonl td_test.jsonl --kind fog --severity 2 --out-dir <dir>
        ->  <dir>/ed_test@fog2.jsonl ...   (the same questions with a "corrupt" field; ids unchanged)
    python corrupt.py --preview <image> --out-dir <dir>      (one corrupted copy per kind, for a visual check)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

KINDS = ("fog", "dark", "blur", "noise", "jpeg", "rain")
HELDOUT = ("motion", "bright", "lowres")
LEVELS = {  # severity 1, 2, 3
    "fog": [0.75, 0.60, 0.45],  # transmission
    "dark": [(0.50, 0.02), (0.35, 0.04), (0.20, 0.06)],  # (gain, noise sigma)
    "blur": [0.0015, 0.0030, 0.0050],  # radius / longer side
    "noise": [0.04, 0.08, 0.12],  # sigma on a 0-1 scale
    "jpeg": [30, 15, 8],  # quality
    "rain": [(0.02, 0.90), (0.04, 0.85), (0.07, 0.80)],  # (share of pixels on a streak, contrast)
    "motion": [0.004, 0.008, 0.014],  # length of the blur / longer side
    "bright": [(1.6, 0.05), (2.2, 0.10), (3.0, 0.15)],  # (gain, lift)
    "lowres": [2, 4, 6],  # downscaling factor
}


def _rng(path, kind):
    return np.random.default_rng(int(hashlib.md5(f"{kind}:{os.path.basename(path)}".encode()).hexdigest()[:8], 16))


def _smooth_field(rng, width, height, cells=6):
    """A low-frequency random field in [0, 1], the size of the image."""
    small = Image.fromarray((rng.random((cells, cells)) * 255).astype(np.uint8))
    return np.asarray(small.resize((width, height), Image.BICUBIC), dtype=np.float32) / 255.0


def apply(image, kind, severity, path=""):
    """image: PIL RGB. Returns the corrupted PIL RGB image of the same size."""
    level = LEVELS[kind][severity - 1]
    rng = _rng(path, kind)
    width, height = image.size
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=level * max(width, height)))
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=level)
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    if kind == "lowres":
        return image.resize((max(1, width // level), max(1, height // level)), Image.BILINEAR).resize((width, height), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    if kind == "motion":  # box blur along the rows
        k = max(3, int(level * max(width, height))) | 1
        c = np.cumsum(np.pad(x, ((0, 0), (k // 2 + 1, k // 2), (0, 0)), mode="edge"), axis=1, dtype=np.float64)
        x = ((c[:, k:] - c[:, :-k]) / k).astype(np.float32)
    elif kind == "bright":
        gain, lift = level
        x = x * gain + lift
    elif kind == "fog":
        t = np.clip(level + 0.25 * (_smooth_field(rng, width, height) - 0.5), 0.05, 1.0)[..., None]
        x = x * t + 0.85 * (1.0 - t)
    elif kind == "dark":
        gain, sigma = level
        x = np.power(x * gain, 1.2) + rng.normal(0.0, sigma, x.shape).astype(np.float32)
    elif kind == "noise":
        x = x + rng.normal(0.0, level, x.shape).astype(np.float32)
    elif kind == "rain":
        density, contrast = level
        length = max(12, int(0.02 * max(width, height)))
        layer = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(layer)
        n = int(density * width * height / length)
        xs, ys = rng.integers(0, width, n), rng.integers(0, height, n)
        for x0, y0 in zip(xs.tolist(), ys.tolist()):  # short slanted lines: streaks fall slightly sideways
            draw.line([(x0, y0), (x0 + int(0.3 * length), y0 + length)], fill=255, width=max(1, width // 1500))
        streaks = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32)[..., None] / 255.0
        x = (x - 0.5) * contrast + 0.5
        x = x * (1 - 0.55 * streaks) + 0.8 * 0.55 * streaks
    else:
        raise ValueError(kind)
    return Image.fromarray((np.clip(x, 0.0, 1.0) * 255).astype(np.uint8))


def enhance(image):
    """A standard enhancement placed in front of the detector branch (control row of the robustness test): gamma
    correction that brings the mean lightness to mid-grey, then contrast-limited adaptive histogram equalisation of the
    lightness channel. The parameters are the usual defaults and were fixed before any result was read."""
    import cv2

    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    light = lab[..., 0].astype(np.float32) / 255.0
    gamma = float(np.clip(np.log(0.5) / np.log(np.clip(light.mean(), 0.05, 0.95)), 0.4, 2.5))  # mean ** gamma = 0.5
    light = (np.clip(light**gamma, 0.0, 1.0) * 255).astype(np.uint8)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def planned(image_path, prob, seed):
    """(kind, severity) of a training image under the augmentation plan of `seed`, or None: with probability `prob` one
    of the six KINDS at a random severity. Depends on the file name and the seed only, so every question file agrees."""
    import random

    rng = random.Random(f"{seed}-{os.path.basename(image_path)}")
    if rng.random() >= prob:
        return None
    return rng.choice(KINDS), rng.choice([1, 2, 3])


def write_plan(items_paths, prob, seed, out_dir, tag, image_lists=()):
    """Training questions with degraded images (P20): <name>@<tag>.jsonl for every question file (the questions on a
    selected image carry a "corrupt" field), and plan_<tag>.jsonl listing the selected images. `image_lists`
    ("<scenario>=<file with one image path per line>") adds images that no question file given here refers to, so that
    one plan covers every training image and any question file can be planned later with the same result."""
    plan = {}
    for spec in image_lists:
        scenario, path = spec.split("=", 1)
        for image in (line.strip() for line in open(path)):
            choice = planned(image, prob, seed) if image else None
            if choice:
                plan[image] = {"image": image, "scenario": scenario, "kind": choice[0], "severity": choice[1]}
    for path in items_paths:
        name = os.path.basename(path)[: -len(".jsonl")]
        n = hit = 0
        with open(os.path.join(out_dir, f"{name}@{tag}.jsonl"), "w") as f:
            for line in open(path):
                item = json.loads(line)
                choice = planned(item["image"], prob, seed)
                if choice:
                    item["corrupt"] = {"kind": choice[0], "severity": choice[1]}
                    plan[item["image"]] = {"image": item["image"], "scenario": item["scenario"], "kind": choice[0], "severity": choice[1]}
                    hit += 1
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        print(f"{name}@{tag}: {n} questions, {hit} on degraded images")
    with open(os.path.join(out_dir, f"plan_{tag}.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in plan.values())
    kinds = {}
    for r in plan.values():
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"plan_{tag}: {len(plan)} images, {kinds}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", nargs="*", default=[])
    parser.add_argument("--kind", choices=KINDS + HELDOUT)
    parser.add_argument("--severity", type=int, default=2)
    parser.add_argument("--preview")
    parser.add_argument("--plan", help="tag of an augmentation plan to write for --items (training questions with degraded images), e.g. aug_s0")
    parser.add_argument("--prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-list", nargs="*", default=[], help="with --plan: <scenario>=<file of image paths>, images to plan besides those of --items")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.plan:
        write_plan(args.items, args.prob, args.seed, args.out_dir, args.plan, args.image_list)
        return
    if args.preview:
        image = Image.open(args.preview).convert("RGB")
        for kind in KINDS + HELDOUT:
            for severity in (1, 2, 3):
                out = apply(image, kind, severity, args.preview)
                out.resize((out.width // 2, out.height // 2)).save(os.path.join(args.out_dir, f"preview_{kind}{severity}.jpg"), quality=85)
        print("previews written to", args.out_dir)
        return
    for path in args.items:
        name = os.path.basename(path)[: -len(".jsonl")]
        out = os.path.join(args.out_dir, f"{name}@{args.kind}{args.severity}.jsonl")
        with open(out, "w") as f:
            n = 0
            for line in open(path):
                item = json.loads(line)
                item["corrupt"] = {"kind": args.kind, "severity": args.severity}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        print(f"{out}: {n} questions")


if __name__ == "__main__":
    main()
