"""Shared pieces: image geometry on the 32-px token grid, region construction, input policies, model wrapper.

An input policy turns one benchmark item into the chat `content` list that is sent to the MLLM.
Policy names (also used as row keys in result files):

  u<cap>      whole image downscaled to at most <cap> visual tokens (u8192 = native for this data)
  o<s>        thumbnail (256 tokens) + annotated evidence regions rendered at <s> x native scale
  ol          as o1, but the region pixels are taken from the thumbnail and upsampled (pointing, no detail)
  os<k>       as o1, region pixels taken from the image downscaled to <k> x the thumbnail size
  r1          thumbnail + regions of the same sizes as o1 placed away from every annotated box
  oa1         as o1, but the regions cover every annotated box regardless of the question
  g:<tag>     thumbnail + regions supplied by a selector (allocation head, detector, attention, ...)
  i<n>        InternVL only (PB_FAMILY=internvl): the whole image as at most <n> tiles of 448 pixels, plus a thumbnail tile when
              n > 1 (i1: one tile, 256 tokens; i12: the family's default high-resolution reading, at most 3,328 tokens)
"""
from __future__ import annotations

import math
import os
import random
import time

import torch
from PIL import Image, ImageOps

CELL = 32  # pixels per visual token side (patch 16 x merge 2)
MIN_PIXELS = 65536  # the processor upscales anything smaller, so a region costs at least 64 tokens
THUMB_TOKENS = 256
REGION_TOKEN_CAP = 576
LETTERS = "ABCDEFGH"
THUMB_TEXT = "以上是整幅图像的缩略图。"
# written answers are decoded greedily and nothing else: a repetition penalty in the generation defaults of a checkpoint would
# count against copying a class name from the evidence list, or against a long list of objects
GREEDY = {"do_sample": False, "repetition_penalty": 1.0}
Image.MAX_IMAGE_PIXELS = None
# Model family: "qwen3vl" (every run up to P31) or "internvl" (P32: InternVL3.5 in the transformers format). InternVL reads an
# image as tiles of 448 pixels and 256 tokens each, chosen by its processor, so its inputs carry a tile count instead of a
# pixel size; the thumbnail of the evidence pathway is one tile.
FAMILY = os.environ.get("PB_FAMILY", "qwen3vl")


def encode_messages(processor, messages):
    """Token ids and pixel inputs of a chat with the generation prompt appended."""
    if FAMILY != "internvl":
        return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    images = [c for m in messages for c in m["content"] if c.get("type") == "image"]
    bare = [{**m, "content": [{"type": "image"} if c.get("type") == "image" else c for c in m["content"]]} for m in messages]
    text = processor.apply_chat_template(bare, tokenize=False, add_generation_prompt=True)
    if not images:
        return processor(text=[text], return_tensors="pt")
    tiles = {c.get("tiles", 1) for c in images}
    assert len(tiles) == 1, "the images of one InternVL input are tiled alike"
    n = tiles.pop()
    return processor(text=[text], images=[c["image"] for c in images], crop_to_patches=n > 1, max_patches=n, return_tensors="pt")


def round_to_cell(value):
    return max(CELL, int(round(value / CELL)) * CELL)


def tokens_of(width, height):
    return (width // CELL) * (height // CELL)


def resize_to_token_cap(image, max_tokens):
    """Downscale (never upscale) so the image costs at most max_tokens."""
    w, h = image.size
    scale = min(1.0, math.sqrt(max_tokens * CELL * CELL / (w * h)))
    nw, nh = round_to_cell(w * scale), round_to_cell(h * scale)
    while tokens_of(nw, nh) > max_tokens:
        if nw >= nh:
            nw -= CELL
        else:
            nh -= CELL
    return image.resize((nw, nh), Image.BICUBIC)


def render_size(box, scale=1.0, cap=REGION_TOKEN_CAP):
    """Pixel size at which a region [x0, y0, x1, y1] is shown: native x scale, on the token grid, within limits."""
    w, h = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
    limit = cap * scale * scale if scale > 1 else cap  # magnification may exceed the native cap proportionally
    shrink = min(1.0, math.sqrt(limit * CELL * CELL / max(w * h, 1.0)))
    nw, nh = round_to_cell(w * shrink), round_to_cell(h * shrink)
    if nw * nh < MIN_PIXELS:
        grow = math.sqrt(MIN_PIXELS / (nw * nh))
        nw, nh = round_to_cell(nw * grow), round_to_cell(nh * grow)
    return nw, nh


def crop(image, box, size):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return image.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))).resize(size, Image.BICUBIC)


def expand(box, width, height, margin=0.5, min_side=256):
    """[x, y, w, h] -> [x0, y0, x1, y1] with context margin, clamped to the image."""
    x, y, w, h = box[:4]
    cx, cy = x + w / 2, y + h / 2
    sw, sh = max(w * (1 + 2 * margin), min_side), max(h * (1 + 2 * margin), min_side)
    return [max(0.0, cx - sw / 2), max(0.0, cy - sh / 2), min(float(width), cx + sw / 2), min(float(height), cy + sh / 2)]


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def merge_regions(regions, k=3):
    """Merge overlapping regions, then the closest pairs, until at most k remain."""
    regions = [list(r) for r in regions]
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if _overlap(regions[i], regions[j]):
                    regions[i] = _union(regions[i], regions.pop(j))
                    merged = True
                    break
            if merged:
                break
    while len(regions) > k:
        best = None
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                u = _union(regions[i], regions[j])
                cost = (u[2] - u[0]) * (u[3] - u[1])
                if best is None or cost < best[0]:
                    best = (cost, i, j)
        _, i, j = best
        regions[i] = _union(regions[i], regions.pop(j))
    return sorted(regions, key=lambda r: (r[1], r[0]))


def evidence_regions(item, k=3, margin=0.5):
    return merge_regions([expand(b, item["width"], item["height"], margin) for b in item["evidence"]], k)


def region_text(box, width, height):
    pct = lambda v, total: int(round(100 * v / total))
    return (
        f"下面是原图中一个区域的高清放大（位置：左 {pct(box[0], width)}%，上 {pct(box[1], height)}%，"
        f"右 {pct(box[2], width)}%，下 {pct(box[3], height)}%）："
    )


def local_path(path):
    """The image file on this machine. Question files, candidates and results keep the paths of the machine they were
    built on as keys; PB_PATH_MAP="old prefix=new prefix;..." says where those files are here."""
    for pair in os.environ.get("PB_PATH_MAP", "").split(";"):
        old, sep, new = pair.partition("=")
        if sep and path.startswith(old):
            return new + path[len(old):]
    return path


def open_image(item):
    image = Image.open(local_path(item["image"]))
    if item.get("exif_transpose"):
        image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    if item.get("corrupt"):  # robustness test: the same corrupted image for every system (corrupt.py)
        import corrupt

        image = corrupt.apply(image, item["corrupt"]["kind"], item["corrupt"]["severity"], item["image"])
    return image


def region_content(image, regions, scale=1.0, source=None, notes=None, thumbnail=True):
    """Thumbnail followed by the given regions. `source`: None = native pixels, or an image
    (a downscaled copy of the whole picture) from which the region pixels are taken instead.
    `thumbnail=False` leaves the whole-image thumbnail out (input ablation: the regions alone)."""
    width, height = image.size
    content = [{"type": "image", "image": resize_to_token_cap(image, THUMB_TOKENS)}, {"type": "text", "text": THUMB_TEXT}] if thumbnail else []
    scales = scale if isinstance(scale, (list, tuple)) else [scale] * len(regions)
    notes = notes or [""] * len(regions)
    for box, region_scale, note in zip(regions, scales, notes):
        size = render_size(box, region_scale)
        if source is None:
            patch = crop(image, box, size)
        else:
            kx, ky = source.size[0] / width, source.size[1] / height
            patch = crop(source, [box[0] * kx, box[1] * ky, box[2] * kx, box[3] * ky], size)
        content += [{"type": "text", "text": region_text(box, width, height) + note}, {"type": "image", "image": patch}]
    return content


def random_regions(item, regions, seed):
    """Regions of the same sizes placed away from every annotated box; None if the image is too crowded."""
    rng = random.Random(seed)
    width, height = item["width"], item["height"]
    placed = []
    for box in regions:
        rw, rh = box[2] - box[0], box[3] - box[1]
        for _ in range(200):
            x0, y0 = rng.uniform(0, max(0.0, width - rw)), rng.uniform(0, max(0.0, height - rh))
            cand = [x0, y0, x0 + rw, y0 + rh]
            if not any(_overlap(cand, [b[0], b[1], b[0] + b[2], b[1] + b[3]]) for b in item["boxes"]):
                placed.append(cand)
                break
        else:
            return None
    return placed


def outlined(image, box, line=None):
    """Copy of the image with the box [x0, y0, x1, y1] outlined in red, so that "this object" is unambiguous."""
    from PIL import ImageDraw

    out = image.copy()
    line = line or max(2, round(max(out.size) / 300))
    ImageDraw.Draw(out).rectangle([box[0] - line, box[1] - line, box[2] + line, box[3] + line], outline=(255, 0, 0), width=line)
    return out


def marked(image, boxes):
    """Copy of the image with every box outlined in red and numbered (1, 2, ...) at its top-left corner."""
    from PIL import ImageDraw, ImageFont

    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=max(12, round(max(out.size) / 48)))
    except TypeError:  # older Pillow: fixed-size bitmap font
        font = ImageFont.load_default()
    for k, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=(255, 0, 0), width=2)
        label = str(k + 1)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tx, ty = max(0, x0 - 2), max(0, y0 - 2 - (bottom - top) - 4)
        draw.rectangle([tx, ty, tx + (right - left) + 4, ty + (bottom - top) + 4], fill=(255, 0, 0))
        draw.text((tx + 2, ty + 1), label, fill=(255, 255, 255), font=font)
    return out


RELATION_TOKEN_CAP = 1024
RELATION_TEXT = "下面是该目标上方直到图像顶部的区域（红框为该目标），用于观察导线与目标的位置关系："
WINDOW_TEXT = "下面是该目标及其周围的区域（红框为该目标）："


def judgment_content(item, policy, image):
    """Inputs for a judgment question about one target object (item["evidence"][0], [x, y, w, h]).
      jb<cap>  whole image at <cap> tokens with the target outlined (knows which object, sees no detail)
      jt       thumbnail with the target outlined + the target region at native scale (detail, no relation)
      jw       thumbnail with the target outlined + a window six times the object's longer side
      jr       thumbnail with the target outlined + the target region + the strip from the top of the image down to
               the object, where the conductors that may pass over it are (detail and relation)
    Regions other than the target itself are shown at up to RELATION_TOKEN_CAP tokens."""
    width, height = image.size
    x, y, w, h = item["evidence"][0][:4]
    target = [x, y, x + w, y + h]
    if policy.startswith("jb"):
        whole = resize_to_token_cap(image, int(policy[2:]))
        k = whole.size[0] / width
        return [{"type": "image", "image": outlined(whole, [v * k for v in target])}]
    thumb = resize_to_token_cap(image, THUMB_TOKENS)
    k = thumb.size[0] / width
    content = [{"type": "image", "image": outlined(thumb, [v * k for v in target], 2)}, {"type": "text", "text": THUMB_TEXT + "红框为所问的目标。"}]
    marked = outlined(image, target)
    if policy in ("jt", "jr"):
        region = expand([x, y, w, h], width, height)
        content += [{"type": "text", "text": region_text(region, width, height)}, {"type": "image", "image": crop(image, region, render_size(region))}]
    if policy == "jw":
        side = 6 * max(w, h)
        cx, cy = x + w / 2, y + h / 2
        region = [max(0.0, cx - side / 2), max(0.0, cy - side / 2), min(float(width), cx + side / 2), min(float(height), cy + side / 2)]
        content += [{"type": "text", "text": WINDOW_TEXT}, {"type": "image", "image": crop(marked, region, render_size(region, cap=RELATION_TOKEN_CAP))}]
    if policy == "jr":
        half = max(1.5 * w, 0.2 * width)
        cx = x + w / 2
        region = [max(0.0, cx - half), 0.0, min(float(width), cx + half), min(float(height), max(y + 1.25 * h, 256.0))]
        content += [{"type": "text", "text": RELATION_TEXT}, {"type": "image", "image": crop(marked, region, render_size(region, cap=RELATION_TOKEN_CAP))}]
    if policy not in ("jt", "jw", "jr"):
        raise ValueError(policy)
    return content


EVIDENCE_CROP_CAP = 256


def evidence_content(image, given):
    """Single-pass evidence packing: what a full-resolution detector branch found, handed to the LLM as compact evidence.
    `given`: {"dets": [{"box": [x0, y0, x1, y1], "conf": c, "name": display name}, ...] sorted by confidence,
              "mode": "full" | "text" | "crops", "band": [lo, hi], "max_crops": m, "thumb_tokens": t,
              "mark": draw the candidates on the thumbnail, "positions": give positions in the text list}
      thumbnail of the whole image (global context, also the fallback when the branch finds nothing)
      + a text list of the candidates: class, confidence, position          (cheap: about 15 tokens each)
      + a magnified crop, with the box outlined, only for candidates whose confidence falls in the uncertain band
    The number of visual tokens does not grow with the resolution of the source image."""
    width, height = image.size
    dets, mode = given["dets"], given.get("mode", "full")
    lo, hi = given.get("band", [0.05, 0.6])
    pct = lambda v, total: int(round(100 * v / total))
    # "uncovered": display names of the classes this question depends on that the branch cannot output (known from the
    # question and the branch's vocabulary alone, never from the answer). The LLM then gets a larger view of the whole
    # image and is told which classes it has to judge by looking (on-demand vision); otherwise the small thumbnail only.
    uncovered = given.get("uncovered") or []
    thumb_tokens = given.get("uncovered_tokens", 1280) if uncovered else given.get("thumb_tokens", THUMB_TOKENS)
    # thumb_tokens 0: the LLM gets no image at all (control: is it reading the picture or only the list?)
    content = []
    if thumb_tokens and FAMILY == "internvl":
        assert not given.get("mark") and not uncovered, "InternVL runs (P32) use the plain thumbnail"
        content = [{"type": "image", "image": image, "tiles": 1}, {"type": "text", "text": THUMB_TEXT}]
    elif thumb_tokens:
        thumb = resize_to_token_cap(image, thumb_tokens)
        if given.get("mark") and dets:  # the candidates drawn on the thumbnail with their list numbers: positions shown, not only told
            thumb = marked(thumb, [[v * thumb.size[0] / width for v in d["box"]] for d in dets])
        content = [{"type": "image", "image": thumb}, {"type": "text", "text": ("以上是整幅图像。" if uncovered else THUMB_TEXT) + ("图上的红框和编号是检测分支给出的候选。" if given.get("mark") and dets else "")}]
    if given.get("vocab") is not None:
        content.append({"type": "text", "text": "检测分支能识别的类别：" + "、".join(v[1] for v in given["vocab"]) + "。"
                        + (f"它不认识{'、'.join('「' + n + '」' for n in uncovered)}，对这一类不会有任何输出，这一类是否存在只能依据图像判断。" if uncovered else "")})
    # Wording of the list. 1: the runs up to 2026-10-09 (the lead-in says "at the resolution of the original image", which
    # the branch does not use, and the question follows the last line without a line break). 2: the final round - the
    # lead-in no longer names a resolution and the list ends with a line break. 3: compact lines for the edge
    # ("name confidence left,top,right,bottom" in percent), about half the tokens. 4: tokenised list (look.py): per candidate a
    # learned class token, the confidence as two digits and a learned position token.
    fmt = given.get("fmt", 1)
    at, end = ("在原图分辨率下", "") if fmt == 1 else ("", "\n")
    if not dets:
        return content + [{"type": "text", "text": "检测候选：无\n" if fmt in (3, 4) else f"检测分支{at}没有给出任何候选目标。{end}"}]
    if fmt == 4:
        return content + [{"type": "text", "text": "检测候选：" + "".join(f"{EVIDENCE_SLOT}{min(99, int(round(100 * d['conf']))):02d}<|quad_start|>" for d in dets) + "\n"}]
    if mode in ("full", "text"):
        where = lambda d: f"，位置（左 {pct(d['box'][0], width)}%，上 {pct(d['box'][1], height)}%，右 {pct(d['box'][2], width)}%，下 {pct(d['box'][3], height)}%）" if given.get("positions", True) else ""
        if fmt == 3:
            lines = [f"{d['name'].split('（')[0]} {d['conf']:.2f} {pct(d['box'][0], width)},{pct(d['box'][1], height)},{pct(d['box'][2], width)},{pct(d['box'][3], height)}" for d in dets]
            head = "检测候选（类别 置信度 左,上,右,下%；可能含误检和重复）："
        elif given.get("conf_text", True):
            lines = [f"{k + 1}. {d['name']}，置信度 {d['conf']:.2f}{where(d)}" for k, d in enumerate(dets)]
            head = f"检测分支{at}给出的候选目标（按置信度从高到低，可能含误检和重复）："
        else:  # the interface of earlier work: detections above a fixed threshold, listed without their confidence
            lines = [f"{k + 1}. {d['name']}{where(d)}" for k, d in enumerate(dets)]
            head = f"检测分支{at}检出的目标："
        content.append({"type": "text", "text": head + "\n" + "\n".join(lines) + end})
    if mode in ("full", "crops"):
        shown = 0
        for k, d in enumerate(dets):
            if shown >= given.get("max_crops", 4) or not (lo <= d["conf"] < hi or mode == "crops"):
                continue
            x0, y0, x1, y1 = d["box"]
            region = expand([x0, y0, x1 - x0, y1 - y0], width, height)
            tokens = max((x1 - x0) * (y1 - y0) / (CELL * CELL), 0.05)
            scale = min(2.0, max(1.0, math.sqrt(16.0 / tokens)))  # a tiny box is magnified until it spans about 16 tokens
            patch = crop(outlined(image, d["box"]), region, render_size(region, scale, cap=EVIDENCE_CROP_CAP))
            content += [{"type": "text", "text": f"候选 {k + 1} 的高清放大（红框为检测框）："}, {"type": "image", "image": patch}]
            shown += 1
    return content


EVIDENCE_SLOT = "<|box_start|>"  # a special token of the tokenizer; each occurrence is replaced by one evidence token (evtok.py)


def evidence_token_content(given):
    """The evidence of a covered question as learned tokens: a short lead-in and one placeholder per candidate."""
    n = len(given["dets"])
    if not n:
        return [{"type": "text", "text": "检测分支没有给出任何候选目标。"}]
    return [{"type": "text", "text": "检测分支的候选目标（置信度从高到低，可能含误检和重复）：" + EVIDENCE_SLOT * n + "。"}]


COT_INSTRUCTION = "请先逐行列出图中全部相关目标（类别，位置），没有则写「无」；再另起一行写「答案：」和选项字母。"
COT_MAX_OBJECTS = 8


def cot_question(item):
    """The question of a list-then-answer sample: the same options, a different last line."""
    assert item["question"].endswith("请只回答选项字母。")
    return item["question"][: -len("请只回答选项字母。")] + COT_INSTRUCTION


def listing_text(item):
    """The annotated objects of the image as text, top to bottom then left to right, in the format of the evidence list
    (used as the target of detection-style supervision for a whole-image model)."""
    import bench  # display names of the classes; bench has no heavy imports

    names = {**bench.ED_NAMES, **getattr(bench, "PL_NAMES", {})}  # PL_NAMES: InsPLAD (P30), only in the copy that ran it
    width, height = item["width"], item["height"]
    pct = lambda v, total: int(round(100 * v / total))
    boxes = sorted(item["boxes"], key=lambda b: (b[1], b[0]))[:COT_MAX_OBJECTS]
    if not boxes:
        return "无"
    return "\n".join(f"{k + 1}. {names.get(b[4], b[4])}，位置（左 {pct(b[0], width)}%，上 {pct(b[1], height)}%，右 {pct(b[0] + b[2], width)}%，下 {pct(b[1] + b[3], height)}%）"
                     for k, b in enumerate(boxes))


def asked_classes(item):
    """The classes a question depends on, as far as the question alone tells (never read off the answer):
    the target of an existence or counting question, every option of a which-class question."""
    if item["qtype"] in ("exist", "count"):
        return [item["target"]]
    if item["qtype"] == "cls":
        return list(item["options"])
    return []


def mask_evidence(given, item, rng):
    """Training-time evidence masking: pretend the branch has never been taught one of the classes this question depends
    on. Its candidates of that class are removed, the class leaves its vocabulary and is named as uncovered, so the
    model has to judge it by looking. For a which-class question the masked class is drawn at random among the options
    (not the answer), so that "uncovered" never tells the answer."""
    if given.get("vocab") is None:
        return given
    known = [v for v in given["vocab"] if v[0] in asked_classes(item)]
    if not known:
        return given
    key, name = rng.choice(known)
    return {**given, "dets": [d for d in given["dets"] if d.get("cls") != key], "vocab": [v for v in given["vocab"] if v[0] != key],
            "uncovered": list(given.get("uncovered") or []) + [name]}


def build_content(item, policy, image=None, given=None):
    """Returns the content list (without the question) for a policy, or None if it does not apply.
    `given`: for policies named g:<tag>, the record {regions, scale} produced by a selector."""
    image = image or open_image(item)
    if policy.startswith("j"):
        return judgment_content(item, policy, image)
    if policy.startswith("g:"):
        if given and "dets" in given:
            return evidence_content(image, given)
        if not given or not given.get("regions"):
            return [{"type": "image", "image": resize_to_token_cap(image, THUMB_TOKENS)}]
        return region_content(image, given["regions"], scale=given.get("scale", 1.0), notes=given.get("notes"), thumbnail=given.get("thumbnail", True))
    if policy.startswith("u"):
        return [{"type": "image", "image": resize_to_token_cap(image, int(policy[1:]))}]
    if policy.startswith("i"):
        assert FAMILY == "internvl", policy
        return [{"type": "image", "image": image, "tiles": int(policy[1:])}]
    if policy == "oa1":  # question-agnostic upper bound: regions over every annotated box, whatever the question asks
        boxes = [b[:4] for b in item["boxes"]]
        return region_content(image, merge_regions([expand(b, item["width"], item["height"]) for b in boxes], 3))
    regions = evidence_regions(item)
    if policy == "ol":
        return region_content(image, regions, source=resize_to_token_cap(image, THUMB_TOKENS))
    if policy.startswith("os"):
        thumb = resize_to_token_cap(image, THUMB_TOKENS)
        k = int(policy[2:])
        if thumb.size[0] * k >= image.size[0]:
            return None
        return region_content(image, regions, source=image.resize((thumb.size[0] * k, thumb.size[1] * k), Image.BICUBIC))
    if policy.startswith("o"):
        return region_content(image, regions, scale=float(policy[1:]))
    if policy == "r1":
        placed = random_regions(item, regions, item["id"])
        return None if placed is None else region_content(image, placed)
    raise ValueError(policy)


class MLLM:
    def __init__(self, model_path, device="cuda:0", lora=None):
        from transformers import AutoProcessor, InternVLForConditionalGeneration, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        family = InternVLForConditionalGeneration if FAMILY == "internvl" else Qwen3VLForConditionalGeneration
        self.model = family.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": device})
        if lora:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora).merge_and_unload()
        self.model.eval()
        self.device = device
        tok = self.processor.tokenizer
        self.letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
        assert all(len(tok.encode(l, add_special_tokens=False)) == 1 for l in LETTERS)
        self.image_token_id = self.model.config.image_token_id

    def encode(self, content):
        return encode_messages(self.processor, [{"role": "user", "content": content}])

    @torch.inference_mode()
    def answer(self, content, n_options):
        """Log-probabilities over the option letters for the first generated token."""
        inputs = self.encode(content).to(self.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = self.model(**inputs).logits[0, -1].float()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        logp = torch.log_softmax(logits[self.letter_ids[:n_options]], dim=-1).tolist()
        return {
            "logp": logp,
            "visual_tokens": int((inputs["input_ids"] == self.image_token_id).sum()),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "seconds": elapsed,
        }

    def load_look_modules(self, lora_dir):
        """The learned input tokens trained with a look-on-doubt adapter (look.py): the branch-state projector and, if the
        list is tokenised, the evidence tokenizer."""
        import evtok
        import look

        table = self.model.get_input_embeddings().weight
        ids = self.processor.tokenizer.convert_tokens_to_ids
        self.slot_ids = {"state": ids(look.STATE_SLOT), "class": ids(look.CLASS_SLOT), "box": ids(look.BOX_SLOT)}
        self.state_projector = self.evidence_tokenizer = None
        path = os.path.join(lora_dir, "state_projector.pt")
        if os.path.exists(path):
            self.state_projector = look.StateProjector(table.shape[1]).to(self.device)
            self.state_projector.load_state_dict(torch.load(path, map_location="cpu"))
            self.state_projector.eval()
        path = os.path.join(lora_dir, "evidence_tokenizer.pt")
        if os.path.exists(path):
            self.evidence_tokenizer = look.EvidenceTokenizer(table.shape[1], len(evtok.CLASSES)).to(self.device)
            self.evidence_tokenizer.load_state_dict(torch.load(path, map_location="cpu"))
            self.evidence_tokenizer.eval()
        self.injector = look.SlotInjector(self.model.get_input_embeddings())

    def look_values(self, item, given):
        """Vectors for the placeholder tokens of one question (branch state, class and position tokens)."""
        import look

        values = {}
        if getattr(self, "state_projector", None) is not None and given.get("state") is not None:
            values[self.slot_ids["state"]] = self.state_projector(torch.tensor(given["state"], dtype=torch.float32, device=self.device))
        if getattr(self, "evidence_tokenizer", None) is not None and given.get("fmt") == 4 and given["dets"]:
            cls, box = look.slot_inputs(given, item["width"], item["height"])
            values[self.slot_ids["class"]], values[self.slot_ids["box"]] = self.evidence_tokenizer(cls.to(self.device), box.to(self.device))
        return values

    @torch.inference_mode()
    def answer_look(self, item, image, given, steps=(1, 2)):
        """Both steps of look-on-doubt for one question (look.py): step 1 from the evidence alone (option letters and X),
        step 2 with the thumbnail. Step 2 is always computed here so that every gate threshold can be evaluated offline
        (steps=(1,) runs the evidence-only step alone and needs no image)."""
        import look

        n = len(item["options"])
        look_id = self.processor.tokenizer.encode(look.LOOK, add_special_tokens=False)[0]
        injector = getattr(self, "injector", None)
        out = {}
        for step in steps:
            inputs = self.processor.apply_chat_template(look.messages(item, given, step, image), tokenize=True, add_generation_prompt=True,
                                                         return_dict=True, return_tensors="pt").to(self.device)
            if injector is not None:
                injector.values = self.look_values(item, given)
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = self.model(**inputs).logits[0, -1].float()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if injector is not None:
                injector.values = None
            ids = self.letter_ids[:n] + ([look_id] if step == 1 else [])
            logp = torch.log_softmax(logits[ids], dim=-1).tolist()
            out[step] = {"logp": logp[:n], "p_look": float(torch.exp(torch.tensor(logp[-1]))) if step == 1 else None,
                         "visual_tokens": int((inputs["input_ids"] == self.image_token_id).sum()), "input_tokens": int(inputs["input_ids"].shape[1]), "seconds": elapsed}
        return out

    @torch.inference_mode()
    def answer_look_cached(self, item, image, given):
        """Look-on-doubt as it is deployed (multiple-choice questions): step 1 on the evidence alone; step 2 as a
        continuation of the same sequence that reuses the key-value cache of step 1, so that only the tokens after the
        step-1 prompt are processed. Both steps are computed and timed whatever step 1 answers; `looks` is the model's
        own decision. Times: prep = building the input (step 2: resizing and preprocessing the thumbnail), run = forward.
        Used for the cost measurements, and to check that the continuation answers like the full forward of answer_look."""
        import look

        n = len(item["options"])
        look_id = self.processor.tokenizer.encode(look.LOOK, add_special_tokens=False)[0]
        sync, now = torch.cuda.synchronize, time.perf_counter
        template = lambda step: self.processor.apply_chat_template(look.messages(item, given, step, image), tokenize=True, add_generation_prompt=True,
                                                                    return_dict=True, return_tensors="pt").to(self.device)
        injector = getattr(self, "injector", None)
        sync()
        t0 = now()
        one = template(1)
        sync()
        t1 = now()
        if injector is not None:
            injector.values = self.look_values(item, given)
        out1 = self.model(input_ids=one["input_ids"], attention_mask=one["attention_mask"], use_cache=True, logits_to_keep=1)
        logp1 = torch.log_softmax(out1.logits[0, -1].float()[self.letter_ids[:n] + [look_id]], dim=-1).tolist()
        sync()
        t2 = now()
        if injector is not None:
            injector.values = None
        two = template(2)
        n1, n2 = one["input_ids"].shape[1], two["input_ids"].shape[1]
        assert torch.equal(two["input_ids"][:, :n1], one["input_ids"]), "the step-1 prompt is not a prefix of the step-2 dialogue"
        sync()
        t3 = now()
        # positions of the whole dialogue (the image has two-dimensional positions), then only its new part runs on top of the cache
        position_ids, _ = self.model.model.get_rope_index(two["input_ids"], two["image_grid_thw"], None, attention_mask=two["attention_mask"])
        out2 = self.model(input_ids=two["input_ids"][:, n1:], attention_mask=two["attention_mask"], position_ids=position_ids[:, :, n1:],
                          past_key_values=out1.past_key_values, cache_position=torch.arange(n1, n2, device=self.device),
                          pixel_values=two["pixel_values"], image_grid_thw=two["image_grid_thw"], use_cache=True, logits_to_keep=1)
        logp2 = torch.log_softmax(out2.logits[0, -1].float()[self.letter_ids[:n]], dim=-1).tolist()
        sync()
        t4 = now()
        return {"logp1": logp1[:n], "p_look": math.exp(logp1[-1]), "looks": logp1[-1] > max(logp1[:n]), "logp": logp2,
                "prep1": t1 - t0, "run1": t2 - t1, "prep2": t3 - t2, "run2": t4 - t3,
                "input_tokens1": n1, "input_tokens": n2, "visual_tokens": int((two["input_ids"] == self.image_token_id).sum())}

    @torch.inference_mode()
    def answer_look_text(self, item, image, given, max_new_tokens=48):
        """Look-on-doubt for an open-ended question: written answers by greedy generation. As in answer_look both steps
        are always computed; step 1 is a request for the image when its text starts with X (look.asks_to_look).
        p_look is the probability of X as the first generated token."""
        import look

        look_id = self.processor.tokenizer.encode(look.LOOK, add_special_tokens=False)[0]
        injector = getattr(self, "injector", None)
        out = {}
        for step in (1, 2):
            inputs = self.processor.apply_chat_template(look.messages(item, given, step, image), tokenize=True, add_generation_prompt=True,
                                                         return_dict=True, return_tensors="pt").to(self.device)
            if injector is not None:
                injector.values = self.look_values(item, given)
            torch.cuda.synchronize()
            start = time.perf_counter()
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **GREEDY, output_logits=True, return_dict_in_generate=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if injector is not None:
                injector.values = None
            new = gen.sequences[0, inputs["input_ids"].shape[1]:]
            first = torch.softmax(gen.logits[0][0].float(), dim=-1)
            out[step] = {"text": self.processor.tokenizer.decode(new, skip_special_tokens=True), "p_look": float(first[look_id]) if step == 1 else None,
                         "generated_tokens": int(new.shape[0]), "visual_tokens": int((inputs["input_ids"] == self.image_token_id).sum()),
                         "input_tokens": int(inputs["input_ids"].shape[1]), "seconds": elapsed}
        return out

    @torch.inference_mode()
    def answer_text(self, content, max_new_tokens=48):
        """Greedy generation of a short written answer (open-ended questions, openform.py)."""
        inputs = self.encode(content).to(self.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **GREEDY)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new = out[0, inputs["input_ids"].shape[1]:]
        return {
            "text": self.processor.tokenizer.decode(new, skip_special_tokens=True), "generated_tokens": int(new.shape[0]),
            "visual_tokens": int((inputs["input_ids"] == self.image_token_id).sum()),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "seconds": elapsed,
        }

    def load_projector(self, path):
        """The evidence-token projector trained with the adapter (evtok.py)."""
        import evtok

        state = torch.load(path, map_location="cpu")
        table = self.model.get_input_embeddings().weight
        self.projector = evtok.EvidenceProjector(table.shape[1], hidden=state["meta"]["hidden"], bands=state["meta"]["bands"]).to(self.device)
        self.projector.load_state_dict(state["state"])
        self.projector.eval()
        self.slot_id = self.processor.tokenizer.convert_tokens_to_ids(EVIDENCE_SLOT)

    @torch.inference_mode()
    def answer_tokens(self, content, cls, feat, n_options):
        """Text-only input whose placeholder positions carry evidence tokens; no vision encoder is run."""
        inputs = self.encode(content).to(self.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        slots = inputs["input_ids"][0] == self.slot_id
        assert int(slots.sum()) == cls.shape[0], (int(slots.sum()), cls.shape)
        if cls.shape[0]:
            embeds[0, slots] = self.projector(cls.to(self.device), feat.to(self.device)).to(embeds.dtype)
        hidden = self.model.model.language_model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"]).last_hidden_state
        logits = self.model.lm_head(hidden[:, -1])[0].float()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {"logp": torch.log_softmax(logits[self.letter_ids[:n_options]], dim=-1).tolist(), "visual_tokens": 0,
                "input_tokens": int(inputs["input_ids"].shape[1]), "seconds": elapsed}

    @torch.inference_mode()
    def answer_cot(self, content, n_options, max_new_tokens=280):
        """List-then-answer: greedy generation of the object list followed by "答案：<letter>". The letter is parsed
        from the text; "parsed" is False when the output has no valid answer line (then it counts as wrong)."""
        inputs = self.encode(content).to(self.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **GREEDY)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.tokenizer.decode(new, skip_special_tokens=True)
        letter = text.rsplit("答案：", 1)[1].strip()[:1] if "答案：" in text else ""
        parsed = letter in LETTERS[:n_options] and letter != ""
        return {
            "logp": [0.0 if parsed and LETTERS[i] == letter else -99.0 for i in range(n_options)],
            "parsed": parsed, "text": text, "generated_tokens": int(new.shape[0]),
            "visual_tokens": int((inputs["input_ids"] == self.image_token_id).sum()),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "seconds": elapsed,
        }
