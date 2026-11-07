"""Build the PowerBrain benchmark (v1) from human detection annotations.

Two scenarios, three question types each; every answer is derived from the
human-drawn boxes by a fixed rule, never from a model.

  ed  external-damage hazards in transmission corridors (COCO json, official train/test split)
  td  tower/line component defects from UAV inspection (Pascal-VOC xml, Roboflow export;
      one variant per source image, split by source)

  cls    which class is present              (images with a single class)      2-5 options
  exist  is class c present                  (positive: present; negative: other classes present)
  count  how many instances of class c       (1 / 2 / 3 / 4 or more)

    python bench.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import xml.etree.ElementTree as ET

D = os.environ.get("PB_GRID_DATA", "data/grid")  # root of the ED and TD source data (under agreement, not released)
ED_DIR = os.environ.get("PB_ED_DIR", f"{D}/ed")  # ED: COCO files annotations/instances_<split>.json and images <split>/
TD_DIR = os.environ.get("PB_TD_DIR", f"{D}/td")  # TD: VOC files Annotations/*.xml and images JPEGImages/
ED_NAMES = {
    "DaoXianYiWu": "导线异物",
    "DiaoChe": "吊车（汽车起重机）",
    "ShiGongJiXie": "施工机械（挖掘机、推土机等）",
    "TaDiao": "塔吊",
    "YanHuo": "烟火",
}
ED_COUNTABLE = {"DiaoChe", "ShiGongJiXie", "TaDiao"}
TD_CLASSES = ["绑扎不规范", "并线线夹保护壳缺失", "耐张线夹保护壳缺失", "横杆腐蚀", "塔头损坏"]
SCENE = {"ed": "这张输电线路通道巡检图像中", "td": "这张输电线路无人机巡检图像中", "pl": "这张输电线路无人机巡检图像中"}
# InsPLAD-det (public; Vieira-e-Silva et al., IJRS 2023): the replication of protocol P30. Categories as in its COCO files
# ("sphere", absent from the paper's 17 classes and from the test towers, is dropped) -> the names used in the questions.
PL_ROOT = os.environ.get("PB_PL_ROOT", "INSPLAD_ROOT")  # the extracted InsPLAD-det folder (annotations/, train/, val/); see README
PL_NAMES = {
    "yoke": "联板", "yoke suspension": "悬垂联板", "spacer": "间隔棒", "stockbridge damper": "防振锤",
    "lightning rod shackle": "避雷线挂环", "lightning rod suspension": "避雷线悬垂线夹", "polymer insulator": "复合绝缘子",
    "glass insulator": "玻璃绝缘子", "tower id plate": "杆塔标识牌", "vari-grip": "预绞式悬垂线夹",
    "polymer insulator lower shackle": "复合绝缘子下端挂环", "polymer insulator upper shackle": "复合绝缘子上端挂环",
    "polymer insulator tower shackle": "复合绝缘子塔侧挂环", "glass insulator big shackle": "玻璃绝缘子大挂环",
    "glass insulator small shackle": "玻璃绝缘子小挂环", "glass insulator tower shackle": "玻璃绝缘子塔侧挂环",
    "spiral damper": "螺旋防振器",
}
AREA_BINS = [0.002, 0.005, 0.01, 0.05]
COUNT_OPTIONS = ["1 处", "2 处", "3 处", "4 处及以上"]


def is_val(image_path):
    """One training image in ten is held out as a validation image (by a hash of its file name).
    Everything that is tuned - thresholds, layers, the tokens-on-evidence requirement - is tuned
    on questions about these images; no model component is trained on them."""
    return int(hashlib.md5(os.path.basename(image_path).encode()).hexdigest(), 16) % 10 == 0


def in_budget(image_path, percent, seed=0):
    """Image-level label budget: keep an image if a seeded hash of its file name falls under `percent`.
    Nested for a fixed seed (the 1% set is inside the 5% set, and so on)."""
    h = int(hashlib.md5(f"{seed}:{os.path.basename(image_path)}".encode()).hexdigest(), 16) % 100000
    return h < percent * 1000


def area_bin(ratio):
    for i, edge in enumerate(AREA_BINS):
        if ratio < edge:
            return i
    return len(AREA_BINS)


ED_SOURCES = ["DiaoChe", "DXYW", "SGJX", "TaDiao", "YanHuo"]


def load_ed(split):
    """The released COCO json numbers the images of each of the five source collections from 1,
    so `image_id` alone is ambiguous (up to five images share one id). Images and annotations are
    both written source by source in the same order; within a source the image ids never decrease.
    The source of an annotation is therefore the index of the run it falls in, and
    (source, image_id) identifies the image. Checked on both splits: exactly five runs, every
    annotation finds its image, and no box leaves its image (it was 4% / 22% of images before)."""
    coco = json.load(open(f"{ED_DIR}/annotations/instances_{split}.json"))
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    prefix = lambda name: os.path.splitext(name)[0].rstrip("0123456789")
    info_of = {(prefix(i["file_name"]), i["id"]): i for i in coco["images"]}
    assert len(info_of) == len(coco["images"])
    boxes, run, previous = collections.defaultdict(list), 0, -1
    for ann in sorted(coco["annotations"], key=lambda a: a["id"]):
        if ann["image_id"] < previous:
            run += 1
        previous = ann["image_id"]
        key = (ED_SOURCES[run], ann["image_id"])
        info = info_of[key]
        x, y, w, h = ann["bbox"]
        assert x + w <= info["width"] + 2 and y + h <= info["height"] + 2, (key, ann["bbox"])
        if w > 1 and h > 1:
            boxes[key].append([x, y, w, h, cat[ann["category_id"]]])
    assert run == len(ED_SOURCES) - 1
    images = []
    for key, info in info_of.items():
        if boxes.get(key):
            images.append({
                "key": f"ed-{os.path.splitext(info['file_name'])[0]}",
                "image": f"{ED_DIR}/{split}/{info['file_name']}",
                "width": info["width"], "height": info["height"], "boxes": boxes[key],
            })
    return images


def load_td():
    """One variant per Roboflow source; 70/30 split by a hash of the source name."""
    by_source = collections.defaultdict(list)
    for path in sorted(glob.glob(f"{TD_DIR}/Annotations/*.xml")):
        stem = os.path.basename(path)[:-4]
        by_source[stem.split(".rf.")[0]].append(stem)
    splits = {"train": [], "test": []}
    for source, stems in sorted(by_source.items()):
        stem = stems[0]
        root = ET.parse(f"{TD_DIR}/Annotations/{stem}.xml").getroot()
        width, height = int(root.find("size/width").text), int(root.find("size/height").text)
        boxes = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            bb = obj.find("bndbox")
            x0, y0, x1, y1 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            if name in TD_CLASSES and x1 - x0 > 1 and y1 - y0 > 1:
                boxes.append([x0, y0, x1 - x0, y1 - y0, name])
        image = f"{TD_DIR}/JPEGImages/{stem}.jpg"
        if not boxes or not os.path.exists(image):
            continue
        split = "test" if int(hashlib.md5(source.encode()).hexdigest(), 16) % 10 < 3 else "train"
        splits[split].append({"key": f"td-{source[:48]}", "image": image, "width": width, "height": height, "boxes": boxes})
    return splits


def load_pl(split):
    """InsPLAD-det: train -> instances_train.json and train/; test -> the towers the authors held out (instances_val.json, val/)."""
    part = "train" if split == "train" else "val"
    coco = json.load(open(f"{PL_ROOT}/annotations/instances_{part}.json"))
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    name_of = {i["id"]: i["file_name"] for i in coco["images"]}
    info = {i["file_name"]: i for i in coco["images"]}
    boxes = collections.defaultdict(list)  # by file name: 46 training images are listed twice, each id holding part of the boxes
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        if cat[ann["category_id"]] in PL_NAMES and w > 1 and h > 1:
            boxes[name_of[ann["image_id"]]].append([x, y, w, h, cat[ann["category_id"]]])
    return [{"key": f"pl-{os.path.splitext(n)[0]}", "image": f"{PL_ROOT}/{part}/{n}", "width": info[n]["width"], "height": info[n]["height"],
             "boxes": boxes[n]} for n in sorted(info) if boxes.get(n) and os.path.exists(f"{PL_ROOT}/{part}/{n}")]


def tower_of(image_path):
    """InsPLAD file names start with the tower number (327-2_DJI_0002.jpg is tower 327/2; CT-1_... another structure):
    the group of P30, as in the authors' tower-grouped split."""
    return "tower-" + os.path.basename(image_path).split("_")[0]


def make_item(scenario, img, qtype, target, options, answer_text, evidence, rng, names):
    options = list(options)
    if qtype != "count":  # count options stay in natural order
        rng.shuffle(options)
    letters = "ABCDE"[: len(options)]
    shown = [names.get(o, o) for o in options]
    if qtype == "cls":
        stem = f"{SCENE[scenario]}存在哪一类{'外力破坏隐患' if scenario == 'ed' else '部件' if scenario == 'pl' else '缺陷'}？"
    elif qtype == "exist":
        stem = f"{SCENE[scenario]}是否存在{names.get(target, target)}{'' if scenario in ('ed', 'pl') else '这类缺陷'}？"
    else:
        stem = f"{SCENE[scenario]}共有几处{names.get(target, target)}？"
    question = stem + "\n" + "\n".join(f"{l}. {s}" for l, s in zip(letters, shown)) + "\n请只回答选项字母。"
    area = img["width"] * img["height"]
    ratios = sorted(b[2] * b[3] / area for b in evidence)
    return {
        "id": f"{img['key']}-{qtype}-{target or 'x'}", "scenario": scenario, "image": img["image"],
        "width": img["width"], "height": img["height"], "qtype": qtype, "target": target,
        "options": options, "answer": letters[options.index(answer_text)], "question": question,
        "evidence": [b[:4] for b in evidence], "boxes": img["boxes"],
        "area_ratio": ratios[len(ratios) // 2], "bin": area_bin(ratios[len(ratios) // 2]),
    }


def candidates(scenario, images, rng):
    """All questions the annotations support, keyed by (qtype, stratum) for balanced sampling."""
    classes = {"ed": list(ED_NAMES), "td": TD_CLASSES, "pl": list(PL_NAMES)}[scenario]
    names = {"ed": ED_NAMES, "td": {}, "pl": PL_NAMES}[scenario]
    countable = {"ed": ED_COUNTABLE, "td": set(TD_CLASSES), "pl": set(PL_NAMES)}[scenario]
    pool = collections.defaultdict(list)
    for img in images:
        present = collections.Counter(b[4] for b in img["boxes"])
        absent = [c for c in classes if c not in present]
        if len(present) == 1:
            c = next(iter(present))
            options = classes if scenario != "pl" else [c] + rng.sample([k for k in classes if k != c], 4)  # 17 classes: the class and 4 others
            pool[("cls", c)].append(make_item(scenario, img, "cls", None, options, c, img["boxes"], rng, names))
        for c in present:
            ev = [b for b in img["boxes"] if b[4] == c]
            pool[("exist", c, "是")].append(make_item(scenario, img, "exist", c, ["是", "否"], "是", ev, rng, names))
            if c in countable:
                n = min(present[c], 4)
                pool[("count", c, n)].append(make_item(scenario, img, "count", c, COUNT_OPTIONS, COUNT_OPTIONS[n - 1], ev, rng, names))
        for c in absent:  # the other objects are what has to be checked and rejected
            pool[("exist", c, "否")].append(make_item(scenario, img, "exist", c, ["是", "否"], "否", img["boxes"], rng, names))
    return pool


def sample(pool, quota, rng, per_image):
    """Round-robin over strata; at most `per_image` questions of one type per image."""
    picked, used = [], collections.Counter()
    for qtype, n in quota.items():
        keys = sorted((k for k in pool if k[0] == qtype), key=str)
        if qtype == "exist":  # keep yes/no balanced: iterate (class, yes) and (class, no) in pairs
            keys = sorted(keys, key=lambda k: (k[1], k[2]))
        for k in keys:
            rng.shuffle(pool[k])
        groups = [[k for k in keys if k[2] == a] for a in ("是", "否")] if qtype == "exist" else [keys]
        for group in groups:  # yes and no get the same quota, so the answer prior is exactly 50/50
            got, target = 0, n // len(groups)
            while got < target and any(pool[k] for k in group):
                for k in group:
                    while pool[k]:
                        item = pool[k].pop()
                        image_key = (qtype, item["image"])
                        if used[image_key] < per_image:
                            used[image_key] += 1
                            picked.append(item)
                            got += 1
                            break
                    if got >= target:
                        break
        if qtype == "exist":  # trim the longer side if one answer ran out of images
            yes = [i for i in picked if i["qtype"] == "exist" and i["options"]["AB".index(i["answer"])] == "是"]
            no = [i for i in picked if i["qtype"] == "exist" and i["options"]["AB".index(i["answer"])] == "否"]
            keep = min(len(yes), len(no))
            drop = {id(i) for i in yes[keep:] + no[keep:]}
            picked = [i for i in picked if id(i) not in drop]
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenarios", nargs="+", default=["ed", "td"], help="pl: the public replication of P30 (InsPLAD-det), built on its own")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    sources = {}
    if "ed" in args.scenarios or "td" in args.scenarios:  # the frozen benchmark: both scenarios in this order, as when it was built
        td = load_td()
        sources = {
            ("ed", "train"): load_ed("train"), ("ed", "test"): load_ed("test"),
            ("td", "train"): td["train"], ("td", "test"): td["test"],
        }
    if "pl" in args.scenarios:
        sources.update({("pl", "train"): load_pl("train"), ("pl", "test"): load_pl("test")})
    quotas = {
        ("ed", "test"): {"cls": 600, "exist": 1000, "count": 600},
        ("td", "test"): {"cls": 600, "exist": 800, "count": 600},
        ("ed", "train"): {"cls": 2000, "exist": 6000, "count": 3000},
        ("td", "train"): {"cls": 2000, "exist": 3000, "count": 2000},
        ("pl", "test"): {"cls": 600, "exist": 1000, "count": 600},
        ("pl", "train"): {"cls": 2000, "exist": 6000, "count": 3000},
    }
    for (scenario, split), images in sources.items():
        pool = candidates(scenario, images, rng)
        items = sample(pool, quotas[(scenario, split)], rng, per_image=1 if split == "test" else 2)
        rng.shuffle(items)
        path = os.path.join(args.out_dir, f"{scenario}_{split}.jsonl")
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"== {scenario} {split}: {len(images)} images -> {len(items)} questions -> {path}")
        if split == "train":  # the fit / validation views of the training questions
            for name, keep in (("fit", False), ("val", True)):
                with open(os.path.join(args.out_dir, f"{scenario}_{name}.jsonl"), "w") as f:
                    part = [i for i in items if is_val(i["image"]) == keep]
                    for item in part:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"   {name}: {len(part)} questions on {len({i['image'] for i in part})} images")
        for qtype in ("cls", "exist", "count"):
            sub = [i for i in items if i["qtype"] == qtype]
            answers = collections.Counter(i["options"]["ABCDE".index(i["answer"])] for i in sub)
            targets = collections.Counter(i["target"] for i in sub)
            bins = collections.Counter(i["bin"] for i in sub)
            print(f"   {qtype}: {len(sub)} | answers {dict(answers)} | targets {dict(targets)} | size bins {dict(sorted(bins.items()))} | images {len({i['image'] for i in sub})}")


if __name__ == "__main__":
    main()
