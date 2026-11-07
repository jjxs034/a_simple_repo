# Statistics of the public InsPLAD-det data for Table II, computed as bench.load_pl reads it, and a rebuild of the P30 questions
# from the official annotations to count the images that carry questions. Needs only the two annotation files and the list
# of image files in InsPLAD-det.zip (Mendeley Data, DOI 10.17632/5n3fjgvfyz.1); no image is read.
#   python3 -I experiments/pb/pub_stats.py <dir with instances_{train,val}.json> <file with the zip's member names> <work dir>
import collections, json, os, statistics, subprocess, sys

ann_dir, names_file, work = sys.argv[1:4]
members = [n.strip() for n in open(names_file) if n.strip()]
root = os.path.join(work, "data")  # a PB_PL_ROOT with the annotations and empty stand-ins for the images, for bench.load_pl
for part in ("train", "val"):
    os.makedirs(os.path.join(root, part), exist_ok=True)
os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
for part in ("train", "val"):
    src = [n for n in os.listdir(ann_dir) if n.endswith(f"instances_{part}.json")]
    assert len(src) == 1, (part, src)
    with open(os.path.join(ann_dir, src[0]), "rb") as f, open(os.path.join(root, "annotations", f"instances_{part}.json"), "wb") as g:
        g.write(f.read())
for n in members:
    parts = n.split("/")
    if len(parts) >= 2 and parts[-2] in ("train", "val") and n.lower().endswith((".jpg", ".jpeg", ".png")):
        open(os.path.join(root, parts[-2], parts[-1]), "a").close()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.environ["PB_PL_ROOT"] = root
import bench  # noqa: E402  (reads PB_PL_ROOT at import)

splits = {"train": bench.load_pl("train"), "test": bench.load_pl("test")}
towers = {k: {bench.tower_of(i["image"]) for i in v} for k, v in splits.items()}
imgs = splits["train"] + splits["test"]
mp = [i["width"] * i["height"] / 1e6 for i in imgs]
share = {k: [b[2] * b[3] / (i["width"] * i["height"]) for i in v for b in i["boxes"]] for k, v in splits.items()}
out = {
    "images": {k: len(v) for k, v in splits.items()},
    "towers": {k: len(v) for k, v in towers.items()}, "towers_shared": len(towers["train"] & towers["test"]),
    "megapixels_median": round(statistics.median(mp), 2), "megapixels_max": round(max(mp), 2),
    "boxes": sum(len(i["boxes"]) for i in imgs),
    "median_box_share_pct": {"pooled": round(100 * statistics.median(share["train"] + share["test"]), 2),
                             "train": round(100 * statistics.median(share["train"]), 2), "test": round(100 * statistics.median(share["test"]), 2)},
}
qdir = os.path.join(work, "bench")
log = subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py"), "--out-dir", qdir,
                      "--scenarios", "pl"], env={**os.environ, "PB_PL_ROOT": root}, capture_output=True, text=True, check=True).stdout
read = lambda name: [json.loads(line) for line in open(os.path.join(qdir, name))]
for name in ("pl_train.jsonl", "pl_fit.jsonl", "pl_val.jsonl", "pl_test.jsonl"):
    items = read(name)
    out[name] = {"questions": len(items), "images": len({i["image"] for i in items})}
print(log)
print("PUB_STATS " + json.dumps(out))
