"""State of the detector branch on an image: is the image inside the range of inputs the branch was trained on?

The features entering the detection head (three pyramid levels) are averaged over space. Their distribution over clean
training images of the detector is summarised, per level and for all levels together, by principal components and
their variances; the squared Mahalanobis distance of a new image in that space is a label-free, degradation-agnostic
score. Only such distances are kept (their logarithms), because they mean the same thing for every detector, whereas
raw feature coordinates do not carry over from the cross-fitted detectors to the full one.

  extract   pooled head-input features of images (question files or an image list; optionally a fixed corruption or a
            per-image corruption plan written by `corrupt.py --plan`)
  stats     principal components and variances from one feature file (clean training images of that detector)
  scalars   the state scalars of the images in feature files: jsonl {image, state: [4 numbers]}
  score     exploratory table: how the total distance separates degraded from clean test images

    python branchstate.py extract --weights <best.pt> --items ed_fit.jsonl --limit 1500 --out bs/ed_full_train.npz
    python branchstate.py stats --features bs/ed_full_train.npz --out bs/ed_full_stats.npz
    python branchstate.py scalars --stats bs/ed_full_stats.npz --features bs/ed_test_clean.npz --out bs/ed_test_clean.jsonl
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

LEVELS = 3
COMPONENTS = 32


class Probe:
    """Online use: hooks on a detector that keep the pooled head-input features of its latest forward pass, and the
    state scalars computed from them with the statistics of that detector (written by `stats`)."""

    def __init__(self, yolo, stats_path):
        layers = yolo.model.model
        self.index = list(layers[-1].f)
        assert len(self.index) == LEVELS, self.index
        self.feats = {}
        self.stats = dict(np.load(stats_path))
        for i in self.index:
            layers[i].register_forward_hook(lambda module, inputs, output, i=i: self.feats.__setitem__(i, output.detach().float().mean((2, 3))[0].cpu().numpy()))

    def state(self):
        """The four state scalars of the image the detector has just seen."""
        x = np.concatenate([self.feats[i] for i in self.index]).astype(np.float64)[None]
        return [round(float(v), 4) for v in state_scalars(self.stats, x)[0]]


def cmd_extract(args):
    from PIL import Image
    from ultralytics import YOLO

    import core
    import corrupt

    model = YOLO(args.weights)
    layers = model.model.model
    index = list(layers[-1].f)  # the layers whose outputs enter the detection head
    assert len(index) == LEVELS, index
    feats = {}
    for i in index:
        layers[i].register_forward_hook(lambda module, inputs, output, i=i: feats.__setitem__(i, output.detach().float().mean((2, 3))[0].cpu().numpy()))
    if args.image_list:
        paths = [line.strip() for line in open(args.image_list) if line.strip()]
    else:
        paths = list({json.loads(line)["image"] for path in args.items for line in open(path)})
    if args.only_items:  # keep the images of these question files only
        wanted = {json.loads(line)["image"] for path in args.only_items for line in open(path)}
        paths = [p for p in paths if p in wanted]
    paths = sorted(paths, key=lambda p: hashlib.md5(p.encode()).hexdigest())
    if args.limit:
        paths = paths[: args.limit]
    paths = paths[args.shard :: args.num_shards]
    plan = {}
    if args.corrupt_plan:  # per-image corruption (P20); images outside the plan stay clean
        plan = {r["image"]: (r["kind"], r["severity"]) for r in map(json.loads, open(args.corrupt_plan))}
    if args.corrupt:
        kind, severity = args.corrupt.split(":")
        plan = {p: (kind, int(severity)) for p in paths}
    out = []
    for n, path in enumerate(paths):
        image = Image.open(core.local_path(path)).convert("RGB")
        if path in plan:
            image = corrupt.apply(image, plan[path][0], plan[path][1], path)
        model.predict(image, imgsz=args.imgsz, conf=0.25, device=args.device, verbose=False)
        out.append(np.concatenate([feats[i] for i in index]))
        if (n + 1) % 500 == 0:
            print(f"{n + 1}/{len(paths)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sizes = [int(feats[i].shape[0]) for i in index] if paths else []
    np.savez(args.out, images=np.array(paths), features=np.stack(out) if out else np.zeros((0, 0)), sizes=np.array(sizes))
    print(f"{args.out}: {len(paths)} images, levels {sizes}")


def load_features(pattern):
    """All shards of a feature file ('x.npz' also matches 'x.s0.npz', 'x.s1.npz' ...)."""
    files = sorted(glob.glob(pattern)) or sorted(glob.glob(pattern[: -len(".npz")] + ".s*.npz"))
    assert files, pattern
    parts = [np.load(f) for f in files]
    parts = [p for p in parts if len(p["images"])]
    return np.concatenate([p["images"] for p in parts]), np.concatenate([p["features"] for p in parts]).astype(np.float64), parts[0]["sizes"]


def cmd_stats(args):
    _, x, sizes = load_features(args.features)
    spans = [(int(sizes[:k].sum()), int(sizes[: k + 1].sum())) for k in range(LEVELS)] + [(0, int(sizes.sum()))]
    out = {"sizes": sizes}
    for k, (a, b) in enumerate(spans):  # three levels, then all levels together
        part = x[:, a:b]
        mu, sd = part.mean(0), part.std(0) + 1e-6
        _, sing, vt = np.linalg.svd((part - mu) / sd, full_matrices=False)
        c = min(COMPONENTS, len(sing))
        out.update({f"mu{k}": mu, f"sd{k}": sd, f"basis{k}": vt[:c].T, f"var{k}": sing[:c] ** 2 / (len(part) - 1)})
    np.savez(args.out, **out)
    print(f"{args.out}: fitted on {len(x)} images")


def state_scalars(stats, x):
    """(n, 4): log of the squared Mahalanobis distance per pyramid level and for all levels together."""
    sizes = stats["sizes"]
    spans = [(int(sizes[:k].sum()), int(sizes[: k + 1].sum())) for k in range(LEVELS)] + [(0, int(sizes.sum()))]
    cols = []
    for k, (a, b) in enumerate(spans):
        z = ((x[:, a:b] - stats[f"mu{k}"]) / stats[f"sd{k}"]) @ stats[f"basis{k}"]
        cols.append(np.log((z**2 / stats[f"var{k}"]).sum(1) + 1e-6))
    return np.stack(cols, 1)


def cmd_scalars(args):
    stats = np.load(args.stats)
    n = 0
    with open(args.out, "w") as out:
        for pattern in args.features:
            images, x, _ = load_features(pattern)
            for path, s in zip(images.tolist(), state_scalars(stats, x)):
                out.write(json.dumps({"image": path, "state": [round(float(v), 4) for v in s]}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{args.out}: {n} images")


def auc(score, y):
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    return (rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(1, pos.sum() * (~pos).sum())


def cmd_score(args):
    """Exploratory: total distance of test images under each condition against clean ones (files <scenario>_test_<cond>.npz
    and <scenario>_fit.npz in --dir, as written by run_branchstate.sh)."""
    for sc in ("ed", "td"):
        _, x, sizes = load_features(os.path.join(args.dir, f"{sc}_fit.npz"))
        mu, sd = x.mean(0), x.std(0) + 1e-6
        _, sing, vt = np.linalg.svd((x - mu) / sd, full_matrices=False)
        k = args.components
        basis, var = vt[:k].T, (sing[:k] ** 2) / (len(x) - 1)
        dist = lambda f: ((((f - mu) / sd) @ basis) ** 2 / var).sum(1)
        d_clean = dist(load_features(os.path.join(args.dir, f"{sc}_test_clean.npz"))[1])
        cut = np.quantile(dist(x), 0.99)
        print(f"## {sc}：检测分支状态得分（拟合用 {len(x)} 幅干净训练图像，阈值取其第 99 百分位）")
        print()
        print("| 条件 | 图像数 | 得分中位数 | 超过阈值的比例 | 与干净测试图像的区分度（AUC） |")
        print("|---|---:|---:|---:|---:|")
        for path in sorted(glob.glob(os.path.join(args.dir, f"{sc}_test_*.npz"))):
            cond = os.path.basename(path)[len(sc) + 6 : -4]
            d = dist(load_features(path)[1])
            sep = auc(np.concatenate([d, d_clean]), np.concatenate([np.ones(len(d)), np.zeros(len(d_clean))])) if cond != "clean" else float("nan")
            print(f"| {cond} | {len(d)} | {np.median(d):.0f} | {100 * np.mean(d > cut):.0f}% | {sep:.2f} |")
        print()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--weights", required=True)
    p.add_argument("--items", nargs="*", default=[])
    p.add_argument("--image-list")
    p.add_argument("--only-items", nargs="*", default=[])
    p.add_argument("--corrupt")
    p.add_argument("--corrupt-plan")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_extract)
    p = sub.add_parser("stats")
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_stats)
    p = sub.add_parser("scalars")
    p.add_argument("--stats", required=True)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_scalars)
    p = sub.add_parser("score")
    p.add_argument("--dir", required=True)
    p.add_argument("--components", type=int, default=64)
    p.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
