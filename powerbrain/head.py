"""Allocation head: from the thumbnail's vision-encoder features and the question, predict
which parts of the image the question depends on (and, in v2, how finely they must be shown).

  feat     thumbnail features (vision encoder + merger output, before the language model) and
           question embeddings (mean of the input-embedding vectors of the question stem)
  train    fit the head on the training questions
  predict  write regions for evaluation: one JSON line per item {id, regions: [[x0,y0,x1,y1],...], scores, scale}

The head never sees the high-resolution image. Its cost at inference is one pass of the vision
encoder over ~256 tokens (needed anyway for the thumbnail) plus ~3M parameters.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random

import torch

import core

DIM = 256


_FOLD_GROUPS = None


def fold_of(image_path, n):
    """Image-level fold for cross-fitting (independent of the validation hash digit). With PB_FOLD_GROUPS set to a json
    written by foldgroups.py, near-duplicate images share a fold: the fold of the first member of their group."""
    import hashlib

    global _FOLD_GROUPS
    if _FOLD_GROUPS is None:
        path = os.environ.get("PB_FOLD_GROUPS")
        _FOLD_GROUPS = json.load(open(path)) if path else {}
    name = _FOLD_GROUPS.get(image_path, os.path.basename(image_path))
    return (int(hashlib.md5(name.encode()).hexdigest(), 16) // 10) % n


def stem_of(item):
    return item["question"].split("\n")[0]


def token_targets(item, gh, gw):
    """Per thumbnail token: (a) fraction of its footprint covered by an evidence box;
    (b) log2 of the number of native-resolution tokens the covering box spans, i.e. how
    large the evidence is, which the allocation step needs to decide the render scale."""
    width, height = item["width"], item["height"]
    cw, ch = width / gw, height / gh
    target, size = torch.zeros(gh, gw), torch.zeros(gh, gw)
    for x, y, w, h in item["evidence"]:
        c0, c1 = max(0, int(x // cw)), min(gw - 1, int((x + w) // cw))
        r0, r1 = max(0, int(y // ch)), min(gh - 1, int((y + h) // ch))
        log_tokens = math.log2(max(w * h, 1.0) / (core.CELL * core.CELL))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                ox = max(0.0, min(x + w, (c + 1) * cw) - max(x, c * cw))
                oy = max(0.0, min(y + h, (r + 1) * ch) - max(y, r * ch))
                # a box much smaller than a token still marks that token fully
                cover = min(1.0, (ox * oy) / min(cw * ch, max(w * h, 1.0)))
                if cover > target[r, c]:
                    target[r, c], size[r, c] = cover, log_tokens
    return target, size


class Head(torch.nn.Module):
    def __init__(self, in_dim, layers=2):
        super().__init__()
        self.vis = torch.nn.Linear(in_dim, DIM)
        self.txt = torch.nn.Linear(in_dim, DIM)
        self.pos = torch.nn.Linear(4, DIM)
        block = torch.nn.TransformerEncoderLayer(DIM, 4, 4 * DIM, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(block, layers)
        self.norm = torch.nn.LayerNorm(DIM)
        self.relevance = torch.nn.Linear(DIM, 1)
        self.size = torch.nn.Linear(DIM, 1)

    def forward(self, feats, pos, question, mask):
        """feats (B, N, in_dim), pos (B, N, 4), question (B, in_dim), mask (B, N) True = real token."""
        x = self.vis(feats) + self.pos(pos)
        q = self.txt(question).unsqueeze(1)
        h = self.encoder(torch.cat([q, x], dim=1), src_key_padding_mask=~torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1))
        h = self.norm(h)
        return self.relevance(h[:, 1:]).squeeze(-1), self.size(h[:, 1:]).squeeze(-1)


def positions(gh, gw):
    r = (torch.arange(gh).float() + 0.5) / gh
    c = (torch.arange(gw).float() + 0.5) / gw
    rr, cc = torch.meshgrid(r, c, indexing="ij")
    return torch.stack([rr, cc, torch.full_like(rr, gh / 16.0), torch.full_like(rr, gw / 16.0)], dim=-1).view(gh * gw, 4)


# --------------------------------------------------------------------------- feat
@torch.inference_mode()
def cmd_feat(args):
    items = [json.loads(line) for path in args.items for line in open(path)]
    images = sorted({(it["image"], bool(it.get("exif_transpose"))) for it in items})[args.shard :: args.num_shards]
    stems = sorted({stem_of(it) for it in items})
    model = core.MLLM(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.shard == 0:
        table = model.model.get_input_embeddings()
        tok = model.processor.tokenizer
        q = {s: table(torch.tensor(tok.encode(s, add_special_tokens=False), device=model.device)).float().mean(0).half().cpu() for s in stems}
        torch.save(q, os.path.join(args.out_dir, "questions.pt"))
    out = {}
    for n, (path, exif) in enumerate(images):
        image = core.open_image({"image": path, "exif_transpose": exif})
        # the head may look at a larger overview than the thumbnail the language model receives:
        # this costs one pass of the vision encoder, not language-model tokens
        thumb = core.resize_to_token_cap(image, args.overview_tokens)
        inputs = model.encode([{"type": "image", "image": thumb}, {"type": "text", "text": "."}]).to(model.device)
        hidden = model.model(**inputs, output_hidden_states=True).hidden_states[0][0]
        feats = hidden[inputs["input_ids"][0] == model.image_token_id]
        gh, gw = thumb.size[1] // core.CELL, thumb.size[0] // core.CELL
        assert feats.shape[0] == gh * gw
        out[path] = (gh, gw, feats.half().cpu())
        if (n + 1) % 200 == 0:
            print(f"[shard {args.shard}] {n + 1}/{len(images)}", flush=True)
    torch.save(out, os.path.join(args.out_dir, f"images_{args.shard}.pt"))
    print(f"[shard {args.shard}] FEAT_DONE", flush=True)


def load_feats(feat_dir):
    images = {}
    for path in sorted(glob.glob(os.path.join(feat_dir, "images_*.pt"))):
        images.update(torch.load(path))
    return images, torch.load(os.path.join(feat_dir, "questions.pt"))


def collate(batch_items, images, questions, device, with_targets=True):
    n = max(images[it["image"]][0] * images[it["image"]][1] for it in batch_items)
    d = next(iter(questions.values())).shape[0]
    feats = torch.zeros(len(batch_items), n, d)
    pos = torch.zeros(len(batch_items), n, 4)
    mask = torch.zeros(len(batch_items), n, dtype=torch.bool)
    target, size = torch.zeros(len(batch_items), n), torch.zeros(len(batch_items), n)
    q = torch.stack([questions[stem_of(it)].float() for it in batch_items])
    for i, it in enumerate(batch_items):
        gh, gw, f = images[it["image"]]
        feats[i, : gh * gw], pos[i, : gh * gw], mask[i, : gh * gw] = f.float(), positions(gh, gw), True
        if with_targets:
            t, z = token_targets(it, gh, gw)
            target[i, : gh * gw], size[i, : gh * gw] = t.view(-1), z.view(-1)
    return feats.to(device), pos.to(device), q.to(device), mask.to(device), target.to(device), size.to(device)


# --------------------------------------------------------------------------- train
def cmd_train(args):
    torch.manual_seed(args.seed)
    items = [json.loads(line) for path in args.items for line in open(path)]
    if args.scenario:
        items = [it for it in items if it["scenario"] == args.scenario]
    if args.holdout_fold >= 0:  # cross-fitting: this head never sees the images of one fold
        items = [it for it in items if fold_of(it["image"], args.num_folds) != args.holdout_fold]
    images, questions = load_feats(args.feat_dir)
    items = [it for it in items if it["image"] in images]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_held = max(200, len(items) // 20)
    device = "cuda:0"
    d = next(iter(questions.values())).shape[0]

    # everything is stacked once; a batch is then plain tensor indexing
    paths = sorted({it["image"] for it in items})
    index = {p: i for i, p in enumerate(paths)}
    n = max(images[p][0] * images[p][1] for p in paths)
    feats = torch.zeros(len(paths), n, d, dtype=torch.half)
    pos = torch.zeros(len(paths), n, 4)
    mask = torch.zeros(len(paths), n, dtype=torch.bool)
    for p, i in index.items():
        gh, gw, f = images[p]
        feats[i, : gh * gw], pos[i, : gh * gw], mask[i, : gh * gw] = f, positions(gh, gw), True
    stems = sorted({stem_of(it) for it in items})
    q_all = torch.stack([questions[s].float() for s in stems]).to(device)
    q_index = {s: i for i, s in enumerate(stems)}
    img_id = torch.tensor([index[it["image"]] for it in items])
    q_id = torch.tensor([q_index[stem_of(it)] for it in items])
    target, size = torch.zeros(len(items), n), torch.zeros(len(items), n)
    for i, it in enumerate(items):
        gh, gw, _ = images[it["image"]]
        t, z = token_targets(it, gh, gw)
        target[i, : gh * gw], size[i, : gh * gw] = t.view(-1), z.view(-1)
    pos, mask, target, size = pos.to(device), mask.to(device), target.to(device), size.to(device)
    held, train = torch.arange(n_held), torch.arange(n_held, len(items))

    head = Head(d).to(device)
    print(f"items {len(items)} | train {len(train)} | held-out {len(held)} | images {len(paths)} | parameters {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    def loss_of(ids):
        im = img_id[ids]
        m, t = mask[im], target[ids.to(device)]
        rel, size_pred = head(feats[im].to(device).float(), pos[im], q_all[q_id[ids].to(device)], m)
        weight = torch.where(t > 0, torch.full_like(t, args.pos_weight), torch.ones_like(t))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(rel, t, weight=weight, reduction="none")
        on = (t > 0.5) & m
        size_loss = torch.nn.functional.smooth_l1_loss(size_pred[on], size[ids.to(device)][on]) if on.any() else rel.sum() * 0
        return (loss * m).sum() / m.sum() + args.size_weight * size_loss

    for epoch in range(args.epochs):
        head.train()
        order = train[torch.randperm(len(train))]
        total, batches = 0.0, 0
        for start in range(0, len(order), args.batch):
            loss = loss_of(order[start : start + args.batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            total, batches = total + loss.item(), batches + 1
        head.eval()
        with torch.no_grad():
            val = [loss_of(held[s : s + args.batch]).item() for s in range(0, len(held), args.batch)]
        print(f"epoch {epoch + 1}/{args.epochs} train {total / batches:.4f} held-out {sum(val) / len(val):.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state": head.state_dict(), "in_dim": d, "args": {k: v for k, v in vars(args).items() if k != "func"}}, args.out)
    print("HEAD_TRAIN_DONE", flush=True)


# --------------------------------------------------------------------------- predict
def components(prob, threshold):
    """4-connected components of prob > threshold: list of (score, r0, c0, r1, c1)."""
    gh, gw = prob.shape
    seen, found = set(), []
    for r in range(gh):
        for c in range(gw):
            if (r, c) in seen or prob[r, c] <= threshold:
                continue
            stack, cells = [(r, c)], []
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < gh and 0 <= nx < gw and (ny, nx) not in seen and prob[ny, nx] > threshold:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            score = float(sum(prob[y, x] for y, x in cells))
            found.append((score, min(y for y, _ in cells), min(x for _, x in cells), max(y for y, _ in cells), max(x for _, x in cells)))
    return sorted(found, reverse=True)


def regions_from_map(prob, width, height, k=3, rel_threshold=0.5, floor=0.1, pad=0.5, min_side=256):
    gh, gw = prob.shape
    comps = components(prob, max(floor, rel_threshold * float(prob.max())))
    if not comps:  # nothing above the floor: still look at the most likely spot
        r, c = divmod(int(prob.argmax()), gw)
        comps = [(float(prob[r, c]), r, c, r, c)]
    cw, ch = width / gw, height / gh
    boxes, scores = [], []
    for score, r0, c0, r1, c1 in comps[: 2 * k]:
        x0, y0, x1, y1 = (c0 - pad) * cw, (r0 - pad) * ch, (c1 + 1 + pad) * cw, (r1 + 1 + pad) * ch
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = max(x1 - x0, min_side), max(y1 - y0, min_side)
        boxes.append([max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(width), cx + w / 2), min(float(height), cy + h / 2)])
        scores.append(score)
    return core.merge_regions(boxes, k), scores[:k]


@torch.no_grad()
def cmd_predict(args):
    items = [json.loads(line) for path in args.items for line in open(path)]
    if args.only_fold >= 0:
        items = [it for it in items if fold_of(it["image"], args.num_folds) == args.only_fold]
    images, questions = load_feats(args.feat_dir)
    ckpt = torch.load(args.head)
    head = Head(ckpt["in_dim"]).to("cuda:0")
    head.load_state_dict(ckpt["state"])
    head.eval()
    covered = collections.defaultdict(list)
    size_err = []
    with open(args.out, "w") as out, open(args.out.replace(".jsonl", ".cands.jsonl"), "w") as cands:
        for start in range(0, len(items), 64):
            batch = items[start : start + 64]
            feats, pos, q, mask, _, _ = collate(batch, images, questions, "cuda:0", with_targets=False)
            rel, size_pred = head(feats, pos, q, mask)
            for i, it in enumerate(batch):
                gh, gw, _ = images[it["image"]]
                prob = torch.sigmoid(rel[i, : gh * gw]).view(gh, gw).cpu()
                size_map = size_pred[i, : gh * gw].view(gh, gw).cpu()
                regions, scores = regions_from_map(prob, it["width"], it["height"], k=args.k, rel_threshold=args.rel_threshold)
                out.write(json.dumps({"id": it["id"], "regions": regions, "scores": scores, "scale": 1.0}) + "\n")
                # raw candidates for the allocation step: token-extent box, score, estimated log2 tokens of the evidence
                cw, ch = it["width"] / gw, it["height"] / gh
                comps = components(prob, max(0.1, args.rel_threshold * float(prob.max())))
                if not comps:
                    r, c = divmod(int(prob.argmax()), gw)
                    comps = [(float(prob[r, c]), r, c, r, c)]
                listed = []
                for score, r0, c0, r1, c1 in comps[:8]:
                    w = prob[r0 : r1 + 1, c0 : c1 + 1]
                    est = float((size_map[r0 : r1 + 1, c0 : c1 + 1] * w).sum() / w.sum())
                    listed.append({"box": [c0 * cw, r0 * ch, (c1 + 1) * cw, (r1 + 1) * ch], "score": score, "peak": float(w.max()), "log2_tokens": est})
                cands.write(json.dumps({"id": it["id"], "max_prob": float(prob.max()), "candidates": listed}) + "\n")
                t, z = token_targets(it, gh, gw)
                on = t > 0.5
                if on.any():
                    size_err.append(float((size_map[on] - z[on]).abs().mean()))
                centers = [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in it["evidence"]]
                hit = [any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in regions) for x, y in centers]
                covered[(it["scenario"], it["qtype"])].append(sum(hit) / len(hit))
                covered[(it["scenario"], "all")].append(sum(hit) / len(hit))
    for key in sorted(covered):
        print(f"evidence boxes whose centre falls in a selected region: {key} {100 * sum(covered[key]) / len(covered[key]):.1f}% (n={len(covered[key])})")
    print(f"evidence size estimate: mean absolute error {sum(size_err) / max(1, len(size_err)):.2f} in log2(tokens) on evidence tokens (n={len(size_err)})")
    print("HEAD_PREDICT_DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("feat")
    f.add_argument("--items", nargs="+", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--shard", type=int, default=0)
    f.add_argument("--num-shards", type=int, default=1)
    f.add_argument("--overview-tokens", type=int, default=core.THUMB_TOKENS)
    f.set_defaults(func=cmd_feat)
    t = sub.add_parser("train")
    t.add_argument("--items", nargs="+", required=True)
    t.add_argument("--feat-dir", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--scenario")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--pos-weight", type=float, default=5.0)
    t.add_argument("--size-weight", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--holdout-fold", type=int, default=-1)
    t.add_argument("--num-folds", type=int, default=2)
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--feat-dir", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rel-threshold", type=float, default=0.5)
    p.add_argument("--only-fold", type=int, default=-1)
    p.add_argument("--num-folds", type=int, default=2)
    p.set_defaults(func=cmd_predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
