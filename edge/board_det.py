"""On the board: time an RKNN model on the NPU, or run the detector branch over a folder of images.

  time     N runs on a random input; prints the median. Works for the detector and for the vision encoder.
  detect   every listed image through read -> letterbox -> NPU -> decode; writes the detections (for scoring on the
           server) and the time of each stage.

    python3 board_det.py time   --model ed_n_1280x1280.rknn --shape 1280 1280
    python3 board_det.py detect --model ed_n_1280x1280 --list images_ed.txt --images ./images_ed --out det_ed_n.jsonl

Needs rknn-toolkit-lite2 and numpy; `detect` also needs OpenCV (preferred, it is what the server uses) or Pillow.
"""
import argparse
import json
import os
import statistics
import time

import numpy as np


def open_model(path, cores):
    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    assert rknn.load_rknn(path) == 0, f"cannot load {path}"
    mask = {"0": RKNNLite.NPU_CORE_0, "0_1": RKNNLite.NPU_CORE_0_1, "0_1_2": RKNNLite.NPU_CORE_0_1_2, "auto": RKNNLite.NPU_CORE_AUTO}[cores]
    assert rknn.init_runtime(core_mask=mask) == 0, "cannot start the NPU runtime (check the driver and librknnrt.so versions)"
    return rknn


def summary(name, values):
    values = sorted(values)
    return f"{name}: median {1000 * statistics.median(values):.1f} ms, fastest {1000 * values[0]:.1f}, slowest tenth above {1000 * values[int(0.9 * (len(values) - 1))]:.1f} (n={len(values)})"


def cmd_time(args):
    rknn = open_model(args.model, args.cores)
    data = np.random.default_rng(0).integers(0, 256, (1, args.shape[0], args.shape[1], 3), dtype=np.uint8)
    for _ in range(args.warmup):
        rknn.inference(inputs=[data])
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        out = rknn.inference(inputs=[data])
        times.append(time.perf_counter() - start)
    print(f"{os.path.basename(args.model)} input {args.shape[0]}x{args.shape[1]} cores {args.cores} output {[tuple(o.shape) for o in out]}")
    print(summary("NPU inference", times))
    print("TIME_DONE")
    rknn.release()


def cmd_detect(args):
    import detpost

    meta = json.load(open(args.model + ".json"))
    rknn = open_model(args.model + ".rknn", args.cores)
    paths = [line.strip() for line in open(args.list) if line.strip()]
    if args.limit:
        paths = paths[: args.limit]
    stages = {"read and decode the file": [], "letterbox": [], "NPU inference": [], "decode and NMS": [], "whole image": []}
    for path in paths[: args.warmup]:  # warm-up passes over the first images, not recorded (protocol P29, item 4)
        image, _ = detpost.load_rgb(os.path.join(args.images, os.path.basename(path) + args.suffix))
        rknn.inference(inputs=[detpost.letterbox(image, meta["size"])[0][None]])
    done = set()
    if os.path.exists(args.out):  # resume
        done = {json.loads(line)["image"] for line in open(args.out)}
    with open(args.out, "a") as out:
        for n, path in enumerate(paths):
            if path in done:
                continue
            t0 = time.perf_counter()
            image, reader = detpost.load_rgb(os.path.join(args.images, os.path.basename(path) + args.suffix))
            t1 = time.perf_counter()
            canvas, r, left, top = detpost.letterbox(image, meta["size"])
            t2 = time.perf_counter()
            output = rknn.inference(inputs=[canvas[None]])[0][0]
            t3 = time.perf_counter()
            dets = detpost.decode(output, r, left, top, image.shape[:2], meta["names"])
            t4 = time.perf_counter()
            for key, value in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                stages[key].append(value)
            out.write(json.dumps({"image": path, "detections": dets, "seconds": [round(v, 4) for v in (t1 - t0, t2 - t1, t3 - t2, t4 - t3)], "reader": reader}, ensure_ascii=False) + "\n")
            out.flush()
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(paths)}: {summary('whole image', stages['whole image'])}", flush=True)
    for key, values in stages.items():
        if values:
            print(summary(key, values))
    print(f"DETECT_DONE {len(paths)} images -> {args.out}")
    rknn.release()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("time")
    p.add_argument("--model", required=True, help="the .rknn file")
    p.add_argument("--shape", type=int, nargs=2, required=True, metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cores", default="0_1_2", choices=["0", "0_1", "0_1_2", "auto"])
    p.set_defaults(func=cmd_time)
    p = sub.add_parser("detect")
    p.add_argument("--model", required=True, help="path without extension: <model>.rknn and <model>.json")
    p.add_argument("--list", required=True, help="text file, one image path per line (the server paths; only the file names are used here)")
    p.add_argument("--images", required=True, help="folder holding the image files")
    p.add_argument("--suffix", default="", help="appended to each file name, e.g. .png for the degraded copies of P29 supplement 2")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cores", default="0_1_2", choices=["0", "0_1", "0_1_2", "auto"])
    p.set_defaults(func=cmd_detect)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
