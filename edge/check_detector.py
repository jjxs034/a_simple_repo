"""Server-side check of the board's detector path: run the exported ONNX model through detpost.py (the code the board
script uses) and write detections in the format of `detector.py detect`, so the usual scripts can score them against
the detections of the detector library itself.

    python check_detector.py --model work/det/ed_n_1280x1280 --items ed_test.jsonl --scenario ed --out det_onnx_ed_n.jsonl
"""
import argparse
import json
import time

import numpy as np

import detpost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path without extension, as given to convert_detector.py")
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    import onnxruntime as ort

    meta = json.load(open(args.model + ".json"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model + ".onnx", options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    paths = sorted({it["image"] for path in args.items for it in map(json.loads, open(path)) if it["scenario"] == args.scenario})
    if args.limit:
        paths = paths[: args.limit]
    start = time.perf_counter()
    with open(args.out, "w") as out:
        for n, path in enumerate(paths):
            image, reader = detpost.load_rgb(path)
            canvas, r, left, top = detpost.letterbox(image, meta["size"])
            tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            output = session.run(None, {name: tensor})[0][0]
            dets = detpost.decode(output, r, left, top, image.shape[:2], meta["names"])
            out.write(json.dumps({"image": path, "detections": dets}, ensure_ascii=False) + "\n")
            if (n + 1) % 200 == 0:
                print(f"{n + 1}/{len(paths)} ({reader}), {(time.perf_counter() - start) / (n + 1):.2f} s per image", flush=True)
    print("CHECK_DETECT_DONE", len(paths), flush=True)


if __name__ == "__main__":
    main()
