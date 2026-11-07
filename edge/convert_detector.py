"""Detector branch -> RKNN for RK3588, without quantisation (half precision on the NPU, so no accuracy question
beyond rounding). Fixed input size. Runs in the rknn environment (.venv-rknn) on the server.

    python convert_detector.py --weights <best.pt> --size 1280 1280 --out work/det/ed_n_1280x1280

Writes <out>.onnx, <out>.rknn and <out>.json (class names, input size). The ONNX file is what check_detector.py runs.
"""
import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--size", type=int, nargs=2, required=True, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--out", required=True, help="output path without extension")
    parser.add_argument("--platform", default="rk3588")
    parser.add_argument("--names", nargs="*", help="class names in the order of the detector's classes (default: the names stored in the weights)")
    args = parser.parse_args()
    from ultralytics import YOLO

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    weights = args.out + ".pt"  # the library exports next to the weights file; work on a copy
    shutil.copy(args.weights, weights)
    model = YOLO(weights)
    names = args.names or [model.names[k] for k in sorted(model.names)]
    exported = model.export(format="onnx", imgsz=tuple(args.size), opset=12, simplify=False, dynamic=False, batch=1, device="cpu")
    os.replace(exported, args.out + ".onnx")
    os.remove(weights)
    print("ONNX_DONE", args.out + ".onnx", flush=True)

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=args.platform)
    assert rknn.load_onnx(model=args.out + ".onnx") == 0, "load_onnx failed"
    assert rknn.build(do_quantization=False) == 0, "build failed"
    assert rknn.export_rknn(args.out + ".rknn") == 0, "export failed"
    rknn.release()
    json.dump({"names": names, "size": args.size, "weights": args.weights, "precision": "fp16", "platform": args.platform}, open(args.out + ".json", "w"), ensure_ascii=False, indent=1)
    print("RKNN_DONE", args.out + ".rknn", flush=True)


if __name__ == "__main__":
    main()
