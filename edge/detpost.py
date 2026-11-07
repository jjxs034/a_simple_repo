"""Pre- and post-processing of the detector branch outside the detector library: letterbox, decoding, per-class NMS.
Shared by the board script (board_det.py) and its server-side check (check_detector.py), so that what runs on the
board is the code that was checked against the library's own output. Needs numpy, and OpenCV or Pillow for images.

Mirrors the library defaults used by `detector.py detect`: best class per box, confidence above 0.01, per-class NMS
at IoU 0.7 through class offsets, at most 300 boxes, boxes mapped back to the original image and clipped.
"""
import numpy as np

PAD = 114


def load_rgb(path):
    """HxWx3 uint8 RGB, EXIF orientation applied (as OpenCV's reader does)."""
    try:
        import cv2

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        assert image is not None, path
        return np.ascontiguousarray(image[:, :, ::-1]), "cv2"
    except ImportError:
        from PIL import Image, ImageOps

        return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB")), "pil"


def resize(image, width, height):
    try:
        import cv2

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def letterbox(image, size):
    """image: HxWx3 uint8; size: (height, width) of the network input. Returns the padded input, the scale, and the
    left and top offsets of the image inside it."""
    h0, w0 = image.shape[:2]
    height, width = size
    r = min(height / h0, width / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    if (nw, nh) != (w0, h0):
        image = resize(image, nw, nh)
    left, top = int(round((width - nw) / 2 - 0.1)), int(round((height - nh) / 2 - 0.1))
    canvas = np.full((height, width, 3), PAD, np.uint8)
    canvas[top : top + nh, left : left + nw] = image
    return canvas, r, left, top


def nms(boxes, scores, iou):
    """Indices kept, highest score first. boxes: (n, 4) corners."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(kind="stable")[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) <= iou]
    return keep


def decode(output, r, left, top, shape, names, conf=0.01, iou=0.7, max_det=300, max_wh=7680.0):
    """output: (4 + classes, anchors) as the exported model writes it: centre x, centre y, width, height in input pixels,
    then one probability per class. shape: (height, width) of the original image.
    Returns [[x0, y0, x1, y1, confidence, class name]] in original-image pixels, highest confidence first."""
    pred = np.asarray(output, np.float32).T
    assert pred.shape[1] == 4 + len(names), pred.shape
    cls = pred[:, 4:].argmax(1)
    score = pred[:, 4:].max(1)
    keep = score > conf
    pred, cls, score = pred[keep], cls[keep], score[keep]
    order = score.argsort(kind="stable")[::-1][:30000]
    pred, cls, score = pred[order], cls[order], score[order]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    idx = nms(boxes + cls[:, None] * max_wh, score, iou)[:max_det]
    boxes = (boxes[idx] - np.array([left, top, left, top], np.float32)) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
    return [[*map(float, b), float(s), names[int(c)]] for b, s, c in zip(boxes, score[idx], cls[idx])]
