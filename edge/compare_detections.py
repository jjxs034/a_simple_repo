import json, sys
import numpy as np
ref = {r["image"]: r["detections"] for r in map(json.loads, open(sys.argv[1]))}
new = {r["image"]: r["detections"] for r in map(json.loads, open(sys.argv[2]))}
def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h + 1e-9)
for thr in (0.05, 0.25):
    total = hit = 0; dconf = []; extra = 0
    for image, dets in new.items():
        r = [d for d in ref[image] if d[4] >= thr]; n = [d for d in dets if d[4] >= thr * 0.8]
        total += len(r)
        for d in r:
            cand = [(iou(d, e), e) for e in n if e[5] == d[5]]
            if cand and max(cand, key=lambda c: c[0])[0] >= 0.9:
                hit += 1; dconf.append(abs(max(cand, key=lambda c: c[0])[1][4] - d[4]))
        extra += sum(1 for e in dets if e[4] >= thr) - len(r)
    print(f"reference boxes with confidence >= {thr}: {total}; matched (same class, IoU >= 0.9): {100 * hit / max(1, total):.1f}%; median |confidence difference| {np.median(dconf):.4f}, 95th percentile {np.quantile(dconf, 0.95):.4f}; box count at this threshold differs by {extra:+d} in total")
print(len(new), "images compared")
