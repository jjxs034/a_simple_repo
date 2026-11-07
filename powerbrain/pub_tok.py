# Share of InsPLAD-det boxes smaller than one token of the 576-token thumbnail, with the resize rule of core.resize_to_token_cap
# (CELL = 32), on the boxes that bench.load_pl reads (as for Table II). Uses the stand-in PB_PL_ROOT that pub_stats.py built.
import math, os, sys
root, repo = sys.argv[1], sys.argv[2]
os.environ["PB_PL_ROOT"] = root
sys.path.insert(0, os.path.join(repo, "experiments", "pb"))
import bench
CELL, CAP = 32, 576
def thumb(w, h):
    s = min(1.0, math.sqrt(CAP * CELL * CELL / (w * h)))
    rc = lambda v: max(CELL, int(round(v / CELL)) * CELL)
    nw, nh = rc(w * s), rc(h * s)
    while (nw // CELL) * (nh // CELL) > CAP:
        if nw >= nh: nw -= CELL
        else: nh -= CELL
    return nw, nh
n = small = 0
for part in ("train", "test"):
    for img in bench.load_pl(part):
        w, h = img["width"], img["height"]
        nw, nh = thumb(w, h)
        for b in img["boxes"]:
            tok = b[2] * b[3] * (nw / w) * (nh / h) / (CELL * CELL)
            n += 1; small += tok < 1
print("boxes", n, "below one token", small, "share %.1f%%" % (100 * small / n), "| thumb of 1920x1080:", thumb(1920, 1080))
