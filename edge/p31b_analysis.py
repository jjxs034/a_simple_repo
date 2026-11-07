# Supplement 2 of protocol P31 (docs/results/round2_protocol.md, written 10-25 22:31; this script 22:33, before any of its
# runs had finished). Run on the board; per-image and per-question rows stay there, only aggregates leave:
#   ssh <board> 'nice -n 19 ~/pb_edge/venv/bin/python - OUTDIR' < edge/p31b_analysis.py
# The knee model is the one frozen at 21:27 (docs/results/board_p31/p31_knee_frozen.json, sha256 bab461ee...); its numbers are
# copied here. Fits and relative errors are those of p31_analysis.py.
import json, os, re, sys
import numpy as np

R1, R2 = os.path.expanduser("~/pb_edge/runs/p31/"), os.path.expanduser("~/pb_edge/runs/p31b/")
out = sys.argv[1] if len(sys.argv) > 1 else R2 + "agg"
os.makedirs(out, exist_ok=True)
KNEE = {"b1": (2.268387615258731, 0.874295429597352, 710), "b2": (0.0007318925217818196, 0.0001191481902733678, 810),
        "g1": (1.0061171214817788, 3.2366627747664913, 710), "g2": (0.009975259467463674, 0.013274496100120521, 950)}  # fix, npu, knee
NPU = [300, 400, 500, 600, 700, 800, 900, 1000]
FIT, HELD = [300, 400, 500, 700, 900, 1000], [600, 800]
GRID_DDR, GRID_NPU = 1320, [300, 400, 500, 600, 700]
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def write(name, header, rows):
    with open(f"{out}/{name}", "w") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")


def done(run, tag):
    p = f"{run}watch_{tag}.json"
    return os.path.exists(p) and json.load(open(p))["valid"]


def rows(run, tag):
    return [d for d in map(json.loads, open(f"{run}{tag}.jsonl")) if not d.get("error")] if done(run, tag) else []


def two_term(x, y):
    (a, b), *_ = np.linalg.lstsq(np.c_[x, x ** 2], y, rcond=None)
    return float(a), float(b)


def lengths(run, tag):
    rs = rows(run, tag)
    if not rs:
        return None
    Ls = sorted({d["prefill_tokens"] for d in rs})
    return np.array(Ls, float), np.array([np.median([d["prefill_ms"] for d in rs if d["prefill_tokens"] == L]) for L in Ls])


def knee_at(c, f, knee=None):
    fix, npu, k = KNEE[c]
    return fix + npu * 1000 / min(f, knee or k)


def knee_fit(fs, ys):  # as in the freeze of supplement 1: one knee on a 10-MHz grid, least squares in relative terms
    best = None
    for k in range(300, 1001, 10):
        x = np.array([1000 / min(f, k) for f in fs]); y = np.array(ys); w = 1 / y
        (a, b), *_ = np.linalg.lstsq(np.c_[np.ones_like(x), x] * w[:, None], y * w, rcond=None)
        s = float(np.sum(((a + b * x - y) / y) ** 2))
        if best is None or s < best[0] - 1e-15:
            best = (s, k, float(a), float(b))
    return best[1:]


# 1) the detector branch per image (read, letterbox, NPU, decode), eight NPU points; knee fitted on six, two held out
say("== criterion 1: detector branch, median ms per image (60 test images per scenario)")
det, det_rows = {}, []
for s in ("ed", "td"):
    for f in NPU:
        rs = rows(R2, f"det_npu{f}_{s}")
        if rs:
            sec = np.array([r["seconds"] for r in rs]) * 1000
            det[s, f] = float(np.median(sec.sum(1)))
            det_rows.append(f"{s} {f} {len(rs)} {det[s, f]:.1f} {np.median(sec[:, 2]):.1f} {np.median(sec[:, 0] + sec[:, 1] + sec[:, 3]):.1f}")
    say(f"  {s}: " + ", ".join(f"{f} MHz {det[s, f]:.0f}" for f in NPU if (s, f) in det))
write("p31b_det.dat", "scenario f n total_ms npu_ms cpu_ms", det_rows)
rel = []
for s in ("ed", "td"):
    if all((s, f) in det for f in FIT + HELD):
        k, a, b = knee_fit(FIT, [det[s, f] for f in FIT])
        for f in HELD:
            p = a + b * 1000 / min(f, k)
            rel.append(abs(p - det[s, f]) / det[s, f] * 100)
            say(f"  {s} {f} MHz: predicted {p:.1f}, measured {det[s, f]:.1f} ({rel[-1]:.2f}%); knee {k} MHz, fixed {a:.1f} ms, NPU part {b:.1f} ms at 1 GHz")
if len(rel) == 4:
    say(f"  held out: median {np.median(rel):.2f}% (line 3%), max {max(rel):.2f}% (line 6%) -> {'PASS' if np.median(rel) <= 3 and max(rel) <= 6 else 'FAIL'}")

# 2) the thumbnail tier: t0 at 300 and 1000 MHz with each frequency's own beta, 600 MHz from the frozen knee model
say("\n== criterion 2: thumbnail tier (language model per question, vision encoder per image)")
own, t0, thumb_rows = {}, {}, []
for f in (300, 1000):
    lm = lengths(R1, f"len_npu{f}")
    if lm is not None:
        own[f] = two_term(*lm)
for f in (300, 1000, 600):   # the two calibrations first: 600 MHz needs both values of t0
    rs = rows(R2, f"thumb_npu{f}")
    if not rs:
        continue
    x = np.array([d["prefill_tokens"] for d in rs], float); y = np.array([d["prefill_ms"] for d in rs], float)
    vis = [(d["n_image_tokens"], d["vision_ms"]) for d in rs if d["vision_ms"] > 0]
    if f in own:
        b1, b2 = own[f]
        t0[f] = float(np.median(y - b1 * x - b2 * x ** 2))
        r = np.abs(y - b1 * x - b2 * x ** 2 - t0[f]) / y * 100
        say(f"  {f} MHz: {len(rs)} questions, {len(vis)} images; t0 {t0[f]:.1f} ms (own beta, in sample: median {np.median(r):.2f}%)")
        thumb_rows.append(f"{f} own {len(rs)} {t0[f]:.2f} {np.median(r):.4f} nan")
    elif len(t0) == 2:
        c_npu = (t0[300] - t0[1000]) / (1000 / 300 - 1)
        t = t0[1000] - c_npu + c_npu * 1000 / f
        p = knee_at("b1", f) * x + knee_at("b2", f) * x ** 2 + t
        r = np.abs(y - p) / y * 100
        n = np.array([v[0] for v in vis], float); vm = np.array([v[1] for v in vis])
        pv = knee_at("g1", f) * n + knee_at("g2", f) * n ** 2
        rv = np.abs(vm - pv) / vm * 100
        say(f"  {f} MHz: {len(rs)} questions; t0 {t:.1f} ms by the rule of criterion 3; language model median {np.median(r):.2f}% (line 5%), "
            f"p95 {np.percentile(r, 95):.2f}% -> {'PASS' if np.median(r) <= 5 else 'FAIL'}")
        say(f"  {f} MHz: {len(vis)} images, vision encoder median {np.median(rv):.2f}% (line 5%) -> {'PASS' if np.median(rv) <= 5 else 'FAIL'}")
        thumb_rows += [f"{f} knee {len(rs)} {t:.2f} {np.median(r):.4f} {np.median(rv):.4f}"]
write("p31b_thumb.dat", "f source n t0 lm_medrel vis_medrel", thumb_rows)

# 3) roofline at DDR 1,320 MHz: below the knee the frozen fix and NPU part (independent of DDR), above it the plateau measured
# at NPU 1 GHz and DDR 1,320 MHz in P31; that plateau sets the knee of each coefficient at this DDR clock
say(f"\n== criterion 3: roofline at DDR {GRID_DDR} MHz")
ref = lengths(R1, f"len_ddr{GRID_DDR}")
grid_rows, rel = [], []
if ref is not None:
    plateau = dict(zip(("b1", "b2"), two_term(*ref)))
    knee = {c: KNEE[c][1] * 1000 / (plateau[c] - KNEE[c][0]) for c in plateau}
    say("  knee at this DDR clock: " + ", ".join(f"{c} {knee[c]:.0f} MHz (plateau {plateau[c]:.4g})" for c in knee))
    for f in GRID_NPU:
        lm = lengths(R2, f"len_npu{f}_ddr{GRID_DDR}")
        if lm is None:
            continue
        L, m = lm
        p = knee_at("b1", f, knee["b1"]) * L + knee_at("b2", f, knee["b2"]) * L ** 2
        r = np.abs(p - m) / m * 100
        rel += list(r)
        a, b = two_term(L, m)
        say(f"  NPU {f} MHz: median {np.median(r):.2f}%, max {r.max():.2f}%; fitted beta1 {a:.4f} (predicted {knee_at('b1', f, knee['b1']):.4f}), "
            f"beta2 {b:.3e} (predicted {knee_at('b2', f, knee['b2']):.3e})")
        grid_rows += [f"{f} {l:.0f} {mm:.1f} {pp:.1f} {rr:.3f}" for l, mm, pp, rr in zip(L, m, p, r)]
    if rel:
        ok = np.median(rel) <= 3 and max(rel) <= 6
        say(f"  {len(rel)} points: median {np.median(rel):.2f}% (line 3%), max {max(rel):.2f}% (line 6%) -> {'PASS' if ok else 'FAIL'}"
            f"{'' if len(rel) == 45 else ' (so far)'}")
write("p31b_grid.dat", "f L measured predicted relpct", grid_rows)
vis_rows = []   # vision encoder at this DDR clock: described only
for f in GRID_NPU:
    for n in (196, 252, 256, 567, 576):
        tag = f"npu{f}_ddr{GRID_DDR}_v{n}"
        if done(R2, tag):
            m = re.search(r"NPU inference: median ([\d.]+) ms", open(f"{R2}{tag}.log", errors="replace").read())
            if m:
                vis_rows.append(f"{f} {n} {float(m.group(1)):.1f}")
write("p31b_grid_vision.dat", "f n ms", vis_rows)
runs = [json.load(open(R2 + p)) for p in sorted(os.listdir(R2)) if p.startswith("watch_") and p.endswith(".json")]
say(f"\n== runs: {len(runs)} watched, {sum(r['valid'] for r in runs)} valid, {sum('_void' in p for p in os.listdir(R2))} void files")
open(f"{out}/p31b_report.txt", "w").write("\n".join(lines) + "\n")
