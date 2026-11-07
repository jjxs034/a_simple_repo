# Protocol P31 (docs/results/round2_protocol.md): the cost model of Section IV-D across NPU and DDR operating points.
# Run on the board; per-question rows stay there and only aggregates leave:
#   ssh <board> 'nice -n 19 ~/pb_edge/venv/bin/python - OUTDIR' < edge/p31_analysis.py
# The fits are those of p29_costfig.py: beta1 L + beta2 L^2 through the origin on the medians of the synthetic lengths,
# gamma1 n + gamma2 n^2 on the medians of the vision sizes, relative errors divided by the measured value. A run counts
# once its watch record says valid, so the script can run while the sweep goes on; criteria with missing runs say so.
# The language-model time of criterion 3 is prefill_ms with t0, as in the per-question errors of P29 (fixed 21:40 on
# 10-25, before any real prompt of P31 had run); wall_ms is reported beside it.
import glob, json, os, re, sys
import numpy as np

R = os.path.expanduser("~/pb_edge/runs/p31/")
out = sys.argv[1] if len(sys.argv) > 1 else R + "agg"
os.makedirs(out, exist_ok=True)
NPU = [300, 400, 500, 600, 700, 800, 900, 1000]
DDR = [534, 1320, 1968, 2400]
CAL = (300, 1000)                      # the two calibrations of the frequency law (criteria 2 and 3)
MID = [f for f in NPU if f not in CAL]
VIS = [196, 252, 256, 567, 576]
THUMBS = [196, 252, 256]
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def write(name, header, rows):
    with open(f"{out}/{name}", "w") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")


def done(tag):
    p = f"{R}watch_{tag}.json"
    return os.path.exists(p) and json.load(open(p))["valid"]


def rows(tag):
    return [d for d in map(json.loads, open(f"{R}{tag}.jsonl")) if not d["error"]] if done(tag) else []


def two_term(x, y):
    (a, b), *_ = np.linalg.lstsq(np.c_[x, x ** 2], y, rcond=None)
    return float(a), float(b)


def maxrel(a, b, x, y):
    return float(np.max(np.abs(a * x + b * x ** 2 - y) / y) * 100)


def lengths(tag):
    """Median prefill per synthetic length."""
    rs = rows(tag)
    if not rs:
        return None
    Ls = sorted({d["prefill_tokens"] for d in rs})
    return np.array(Ls, float), np.array([np.median([d["prefill_ms"] for d in rs if d["prefill_tokens"] == L]) for L in Ls])


def vision(pre, sizes):
    """Median of the timed runs per size, from the logs of board_det.py ('<model>.rknn input ... output [(n, 2048), ...]'
    followed by 'NPU inference: median X ms')."""
    got = []
    for n in sizes:
        tag = f"{pre}_v{n}"
        if not done(tag):
            continue
        txt = open(f"{R}{tag}.log", errors="replace").read().splitlines()
        for a, b in zip(txt, txt[1:]):
            m = re.search(r"vision_\w+\.rknn input .* output \[\((\d+), \d+\)", a)
            t = re.search(r"NPU inference: median ([\d.]+) ms", b)
            if m and t:
                got.append((int(m.group(1)), float(t.group(1))))
    return (np.array([g[0] for g in got], float), np.array([g[1] for g in got])) if got else None


# 1) every operating point on its own: medians, two-term fits and their largest error (criterion 1)
pts = [("npu", f) for f in NPU] + [("ddr", d) for d in DDR]
fit, pre_rows, vis_rows, fit_rows = {}, [], [], []
for kind, f in pts:
    lm = lengths(f"len_{kind}{f}")
    vi = vision(f"{kind}{f}", VIS if kind == "npu" else THUMBS)
    if lm is None and vi is None:
        continue
    e = {}
    if lm is not None:
        L, m = lm
        e["b1"], e["b2"] = two_term(L, m)
        e["lm_err"] = maxrel(e["b1"], e["b2"], L, m)
        e["L"], e["m"] = L, m
        pre_rows += [f"{kind} {f} {l:.0f} {v:.1f}" for l, v in zip(L, m)]
    if vi is not None and len(vi[0]) >= 3:
        n, t = vi
        e["g1"], e["g2"] = two_term(n, t)
        e["vis_err"] = maxrel(e["g1"], e["g2"], n, t)
        e["n"], e["t"] = n, t
        vis_rows += [f"{kind} {f} {a:.0f} {b:.1f}" for a, b in zip(n, t)]
    fit[kind, f] = e
    fit_rows.append(f"{kind} {f} {e.get('b1', np.nan):.6g} {e.get('b2', np.nan):.6g} {e.get('lm_err', np.nan):.4f} "
                    f"{e.get('g1', np.nan):.6g} {e.get('g2', np.nan):.6g} {e.get('vis_err', np.nan):.4f} {len(e.get('n', []))}")
write("p31_prefill.dat", "kind f L ms", pre_rows)
write("p31_vision.dat", "kind f n ms", vis_rows)
write("p31_fit.dat", "kind f beta1 beta2 lm_maxrel gamma1 gamma2 vis_maxrel nvis", fit_rows)

say("== criterion 1: the two-term form at every operating point (largest relative error over the medians)")
for kind, f in pts:
    e = fit.get((kind, f))
    if e:
        say(f"  {kind} {f:4d} MHz: prefill beta1 {e.get('b1', np.nan):.4f} ms/token, beta2 {e.get('b2', np.nan):.3e}, "
            f"max {e.get('lm_err', np.nan):.2f}% | vision ({len(e.get('n', []))} sizes) gamma1 {e.get('g1', np.nan):.3f}, "
            f"gamma2 {e.get('g2', np.nan):.3e}, max {e.get('vis_err', np.nan):.2f}%")
lm_pts = [k for k in pts if "lm_err" in fit.get(k, {})]
vis_pts = [("npu", f) for f in NPU if len(fit.get(("npu", f), {}).get("n", [])) == 5]
miss = [f"{k}{f}" for k, f in pts if (k, f) not in lm_pts] + [f"npu{f} vision" for f in NPU if ("npu", f) not in vis_pts]
worst_lm = max([fit[k]["lm_err"] for k in lm_pts], default=np.nan)
worst_vis = max([fit[k]["vis_err"] for k in vis_pts], default=np.nan)
say(f"  prefill: worst {worst_lm:.2f}% over {len(lm_pts)} points (line 3%) -> {'PASS' if worst_lm <= 3 else 'FAIL'}"
    f"{'' if not miss else '; missing ' + ', '.join(miss)}")
say(f"  vision: worst {worst_vis:.2f}% over {len(vis_pts)} NPU points (line 5%) -> {'PASS' if worst_vis <= 5 else 'FAIL'}")

# 2) the frequency law c(f) = c_fix + c_npu (1 GHz / f), set by the calibrations at 300 and 1000 MHz only
def law(c_lo, c_hi):
    k = (1000 / CAL[0] - 1)
    c_npu = (c_lo - c_hi) / k
    return c_hi - c_npu, c_npu


def at(fix_npu, f):
    return fix_npu[0] + fix_npu[1] * 1000 / f


lo, hi = fit.get(("npu", CAL[0]), {}), fit.get(("npu", CAL[1]), {})
lawc, law_rows, pred_rows = {}, [], []
for c in ("b1", "b2", "g1", "g2"):
    if c in lo and c in hi:
        lawc[c] = law(lo[c], hi[c])
        law_rows.append(f"{c} {lawc[c][0]:.6g} {lawc[c][1]:.6g} {lawc[c][0] / hi[c] * 100:.2f}")
say("\n== criterion 2: frequency law from 300 and 1000 MHz only")
for r in law_rows:
    c, fx, nu, share = r.split()
    say(f"  {c}: fixed {float(fx):.4g}, NPU part {float(nu):.4g} at 1 GHz; fixed share of the 1-GHz value {share}%")
for what, cs, key_x, key_y, line_med, line_max in (("prefill", ("b1", "b2"), "L", "m", 3, 6), ("vision", ("g1", "g2"), "n", "t", 5, 8)):
    if not all(c in lawc for c in cs):
        continue
    rel, have = [], []
    for f in MID:
        e = fit.get(("npu", f), {})
        if key_x not in e:
            continue
        x, y = e[key_x], e[key_y]
        p = at(lawc[cs[0]], f) * x + at(lawc[cs[1]], f) * x ** 2
        r = np.abs(p - y) / y * 100
        rel += list(r)
        have.append(f)
        pred_rows += [f"{what} {f} {a:.0f} {b:.1f} {c:.1f} {d:.3f}" for a, b, c, d in zip(x, y, p, r)]
        say(f"  {what} {f} MHz: median {np.median(r):.2f}%, max {np.max(r):.2f}%")
    if rel:
        med, mx = float(np.median(rel)), float(np.max(rel))
        ok = med <= line_med and mx <= line_max
        say(f"  {what}: {len(rel)} points at {have}: median {med:.2f}% (line {line_med}%), max {mx:.2f}% (line {line_max}%) -> "
            f"{'PASS' if ok else 'FAIL'}{'' if len(have) == len(MID) else ' (so far; ' + str(len(MID) - len(have)) + ' frequencies missing)'}")
# descriptive: the same law fitted by least squares over all measured NPU points, coefficient by coefficient
for c in ("b1", "b2", "g1", "g2"):
    fs = [f for f in NPU if c in fit.get(("npu", f), {})]
    if len(fs) >= 3:
        X = np.c_[np.ones(len(fs)), [1000 / f for f in fs]]
        y = np.array([fit["npu", f][c] for f in fs])
        (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
        r = np.abs(X @ [a, b] - y) / y * 100
        say(f"  (all {len(fs)} points, least squares) {c}: fixed {a:.4g}, NPU part {b:.4g}, max deviation {r.max():.2f}%")

# 3) real prompts at 300, 600 and 1000 MHz; t0 of each calibration point is its median residual, as in P29
say("\n== criterion 3: real prompts (list only, 404) at an uncalibrated frequency")
real, t0, err_rows, bin_rows = {}, {}, [], []
for f in (300, 600, 1000):
    rs = rows(f"real_npu{f}")
    if rs:
        real[f] = (np.array([d["prefill_tokens"] for d in rs], float), np.array([d["prefill_ms"] for d in rs], float),
                   np.array([d["wall_ms"] for d in rs], float))
for f in CAL:
    if f in real and "b1" in fit.get(("npu", f), {}):
        x, y, w = real[f]
        e = fit["npu", f]
        t0[f] = float(np.median(y - (e["b1"] * x + e["b2"] * x ** 2)))
        r = np.abs(y - (e["b1"] * x + e["b2"] * x ** 2 + t0[f])) / y * 100
        err_rows.append(f"{f} own {len(x)} {t0[f]:.2f} {np.median(r):.4f} {np.percentile(r, 95):.4f} {r.max():.4f}")
        say(f"  {f} MHz (own calibration, t0 {t0[f]:.1f} ms, in sample): median {np.median(r):.2f}%, p95 {np.percentile(r, 95):.2f}%")
if len(t0) == 2 and "b1" in lawc and 600 in real:
    t0law = law(t0[CAL[0]], t0[CAL[1]])
    law_rows.append(f"t0 {t0law[0]:.6g} {t0law[1]:.6g} {t0law[0] / t0[CAL[1]] * 100:.2f}")
    x, y, w = real[600]
    p = at(lawc["b1"], 600) * x + at(lawc["b2"], 600) * x ** 2 + at(t0law, 600)
    r = np.abs(y - p) / y * 100
    rw = np.abs(w - p) / w * 100
    err_rows.append(f"600 law {len(x)} {at(t0law, 600):.2f} {np.median(r):.4f} {np.percentile(r, 95):.4f} {r.max():.4f}")
    say(f"  600 MHz from the law (t0 {at(t0law, 600):.1f} ms): median {np.median(r):.2f}% (line 5%), p95 {np.percentile(r, 95):.2f}%, "
        f"max {r.max():.2f}% -> {'PASS' if np.median(r) <= 5 else 'FAIL'}; against wall_ms: median {np.median(rw):.2f}%")
    if "b1" in fit.get(("npu", 600), {}):   # descriptive: had 600 MHz been calibrated itself
        e = fit["npu", 600]
        q = e["b1"] * x + e["b2"] * x ** 2
        rq = np.abs(y - q - np.median(y - q)) / y * 100
        say(f"  (descriptive) 600 MHz with its own calibration and own t0: median {np.median(rq):.2f}%")
for f in real:   # six bins of equal counts per frequency: median L, measured, predicted with t0 (law at 600)
    x, y, w = real[f]
    if f in t0:
        p = fit["npu", f]["b1"] * x + fit["npu", f]["b2"] * x ** 2 + t0[f]
    elif len(t0) == 2 and "b1" in lawc:
        p = at(lawc["b1"], f) * x + at(lawc["b2"], f) * x ** 2 + at(law(t0[CAL[0]], t0[CAL[1]]), f)
    else:
        continue
    edges = np.unique(np.percentile(x, np.linspace(0, 100, 7)).round())
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() >= 20:
            bin_rows.append(f"{f} {np.median(x[m]):.0f} {np.median(y[m]):.1f} {np.median(p[m]):.1f} {int(m.sum())}")
write("p31_law.dat", "coef fix npu fixshare_pct_at_1GHz", law_rows)
write("p31_pred.dat", "what f x measured predicted relpct", pred_rows)
write("p31_realerr.dat", "f source n t0 medrel p95rel maxrel", err_rows)
write("p31_real.dat", "f L ms pred n", bin_rows)

# 4) beyond the calibrated range: medians at 1,024-2,000 tokens against the curve of the same frequency (report only)
say("\n== item 4: beyond the calibrated range (measured above the curve, % of measured)")
long_rows = []
for f in (300, 500, 700, 1000):
    lm, e = lengths(f"long_npu{f}"), fit.get(("npu", f), {})
    if lm is None or "b1" not in e:
        continue
    L, m = lm
    cv = e["b1"] * L + e["b2"] * L ** 2
    gap = (m - cv) / m * 100
    LL, MM = np.r_[e["L"], L], np.r_[e["m"], m]
    c1, c2 = two_term(LL, MM)
    long_rows += [f"{f} {l:.0f} {a:.1f} {b:.1f} {g:.3f}" for l, a, b, g in zip(L, m, cv, gap)]
    say(f"  {f} MHz: gap {gap.min():.1f}% to {gap.max():.1f}%; one fit over all {len(LL)} lengths: max {maxrel(c1, c2, LL, MM):.2f}%")
write("p31_long.dat", "f L ms curve gappct", long_rows)

# 5) DDR operating points at an NPU of 1 GHz (description only): coefficients against those at 2,400 MHz, and what is
# left of each after the NPU part of the law (taken as independent of the DDR clock)
say("\n== item 5: DDR operating points (NPU 1 GHz)")
ref = fit.get(("ddr", 2400), {})
for d in DDR:
    e = fit.get(("ddr", d), {})
    if not e or not ref:
        continue
    parts = []
    for c in ("b1", "b2", "g1", "g2"):
        if c in e and c in ref:
            rest = f", minus NPU part {e[c] - lawc[c][1]:.4g}" if c in lawc else ""
            parts.append(f"{c} {e[c]:.4g} (x{e[c] / ref[c]:.2f}{rest})")
    say(f"  ddr {d:4d}: " + "; ".join(parts))
if "b1" in fit.get(("ddr", 2400), {}) and "b1" in fit.get(("npu", 1000), {}):
    a, b = fit["ddr", 2400], fit["npu", 1000]
    say(f"  repeat of the 1000/2400 point: prefill medians differ by at most "
        f"{np.max(np.abs(a['m'] - b['m']) / b['m']) * 100:.2f}% between the two runs")

# memory and the run records
say("\n== memory reported by the runtime at the longest synthetic length, per operating point (MiB)")
mem = []
for kind, f in pts:
    rs = rows(f"len_{kind}{f}")
    if rs:
        top = max(d["prefill_tokens"] for d in rs)
        mem.append(f"{kind}{f} {np.median([d['memory_mb'] for d in rs if d['prefill_tokens'] == top]):.1f}")
say("  " + ", ".join(mem))
runs = sorted(glob.glob(R + "watch_*.json"))
recs = [json.load(open(p)) for p in runs]
void = [p for p in runs if "_void" in p]
say(f"\n== runs: {len(recs)} watched, {sum(r['valid'] for r in recs)} valid, {len(void)} set aside as void; "
    f"hottest sensor {max(max(t['max'] for t in r['temps'].values()) for r in recs):.1f} C; NPU and DDR at their set "
    f"clock in {min(min(r['clocks'][k]['at_fixed'] for k in ('fdab0000.npu', 'dmc')) for r in recs) * 100:.0f}% of samples at worst")
open(f"{out}/p31_report.txt", "w").write("\n".join(lines) + "\n")
