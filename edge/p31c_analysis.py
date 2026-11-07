# Supplement 3 of protocol P31 (docs/results/round2_protocol.md, written 10-27 before any of its runs): criterion 3 of
# supplement 2, the knee rule, at DDR 1,968 and 534 MHz. Run on the board; per-run rows stay there, only aggregates leave:
#   ssh <board> 'nice -n 19 ~/pb_edge/venv/bin/python - OUTDIR' < edge/p31c_analysis.py
# The knee model is the one frozen at 21:27 on 10-25 (docs/results/board_p31/p31_knee_frozen.json, sha256 bab461ee...); its
# numbers are copied here. Fits and relative errors are those of p31_analysis.py and p31b_analysis.py.
import json, os, re, sys
import numpy as np

R1, R3 = os.path.expanduser("~/pb_edge/runs/p31/"), os.path.expanduser("~/pb_edge/runs/p31c/")
out = sys.argv[1] if len(sys.argv) > 1 else R3 + "agg"
os.makedirs(out, exist_ok=True)
KNEE = {"b1": (2.268387615258731, 0.874295429597352, 710), "b2": (0.0007318925217818196, 0.0001191481902733678, 810)}  # fix, npu, knee
GRID = {1968: [300, 400, 500, 600, 700], 534: [300, 500, 700]}
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


def knee_at(c, f, knee):
    fix, npu, _ = KNEE[c]
    return fix + npu * 1000 / min(f, knee)


# below the knee the frozen fix and NPU part (independent of DDR), above it the plateau measured at NPU 1 GHz and the same
# DDR clock in P31; that plateau sets the knee of each coefficient at this DDR clock
grid_rows, fit_rows = [], []
for d, fs in GRID.items():
    say(f"== roofline at DDR {d} MHz")
    plateau = dict(zip(("b1", "b2"), two_term(*lengths(R1, f"len_ddr{d}"))))
    knee = {c: KNEE[c][1] * 1000 / (plateau[c] - KNEE[c][0]) for c in plateau}
    say("  knee at this DDR clock: " + ", ".join(f"{c} {knee[c]:.0f} MHz (plateau {plateau[c]:.4g})" for c in knee))
    rel = []
    for f in fs:
        lm = lengths(R3, f"len_npu{f}_ddr{d}")
        if lm is None:
            continue
        L, m = lm
        p = knee_at("b1", f, knee["b1"]) * L + knee_at("b2", f, knee["b2"]) * L ** 2
        r = np.abs(p - m) / m * 100
        rel += list(r)
        a, b = two_term(L, m)
        own = np.abs(a * L + b * L ** 2 - m) / m * 100
        say(f"  NPU {f} MHz: median {np.median(r):.2f}%, max {r.max():.2f}%; fitted beta1 {a:.4f} (predicted {knee_at('b1', f, knee['b1']):.4f}), "
            f"beta2 {b:.3e} (predicted {knee_at('b2', f, knee['b2']):.3e}); two-term fit in sample, max {own.max():.2f}%")
        grid_rows += [f"{d} {f} {l:.0f} {mm:.1f} {pp:.1f} {rr:.3f}" for l, mm, pp, rr in zip(L, m, p, r)]
        fit_rows.append(f"{d} {f} {a:.6f} {b:.6e} {own.max():.4f}")
    if rel:
        n = 9 * len(fs)
        ok = np.median(rel) <= 3 and max(rel) <= 6
        say(f"  {len(rel)} points: median {np.median(rel):.2f}% (line 3%), max {max(rel):.2f}% (line 6%) -> {'PASS' if ok else 'FAIL'}"
            f"{'' if len(rel) == n else ' (so far)'}")
write("p31c_grid.dat", "ddr f L measured predicted relpct", grid_rows)
write("p31c_fit.dat", "ddr f beta1 beta2 maxrel", fit_rows)
vis_rows = []   # the 256-token vision encoder: described only
for d, fs in GRID.items():
    for f in fs:
        tag = f"npu{f}_ddr{d}_v256"
        if done(R3, tag):
            m = re.search(r"NPU inference: median ([\d.]+) ms", open(f"{R3}{tag}.log", errors="replace").read())
            if m:
                vis_rows.append(f"{d} {f} 256 {float(m.group(1)):.1f}")
write("p31c_vision.dat", "ddr f n ms", vis_rows)
names = sorted(os.listdir(R3)) if os.path.isdir(R3) else []
runs = [json.load(open(R3 + p)) for p in names if p.startswith("watch_") and p.endswith(".json")]
say(f"\n== runs: {len(runs)} watched, {sum(r['valid'] for r in runs)} valid, {sum('_void' in p for p in names)} void files")
open(f"{out}/p31c_report.txt", "w").write("\n".join(lines) + "\n")
