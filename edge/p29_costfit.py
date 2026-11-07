# Cost model of Section IV-D, checked on the board (protocol P29 outputs in ~/pb_edge/runs/p29/; run on the board,
# raw rows stay there):  ssh <board> 'nice -n 19 ~/pb_edge/venv/bin/python -' < edge/p29_costfit.py
import json, os, numpy as np
R = os.path.expanduser("~/pb_edge/runs/p29/")
def load(f):
    return [json.loads(l) for l in open(R + f)]

syn = [d for d in load("llm_lengths.jsonl") if not d["error"]]
lst = [d for d in load("llm_serverlists.jsonl") if not d["error"]]
mm = [d for d in load("mm_serverlists_all.jsonl") if not d["error"]]

# 1) fit on the medians of the synthetic lengths (as in the summary)
Ls = sorted({d["prefill_tokens"] for d in syn})
med = np.array([np.median([d["prefill_ms"] for d in syn if d["prefill_tokens"] == L]) for L in Ls])
L = np.array(Ls, float)
A = np.c_[L, L**2]
(a, b), *_ = np.linalg.lstsq(A, med, rcond=None)
pred = a * L + b * L**2
print("synthetic medians:", dict(zip(Ls, med.round(1))))
print(f"fit a={a:.4f} ms/token b={b:.4e} ms/token^2  max rel err {np.max(np.abs(pred-med)/med)*100:.2f}%")
A1 = np.c_[np.ones_like(L), L, L**2]
(c0, a1, b1), *_ = np.linalg.lstsq(A1, med, rcond=None)
print(f"with intercept: c={c0:.2f} a={a1:.4f} b={b1:.4e}")
f = lambda x: a * x + b * x**2

# 2) real prompts, list only (text path)
def check(name, rows, key_ms="prefill_ms", key_L="prefill_tokens"):
    x = np.array([d[key_L] for d in rows], float); y = np.array([d[key_ms] for d in rows], float)
    p = f(x); r = y - p; rel = r / p
    print(f"\n{name}: n={len(x)} L range {x.min():.0f}-{x.max():.0f} median L {np.median(x):.0f}")
    print(f"  measured median {np.median(y):.1f} ms, predicted at median L {f(np.median(x)):.1f} ms")
    print(f"  residual (measured - predicted): median {np.median(r):.1f} ms, IQR {np.percentile(r,25):.1f}..{np.percentile(r,75):.1f}, "
          f"rel median {np.median(rel)*100:.2f}%, |rel| p95 {np.percentile(np.abs(rel),95)*100:.2f}%")
    s, i = np.polyfit(x, r, 1)
    print(f"  residual vs L: slope {s:.4f} ms/token, intercept {i:.1f} ms")
    e = np.median(r)
    rel2 = (y - p - e) / y
    print(f"  with constant e={e:.1f} ms: |rel| median {np.median(np.abs(rel2))*100:.2f}%, p95 {np.percentile(np.abs(rel2),95)*100:.2f}%, max {np.max(np.abs(rel2))*100:.2f}%")
    edges = np.unique(np.percentile(x, [0, 10, 25, 50, 75, 90, 100]).round())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum():
            print(f"    L {lo:.0f}-{hi:.0f}: n={m.sum()} measured med {np.median(y[m]):.1f} predicted med {np.median(p[m]):.1f} (+e {np.median(p[m])+e:.1f})")
    return x, y

check("list only, real prompts", lst)
check("thumbnail tier, real prompts (image tokens included in L)", mm)
for sc in ("ed", "td"):
    check(f"  thumbnail tier, {sc}", [d for d in mm if d["scenario"] == sc])
    check(f"  list only, {sc}", [d for d in lst if d["scenario"] == sc])

# 3) memory reported by the runtime
def mem(name, rows):
    x = np.array([d["prefill_tokens"] for d in rows], float); y = np.array([d["memory_mb"] for d in rows], float)
    s, i = np.polyfit(x, y, 1)
    print(f"\nmemory {name}: n={len(x)} min {y.min():.1f} max {y.max():.1f} MB; fit {i:.1f} + {s:.4f} L (MB), "
          f"resid max {np.max(np.abs(y-(i+s*x))):.1f} MB")
mem("synthetic", syn)
for L0 in Ls:
    ys = [d["memory_mb"] for d in syn if d["prefill_tokens"] == L0]
    print(f"   L={L0}: memory median {np.median(ys):.1f} min {min(ys):.1f} max {max(ys):.1f}")
mem("list only", lst)
mem("thumbnail tier", mm)

# 4) the first call at each synthetic length is slower than the rest (one-time cost of a new input shape)
by = {}
for d in syn: by.setdefault(d["prefill_tokens"], []).append(d["prefill_ms"])
print("\nfirst call minus median of the other nine:", {k: round(v[0] - float(np.median(v[1:])), 1) for k, v in sorted(by.items())})
