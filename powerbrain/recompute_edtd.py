#!/usr/bin/env python3
"""Recompute the ED/TD numbers of Tables II-IV, Fig. 4 and the strata and near-duplicate checks of the text from the de-identified per-question records alone
(data/edtd/questions.csv.gz, outputs.csv.gz, outputs_rules.csv.gz). Same arithmetic as maintable.py, robustreport.py,
xreport.py and openformreport.py: accuracies, seed means and standard deviations, and 95% intervals from 2,000 resamples of
test images (dupeffect.clustered_ci, seed 0, images in the order of the question files).

    python recompute_edtd.py --data ../data/edtd --out edtd_tables.json
"""
import argparse, collections, csv, gzip, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_x  # FOLDS and FORMS of the question-form experiment

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
D = args.data
Q = collections.defaultdict(list)
for r in csv.DictReader(gzip.open(os.path.join(D, "questions.csv.gz"), "rt", encoding="utf-8")):
    Q[r["file"]].append(r)
OUT = collections.defaultdict(list)
for name in ("outputs.csv.gz", "outputs_rules.csv.gz"):
    for r in csv.DictReader(gzip.open(os.path.join(D, name), "rt", encoding="utf-8")):
        OUT[r["run"]].append((r["policy"], r["qh"], float(r["correct"])))
def load_runs(runs, policy):
    seeds = []
    for run in runs:
        d = {}
        for pol, qh, c in OUT.get(run, []):
            if not policy or policy == pol:
                d[qh] = c
        seeds.append(d)
    return seeds
def ci(items, values, reps=2000, seed=0):
    by = collections.defaultdict(list)
    for it, v in zip(items, values):
        by[it["ih"]].append(v)
    s = np.array([sum(v) for v in by.values()], float); n = np.array([len(v) for v in by.values()], float)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
s3 = lambda n: [f"f_{n}_s0", f"f_{n}_s1", f"f_{n}_s2"]
RESULT = {}

def main_table(rows, ref):
    res = {}
    for sc in ("ed", "td"):
        items = Q[f"{sc}_test"]
        seeds_of = {name: load_runs(runs, pol) for name, runs, pol in rows}
        mean_ok = {name: {it["qh"]: float(np.mean([s[it["qh"]] for s in seeds])) for it in items}
                   for name, seeds in seeds_of.items() if all(it["qh"] in s for s in seeds for it in items)}
        for name, seeds in seeds_of.items():
            if name not in mean_ok:
                res[f"{sc}/{name}"] = "missing"; continue
            ok = mean_ok[name]; v = [ok[it["qh"]] for it in items]; lo, hi = ci(items, v)
            per_seed = [100 * np.mean([s[it["qh"]] for it in items]) for s in seeds]
            d = None
            if name != ref and ref in mean_ok:
                dd = [mean_ok[ref][it["qh"]] - ok[it["qh"]] for it in items]; dlo, dhi = ci(items, dd)
                d = [round(100 * float(np.mean(dd)), 4), round(100 * dlo, 4), round(100 * dhi, 4)]
            res[f"{sc}/{name}"] = {"accuracy": round(100 * float(np.mean(v)), 4), "ci": [round(100 * lo, 4), round(100 * hi, 4)],
                                   **{t: round(float(100 * np.mean([ok[it["qh"]] for it in items if it["form"] == t])), 4) for t in ("cls", "exist", "count")},
                                   "seeds": [round(float(x), 4) for x in per_seed],
                                   "sd": round(float(np.std(per_seed, ddof=1)), 4) if len(per_seed) > 1 else None, "ref_minus_this": d}
    return res
ROWS = [("检测分支按规则", ["detector_v2_test"], ""), ("检测分支 + 小汇总网络", ["aggregator_v1", "aggregator_v1_s1", "aggregator_v1_s2"], ""),
        ("整图 576 令牌（4B）", s3("w576"), "u576"), ("整图原始分辨率（4B）", s3("wnat"), "u8192"), ("只读清单（4B）", s3("list"), "g:e"),
        ("本文（4B）", s3("one576"), "g:e"), ("端侧配置，只读清单（2B，纳米分支）", s3("listn2b"), "g:e"), ("端侧配置，带缩略图（2B，纳米分支）", s3("onen2b"), "g:e"),
        ("缩略图 256 令牌", ["f_one256_s0"], "g:e"), ("标准文字清单", ["f_abl_w2"], "g:e"), ("不做退化感知的对齐", ["f_abl_noaug"], "g:e"),
        ("只退化图像，清单取干净图像", ["f_abl_imgaug"], "g:e"), ("样本内清单训练", ["f_abl_ins"], "g:e"), ("人工框清单训练", ["f_abl_gt"], "g:e"),
        ("最接近已有工作的接口", ["f_prior_s0"], "g:e"), ("2B，中型分支，带缩略图", ["f_one2b_s0"], "g:e"), ("2B，中型分支，只读清单", ["f_list2b_s0"], "g:e"),
        ("4B，纳米分支，带缩略图", ["f_onen_s0"], "g:e"), ("检测分支 + 小汇总网络（带退化的训练清单）", ["f_aggaug"], ""),
        ("整图 576 令牌（2B）", ["f_w576_2b"], "u576"), ("整图原始分辨率（2B）", ["f_wnat_2b"], "u8192"), ("整图原始分辨率，视觉编码器一并适配", ["f_wnat_vis"], "u8192"),
        ("整图原始分辨率，剪枝留 50%", ["f_wnat_s0_fv05"], "u8192"), ("整图原始分辨率，剪到 576 个令牌", ["f_wnat_s0_fv576"], "u8192"),
        ("未微调 4B，576 令牌", ["base"], "u576"), ("未微调 4B，原始分辨率", ["base"], "u8192")]
ROWS_P32 = [("本文（InternVL3.5-4B）", ["p32_ivl_one_s0"], "g:e"), ("整图一块 256 令牌（InternVL3.5-4B）", ["p32_ivl_w1_s0"], "i1"),
        ("整图动态切块（InternVL3.5-4B）", ["p32_ivl_w12_s0"], "i12"), ("未微调，一块（InternVL3.5-4B）", ["p32_ivl_zs"], "i1"),
        ("未微调，动态切块（InternVL3.5-4B）", ["p32_ivl_zs"], "i12"), ("本文（Qwen3-VL-4B）", s3("one576"), "g:e"),
        ("整图 576 令牌（Qwen3-VL-4B）", s3("w576"), "u576"), ("整图原始分辨率（Qwen3-VL-4B）", s3("wnat"), "u8192")]
RESULT["table2_and_table4"] = main_table(ROWS, "本文（4B）")
RESULT["table2_internvl"] = main_table(ROWS_P32, "本文（InternVL3.5-4B）")
SYS = [("检测分支按规则", "detector_v2_test", "detector_c{C}", ""), ("检测分支按规则，前置图像增强", "detector_enh", "detector_enh_c{C}", ""),
       ("检测分支 + 小汇总网络", "aggregator_v1", "aggregator_c{C}", ""), ("检测分支 + 小汇总网络（带退化的训练清单）", "f_aggaug", "f_aggaug_c{C}", ""),
       ("只读清单（4B）", "f_list_s0", "f_list_s0_c{C}", "g:e"), ("本文（4B）", "f_one576_s0", "f_one576_s0_c{C}", "g:e"),
       ("缩略图 256 令牌", "f_one256_s0", "f_one256_s0_c{C}", "g:e"), ("不做退化感知的对齐", "f_abl_noaug", "f_abl_noaug_c{C}", "g:e"),
       ("只退化图像，清单取干净图像", "f_abl_imgaug", "f_abl_imgaug_c{C}", "g:e"), ("端侧配置，只读清单（2B，纳米分支）", "f_listn2b_s0", "f_listn2b_s0_c{C}", "g:e"),
       ("端侧配置，带缩略图（2B，纳米分支）", "f_onen2b_s0", "f_onen2b_s0_c{C}", "g:e"), ("整图 576 令牌（4B）", "f_w576_s0", "f_w576_s0_c{C}", "u576"),
       ("整图原始分辨率（4B）", "f_wnat_s0", "f_wnat_s0_c{C}", "u8192")]
robust = {}
for kind, tags in (("seen", ["fog2", "dark2", "blur2", "noise2", "jpeg2", "rain2"]), ("heldout", ["motion2", "bright2", "lowres2"])):
    for sc in ("ed", "td"):
        items = Q[f"{sc}_test"]
        for name, clean, dirty, pol in SYS:
            c = load_runs([clean], pol)[0]
            if any(it["qh"] not in c for it in items):
                robust[f"{kind}/{sc}/{name}"] = "missing clean"; continue
            base = 100 * np.mean([c[it["qh"]] for it in items]); cells = {"干净": f"{base:.1f}"}; vals = []
            for t in tags:
                dd = load_runs([dirty.format(C=t)], pol)[0]
                if any(it["qh"] not in dd for it in items):
                    cells[t] = "missing"; continue
                a = 100 * np.mean([dd[it["qh"]] for it in items]); vals.append(a); cells[t] = f"{a:.1f}"
            if len(vals) == len(tags):
                cells["退化平均"] = f"{np.mean(vals):.1f}"; cells["平均下降"] = f"{base - np.mean(vals):.1f}"
            robust[f"{kind}/{sc}/{name}"] = cells
RESULT["fig4_and_table4_robustness"] = robust
r1 = lambda x: round(100 * x, 1)
def load(run, policy):
    return load_runs([run], policy)[0]
out = {"pairs": {}, "forms": {}, "open": {}}
SYS = {"检测分支按规则": ("detector_v2_test", "detector_c{C}", ""), "检测分支按规则，前置图像增强": ("detector_enh", "detector_enh_c{C}", ""),
       "检测分支 + 小汇总网络": ("aggregator_v1", "aggregator_c{C}", ""), "检测分支 + 小汇总网络（带退化的训练清单）": ("f_aggaug", "f_aggaug_c{C}", ""),
       "只读清单（4B）": ("f_list_s0", "f_list_s0_c{C}", "g:e"), "本文（4B）": ("f_one576_s0", "f_one576_s0_c{C}", "g:e"),
       "缩略图 256 令牌": ("f_one256_s0", "f_one256_s0_c{C}", "g:e"), "不做退化感知的对齐": ("f_abl_noaug", "f_abl_noaug_c{C}", "g:e"),
       "只退化图像，清单取干净图像": ("f_abl_imgaug", "f_abl_imgaug_c{C}", "g:e"), "端侧配置，只读清单（2B，纳米分支）": ("f_listn2b_s0", "f_listn2b_s0_c{C}", "g:e"),
       "端侧配置，带缩略图（2B，纳米分支）": ("f_onen2b_s0", "f_onen2b_s0_c{C}", "g:e"), "整图 576 令牌（4B）": ("f_w576_s0", "f_w576_s0_c{C}", "u576"),
       "整图原始分辨率（4B）": ("f_wnat_s0", "f_wnat_s0_c{C}", "u8192")}
RP = [("本文（4B）", "检测分支 + 小汇总网络"), ("本文（4B）", "检测分支 + 小汇总网络（带退化的训练清单）"), ("只读清单（4B）", "检测分支 + 小汇总网络（带退化的训练清单）"),
      ("检测分支 + 小汇总网络（带退化的训练清单）", "检测分支 + 小汇总网络"), ("本文（4B）", "只读清单（4B）"), ("本文（4B）", "不做退化感知的对齐"),
      ("本文（4B）", "只退化图像，清单取干净图像"), ("本文（4B）", "缩略图 256 令牌"), ("本文（4B）", "整图 576 令牌（4B）"), ("本文（4B）", "整图原始分辨率（4B）"),
      ("端侧配置，带缩略图（2B，纳米分支）", "端侧配置，只读清单（2B，纳米分支）"), ("整图 576 令牌（4B）", "检测分支按规则"),
      ("检测分支按规则，前置图像增强", "检测分支按规则"), ("本文（4B）", "检测分支按规则，前置图像增强")]
for kind, tags in (("seen", ["fog2", "dark2", "blur2", "noise2", "jpeg2", "rain2"]), ("heldout", ["motion2", "bright2", "lowres2"])):
    clean = {n: load(c, p) for n, (c, _, p) in SYS.items()}
    dirty = {n: {t: load(d.format(C=t), p) for t in tags} for n, (_, d, p) in SYS.items()}
    for sc in ("ed", "td"):
        items = Q[f"{sc}_test"]
        for a, b in RP:
            if not all(it["qh"] in dirty[a][t] and it["qh"] in dirty[b][t] for t in tags for it in items):
                continue
            pi, d, dd = [], [], []
            for t in tags:
                for it in items:
                    pi.append(it); d.append(dirty[a][t][it["qh"]] - dirty[b][t][it["qh"]])
                    dd.append((clean[b][it["qh"]] - dirty[b][t][it["qh"]]) - (clean[a][it["qh"]] - dirty[a][t][it["qh"]]))
            lo, hi = ci(pi, d); lo2, hi2 = ci(pi, dd)
            out["pairs"][f"{kind}/{sc}/{a} − {b}"] = {"退化图像上": [r1(np.mean(d)), r1(lo), r1(hi)], "前者少掉": [r1(np.mean(dd)), r1(lo2), r1(hi2)]}
# forms (rows with a language model)
half = {q: h for h, forms in bench_x.FOLDS.items() for q in forms}
items = {sc: Q[f"{sc}_testx"] + Q[f"{sc}_testx2"] for sc in ("ed", "td")}
def split(first, second, which):
    a, b = load(*first), load(*second); ok = {}
    for it in items["ed"] + items["td"]:
        tr = a if half[it["form"]] == "A" else b; other = b if half[it["form"]] == "A" else a
        src = other if which == "heldout" else tr
        if it["qh"] in src: ok[it["qh"]] = src[it["qh"]]
    return ok
rows = collections.OrderedDict()
rows["本文，只训练基础问法"] = load("f_one576_s0_x", "g:e")
rows["本文，多问法训练（该问法未参与训练）"] = split(("f_forms_oneA_x", "g:e"), ("f_forms_oneB_x", "g:e"), "heldout")
rows["整图 576 令牌，多问法训练（该问法未参与训练）"] = split(("sft_forms_uniA_x", "u576"), ("sft_forms_uniB_x", "u576"), "heldout")
rows["整图原始分辨率，多问法训练（该问法未参与训练）"] = split(("sft_forms_uniA_x", "u8192"), ("sft_forms_uniB_x", "u8192"), "heldout")
rows["本文，多问法训练（该问法参与了训练）"] = split(("f_forms_oneA_x", "g:e"), ("f_forms_oneB_x", "g:e"), "seen")
rows["整图原始分辨率，多问法训练（该问法参与了训练）"] = split(("sft_forms_uniA_x", "u8192"), ("sft_forms_uniB_x", "u8192"), "seen")
XTYPES = bench_x.FORMS["old"] + bench_x.FORMS["new"]
for sc in ("ed", "td"):
    sub = items[sc]; forms = [q for q in XTYPES if any(it["form"] == q for it in sub)]
    for name, ok in rows.items():
        if any(it["qh"] not in ok for it in sub):
            out["forms"][f"{sc}/{name}"] = "missing"; continue
        v = [ok[it["qh"]] for it in sub]; lo, hi = ci(sub, v)
        cells = {"全部（95% 区间，按图像重采样）": f"{100 * np.mean(v):.1f}（{100 * lo:.1f} 到 {100 * hi:.1f}）"}
        for q in forms:
            cells[f"{q[2:]}（{sum(it['form'] == q for it in sub)}）"] = f"{100 * np.mean([ok[it['qh']] for it in sub if it['form'] == q]):.1f}"
        out["forms"][f"{sc}/{name}"] = cells
# open-form (rows with a language model)
ORows = collections.OrderedDict([("整图 576 令牌", load("f_open_w576", "u576")), ("整图原始分辨率", load("f_open_wnat", "u8192")),
                                 ("端侧配置，只读清单（2B，纳米分支）", load("f_open_listn2b", "g:e")), ("本文（4B）", load("f_open_one", "g:e"))])
def cell(its, ok):
    v = [ok[it["qh"]] for it in its]; lo, hi = ci(its, v); return f"{100 * np.mean(v):.1f}（{100 * lo:.1f}–{100 * hi:.1f}）"
def diff(its, a, b):
    d = [a[it["qh"]] - b[it["qh"]] for it in its]; lo, hi = ci(its, d); return f"{100 * np.mean(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）"
for sc in ("ed", "td"):
    its = Q[f"{sc}_test_open"]; base = [it for it in its if it["form"] in ("cls", "exist", "count")]; inv = [it for it in its if it["form"] == "inventory"]
    ref = ORows["本文（4B）"]
    for name, ok in ORows.items():
        if any(it["qh"] not in ok for it in its):
            out["open"][f"{sc}/{name}"] = "missing"; continue
        c = {"三类基础题合计": cell(base, ok), "类别": f"{100 * np.mean([ok[it['qh']] for it in its if it['form'] == 'cls']):.1f}",
             "是否存在": f"{100 * np.mean([ok[it['qh']] for it in its if it['form'] == 'exist']):.1f}",
             "计数": f"{100 * np.mean([ok[it['qh']] for it in its if it['form'] == 'count']):.1f}", "清点": cell(inv, ok)}
        c["基础题：本文（4B） 减该行"] = "—" if name == "本文（4B）" else diff(base, ref, ok)
        c["清点题：本文（4B） 减该行"] = "—" if name == "本文（4B）" else diff(inv, ref, ok)
        out["open"][f"{sc}/{name}"] = c
RESULT["robustness_pairs"] = out["pairs"]; RESULT["table3_forms"] = out["forms"]; RESULT["table3_open"] = out["open"]
RU = {'rules_forms': load_runs(['rules_forms'], '')[0], 'rules_open': load_runs(['rules_open'], '')[0]}
ONE = load_runs(['f_open_one'], 'g:e')[0]
rule_out = {}
XT = bench_x.FORMS["old"] + bench_x.FORMS["new"]
for sc in ("ed", "td"):
    sub = Q[f"{sc}_testx"] + Q[f"{sc}_testx2"]; ok = RU["rules_forms"]
    v = [ok[it["qh"]] for it in sub]; lo, hi = ci(sub, v)
    c = {"全部（95% 区间，按图像重采样）": f"{100 * np.mean(v):.1f}（{100 * lo:.1f} 到 {100 * hi:.1f}）"}
    for q in [q for q in XT if any(it["form"] == q for it in sub)]:
        c[f"{q[2:]}（{sum(it['form'] == q for it in sub)}）"] = f"{100 * np.mean([ok[it['qh']] for it in sub if it['form'] == q]):.1f}"
    rule_out[f"forms/{sc}"] = c
    its = Q[f"{sc}_test_open"]; base = [it for it in its if it["form"] in ("cls", "exist", "count")]; inv = [it for it in its if it["form"] == "inventory"]; ok = RU["rules_open"]
    def cell(x):
        vv = [ok[it["qh"]] for it in x]; l, h = ci(x, vv); return f"{100 * np.mean(vv):.1f}（{100 * l:.1f}–{100 * h:.1f}）"
    def diff(x):
        d = [ONE[it["qh"]] - ok[it["qh"]] for it in x]; l, h = ci(x, d); return f"{100 * np.mean(d):+.1f}（{100 * l:+.1f} 到 {100 * h:+.1f}）"
    rule_out[f"open/{sc}"] = {"三类基础题合计": cell(base), "类别": f"{100 * np.mean([ok[it['qh']] for it in its if it['form'] == 'cls']):.1f}",
                         "是否存在": f"{100 * np.mean([ok[it['qh']] for it in its if it['form'] == 'exist']):.1f}",
                         "计数": f"{100 * np.mean([ok[it['qh']] for it in its if it['form'] == 'count']):.1f}", "清点": cell(inv),
                         "基础题：本文（4B） 减该行": diff(base), "清点题：本文（4B） 减该行": diff(inv)}
RESULT["table3_rule_rows"] = rule_out

# strata of Table II's text (strata.py) and the near-duplicate check (dupeffect.py), from the three derived fields
SROWS = [("本文（4B）", s3("one576"), "g:e"), ("整图原始分辨率（4B）", s3("wnat"), "u8192"), ("整图 576 令牌（4B）", s3("w576"), "u576"),
         ("检测分支 + 小汇总网络", ["aggregator_v1", "aggregator_v1_s1", "aggregator_v1_s2"], ""), ("检测分支按规则", ["detector_v2_test"], ""), ("只读清单（4B）", s3("list"), "g:e")]
SPAIRS = [("本文（4B）", "整图原始分辨率（4B）"), ("本文（4B）", "整图 576 令牌（4B）"), ("本文（4B）", "检测分支 + 小汇总网络"), ("本文（4B）", "只读清单（4B）"),
          ("整图原始分辨率（4B）", "整图 576 令牌（4B）")]
FRAME_BACK = {"<=1.5 MP": "≤1.5 百万像素", "1.5-4 MP": "1.5–4 百万像素", "4-6 MP": "4–6 百万像素", ">6 MP": ">6 百万像素"}
SIZE_ORDER = ["<0.2%", "0.2–0.5%", "0.5–1%", "1–5%", "≥5%"]
def pooled(runs, pol):
    seeds = load_runs(runs, pol)
    return {k: float(np.mean([s[k] for s in seeds])) for k in seeds[0] if all(k in s for s in seeds)}
srows = collections.OrderedDict((n, pooled(r, p)) for n, r, p in SROWS)
for by, keyf, order in (("size", lambda it: it["evidence_area_bin"], SIZE_ORDER),
                        ("pixels", lambda it: FRAME_BACK.get(it["frame_size_bin"], ""), ["≤1.5 百万像素", "1.5–4 百万像素", "4–6 百万像素", ">6 百万像素"]),
                        ("qtype", lambda it: it["form"], None)):
    sec = {}
    for sc in ("ed", "td"):
        items = Q[f"{sc}_test"]
        groups = collections.OrderedDict()
        for k in (order or sorted({keyf(it) for it in items})):
            sub = [it for it in items if keyf(it) == k]
            if sub: groups[k] = sub
        groups["全部"] = items
        cols = {k: f"{k}（{len(v)} 题，{len({it['ih'] for it in v})} 图）" for k, v in groups.items()}
        sec[f"{sc}/columns"] = list(cols.values())
        for name, ok in srows.items():
            sec[f"{sc}/{name}"] = {cols[k]: f"{100 * np.mean([ok[it['qh']] for it in sub]):.1f}" for k, sub in groups.items()}
        for a, b in SPAIRS:
            cells = {}
            for k, sub in groups.items():
                d = [srows[a][it["qh"]] - srows[b][it["qh"]] for it in sub]; lo, hi = ci(sub, d)
                cells[k] = [round(100 * float(np.mean(d)), 1), round(100 * lo, 1), round(100 * hi, 1)]
            sec[f"{sc}/{a} − {b}"] = cells
    RESULT[f"strata_{by}"] = sec
STRATA = ["无近重复", "相似场景（8 位以上）", "几乎相同（≤7 位）"]
DPAIRS = [("本文（4B）", "整图原始分辨率（4B）"), ("本文（4B）", "整图 576 令牌（4B）"), ("本文（4B）", "检测分支 + 小汇总网络"), ("本文（4B）", "检测分支按规则")]
dsec = {}
for sc in ("ed", "td"):
    items = Q[f"{sc}_test"]
    groups = [[it for it in items if it["near_duplicate"] == str(k)] for k in range(3)] + [items]
    dsec[f"{sc}/题数（图像数）"] = {c: f"{len(g)}（{len({it['ih'] for it in g})}）" for c, g in zip(STRATA + ["全部"], groups)}
    for name, ok in srows.items():
        dsec[f"{sc}/{name}"] = {c: (f"{100 * np.mean([ok[it['qh']] for it in g]):.1f}" if g else "—") for c, g in zip(STRATA + ["全部"], groups)}
    for a, b in DPAIRS:
        cells = {}
        for c, g in zip(STRATA + ["全部"], groups):
            d = [srows[a][it["qh"]] - srows[b][it["qh"]] for it in g]; lo, hi = ci(g, d)
            cells[c] = f"{100 * np.mean(d):+.1f}（{100 * lo:+.1f} 到 {100 * hi:+.1f}）" if d else "—"
        dsec[f"{sc}/{a} − {b}"] = cells
RESULT["dupeffect"] = dsec
json.dump(RESULT, open(args.out, "w"), ensure_ascii=False, indent=1)
print("written", args.out)
