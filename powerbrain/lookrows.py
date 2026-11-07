"""Ordinary result rows from the stored outputs of a look-on-doubt model (look.py).

evaluate.py stores both steps of every question (policy look:<tag>), so that the gate can be studied offline. For one
model this script writes one result folder per evaluated run and operating mode, in the format every report reads
(dupeffect.load_row, maintable.py, robustreport.py, xreport.py, openformreport.py):

  never    step 1 only: its best option; for written answers its text, a request for the image counting as wrong
  always   step 2 for every question
  self     the model's own decision: step 2 when step 1 answers X          <- the row reported as the method
  t10/t20  step 2 when P(X) exceeds the threshold at which 10% / 20% of the clean validation questions are looked at
           (per scenario; from the validation run <name>_val, never from test data)
  c05/c10/c20   step 2 when P(X) exceeds 0.05 / 0.10 / 0.20, thresholds fixed in advance

    python lookrows.py --results <dir> --name f_look_s0 --suffixes "" _val _cfog2 _x [--policy look:e]
        -> <dir>/f_look_s0.self/shard0.jsonl, <dir>/f_look_s0_cfog2.self/shard0.jsonl, ...
    python lookrows.py selftest

A row keeps id, policy, scenario, qtype, target, bin and truth; pred / correct of the step that is used; looked (0/1);
visual_tokens and input_tokens of the computation actually needed (step 2 continues the sequence of step 1, so its
length is the total); seconds = step 1 plus, when looked, the full forward of step 2. The time is an upper bound,
because the stored step 2 does not reuse the cache of step 1; the costs in the paper are measured separately.
"""
import argparse
import glob
import json
import math
import os

MODES = ("never", "always", "self", "t10", "t20", "c05", "c10", "c20")
SHARE = {"t10": 0.10, "t20": 0.20}
FIXED = {"c05": 0.05, "c10": 0.10, "c20": 0.20}  # look when P(X) exceeds a threshold fixed in advance (no data involved)
KEEP = ("id", "policy", "scenario", "qtype", "target", "bin", "truth")


def load(path, policy):
    rows = {}
    for f in glob.glob(os.path.join(path, "*shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("policy") == policy:
                rows[r["id"]] = r
    return rows


def quantile(values, q):
    """Linear interpolation between order statistics (the default of numpy.quantile)."""
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def thresholds(val_rows):
    """{(scenario, mode): threshold on P(X)} from the clean validation questions."""
    out = {}
    for sc in sorted({r["scenario"] for r in val_rows.values()}):
        p = [r["p_look"] for r in val_rows.values() if r["scenario"] == sc]
        for mode, share in SHARE.items():
            out[(sc, mode)] = quantile(p, 1 - share)
    return out


def own_decision(row):
    """Does step 1 itself ask for the image? Written answers: the text starts with X. Option letters: X is more
    probable than every option."""
    if "look1" in row:
        return bool(row["look1"])
    return row["p_look"] > max(math.exp(v) for v in row["logp1"])


def looks(row, mode, thr):
    if mode in ("never", "always"):
        return mode == "always"
    if mode == "self":
        return own_decision(row)
    if mode in FIXED:
        return row["p_look"] > FIXED[mode]
    return row["p_look"] > thr[(row["scenario"], mode)]


def derive(row, mode, thr):
    look = looks(row, mode, thr)
    out = {k: row[k] for k in KEEP if k in row}
    out.update({
        "pred": row["pred"] if look else row["pred1"], "correct": row["correct"] if look else row["correct1"], "looked": int(look),
        "visual_tokens": row["visual_tokens"] if look else 0, "input_tokens": row["input_tokens"] if look else row["input_tokens1"],
        "seconds": round(row["seconds1"] + (row["seconds"] if look else 0.0), 4),
    })
    return out


def write(results, name, suffix, policy, thr, modes=MODES):
    rows = load(os.path.join(results, name + suffix), policy)
    if not rows:
        print(f"{name}{suffix}: no rows with policy {policy}")
        return {}
    summary = {}
    for mode in modes:
        if mode in SHARE and not thr:
            continue
        derived = [derive(r, mode, thr) for r in rows.values()]
        target = os.path.join(results, f"{name}{suffix}.{mode}")
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "shard0.jsonl"), "w") as f:
            f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in derived)
        for sc in sorted({d["scenario"] for d in derived}):
            sub = [d for d in derived if d["scenario"] == sc]
            summary[(sc, mode)] = (100 * sum(d["correct"] for d in sub) / len(sub), 100 * sum(d["looked"] for d in sub) / len(sub), len(sub))
    scenarios = sorted({sc for sc, _ in summary})
    print(f"{name}{suffix} ({', '.join(f'{sc} {summary[(sc, MODES[0])][2]}' for sc in scenarios)} questions; accuracy / share looked at): " + "; ".join(
        f"{mode} " + " | ".join(f"{summary[(sc, mode)][0]:.1f} / {summary[(sc, mode)][1]:.0f}%" for sc in scenarios) for mode in modes if (scenarios[0], mode) in summary))
    return summary


def selftest():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        def put(run, rows):
            os.makedirs(os.path.join(tmp, run))
            with open(os.path.join(tmp, run, "shard0.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)

        def row(i, p_look, best, correct1, correct):  # two options; `best` is the probability of the better one
            rest = 1 - p_look
            return {"id": f"q{i}", "policy": "look:e", "scenario": "ed", "qtype": "exist", "target": "t", "bin": "b", "truth": "是", "pred1": "是" if correct1 else "否",
                    "pred": "是" if correct else "否", "correct1": correct1, "correct": correct, "p_look": p_look,
                    "logp1": [math.log(rest * best), math.log(rest * (1 - best))], "visual_tokens": 256, "input_tokens": 400, "input_tokens1": 90, "seconds": 0.05, "seconds1": 0.02}

        put("m_val", [row(i, i / 100, 0.9, 1, 1) for i in range(100)])  # P(X) = 0.00 ... 0.99
        put("m", [row(0, 0.05, 0.9, 1, 1), row(1, 0.60, 0.9, 0, 1), row(2, 0.85, 0.9, 0, 1), row(3, 0.95, 0.9, 1, 0),
                  {"id": "q4", "policy": "look:e", "scenario": "ed", "qtype": "count", "target": "t", "bin": "b", "truth": "2", "pred1": "X", "pred": "2", "correct1": 0, "correct": 1,
                   "look1": 1, "p_look": 0.4, "text": "2", "visual_tokens": 256, "input_tokens": 400, "input_tokens1": 90, "seconds": 0.05, "seconds1": 0.02},
                  {"id": "q5", "policy": "g:e", "scenario": "ed", "qtype": "count", "correct": 1}])
        thr = thresholds(load(os.path.join(tmp, "m_val"), "look:e"))
        assert abs(thr[("ed", "t10")] - 0.891) < 1e-9 and abs(thr[("ed", "t20")] - 0.792) < 1e-9, thr
        summary = write(tmp, "m", "", "look:e", thr)
        get = lambda mode: load(os.path.join(tmp, "m." + mode), "look:e")
        assert set(get("self")) == {"q0", "q1", "q2", "q3", "q4"}  # the row of another policy is left out
        assert [get("never")[f"q{i}"]["correct"] for i in range(5)] == [1, 0, 0, 1, 0] and all(r["looked"] == 0 and r["visual_tokens"] == 0 for r in get("never").values())
        assert [get("always")[f"q{i}"]["correct"] for i in range(5)] == [1, 1, 1, 0, 1] and all(r["looked"] == 1 for r in get("always").values())
        # own decision: X beats the best option (0.9 of the rest) when P(X) > 0.9 (1 - P(X)), that is P(X) > 0.474; the written answer by its text
        assert [get("self")[f"q{i}"]["looked"] for i in range(5)] == [0, 1, 1, 1, 1] and [get("self")[f"q{i}"]["correct"] for i in range(5)] == [1, 1, 1, 0, 1]
        assert [get("t10")[f"q{i}"]["looked"] for i in range(5)] == [0, 0, 0, 1, 0] and [get("t20")[f"q{i}"]["looked"] for i in range(5)] == [0, 0, 1, 1, 0]
        one = get("self")["q1"]
        assert one["input_tokens"] == 400 and one["visual_tokens"] == 256 and abs(one["seconds"] - 0.07) < 1e-9 and get("self")["q0"]["input_tokens"] == 90
        assert summary[("ed", "self")] == (80.0, 80.0, 5)
        assert [get("c10")[f"q{i}"]["looked"] for i in range(5)] == [0, 1, 1, 1, 1] and [get("c20")[f"q{i}"]["looked"] for i in range(5)] == [0, 1, 1, 1, 1]
        assert write(tmp, "m", "", "look:e", {}).keys() == {("ed", m) for m in ("never", "always", "self", "c05", "c10", "c20")}  # without a validation run: no share modes
    print("LOOKROWS_SELFTEST_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?")
    parser.add_argument("--results")
    parser.add_argument("--name")
    parser.add_argument("--suffixes", nargs="*", default=[""])
    parser.add_argument("--policy", default="look:e")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    thr = thresholds(load(os.path.join(args.results, args.name + "_val"), args.policy))
    if not thr:
        print(f"{args.name}_val: no validation run, the threshold modes are skipped")
    for suffix in args.suffixes:
        write(args.results, args.name, suffix, args.policy, thr)


if __name__ == "__main__":
    main()
