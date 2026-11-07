"""Consistency audit of the final round, run before the numbers are frozen. It reads only files on disk and reports
every deviation; nothing is repaired.

  training    every adapter f_*: effective batch (processes x accumulation) 4, one epoch, rank 16, learning rate 1e-4,
              the seed its name says, and the number of training questions of its question files
  results     every result folder f_*: each policy answers every question of its question files exactly once, no
              question outside them, correctness is 0 or 1; the rows derived by lookrows.py have as many questions as
              the stored two-step outputs they come from
  seeds       the seeds of one row answer the same set of questions

    python audit_final.py --bench <bench dir> --lora <dir> --results <dir> [--prefix f_]
    python audit_final.py selftest
Exit status 1 if anything deviates.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

MODES = ("never", "always", "self", "t10", "t20")
EXPECT = {"accumulate_x_processes": 4, "epochs": 1.0, "rank": 16, "lr": 1e-4}


def ids_of(paths):
    out = set()
    for path in paths:
        for line in open(path):
            out.add(json.loads(line)["id"])
    return out


def question_files(bench, name):
    """The question files a result folder was evaluated on, from the naming convention of final_round.sh."""
    b = lambda *names: [os.path.join(bench, n) for n in names]
    if re.search(r"_c[a-z]+\d$", name):  # a degradation: the test questions with a "corrupt" field, ids unchanged
        return b("ed_test.jsonl", "td_test.jsonl")
    if name.endswith("_val"):
        return b("ed_val_open.jsonl", "td_val_open.jsonl") if "_open_" in name else b("ed_val.jsonl", "td_val.jsonl")
    if name.endswith("_x"):
        return b("ed_testx.jsonl", "td_testx.jsonl", "ed_testx2.jsonl", "td_testx2.jsonl")
    if "_open_" in name:
        return b("ed_test_open.jsonl", "td_test_open.jsonl")
    m = re.match(r".*_loco_[a-z0-9]+_([A-Za-z]+)$", name)
    if m:
        return b(f"ed_test_only{m.group(1)}.jsonl")
    return b("ed_test.jsonl", "td_test.jsonl")


def audit_training(lora, prefix):
    problems, table = [], []
    for folder in sorted(glob.glob(os.path.join(lora, prefix + "*"))):
        name = os.path.basename(folder)
        path = os.path.join(folder, "train_args.json")
        if not os.path.exists(os.path.join(folder, "adapter_model.safetensors")):
            problems.append(f"{name}: no adapter (training unfinished or failed)")
            continue
        if not os.path.exists(path):
            problems.append(f"{name}: no train_args.json")
            continue
        a = json.load(open(path))
        got = {"accumulate_x_processes": a["accumulate"] * a["processes"], "epochs": a["epochs"], "rank": a["rank"], "lr": a["lr"]}
        for key, want in EXPECT.items():
            if got[key] != want:
                problems.append(f"{name}: {key} is {got[key]}, expected {want}")
        m = re.search(r"_s(\d)$", name)
        if a["seed"] != (int(m.group(1)) if m else 0):
            problems.append(f"{name}: seed {a['seed']} does not match the name")
        missing = [p for p in a["items"] if not os.path.exists(p)]
        if not missing:
            n = sum(1 for p in a["items"] for _ in open(p))
            if n != a["num_items"]:
                problems.append(f"{name}: trained on {a['num_items']} questions, its question files hold {n}")
        table.append((name, a["formats"], a["seed"], a["processes"], a["accumulate"], a["num_items"], "+".join(os.path.basename(p) for p in a["items"])))
    return problems, table


def audit_results(bench, results, prefix):
    problems, counts = [], {}
    cache = {}
    for folder in sorted(glob.glob(os.path.join(results, prefix + "*"))):
        name = os.path.basename(folder)
        if not os.path.isdir(folder) or name == "frozen":
            continue
        base, _, mode = name.rpartition(".")
        derived = mode in MODES and bool(base)
        files = tuple(question_files(bench, base if derived else name))
        if any(not os.path.exists(f) for f in files):
            problems.append(f"{name}: question files not found ({', '.join(os.path.basename(f) for f in files)})")
            continue
        if files not in cache:
            cache[files] = ids_of(files)
        want = cache[files]
        seen = collections.defaultdict(collections.Counter)
        for path in glob.glob(os.path.join(folder, "*shard*.jsonl")):
            for line in open(path):
                r = json.loads(line)
                seen[r["policy"]][r["id"]] += 1
                if r["correct"] not in (0, 1):
                    problems.append(f"{name}: correctness {r['correct']!r} for {r['id']}")
        if not seen:
            problems.append(f"{name}: no rows")
        for policy, c in seen.items():
            extra, missing, twice = set(c) - want, want - set(c), [i for i, k in c.items() if k > 1]
            if extra or missing or twice:
                problems.append(f"{name} [{policy}]: {len(missing)} questions missing, {len(extra)} outside the question files, {len(twice)} answered more than once")
            counts[(name, policy)] = set(c)
    for (name, policy), ids in counts.items():  # derived rows against their source
        base, _, mode = name.rpartition(".")
        if mode in MODES and (base, policy) in counts and counts[(base, policy)] != ids:
            problems.append(f"{name} [{policy}]: differs from the stored two-step outputs of {base}")
    groups = collections.defaultdict(dict)  # the seeds of one row
    for (name, policy), ids in counts.items():
        m = re.match(r"(.*)_s(\d)(.*)$", name)
        if m:
            groups[(m.group(1) + "_s*" + m.group(3), policy)][m.group(2)] = ids
    for (pattern, policy), by_seed in groups.items():
        if len({frozenset(v) for v in by_seed.values()}) > 1:
            problems.append(f"{pattern} [{policy}]: the seeds {sorted(by_seed)} do not answer the same questions")
    return problems, counts


def run(bench, lora, results, prefix):
    p1, table = audit_training(lora, prefix)
    p2, counts = audit_results(bench, results, prefix)
    print(f"{len(table)} adapters, {len(counts)} result rows checked")
    print("| 适配器 | 格式 | 种子 | 进程数 | 梯度累积 | 训练题数 | 训练题文件 |")
    print("|---|---|---:|---:|---:|---:|---|")
    for row in table:
        print("| " + " | ".join(str(v) for v in row) + " |")
    print()
    for line in p1 + p2:
        print("DEVIATION:", line)
    print("AUDIT_OK" if not p1 + p2 else f"AUDIT_FAILED: {len(p1 + p2)} deviations")
    return not p1 + p2


def selftest():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bench, lora, results = (os.path.join(tmp, d) for d in ("bench", "lora", "results"))
        os.makedirs(bench)
        write = lambda path, rows: (os.makedirs(os.path.dirname(path), exist_ok=True), open(path, "w").writelines(json.dumps(r) + "\n" for r in rows))
        for sc in ("ed", "td"):
            write(os.path.join(bench, f"{sc}_test.jsonl"), [{"id": f"{sc}-t{i}"} for i in range(5)])
            write(os.path.join(bench, f"{sc}_val.jsonl"), [{"id": f"{sc}-v{i}"} for i in range(3)])
            write(os.path.join(bench, f"{sc}_fit.jsonl"), [{"id": f"{sc}-f{i}"} for i in range(7)])
        fit = [os.path.join(bench, "ed_fit.jsonl"), os.path.join(bench, "td_fit.jsonl")]
        test_ids = [f"{sc}-t{i}" for sc in ("ed", "td") for i in range(5)]

        def adapter(name, **over):
            os.makedirs(os.path.join(lora, name))
            open(os.path.join(lora, name, "adapter_model.safetensors"), "w").close()
            args = {"accumulate": 2, "processes": 2, "epochs": 1.0, "rank": 16, "lr": 1e-4, "seed": 0, "items": fit, "num_items": 14, "formats": "look"}
            json.dump({**args, **over}, open(os.path.join(lora, name, "train_args.json"), "w"))

        rows = lambda ids, policy: [{"id": i, "policy": policy, "correct": 1} for i in ids]
        adapter("f_look_s0")
        adapter("f_look_s1", seed=1)
        write(os.path.join(results, "f_look_s0", "shard0.jsonl"), rows(test_ids, "look:e"))
        write(os.path.join(results, "f_look_s1", "shard0.jsonl"), rows(test_ids, "look:e"))
        write(os.path.join(results, "f_look_s0.self", "shard0.jsonl"), rows(test_ids, "look:e"))
        write(os.path.join(results, "f_look_s0_val", "shard0.jsonl"), rows([f"{sc}-v{i}" for sc in ("ed", "td") for i in range(3)], "look:e"))
        write(os.path.join(results, "f_look_s0_cfog2", "shard0.jsonl"), rows(test_ids, "look:e"))
        assert run(bench, lora, results, "f_")
        adapter("f_bad_s2", seed=0, accumulate=1, num_items=13)  # three deviations: seed, effective batch, number of questions
        write(os.path.join(results, "f_look_s1", "shard0.jsonl"), rows(test_ids[:-1] + ["ed-t0"], "look:e"))  # one missing, one twice; seeds differ
        write(os.path.join(results, "f_look_s0.self", "shard0.jsonl"), rows(test_ids[:-2], "look:e"))  # derived rows incomplete
        problems = audit_training(lora, "f_")[0] + audit_results(bench, results, "f_")[0]
        assert len(problems) == 7, problems
        assert not run(bench, lora, results, "f_")
    print("AUDIT_SELFTEST_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?")
    parser.add_argument("--bench")
    parser.add_argument("--lora")
    parser.add_argument("--results")
    parser.add_argument("--prefix", default="f_")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    sys.exit(0 if run(args.bench, args.lora, args.results, args.prefix) else 1)


if __name__ == "__main__":
    main()
