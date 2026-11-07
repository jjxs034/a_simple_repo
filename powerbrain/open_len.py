# Length of the open answers (generated tokens) in the frozen open-ended runs, by scenario and question type (aggregates only).
import collections, glob, json, os, statistics
R = os.path.join(os.environ.get("PB_E", "."), "results")
for run in ("f_open_one", "f_open_listn2b"):
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{R}/{run}/shard*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            n = r.get("generated_tokens")
            if n is None:
                continue
            groups[(r.get("scenario"), r.get("qtype"))].append(n)
            groups[(r.get("scenario"), "ALL")].append(n)
            groups[("ALL", "ALL")].append(n)
    print(run)
    for k in sorted(groups, key=str):
        v = sorted(groups[k])
        p90 = v[int(0.9 * (len(v) - 1))]
        print(f"  {k[0]!s:4} {k[1]!s:12} n={len(v):5d} median={statistics.median(v):4.1f} mean={statistics.mean(v):5.2f} p90={p90} max={v[-1]}")
