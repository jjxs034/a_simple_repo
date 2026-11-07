# Size distribution of the ED and TD images as bench.py loads them, for the resolution sentences of Section III-A (aggregates only).
#   cd $PB_CODE && nice -n 19 python - < size_dist.py      (on the machine that holds the ED and TD data)
import collections, os, sys
sys.path.insert(0, os.environ.get("PB_CODE", "."))
import bench
td = bench.load_td()
sets = {"ed": {"train": bench.load_ed("train"), "test": bench.load_ed("test")}, "td": td}
edges = [(0, 1.5), (1.5, 4), (4, 6), (6, 8.3), (8.3, 1e9)]
for sc, splits in sets.items():
    for part in ("all", "test"):
        imgs = splits["train"] + splits["test"] if part == "all" else splits["test"]
        mp = [i["width"] * i["height"] / 1e6 for i in imgs]
        n = len(mp)
        cnt = [sum(1 for m in mp if lo < m <= hi) for lo, hi in edges]
        sizes = collections.Counter((i["width"], i["height"]) for i in imgs).most_common(6)
        print(sc, part, "n", n, "| MP bins <=1.5, 1.5-4, 4-6, 6-8.3, >8.3:", cnt, [round(100 * c / n, 1) for c in cnt],
              "| <=1080p %.1f%%" % (100 * sum(1 for m in mp if m <= 2.0736) / n), "| min %.2f max %.2f" % (min(mp), max(mp)), "| top sizes", sizes)
