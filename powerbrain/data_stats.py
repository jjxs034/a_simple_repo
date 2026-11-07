# Image and box statistics of the source splits (all images, as bench.py loads them; boxes are [x, y, w, h, class]), for Table I.
#   cd $PB_CODE && nice -n 19 python - < data_stats.py      (on the machine that holds the ED and TD data)
import os, statistics, sys
sys.path.insert(0, os.environ.get("PB_CODE", "."))
import bench
td = bench.load_td()
for sc, splits in (("ed", {"train": bench.load_ed("train"), "test": bench.load_ed("test")}), ("td", td)):
    imgs = splits["train"] + splits["test"]
    mp = [i["width"] * i["height"] / 1e6 for i in imgs]
    share = [b[2] * b[3] / (i["width"] * i["height"]) for i in imgs for b in i["boxes"]]
    print(sc, "images train/test", len(splits["train"]), len(splits["test"]), "| megapixels median %.2f max %.2f" % (statistics.median(mp), max(mp)),
          "| boxes", len(share), "| median box share %.3f%%" % (100 * statistics.median(share)))
