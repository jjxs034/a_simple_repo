"""Training cost of the final round, from metadata only (run on the training server with nice -n 19; reads no data):
  LoRA runs   GPUs and questions from <lora>/<name>/train_args.json, trainable parameters from the PEFT line of
              logs/train_<name>.log, wall time from the creation of that log to the write of adapter_model.safetensors;
  detectors   epochs, batch and devices from args.yaml, training time from the last "time" of results.csv.
GPU hours = wall time x GPUs. Record: docs/results/train_cost.md."""
import glob, json, os, re, sys
W = os.environ.get("PB_W", ".")
L = os.path.join(os.environ.get("PB_E", "."), "lora")
rows = []
for d in sorted(glob.glob(f"{L}/f_*")):
    name = os.path.basename(d)
    ad, ta, lg = f"{d}/adapter_model.safetensors", f"{d}/train_args.json", f"{W}/logs/train_{name}.log"
    if not (os.path.exists(ad) and os.path.exists(ta) and os.path.exists(lg)):
        continue
    args = json.load(open(ta))
    text = open(lg, errors="replace").read()
    m = re.search(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+)", text)
    born = os.stat(lg).st_birthtime if hasattr(os.stat(lg), "st_birthtime") else None
    if born is None:
        import subprocess
        born = int(subprocess.run(["stat", "-c", "%W", lg], capture_output=True, text=True).stdout.strip() or 0) or None
    end = os.path.getmtime(ad)
    hours = (end - born) / 3600 if born else float("nan")
    rows.append((name, args.get("processes"), args.get("num_items"), os.path.basename(str(args.get("model", ""))),
                 m.group(1) if m else "-", m.group(2) if m else "-", hours, int(text.count("TRAIN_DONE")), bool(args.get("vision_lora"))))
print("name\tgpus\titems\tmodel\ttrainable\tall\twall_h\tdone\tvision_lora")
for r in rows:
    print("\t".join(str(x) if not isinstance(x, float) else f"{x:.2f}" for x in r))


import csv, glob, json, os, re, subprocess
W = os.environ.get("PB_W", ".")
E = os.environ.get("PB_E", ".")
born = lambda p: int(subprocess.run(["stat", "-c", "%W", p], capture_output=True, text=True).stdout.strip() or 0)
print("== InternVL adapters")
for ta in sorted(glob.glob(f"{E}/lora/*/train_args.json")):
    a = json.load(open(ta))
    if "internvl" not in str(a.get("model", "")).lower():
        continue
    d = os.path.dirname(ta); name = os.path.basename(d)
    logs = [p for p in glob.glob(f"{W}/logs/*{name}*.log") if os.path.basename(p).startswith("train")]
    lg = logs[0] if logs else None
    text = open(lg, errors="replace").read() if lg else ""
    m = re.search(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+)", text)
    end = os.path.getmtime(f"{d}/adapter_model.safetensors")
    h = (end - born(lg)) / 3600 if lg and born(lg) else float("nan")
    print(name, a.get("processes"), a.get("num_items"), os.path.basename(a["model"]), m.group(1) if m else "-", m.group(2) if m else "-", f"{h:.2f}", os.path.basename(lg) if lg else "-")
print("== detectors")
for args_yaml in sorted(glob.glob(f"{E}/detectors_v2*/*/args.yaml") + glob.glob(f"{E}/detectors_cf*/*/args.yaml")):
    d = os.path.dirname(args_yaml)
    y = open(args_yaml).read()
    get = lambda k: (re.search(rf"^{k}: (.*)$", y, re.M) or [None, "?"])[1]
    res = f"{d}/results.csv"
    rows = list(csv.DictReader(open(res))) if os.path.exists(res) else []
    t = rows[-1].get("time") if rows and "time" in rows[-1] else None
    t = float(t) / 3600 if t else (os.path.getmtime(res) - born(args_yaml)) / 3600 if rows and born(args_yaml) else float("nan")
    print(d.replace(E + "/", ""), "model", os.path.basename(get("model")), "epochs", get("epochs"), "imgsz", get("imgsz"), "batch", get("batch"), "device", get("device"), "done_epochs", len(rows), f"hours {t:.2f}")
