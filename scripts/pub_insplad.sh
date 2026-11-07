#!/usr/bin/env bash
# Protocol P30: the public replication on InsPLAD-det, run on one 4-GPU node of an H100 cluster.
# Same question rules, detector, list wording, LoRA settings and evaluation as the final round; one seed; clean images only.
#   bash pub_insplad.sh            all steps; a step whose output exists is skipped, so a stopped job can be started again
# Inputs prepared on the dev machine before the job: $PUB/bench/pl_{fit,val,test}.jsonl and fold_groups.json (bench.py --scenarios pl),
# the code in $PUB/code (experiments/pb), the weights in $PUB/weights, the environment $ENV (packages as in requirements.txt).
set -u
PUB=${PB_PUB:-$PWD/pub}; C=$PUB/code; ENV=${PB_ENV:-$PUB/env}; PY=$ENV/bin/python
B=$PUB/bench; G=$PUB/regions; R=$PUB/results; L=$PUB/lora; DD=$PUB/det_data; DT=$PUB/detectors; LOG=$PUB/logs
M4=$PUB/weights/Qwen3-VL-4B-Instruct
export PB_PL_ROOT=$PUB/data PB_SCENARIOS=pl PB_FOLD_GROUPS=$B/fold_groups.json OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_CONFIG_DIR=$PUB/ultralytics
mkdir -p $G $R $L $DD $DT $LOG $YOLO_CONFIG_DIR
step() { echo "[p30 $(date +%m-%d\ %T)] $*" | tee -a $LOG/p30.log; }
cd $C
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | tee -a $LOG/p30.log
$PY -c "import torch, transformers, peft, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'transformers', transformers.__version__, 'peft', peft.__version__, 'ultralytics', ultralytics.__version__)" | tee -a $LOG/p30.log

# 1. detector data: the full detector (fit images; validation images held out) and the two cross-fitting folds (grouped by tower)
for k in -1 0 1; do
  name=pl; [ $k -ge 0 ] && name=pl_cf$k
  [ -s $DD/$name/data.yaml ] || $PY detector.py prep --scenario pl --holdout-fold $k --out-dir $DD >> $LOG/p30.log 2>&1
done
step "detector data: $(for n in pl pl_cf0 pl_cf1; do echo -n "$n $(ls $DD/$n/images/train | wc -l)/$(ls $DD/$n/images/val | wc -l) "; done)"

# 2. three detectors at once (YOLO11m, longer side 1280, 30 epochs as the final round's medium detector of the larger scenario)
g=0  # ultralytics runs its AMP check with yolo11n.pt from the working directory ($C): copied there, nothing is downloaded
for name in pl pl_cf0 pl_cf1; do
  [ -s $DT/$name/weights/best.pt ] || ( cd $C && $PY detector.py train --scenario pl --name $name --data-dir $DD --weights $PUB/weights/yolo11m.pt --project $DT \
      --devices $g --epochs 30 --batch 16 > $LOG/det_train_$name.log 2>&1 ) &
  g=$((g + 1))
done
wait
cd $C
for name in pl pl_cf0 pl_cf1; do [ -s $DT/$name/weights/best.pt ] || { step "DETECTOR $name FAILED"; exit 1; }; done
step "detectors trained"

# 3. detections: the full detector on the evaluation images (test and validation) and, for the in-sample control, on its own
#    training images; each fold detector on the fit images it never saw
$PY - <<PY
import json, sys
sys.path.insert(0, "$C")
import head
fit = json.load(open("$DD/pl/images.json"))["train"]
for k in (0, 1):
    seen = set(json.load(open("$DD/pl_cf%d/images.json" % k))["train"])
    held = [p for p in fit if p not in seen]
    assert all(head.fold_of(p, 2) == k for p in held)
    open("$DT/heldout_cf%d.txt" % k, "w").write("\n".join(held) + "\n")
open("$DT/fit_all.txt", "w").write("\n".join(fit) + "\n")
print("held-out fit images per fold:", [sum(1 for _ in open("$DT/heldout_cf%d.txt" % k)) for k in (0, 1)], "fit images:", len(fit))
PY
det() {  # <gpu> <weights> <out> <detector.py detect arguments...>
  local gpu=$1 w=$2 out=$3; shift 3
  [ -s $out ] || ( CUDA_VISIBLE_DEVICES=$gpu $PY detector.py detect --scenario pl --weights $w --out $out.part --devices 0 "$@" > $LOG/det_$(basename $out .jsonl).log 2>&1 && mv $out.part $out ) &
}
det 0 $DT/pl/weights/best.pt $DT/pl_det_eval.jsonl --items $B/pl_test.jsonl $B/pl_val.jsonl
det 1 $DT/pl/weights/best.pt $DT/pl_det_ins.jsonl --image-list $DT/fit_all.txt
det 2 $DT/pl_cf0/weights/best.pt $DT/pl_det_cf0.jsonl --image-list $DT/heldout_cf0.txt
det 3 $DT/pl_cf1/weights/best.pt $DT/pl_det_cf1.jsonl --image-list $DT/heldout_cf1.txt
wait
for f in eval ins cf0 cf1; do [ -s $DT/pl_det_$f.jsonl ] || { step "DETECTION $f FAILED"; exit 1; }; done
step "detections: $(for f in eval ins cf0 cf1; do echo -n "$f $(wc -l < $DT/pl_det_$f.jsonl) "; done)"

# 4. candidates and evidence files (CPU): out-of-fold lists for training (the method), in-sample and human-box lists (the two controls)
[ -s $G/pl_cands_eval.jsonl ] || $PY verify.py cands --detections $DT/pl_det_eval.jsonl --out $G/pl_cands_eval.jsonl >> $LOG/p30.log 2>&1
[ -s $G/pl_cands_fit.jsonl ] || $PY verify.py cands --detections $DT/pl_det_cf0.jsonl $DT/pl_det_cf1.jsonl --out $G/pl_cands_fit.jsonl >> $LOG/p30.log 2>&1
[ -s $G/pl_cands_ins.jsonl ] || $PY verify.py cands --detections $DT/pl_det_ins.jsonl --out $G/pl_cands_ins.jsonl >> $LOG/p30.log 2>&1
mk() { local out=$B/f_evid_$1.jsonl; shift; [ -s $out ] || { $PY verify.py evidence --mode text --require-cands --fmt 3 --thumb-tokens 576 "$@" --out $out.part >> $LOG/p30.log 2>&1 && mv $out.part $out; }; }
mk pl_eval_m576 --cands $G/pl_cands_eval.jsonl --items $B/pl_test.jsonl $B/pl_val.jsonl
mk pl_fit_m576 --cands $G/pl_cands_fit.jsonl --items $B/pl_fit.jsonl
mk pl_fit_m576_ins --cands $G/pl_cands_ins.jsonl --items $B/pl_fit.jsonl
mk pl_fit_m576_gt --from-gt --items $B/pl_fit.jsonl
for f in pl_eval_m576 pl_fit_m576 pl_fit_m576_ins pl_fit_m576_gt; do [ -s $B/f_evid_$f.jsonl ] || { step "EVIDENCE $f FAILED"; exit 1; }; done
step "evidence files ready"

# 5. four adapters at once, one GPU each (effective batch 4 = 1 GPU x 4 accumulation steps; LoRA defaults of train_lora.py)
train() {  # <gpu> <name> <format> [--selector-regions evidence]
  local gpu=$1 name=$2 fmt=$3; shift 3
  [ -f $L/$name/adapter_model.safetensors ] || ( CUDA_VISIBLE_DEVICES=$gpu $ENV/bin/accelerate launch --num_processes 1 --mixed_precision bf16 train_lora.py \
      --items $B/pl_fit.jsonl --model $M4 --out $L/$name --formats $fmt --accumulate 4 --seed 0 "$@" > $LOG/train_$name.log 2>&1 ) &
}
train 0 pl_one576 pack --selector-regions $B/f_evid_pl_fit_m576.jsonl
train 1 pl_abl_ins pack --selector-regions $B/f_evid_pl_fit_m576_ins.jsonl
train 2 pl_abl_gt pack --selector-regions $B/f_evid_pl_fit_m576_gt.jsonl
train 3 pl_w576 u576only
wait
for n in pl_one576 pl_abl_ins pl_abl_gt pl_w576; do [ -f $L/$n/adapter_model.safetensors ] || { step "TRAINING $n FAILED"; exit 1; }; done
step "adapters trained"

# 6. the detector with rules (thresholds tuned on the validation questions), then the four models on the test questions
[ -s $R/pl_detector.jsonl ] || $PY detector.py answer --detections $DT/pl_det_eval.jsonl --items $B/pl_test.jsonl --tune-items $B/pl_val.jsonl --out $R/pl_detector.jsonl > $LOG/pl_detector.log 2>&1
ev() {  # <gpu> <name> <policy> [--regions ...]
  local gpu=$1 name=$2 pol=$3; shift 3
  mkdir -p $R/$name
  [ -s $R/$name/shard0.jsonl ] && [ $(wc -l < $R/$name/shard0.jsonl) -ge $(wc -l < $B/pl_test.jsonl) ] || \
    ( CUDA_VISIBLE_DEVICES=$gpu $PY evaluate.py --items $B/pl_test.jsonl --out-dir $R/$name --model $M4 --lora $L/$name --policies $pol "$@" --shard 0 --num-shards 1 > $LOG/eval_$name.log 2>&1 ) &
}
ev 0 pl_one576 g:e --regions e=$B/f_evid_pl_eval_m576.jsonl
ev 1 pl_abl_ins g:e --regions e=$B/f_evid_pl_eval_m576.jsonl
ev 2 pl_abl_gt g:e --regions e=$B/f_evid_pl_eval_m576.jsonl
ev 3 pl_w576 u576
wait
step "evaluated: $(for n in pl_one576 pl_abl_ins pl_abl_gt pl_w576; do echo -n "$n $(cat $R/$n/*.jsonl 2>/dev/null | wc -l) "; done)"
echo P30_DONE | tee -a $LOG/p30.log
