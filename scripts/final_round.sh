#!/usr/bin/env bash
# 最终一轮（设定见 docs/results/round2_protocol.md 的 P16、P23、P27 与 P28）：论文里的每个数字都出自同一套设置。
# 全部新产物以 f_ 开头（适配器 $L/f_*，结果 $R/f_*，证据文件 $B/f_evid_*，带退化的训练题 $B/f_aug/），不覆盖此前任何结果。
# 方法行是单步模型（训练格式 pack：检测分支的候选清单加一幅缩略图，一次作答）。「存疑再看」两轮都未过线（P21、P26），不在这一轮里；
# 它的几个函数（mkl、tvl、robust_look）留着，只在作者决定报告「少看图」的档位时才用。
#
#   bash final_round.sh config KEY=VALUE ...    写定主配置，此后各命令都读它。可以分几次写（先写 AUG、AUGBASE 让整图基线先跑；再写 FMT、STATE；
#                                               THUMB 等 thumb 组读完、按 P27 第 4 条判定后再写）；已写定的项不能改成别的值：
#         FMT      清单写法：2 文字，3 紧凑文字（单步模型不支持证据令牌）
#         THUMB    本文（4B）缩略图的令牌数（256 或 576）
#         STATE    只能是 0：单步模型不带分支状态令牌
#         AUG      训练图像的退化方案（如 aug_s0：三成训练图像加合成退化，种子 0；- 表示不用）
#         AUGBASE  1 整图基线也用同一份带退化的训练图像，0 整图基线用干净训练图像
#   bash final_round.sh jobs [组...]            打印任务表，追加到任务表文件即入队。组：prep base thumb tiers main ablation robust forms open loco large cost
#                                               （prep、base、large 只用到 AUG 与 AUGBASE；thumb、tiers 还要 FMT、STATE；其余各组还要 THUMB）
#                                               （cost 组测单卡耗时，须等其余任务都跑完、机器空闲时单独入队，用一张显卡的一个工作进程跑）
#                                               一行一个任务：名称[@依赖1,依赖2]<TAB>命令；依赖全部完成后才会被认领
#   bash final_round.sh worker "0 1 2 3" A      工作进程：反复认领任务表里下一个可跑的任务，用给定的显卡跑到没有任务为止
#                                               （PB_JOBS 可另指一份任务表，完成标记是共用的：准备步骤占显存很少，可以单开一个工作进程与训练共用显卡）
#   bash final_round.sh status [行数]
#
# 约定：每个训练的等效批量为 4（显卡数 × 梯度累积步数 = 4），一轮，其余设置同 train_lora.py 默认。
# prep 组要运行检测器（退化训练图像上的折外检测、分支状态），须在放着检测器权重的机器上跑；其余各组只需要题目、证据文件、图像与底座。
# 正在运行的工作进程读着这个文件：服务器上不要原地覆盖它，改过的版本换一个文件名上传（任务表里的命令只用到 train、tv、evalrow 等函数，新旧版本通用；
# 新增的函数只有新版本有，所以含新组的任务表要另存一份，只给用新版本启动的工作进程读）。
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
# 换一台机器时用环境变量改三处路径：PB_W（代码与日志）、PB_E（数据与产物）、PB_M4（4B 底座）；图像路径的映射见 core.local_path（PB_PATH_MAP）。
# 大显存的卡（如 80 GB）：一个工作进程给一张卡即可（单进程、梯度累积 4 步），同一张卡可以再开第二个工作进程；PB_EVAL_SHARDS 设每张卡的评测进程数。
W=${PB_W:-$HOME/powerbrain-exp}
E=${PB_E:-$HOME/powerbrain_exp}
B=$E/bench_v1; G=$E/regions; R=$E/results; L=$E/lora
F=$W/final; mkdir -p $F/claim $F/done $W/logs
JF=${PB_JOBS:-$F/jobs.txt}
PY=$W/.venv/bin/python
LOG=$W/logs/final_round.log
FMT=""; THUMB=""; STATE=""; AUG=""; AUGBASE=""
case $1 in jobs|worker|status) [ -f $F/config ] || { echo "run: final_round.sh config KEY=VALUE ... first"; exit 1; }; . $F/config ;; esac
case $1 in jobs|worker) [ -n "$AUG" ] && [ -n "$AUGBASE" ] || { echo "config: AUG and AUGBASE are not set"; exit 1; } ;; esac
need() { local k; for k in "$@"; do [ -n "${!k}" ] || { echo "config: $k is not set yet" >&2; return 1; }; done; }   # 主配置里这几项是否都已写定
structure() { need FMT STATE THUMB; }

FA=$B/f_aug                                                                           # 带退化的训练题文件、退化方案、交叉拟合各折的图像清单
aug() { local f out=""; for f in "$@"; do [ "$AUG" = - ] && out="$out $f" || out="$out $FA/$(basename ${f%.jsonl})@$AUG.jsonl"; done; echo $out; }   # 训练题文件 → 带退化的版本
FIT="$B/ed_fit.jsonl $B/td_fit.jsonl"; FITA=$(aug $FIT)
VAL="$B/ed_val.jsonl $B/td_val.jsonl"; TEST="$B/ed_test.jsonl $B/td_test.jsonl"
FITX="$B/ed_fitx.jsonl $B/td_fitx.jsonl"
XFORMS="$B/ed_testx.jsonl $B/td_testx.jsonl $B/ed_testx2.jsonl $B/td_testx2.jsonl"   # 十二种新问法的测试题
KINDS6="fog2 dark2 blur2 noise2 jpeg2 rain2"                                         # 六种已用过的退化
KINDS9="$KINDS6 motion2 bright2 lowres2"                                             # 加三种留出的（只在主配置全部写定之后评测）
LOCO="TaDiao ShiGongJiXie"
OFIT="$B/ed_fit_open.jsonl $B/td_fit_open.jsonl"; OTEST="$B/ed_test_open.jsonl $B/td_test_open.jsonl"; OVAL="$B/ed_val_open.jsonl $B/td_val_open.jsonl"   # 开放式题目（openform.py）
basefit() { [ "$AUGBASE" = 1 ] && aug "$@" || echo "$@"; }                           # 整图基线的训练题
# 候选文件。训练图像：折外（中型、纳米）、样本内；带 A 的是退化训练图像上的（后给的文件覆盖先给的，所以被选中退化的图用退化图像上的候选）
CM=$G/cands_fit_v1.jsonl; CN=$G/cands_fit_nano.jsonl; CI=$G/cands_insample_v1.jsonl
CMA=""; CNA=""; CIA=""; [ "$AUG" = - ] || { CMA=$G/f_cands_fit_m_$AUG.jsonl; CNA=$G/f_cands_fit_n_$AUG.jsonl; CIA=$G/f_cands_ins_$AUG.jsonl; }
EM=$G/cands_eval_v1.jsonl; EN=$G/cands_eval_nano.jsonl
SA=$([ "$AUG" = - ] && echo clean || echo aug)                                        # 训练图像的分支状态取哪一份
cd $W/pb
log() { echo "[final ${WORKER:-main} $(date +%m-%d\ %T)] $*" >> $LOG; }
ev() { echo $B/f_evid_$1.jsonl; }
model() {  # 底座路径：2b、4b（默认）、8b（只在大显存的卡上训练得动原始分辨率）
  case $1 in
    2b) echo $E/weights/Qwen3-VL-2B-Instruct ;;
    8b) echo ${PB_M8:-$E/weights/Qwen3-VL-8B-Instruct} ;;
    *)  echo ${PB_M4:-$E/weights/Qwen3-VL-4B-Instruct} ;;
  esac
}

# ---------------------------------------------------------------------------------------------- 证据文件
mk() {  # 单步模型的证据文件：<名称> <verify.py evidence 的参数...>（清单写法 --fmt 由调用处写明，这里不设默认值）
  local out; out=$(ev $1); shift
  [ -s $out ] && return 0
  $PY verify.py evidence --mode text --require-cands "$@" --out $out.part >> $LOG 2>&1 && mv $out.part $out
}
mkl() {  # 存疑再看的证据文件：<名称> <labels 或 -> <状态：检测分支 m|n，或 - 不带> <状态文件名，如 fit_aug、eval_clean、test_fog2> <题目文件（引号内）> <verify.py evidence 的其余参数...>
  local out lab=$2 sd=$3 sn=$4 items=$5 tmp=$B/.f_tmp_$1.jsonl st=""; out=$(ev $1); shift 5
  [ -s $out ] && return 0
  [ "$sd" != - ] && st="--state $E/f_state_$sd/state_ed_$sn.jsonl $E/f_state_$sd/state_td_$sn.jsonl --require-state"
  $PY verify.py evidence --mode text --require-cands --items $items "$@" --out $tmp >> $LOG 2>&1 &&
    $PY looklabel.py --evidence $tmp --items $items $st $([ "$lab" = labels ] && echo --labels) --out $out.part >> $LOG 2>&1 && mv $out.part $out
  local ok=$?; rm -f $tmp; return $ok
}

# ---------------------------------------------------------------------------------------------- 准备（prep 组）
prep_plan() {  # 交叉拟合各折的图像清单；退化方案（覆盖全部训练图像）；各训练题文件带退化的版本
  mkdir -p $FA
  local X sc k lists=""
  for X in $LOCO; do [ -s $B/ed_fit_no${X}_strict.jsonl ] || $PY loco.py --bench $B --scenario ed --class $X --strict >> $LOG 2>&1 || return 1; done
  [ -s $FA/fold_td_1.txt ] || $PY - <<PY || return 1
import json, sys
sys.path.insert(0, ".")
import head
for sc in ("ed", "td"):
    fit = json.load(open("$W/det_data/%s/images.json" % sc))["train"]
    for k in (0, 1):
        open("$FA/fold_%s_%d.txt" % (sc, k), "w").write("\n".join(p for p in fit if head.fold_of(p, 2) == k) + "\n")
PY
  for sc in ed td; do cat $FA/fold_${sc}_0.txt $FA/fold_${sc}_1.txt > $FA/all_$sc.txt; for k in 0 1; do lists="$lists $sc=$FA/fold_${sc}_$k.txt"; done; done
  [ "$AUG" = - ] && return 0
  $PY corrupt.py --plan $AUG --seed ${AUG##*_s} --prob 0.3 --image-list $lists --out-dir $FA \
    --items $FIT $FITX $B/ed_fitx_A.jsonl $B/td_fitx_A.jsonl $B/ed_fitx_B.jsonl $B/td_fitx_B.jsonl $OFIT $(for X in $LOCO; do echo $B/ed_fit_no${X}_strict.jsonl; done) >> $LOG 2>&1 || return 1
  log "prep_plan: $(wc -l < $FA/plan_$AUG.jsonl) degraded training images"
}

prep_det() {  # 折外检测器（中型、纳米）与见过这些图的全量检测器（样本内对照）在退化训练图像上的检测结果 → 候选文件
  [ "$AUG" = - ] && return 0
  local g=($GPUS) n=0 size sfx sc k
  det() {  # <权重> <场景> <图像清单> <输出>：后台运行，已存在则跳过
    [ -s $4 ] && return 0
    ( CUDA_VISIBLE_DEVICES=${g[$((n % ${#g[@]}))]} $PY detector.py detect --scenario $2 --image-list $3 --corrupt-plan $FA/plan_$AUG.jsonl --weights $1 --out $4.part --devices 0 \
        > $W/logs/f_det_$(basename $4 .jsonl).log 2>&1 && mv $4.part $4 ) &
    n=$((n+1)); [ $((n % (2 * ${#g[@]}))) = 0 ] && wait
  }
  for size in m n; do sfx=$([ $size = n ] && echo n)
    for sc in ed td; do for k in 0 1; do det $E/detectors_cf$sfx/${sc}_cf$k/weights/best.pt $sc $FA/fold_${sc}_$k.txt $E/detectors_cf$sfx/${sc}_cf${k}_det_f_$AUG.jsonl; done; done
  done
  for sc in ed td; do det $E/detectors_v2/$sc/weights/best.pt $sc $FA/all_$sc.txt $E/detectors_v2/${sc}_det_f_ins_$AUG.jsonl; done
  wait
  for size in m n; do sfx=$([ $size = n ] && echo n)
    [ -s $G/f_cands_fit_${size}_$AUG.jsonl ] || $PY verify.py cands --detections $E/detectors_cf$sfx/{ed,td}_cf{0,1}_det_f_$AUG.jsonl --out $G/f_cands_fit_${size}_$AUG.jsonl >> $LOG 2>&1 || return 1
  done
  [ -s $CIA ] || $PY verify.py cands --detections $E/detectors_v2/{ed,td}_det_f_ins_$AUG.jsonl --out $CIA >> $LOG 2>&1 || return 1
  log "prep_det: $(wc -l < $CMA) / $(wc -l < $CNA) / $(wc -l < $CIA) degraded training images with candidates (out-of-fold medium / nano / in-sample)"
}

prep_state() {  # 分支状态标量：训练图像（由没见过它的折外检测器算；干净与带退化）、样本内对照、验证与测试图像、九种退化下的测试图像
  local g=($GPUS) n=0 size sfx S sc k C v FULL CF plan=""
  [ "$AUG" = - ] || plan="--corrupt-plan $FA/plan_$AUG.jsonl"
  ex() {  # <输出名> <branchstate.py extract 的参数...>：后台运行，已存在则跳过
    local name=$1; shift
    ( [ -s $S/$name.npz ] || CUDA_VISIBLE_DEVICES=${g[$((n % ${#g[@]}))]} $PY branchstate.py extract "$@" --out $S/$name.npz > $W/logs/f_state_${size}_$name.log 2>&1 ) &
    n=$((n+1)); [ $((n % (2 * ${#g[@]}))) = 0 ] && wait
  }
  for size in m n; do sfx=$([ $size = n ] && echo n); S=$E/f_state_$size; mkdir -p $S
    for sc in ed td; do
      FULL=$E/detectors_v2$sfx/$sc/weights/best.pt
      ex ${sc}_full_train --weights $FULL --image-list $FA/all_$sc.txt --limit 1500
      for k in 0 1; do
        CF=$E/detectors_cf$sfx/${sc}_cf$k/weights/best.pt
        ex ${sc}_cf${k}_train --weights $CF --image-list $FA/fold_${sc}_$((1-k)).txt --limit 1500      # 统计量取自该检测器自己的干净训练图像
        ex ${sc}_fit_clean_f$k --weights $CF --image-list $FA/fold_${sc}_$k.txt
        [ -n "$plan" ] && ex ${sc}_fit_aug_f$k --weights $CF --image-list $FA/fold_${sc}_$k.txt $plan
      done
      if [ $size = m ]; then
        ex ${sc}_ins_clean --weights $FULL --image-list $FA/all_$sc.txt
        [ -n "$plan" ] && ex ${sc}_ins_aug --weights $FULL --image-list $FA/all_$sc.txt $plan
      fi
      ex ${sc}_eval_clean --weights $FULL --items $B/${sc}_val.jsonl $B/${sc}_test.jsonl $B/${sc}_testx.jsonl $B/${sc}_testx2.jsonl
      for C in $KINDS9; do for k in 0 1; do
        ex ${sc}_test_$C.s$k --weights $FULL --items $B/${sc}_test.jsonl --corrupt ${C%2}:2 --shard $k --num-shards 2
      done; done
    done
  done
  wait
  for size in m n; do S=$E/f_state_$size
    for sc in ed td; do
      $PY branchstate.py stats --features $S/${sc}_full_train.npz --out $S/${sc}_full_stats.npz >> $LOG 2>&1 || return 1
      for k in 0 1; do $PY branchstate.py stats --features $S/${sc}_cf${k}_train.npz --out $S/${sc}_cf${k}_stats.npz >> $LOG 2>&1 || return 1; done
      for v in clean $([ -n "$plan" ] && echo aug); do
        for k in 0 1; do $PY branchstate.py scalars --stats $S/${sc}_cf${k}_stats.npz --features $S/${sc}_fit_${v}_f$k.npz --out $S/.state_${sc}_fit_${v}_f$k.jsonl >> $LOG 2>&1 || return 1; done
        cat $S/.state_${sc}_fit_${v}_f0.jsonl $S/.state_${sc}_fit_${v}_f1.jsonl > $S/state_${sc}_fit_$v.jsonl; rm -f $S/.state_${sc}_fit_${v}_f?.jsonl
        [ $size = m ] && { $PY branchstate.py scalars --stats $S/${sc}_full_stats.npz --features $S/${sc}_ins_$v.npz --out $S/state_${sc}_ins_$v.jsonl >> $LOG 2>&1 || return 1; }
      done
      $PY branchstate.py scalars --stats $S/${sc}_full_stats.npz --features $S/${sc}_eval_clean.npz --out $S/state_${sc}_eval_clean.jsonl >> $LOG 2>&1 || return 1
      for C in $KINDS9; do $PY branchstate.py scalars --stats $S/${sc}_full_stats.npz --features $S/${sc}_test_$C.npz --out $S/state_${sc}_test_$C.jsonl >> $LOG 2>&1 || return 1; done
    done
  done
  log "prep_state: $(ls $E/f_state_m/state_*.jsonl | wc -l) / $(ls $E/f_state_n/state_*.jsonl | wc -l) state files (medium / nano)"
}

prep_evid0() {  # 判定缩略图大小之前就能备好的证据文件（只用 CPU）：本文的两种缩略图、只读清单、端侧两档。训练题：折外清单、带退化；评测题：全量检测器的清单
  need FMT STATE || return 1
  local t EVALI="$TEST $VAL"
  for t in 256 576 0; do
    mk pk_fit_m$t --cands $CM $CMA --items $FITA --fmt $FMT --thumb-tokens $t && mk pk_eval_m$t --cands $EM --items $EVALI --fmt $FMT --thumb-tokens $t || return 1
  done
  for t in 256 0; do
    mk pk_fit_n$t --cands $CN $CNA --items $FITA --fmt $FMT --thumb-tokens $t && mk pk_eval_n$t --cands $EN --items $EVALI --fmt $FMT --thumb-tokens $t || return 1
  done
  log "prep_evid0: $(ls $B/f_evid_pk_fit_[mn]*.jsonl $B/f_evid_pk_eval_[mn]*.jsonl | wc -l) evidence files"
}

prep_evid() {  # 其余的训练与干净评测证据文件（依赖缩略图大小；只用 CPU）
  structure || return 1
  local T=$THUMB X EVALI="$TEST $VAL" OW=$([ "$FMT" = 3 ] && echo 2 || echo 3)
  local main="--fmt $FMT --thumb-tokens $T"
  # 消融：另一种文字写法；不做退化感知的对齐（干净训练图像）；只退化图像、清单仍取干净图像上的折外候选；样本内清单；人工框做的清单；
  # 最接近已有工作的接口（样本内清单、阈值 0.3、不写置信度、干净训练图像、标准文字）；4B 配纳米分支
  mk pk_fit_m${T}_w$OW --cands $CM $CMA --items $FITA --fmt $OW --thumb-tokens $T && mk pk_eval_m${T}_w$OW --cands $EM --items $EVALI --fmt $OW --thumb-tokens $T || return 1
  if [ "$AUG" != - ]; then
    mk pk_fit_m${T}_noaug --cands $CM --items $FIT $main && mk pk_fit_m${T}_imgaug --cands $CM --items $FITA $main || return 1
  fi
  mk pk_fit_m${T}_ins --cands $CI $CIA --items $FITA $main || return 1
  mk pk_fit_m${T}_gt --from-gt --items $FITA $main || return 1
  mk pk_fit_s0_$T --cands $CI --items $FIT --fmt 2 --thumb-tokens $T --min-conf 0.3 --no-conf && mk pk_eval_s0_$T --cands $EM --items $EVALI --fmt 2 --thumb-tokens $T --min-conf 0.3 --no-conf || return 1
  mk pk_fit_n$T --cands $CN $CNA --items $FITA $main && mk pk_eval_n$T --cands $EN --items $EVALI $main || return 1
  # 没训练过的问法：训练题为基础题加十二种新问法（各模型只取其中一半），测试题为基础题加全部新问法
  mk pk_fit_m${T}_forms --cands $CM $CMA --items $FITA $(aug $FITX) $main || return 1
  mk pk_eval_m${T}_forms --cands $EM $G/cands_testx_v1.jsonl $G/cands_testx2_v1.jsonl --items $TEST $XFORMS $main || return 1
  # 开放式题目（不带选项）：本文；端侧只读清单
  mk pk_fit_m${T}_open --cands $CM $CMA --items $(aug $OFIT) $main && mk pk_eval_m${T}_open --cands $EM --items $OTEST $OVAL $main || return 1
  mk pk_fit_n0_open --cands $CN $CNA --items $(aug $OFIT) --fmt $FMT --thumb-tokens 0 && mk pk_eval_n0_open --cands $EN --items $OTEST $OVAL --fmt $FMT --thumb-tokens 0 || return 1
  # 留出类别（严格版：训练题的选项里也没有这一类）：训练里没有这一类
  for X in $LOCO; do
    mk pk_fit_loco_${X}_m$T --cands $CM $CMA --items $(aug $B/ed_fit_no${X}_strict.jsonl $B/td_fit.jsonl) $main --drop-class $X || return 1
    mk pk_eval_loco_${X}_m$T --cands $EM --items $B/ed_test_only$X.jsonl $main --drop-class $X || return 1
  done
  log "prep_evid: $(ls $B/f_evid_pk_*.jsonl | wc -l) evidence files"
}

# ---------------------------------------------------------------------------------------------- 训练与评测
train() {  # <名称> <底座 4b|2b|8b> <格式> <种子> <证据文件或 -> <训练题文件（引号内空格分隔）> [其余参数]
  local name=$1 base=$2 fmt=$3 seed=$4 ev=$5 items=$6; shift 6
  [ -f $L/$name/adapter_model.safetensors ] && return 0
  local np; np=$(echo $GPUS | wc -w)
  case $np in 1|2|4) ;; *) log "$name: $np GPUs cannot give an effective batch of 4"; return 1 ;; esac
  for attempt in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$(echo $GPUS | tr ' ' ',') $W/.venv/bin/accelerate launch --num_processes $np --mixed_precision bf16 train_lora.py --items $items --model $(model $base) \
      --out $L/$name --formats $fmt $([ "$ev" != - ] && echo --selector-regions $ev) --accumulate $((4 / np)) --seed $seed "$@" > $W/logs/train_$name.log 2>&1
    [ -f $L/$name/adapter_model.safetensors ] && break; sleep 20
  done
  [ -f $L/$name/adapter_model.safetensors ] || { log "$name: TRAINING FAILED"; return 1; }
}

evalrow() {  # <结果名> <适配器名或 -> <底座> <题目文件（空串为测试集）> <证据文件或 -> <策略...>
  local res=$1 name=$2 base=$3 items=$4 ev=$5; shift 5
  PB_MODEL=$(model $base) PB_ITEMS="$items" PB_REGIONS="$([ "$ev" != - ] && echo "--regions e=$ev")" \
    $W/run_eval3.sh $res $([ "$name" = - ] && echo - || echo $L/$name) "${EVAL_GPUS:-$(shards)}" "$@"
}
shards() { local i out=""; for i in $(seq ${PB_EVAL_SHARDS:-1}); do out="$out $GPUS"; done; echo $out; }   # 大显存的卡上一张卡放多个评测进程

tv() {  # 单步模型或整图基线：<名称> <底座> <证据文件或 -> <策略...>：测试集与验证集，并把测试集准确率记入日志
  local name=$1 base=$2 ev=$3; shift 3
  evalrow $name $name $base "" $ev "$@" && evalrow ${name}_val $name $base "$VAL" $ev "$@" || return 1
  $PY report.py --results $R/$name --out $R/$name/report.md > /dev/null 2>&1
  log "$name test: $(grep -A8 '：准确率' $R/$name/report.md | grep '^| g:e\|^| u' | cut -d'|' -f2,4,5,7,8,9,10,11 | tr '\n' ';')"
}

tvl() {  # 存疑再看的模型：<名称> <底座> <证据文件>：测试集与验证集各题的两步输出，再换算成各工作点的普通结果行（lookrows.py）
  local name=$1 base=$2 ev=$3
  evalrow $name $name $base "" $ev look:e && evalrow ${name}_val $name $base "$VAL" $ev look:e || return 1
  log "$($PY lookrows.py --results $R --name $name --suffixes "" _val 2>&1 | tr '\n' ' ')"
}

robust_det() {  # 九种退化里与结构无关的部分：题目文件、两个检测分支在退化图像上的候选、不含语言模型的各行（含前置图像增强的对照）
  local g0 g1 kind size dir C; read -r g0 g1 _ <<< "$GPUS"; g1=${g1:-$g0}
  mkdir -p $B/corrupt
  # 对照（P24）：检测分支前加标准图像增强（伽马校正加 CLAHE；检测器不重训），仍按规则答题；阈值在同样增强过的验证图像上定
  local DE=$E/detectors_v2 ENH
  enh() {  # <场景> <显卡> <输出> <detector.py detect 的其余参数...>
    local sc=$1 gpu=$2 out=$3; shift 3
    [ -s $out ] || CUDA_VISIBLE_DEVICES=$gpu $PY detector.py detect --scenario $sc --weights $DE/$sc/weights/best.pt --enhance --out $out --devices 0 "$@" > $W/logs/f_$(basename $out .jsonl).log 2>&1
  }
  enh ed $g0 $DE/ed_detections_enh.jsonl --items $B/ed_val.jsonl $B/ed_test.jsonl & enh td $g1 $DE/td_detections_enh.jsonl --items $B/td_val.jsonl $B/td_test.jsonl & wait
  ENH="$DE/ed_detections_enh.jsonl $DE/td_detections_enh.jsonl"
  [ -s $R/detector_enh.jsonl ] || $PY detector.py answer --detections $ENH --items $TEST --tune-items $VAL --out $R/detector_enh.jsonl > $W/logs/detector_answer_enh.log 2>&1
  for C in $KINDS9; do
    kind=${C%2}
    [ -s "$B/corrupt/td_test@$C.jsonl" ] || $PY corrupt.py --items $TEST --kind $kind --severity 2 --out-dir $B/corrupt >> $LOG 2>&1
    enh ed $g0 $DE/ed_detections_enh_c$C.jsonl --items $B/ed_test.jsonl --corrupt $kind:2 & enh td $g1 $DE/td_detections_enh_c$C.jsonl --items $B/td_test.jsonl --corrupt $kind:2 & wait
    [ -s $R/detector_enh_c$C.jsonl ] || $PY detector.py answer --detections $ENH $DE/ed_detections_enh_c$C.jsonl $DE/td_detections_enh_c$C.jsonl --items $TEST --tune-items $VAL --out $R/detector_enh_c$C.jsonl > $W/logs/detector_answer_enh_c$C.log 2>&1
    for size in m n; do
      dir=$E/detectors_v2; [ $size = n ] && dir=$E/detectors_v2n
      [ -s $dir/ed_detections_c$C.jsonl ] || CUDA_VISIBLE_DEVICES=$g0 $PY detector.py detect --scenario ed --items $B/ed_test.jsonl --weights $dir/ed/weights/best.pt --corrupt $kind:2 --out $dir/ed_detections_c$C.jsonl --devices 0 > $W/logs/f_detect_${size}_c${C}_ed.log 2>&1 &
      [ -s $dir/td_detections_c$C.jsonl ] || CUDA_VISIBLE_DEVICES=$g1 $PY detector.py detect --scenario td --items $B/td_test.jsonl --weights $dir/td/weights/best.pt --corrupt $kind:2 --out $dir/td_detections_c$C.jsonl --devices 0 > $W/logs/f_detect_${size}_c${C}_td.log 2>&1 &
      wait
    done
    [ -s $G/cands_eval_c$C.jsonl ] || $PY verify.py cands --detections $E/detectors_v2/ed_detections_c$C.jsonl $E/detectors_v2/td_detections_c$C.jsonl --out $G/cands_eval_c$C.jsonl >> $LOG 2>&1
    [ -s $G/cands_eval_nano_c$C.jsonl ] || $PY verify.py cands --detections $E/detectors_v2n/ed_detections_c$C.jsonl $E/detectors_v2n/td_detections_c$C.jsonl --out $G/cands_eval_nano_c$C.jsonl >> $LOG 2>&1
    # 不含语言模型的两行：阈值与小网络沿用干净数据（验证图像用干净检测结果，测试图像用退化后的检测结果）
    [ -s $R/detector_c$C.jsonl ] || $PY detector.py answer --detections $E/detectors_v2/ed_detections.jsonl $E/detectors_v2/td_detections.jsonl $E/detectors_v2/ed_detections_c$C.jsonl $E/detectors_v2/td_detections_c$C.jsonl \
      --items $TEST --tune-items $VAL --out $R/detector_c$C.jsonl > $W/logs/detector_answer_c$C.log 2>&1
    [ -s $R/aggregator_c$C.jsonl ] || $PY aggregator.py --fit-cands $G/cands_fit_v1.jsonl --eval-cands $G/cands_eval_v1.jsonl $G/cands_eval_c$C.jsonl --bench $B --out $R/aggregator_c$C.jsonl > $W/logs/aggregator_c$C.log 2>&1
    log "robust_det $C: candidates and the rows without a language model"
  done
  [ "$(ls $G/cands_eval_nano_c*2.jsonl 2>/dev/null | wc -l)" -ge 9 ]
}

robust_prep0() {  # 六种见过的退化下、本文两种缩略图的证据文件（判定缩略图大小用；候选由 robust_det 备好）
  need FMT STATE || return 1
  local C t XI
  for C in $KINDS6; do
    XI="$B/corrupt/ed_test@$C.jsonl $B/corrupt/td_test@$C.jsonl"
    for t in 256 576; do mk pk_c${C}_m$t --cands $G/cands_eval_c$C.jsonl --items $XI --fmt $FMT --thumb-tokens $t || return 1; done
  done
  log "robust_prep0: $(ls $B/f_evid_pk_c*_m256.jsonl $B/f_evid_pk_c*_m576.jsonl 2>/dev/null | wc -l) evidence files"
}

robust_prep() {  # 九种退化下各行的证据文件（候选由 robust_det 备好）
  need FMT STATE || return 1
  local C t XI
  for C in $KINDS9; do
    XI="$B/corrupt/ed_test@$C.jsonl $B/corrupt/td_test@$C.jsonl"
    for t in 256 576 0; do mk pk_c${C}_m$t --cands $G/cands_eval_c$C.jsonl --items $XI --fmt $FMT --thumb-tokens $t || return 1; done
    for t in 256 0; do mk pk_c${C}_n$t --cands $G/cands_eval_nano_c$C.jsonl --items $XI --fmt $FMT --thumb-tokens $t || return 1; done
  done
  log "robust_prep: $(ls $B/f_evid_pk_c*.jsonl 2>/dev/null | wc -l) evidence files"
  [ "$(ls $B/f_evid_pk_c*_n0.jsonl 2>/dev/null | wc -l)" = 9 ]
}

all_answered() {  # <结果名> <策略> <题目文件...>：这个结果里是否每道题都已有该策略的答案。评完的不再评，免得换了显卡数（分片数）之后同一道题被答两次
  local res=$1 pol=$2; shift 2
  $PY - "$R/$res" "$pol" "$@" <<'PY'
import glob, json, sys
want = {json.loads(line)["id"] for path in sys.argv[3:] for line in open(path)}
have = set()
for path in glob.glob(sys.argv[1] + "/*shard*.jsonl"):
    for line in open(path):
        row = json.loads(line)
        if row.get("policy") == sys.argv[2]:
            have.add(row["id"])
sys.exit(0 if want and want <= have else 1)
PY
}

robust_row() {  # 单步模型或整图基线：<结果名前缀> <适配器名> <底座> <证据键（如 pk_c@_m256，@ 处填退化名）或 -> <策略> [每张卡的评测进程数，默认 2] [退化种类，默认九种]
  local res=$1 name=$2 base=$3 key=$4 pol=$5 kinds=${7:-$KINDS9} ev=- C XI
  local EVAL_GPUS=$GPUS; [ "${6:-2}" = 2 ] && EVAL_GPUS="$GPUS $GPUS"   # 退化在 CPU 上现算，评测受 CPU 限制，一张卡放两个进程
  for C in $kinds; do
    XI="$B/corrupt/ed_test@$C.jsonl $B/corrupt/td_test@$C.jsonl"
    all_answered ${res}_c$C $pol $XI && continue
    [ "$key" != - ] && ev=$(ev ${key/@/$C})
    evalrow ${res}_c$C $name $base "$XI" $ev $pol || return 1
  done
  log "$res: evaluated under $kinds"
}

robust_look() {  # 存疑再看的模型：<模型名> <底座> <证据键（如 lk_c@_m）>；评完后换算成各工作点的结果行（阈值来自该模型的干净验证集）
  local name=$1 base=$2 key=$3 C EVAL_GPUS="$GPUS $GPUS"
  for C in $KINDS9; do
    evalrow ${name}_c$C $name $base "$B/corrupt/ed_test@$C.jsonl $B/corrupt/td_test@$C.jsonl" $(ev ${key/@/$C}) look:e || return 1
  done
  $PY lookrows.py --results $R --name $name --suffixes $(for C in $KINDS9; do echo _c$C; done) >> $LOG 2>&1
  log "$name: nine degradations evaluated"
}

robust_aggaug() {  # 对照（P28）：小汇总网络改用与大模型各行同一份带退化的训练清单拟合（结构、特征、模型选择不变）；干净测试集与九种退化各出一份结果。纯 CPU
  [ "$AUG" = - ] && return 0
  local C
  fitagg() {  # <输出> <评测用的候选文件...>：被退化方案选中的训练图像用退化图像上的折外候选（后给的文件覆盖先给的）
    local out=$1; shift
    [ -s $out ] && return 0
    $PY aggregator.py --fit-cands $CM $CMA --eval-cands "$@" --bench $B --out $out.part > $W/logs/$(basename $out .jsonl).log 2>&1 && mv $out.part $out
  }
  fitagg $R/f_aggaug.jsonl $EM || return 1
  for C in $KINDS9; do fitagg $R/f_aggaug_c$C.jsonl $EM $G/cands_eval_c$C.jsonl || return 1; done
  log "robust_aggaug: $(ls $R/f_aggaug.jsonl $R/f_aggaug_c*.jsonl 2>/dev/null | wc -l) result files"
}

thumb_table() {  # 判定缩略图大小用的表（P27 第 4 条）：256 与 576 令牌两个单步模型在干净测试集与六种见过的退化上的准确率及配对差值
  $PY robustreport.py --bench $B --results $R --runs $(for C in $KINDS6; do echo ${C%2}:2; done) \
    --system "缩略图 256 令牌=f_one256_s0:g:e|f_one256_s0_c{C}:g:e" --system "缩略图 576 令牌=f_one576_s0:g:e|f_one576_s0_c{C}:g:e" \
    --pair "缩略图 576 令牌" "缩略图 256 令牌" > $R/f_thumb.md 2>> $LOG || return 1
  log "thumb table written: $R/f_thumb.md (576 if its six-kind mean is at least 1 point higher in one scenario and nothing else is more than 0.5 lower; else 256)"
}

prune_row() {  # 通用令牌剪枝基线（P10）：原始分辨率整图模型在语言模型第 3 层之后只保留一半视觉令牌（原文默认档）
  PB_MODEL=$(model 4b) PB_ITEMS="" PB_REGIONS="--fastv 3:0.5" $W/run_eval3.sh f_wnat_s0_fv05 $L/f_wnat_s0 "$(shards)" u8192
}

cost_rows() {  # 单卡在线端到端代价：每题走一遍完整流程（检测分支、组装、预处理、前向），并与精度评测的答案逐题核对；除剪枝一行外，各行另测生成 16 个令牌的回答的耗时（回应「耗时主要在解码」）
  structure || return 1
  local g rows=$R/f_cost_rows.jsonl DM=$E/detectors_v2 DN=$E/detectors_v2n T=$THUMB; read -r g _ <<< "$GPUS"
  local WM="--weights ed=$DM/ed/weights/best.pt td=$DM/td/weights/best.pt" WN="--weights ed=$DN/ed/weights/best.pt td=$DN/td/weights/best.pt"
  c() { CUDA_VISIBLE_DEVICES=$g $PY cost_pack.py run --items $TEST --rows $rows "$@" >> $W/logs/f_cost.log 2>&1; }
  rm -f $rows
  c --model $(model 4b) --lora $L/f_one${T}_s0 --policies ours $WM --fmt $FMT --thumb-tokens $T --evaluated $R/f_one${T}_s0:g:e --name "本文（4B）" --gen-tokens 16 &&
  c --model $(model 4b) --lora $L/f_list_s0 --policies ours $WM --fmt $FMT --thumb-tokens 0 --evaluated $R/f_list_s0:g:e --name "只读清单（4B）" --gen-tokens 16 &&
  c --model $(model 2b) --lora $L/f_onen2b_s0 --policies ours $WN --fmt $FMT --thumb-tokens 256 --evaluated $R/f_onen2b_s0:g:e --name "端侧配置，带缩略图（2B，纳米分支）" --gen-tokens 16 &&
  c --model $(model 2b) --lora $L/f_listn2b_s0 --policies ours $WN --fmt $FMT --thumb-tokens 0 --evaluated $R/f_listn2b_s0:g:e --name "端侧配置，只读清单（2B，纳米分支）" --gen-tokens 16 &&
  c --model $(model 4b) --lora $L/f_w576_s0 --policies u576 --evaluated $R/f_w576_s0 --name "整图 576 令牌（4B）" --gen-tokens 16 &&
  c --model $(model 4b) --lora $L/f_wnat_s0 --policies u8192 --evaluated $R/f_wnat_s0 --name "整图原始分辨率（4B）" --gen-tokens 16 &&
  c --model $(model 4b) --lora $L/f_wnat_s0 --policies u8192 --fastv 3:0.5 --evaluated $R/f_wnat_s0_fv05 --name "整图原始分辨率，剪枝留 50%（4B）" || return 1
  # 整图基线里最强的一行（视觉编码器一并适配）：正文「耗时约为其几分之一」要用端到端口径，此前只有前向耗时。放在最后，这一行失败不拦住整张表
  c --model $(model 4b) --lora $L/f_wnat_vis --policies u8192 --evaluated $R/f_wnat_vis --name "整图原始分辨率，视觉编码器一并适配（4B）" --gen-tokens 16 ||
    log "cost: the row of the vision-adapted whole-image baseline failed (see $W/logs/f_cost.log); the table is written without it"
  $PY cost_pack.py table --rows $rows --out $R/f_cost.md >> $W/logs/f_cost.log 2>&1
}

case $1 in
config)
  shift; touch $F/config
  for kv in "$@"; do
    key=${kv%%=*}; val=${kv#*=}
    case $kv in FMT=2|FMT=3|THUMB=256|THUMB=576|STATE=0|AUG=*|AUGBASE=0|AUGBASE=1) ;;
      *) echo "not accepted: $kv (FMT 2 or 3, THUMB 256 or 576, STATE 0, AUGBASE 0 or 1)"; exit 1 ;; esac
    old=$(grep "^$key=" $F/config | cut -d= -f2)
    if [ -z "$old" ]; then echo "$kv" >> $F/config
    elif [ "$old" != "$val" ]; then echo "$key is already fixed at $old; refusing to change it to $val"; exit 1; fi
  done
  . $F/config
  echo "FMT=$FMT THUMB=$THUMB STATE=$STATE AUG=$AUG AUGBASE=$AUGBASE"; log "config: FMT=$FMT THUMB=$THUMB STATE=$STATE AUG=$AUG AUGBASE=$AUGBASE" ;;

jobs)
  shift; GROUPS_WANTED="${*:-prep base}"
  job() { printf '%s\t%s\n' "$1" "$2"; }
  for g in $GROUPS_WANTED; do case $g in prep|base|large) ;; thumb|tiers) need FMT STATE || exit 1 ;; *) structure || exit 1 ;; esac; done
  T=$THUMB; OT=$([ "$T" = 256 ] && echo 576 || echo 256); OW=$([ "$FMT" = 3 ] && echo 2 || echo 3)
  BF=$(basefit $FIT)
  pk() { echo "train $1 $2 pack $3 $(ev $4) \"${6:-$FITA}\" && tv $1 $2 $(ev $5) g:e"; }                       # <名称> <底座> <种子> <训练证据> <评测证据> [训练题]
  wi() { echo "train $1 $2 $3 $4 - \"$BF\" ${6:-} && tv $1 $2 - $5"; }                                      # <名称> <底座> <格式> <种子> <策略> [训练的其余参数]
  for g in $GROUPS_WANTED; do
    case $g in
    prep)  # 要运行检测器，须在放着检测器权重的机器上跑；这一组与主配置的后三项无关
      job prep_plan "prep_plan"
      job prep_det@prep_plan "prep_det"
      job prep_state@prep_plan "prep_state"
      job robust_det "robust_det" ;;
    base)  # 整图基线各行：与结构无关，主配置的后三项没定也可以先跑
      for s in 0 1 2; do job w576_s$s@prep_plan "$(wi f_w576_s$s 4b u576only $s u576)"; done
      job w576_2b@prep_plan "$(wi f_w576_2b 2b u576only 0 u576)"
      for s in 0 1 2; do job wnat_s$s@prep_plan "$(wi f_wnat_s$s 4b native $s u8192)"; done
      job prune@wnat_s0     "prune_row"
      job wnat_2b@prep_plan  "$(wi f_wnat_2b 2b native 0 u8192)"
      job wnat_vis@prep_plan "$(wi f_wnat_vis 4b native 0 u8192 --vision-lora)"      # 整图基线里最强的一行：视觉编码器一并适配
      OB=$(basefit $OFIT)
      ow() { echo "train $1 4b $2 0 - \"$OB\" && evalrow $1 $1 4b \"$OTEST\" - $3 && evalrow ${1}_val $1 4b \"$OVAL\" - $3"; }
      job open_w576@prep_plan "$(ow f_open_w576 u576only u576)"
      job open_wnat@prep_plan "$(ow f_open_wnat native u8192)"
      for X in $LOCO; do
        LB=$(basefit $B/ed_fit_no${X}_strict.jsonl $B/td_fit.jsonl); ONLY=$B/ed_test_only$X.jsonl
        job loco_w576_$X@prep_plan "train f_loco_w576_$X 4b u576only 0 - \"$LB\" && evalrow f_loco_w576_$X f_loco_w576_$X 4b $ONLY - u576"
        job loco_wnat_$X@prep_plan "train f_loco_wnat_$X 4b native 0 - \"$LB\" && evalrow f_loco_wnat_$X f_loco_wnat_$X 4b $ONLY - u8192"
      done
      job robust_w576@robust_det,w576_s0 "robust_row f_w576_s0 f_w576_s0 4b - u576"
      job robust_wnat@robust_det,wnat_s0 "robust_row f_wnat_s0 f_wnat_s0 4b - u8192 1" ;;
    thumb)  # 判定缩略图大小（P27 第 4 条）：256 与 576 令牌各一个单步模型（种子 0），评干净测试集、验证集与六种见过的退化。
            # 三种留出的退化等 THUMB 写定之后在 robust 组里评。入选的一个即本文的种子 0，落选的一个即消融行「另一种缩略图大小」
      job prep_evid0@prep_det "prep_evid0"
      job robust_prep0@robust_det "robust_prep0"
      for t in 256 576; do
        job one${t}_s0@prep_evid0 "$(pk f_one${t}_s0 4b 0 pk_fit_m$t pk_eval_m$t)"
        job seen_one$t@one${t}_s0,robust_prep0 "robust_row f_one${t}_s0 f_one${t}_s0 4b pk_c@_m$t g:e 2 \"$KINDS6\""
      done
      job thumb_table@seen_one256,seen_one576 "thumb_table" ;;
    tiers)  # 主表里与缩略图大小无关的三行（各三个种子）：只读清单（4B）；端侧只读清单、端侧带缩略图（2B、纳米检测分支；缩略图固定 256 令牌）
      for s in 0 1 2; do
        job list_s$s@prep_evid0    "$(pk f_list_s$s 4b $s pk_fit_m0 pk_eval_m0)"
        job listn2b_s$s@prep_evid0 "$(pk f_listn2b_s$s 2b $s pk_fit_n0 pk_eval_n0)"
        job onen2b_s$s@prep_evid0  "$(pk f_onen2b_s$s 2b $s pk_fit_n256 pk_eval_n256)"
      done ;;
    main)  # 本文（4B，清单加缩略图）的另两个种子；其余证据文件
      job prep_evid@prep_evid0 "prep_evid"
      for s in 1 2; do job one${T}_s$s@prep_evid0 "$(pk f_one${T}_s$s 4b $s pk_fit_m$T pk_eval_m$T)"; done ;;
    ablation)  # 单个种子；除注明外为 4B、中型检测分支、带缩略图
      job abl_w$OW@prep_evid "$(pk f_abl_w$OW 4b 0 pk_fit_m${T}_w$OW pk_eval_m${T}_w$OW)"      # 另一种文字写法
      if [ "$AUG" != - ]; then
        job abl_noaug@prep_evid  "$(pk f_abl_noaug 4b 0 pk_fit_m${T}_noaug pk_eval_m$T "$FIT")"  # 不做退化感知的对齐：干净训练图像
        job abl_imgaug@prep_evid "$(pk f_abl_imgaug 4b 0 pk_fit_m${T}_imgaug pk_eval_m$T)"       # 只退化图像，清单仍取干净图像上的折外候选
      fi
      job abl_ins@prep_evid  "$(pk f_abl_ins 4b 0 pk_fit_m${T}_ins pk_eval_m$T)"                 # 样本内清单
      job abl_gt@prep_evid   "$(pk f_abl_gt 4b 0 pk_fit_m${T}_gt pk_eval_m$T)"                   # 人工框做的清单
      job prior_s0@prep_evid "$(pk f_prior_s0 4b 0 pk_fit_s0_$T pk_eval_s0_$T "$FIT")"           # 最接近已有工作的接口
      job one2b_s0@prep_evid0  "$(pk f_one2b_s0 2b 0 pk_fit_m$T pk_eval_m$T)"                     # 2B 配中型分支：带缩略图
      job list2b_s0@prep_evid0 "$(pk f_list2b_s0 2b 0 pk_fit_m0 pk_eval_m0)"                      # 2B 配中型分支：只读清单
      job onen_s0@prep_evid   "$(pk f_onen_s0 4b 0 pk_fit_n$T pk_eval_n$T)" ;;                   # 4B 配纳米分支
    robust)  # 九种退化下的准确率（各行取种子 0）。本文两种缩略图的六种见过的退化已在 thumb 组评过，这里只补三种留出的
      job robust_prep@robust_det,robust_prep0 "robust_prep"
      job robust_one@robust_prep,seen_one$T      "robust_row f_one${T}_s0 f_one${T}_s0 4b pk_c@_m$T g:e"
      job robust_other@robust_prep,seen_one$OT   "robust_row f_one${OT}_s0 f_one${OT}_s0 4b pk_c@_m$OT g:e"
      job robust_list@robust_prep,list_s0        "robust_row f_list_s0 f_list_s0 4b pk_c@_m0 g:e"
      job robust_listn2b@robust_prep,listn2b_s0  "robust_row f_listn2b_s0 f_listn2b_s0 2b pk_c@_n0 g:e"
      job robust_onen2b@robust_prep,onen2b_s0    "robust_row f_onen2b_s0 f_onen2b_s0 2b pk_c@_n256 g:e"
      if [ "$AUG" != - ]; then
        job robust_noaug@robust_prep,abl_noaug   "robust_row f_abl_noaug f_abl_noaug 4b pk_c@_m$T g:e"
        job robust_imgaug@robust_prep,abl_imgaug "robust_row f_abl_imgaug f_abl_imgaug 4b pk_c@_m$T g:e"
        job robust_aggaug@robust_det,prep_det    "robust_aggaug"                                 # 小汇总网络用同一份带退化的训练清单拟合（P28；纯 CPU）
      fi ;;
    forms)  # 没训练过的问法：两半互换；整图两行沿用 sft_forms_uniA / uniB（干净训练图像）
      for h in A B; do
        HF="$FITA $(aug $B/ed_fitx_$h.jsonl $B/td_fitx_$h.jsonl)"
        job forms_one$h@prep_evid "train f_forms_one$h 4b pack 0 $(ev pk_fit_m${T}_forms) \"$HF\" && evalrow f_forms_one${h}_x f_forms_one$h 4b \"$XFORMS\" $(ev pk_eval_m${T}_forms) g:e && evalrow f_forms_one$h f_forms_one$h 4b \"\" $(ev pk_eval_m${T}_forms) g:e"
      done
      job forms_base@prep_evid,one${T}_s0 "evalrow f_one${T}_s0_x f_one${T}_s0 4b \"$XFORMS\" $(ev pk_eval_m${T}_forms) g:e" ;;
    open)  # 开放式评测：各用开放式训练题单独训练，贪心生成，按规则判分
      OA=$(aug $OFIT)
      op() { echo "train $1 $2 pack 0 $(ev $3) \"$OA\" && evalrow $1 $1 $2 \"$OTEST\" $(ev $4) g:e && evalrow ${1}_val $1 $2 \"$OVAL\" $(ev $4) g:e"; }
      job open_one@prep_evid     "$(op f_open_one 4b pk_fit_m${T}_open pk_eval_m${T}_open)"
      job open_listn2b@prep_evid "$(op f_open_listn2b 2b pk_fit_n0_open pk_eval_n0_open)" ;;
    cost)  # 单卡耗时：机器空闲时单独跑
      job cost@one${T}_s0,list_s0,listn2b_s0,onen2b_s0,w576_s0,wnat_s0,prune,wnat_vis "cost_rows" ;;
    large)  # 更大的整图基线（8B）：需要 80 GB 一级的显卡
      job w576_8b@prep_plan "$(wi f_w576_8b 8b u576only 0 u576)"
      job wnat_8b@prep_plan "$(wi f_wnat_8b 8b native 0 u8192)" ;;
    loco)  # 留出类别：训练里没有这一类的单步模型与整图模型，只测关于这一类的题
      for X in $LOCO; do
        LF=$(aug $B/ed_fit_no${X}_strict.jsonl $B/td_fit.jsonl); ONLY=$B/ed_test_only$X.jsonl
        job loco_one_$X@prep_evid  "train f_loco_one_$X 4b pack 0 $(ev pk_fit_loco_${X}_m$T) \"$LF\" && evalrow f_loco_one_$X f_loco_one_$X 4b $ONLY $(ev pk_eval_loco_${X}_m$T) g:e u576"
      done ;;
    *) echo "unknown group: $g" >&2; exit 1 ;;
    esac
  done ;;

worker)
  GPUS="$2"; WORKER=${3:-w}; export GPUS WORKER
  log "worker on GPUs $GPUS"
  while :; do
    picked=""; blocked=0
    while IFS=$'\t' read -r spec cmd; do
      [ -z "$spec" ] && continue
      name=${spec%%@*}; deps=""; [ "$spec" != "$name" ] && deps=${spec#*@}
      [ -e $F/done/$name ] && continue
      [ -d $F/claim/$name ] && continue
      ready=1; dead=""
      for d in ${deps//,/ }; do
        [ -e $F/done/$d ] && continue
        ready=0; { [ -e $F/claim/$d/FAILED ] || [ -e $F/claim/$d/BLOCKED ]; } && dead=$d
      done
      if [ -n "$dead" ]; then   # 依赖失败的任务不再等，并标成 BLOCKED，让依赖它的任务照此传递下去（否则隔一层的任务会一直等）
        mkdir $F/claim/$name 2>/dev/null && { echo "$dead" > $F/claim/$name/BLOCKED; log "blocked $name: $dead did not finish (after the repair remove $F/claim/$name)"; }
        continue
      fi
      [ $ready = 1 ] || { blocked=1; continue; }
      mkdir $F/claim/$name 2>/dev/null || continue
      picked=$name; break
    done < $JF
    if [ -z "$picked" ]; then
      [ $blocked = 1 ] && { sleep ${PB_POLL:-60}; continue; }
      break
    fi
    . $F/config   # re-read: an item written after this worker started (THUMB on 2026-10-13) must reach the jobs that need it at run time
    log "start $picked"
    if eval "$cmd" < /dev/null; then touch $F/done/$picked; log "done $picked"; else touch $F/claim/$picked/FAILED; log "FAILED $picked (claim kept; remove $F/claim/$picked to retry)"; fi
  done
  log "worker finished: nothing left to run" ;;

status)
  total=$(grep -c . $JF 2>/dev/null); echo "jobs $total, done $(ls $F/done | wc -l), claimed $(ls $F/claim | wc -l)"
  while IFS=$'\t' read -r spec cmd; do
    [ -z "$spec" ] && continue
    name=${spec%%@*}
    if [ -e $F/done/$name ]; then s=done; elif [ -e $F/claim/$name/FAILED ]; then s=FAILED; elif [ -e $F/claim/$name/BLOCKED ]; then s="blocked by $(cat $F/claim/$name/BLOCKED)"; elif [ -d $F/claim/$name ]; then s=RUNNING; else s=waiting; fi
    printf '%-22s %s\n' "$name" "$s"
  done < $JF
  grep "test:\|FAILED\|evaluated\|accuracy / share\|prep_" $LOG | tail -n ${2:-12} | cut -c1-240 ;;
esac
