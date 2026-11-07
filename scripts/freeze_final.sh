#!/usr/bin/env bash
# 最终一轮的冻结：由逐题输出一次生成全部表格与稿件数字宏（稿件、回复信、封面信只引用这里的产物）。
# 用法：bash freeze_final.sh [标签，默认 v4]。缺结果的行会在表里标明缺多少题，不会被悄悄略过。
# 方法行是单步模型（清单加缩略图，协议 P27）；只读清单、端侧两档与各消融并列给出。
export OMP_NUM_THREADS=8
W=${PB_W:-$HOME/powerbrain-exp}
E=${PB_E:-$HOME/powerbrain_exp}
B=$E/bench_v1; R=$E/results; D=$E/detectors_v2
PY=$W/.venv/bin/python
V=${1:-v4}; O=$R/frozen; mkdir -p $O
FMT=""; THUMB=""; STATE=""; AUG=""; AUGBASE=""
. $W/final/config
[ -n "$FMT" ] && [ -n "$THUMB" ] || { echo "final/config: FMT and THUMB are not set"; exit 1; }
cd $W/pb
seeds() { echo "$R/f_$1_s0$2+$R/f_$1_s1$2+$R/f_$1_s2$2"; }   # <行名> [结果名后缀，如 _val]
GE=g:e
T=$THUMB; OT=$([ "$T" = 256 ] && echo 576 || echo 256); OW=$([ "$FMT" = 3 ] && echo 2 || echo 3)
WORDING=([2]="标准文字清单" [3]="紧凑文字清单")

OURS="本文（4B）=$(seeds one$T):$GE"
LIST="只读清单（4B）=$(seeds list):$GE"
EDGE0="端侧配置，只读清单（2B，纳米分支）=$(seeds listn2b):$GE"
EDGE1="端侧配置，带缩略图（2B，纳米分支）=$(seeds onen2b):$GE"
W576="整图 576 令牌（4B）=$(seeds w576):u576"
WNAT="整图原始分辨率（4B）=$(seeds wnat):u8192"
RULE="检测分支按规则=$R/detector_v2_test.jsonl"
AGG="检测分支 + 小汇总网络=$R/aggregator_v1.jsonl+$R/aggregator_v1_s1.jsonl+$R/aggregator_v1_s2.jsonl"
AGGA="检测分支 + 小汇总网络（带退化的训练清单）"   # 对照（P28）：与大模型各行同一份训练清单，种子 0
ABL=(--row "缩略图 $OT 令牌=$R/f_one${OT}_s0:$GE" --row "${WORDING[$OW]}=$R/f_abl_w$OW:$GE"
     --row "不做退化感知的对齐=$R/f_abl_noaug:$GE" --row "只退化图像，清单取干净图像=$R/f_abl_imgaug:$GE"
     --row "样本内清单训练=$R/f_abl_ins:$GE" --row "人工框清单训练=$R/f_abl_gt:$GE" --row "最接近已有工作的接口=$R/f_prior_s0:$GE"
     --row "2B，中型分支，带缩略图=$R/f_one2b_s0:$GE" --row "2B，中型分支，只读清单=$R/f_list2b_s0:$GE" --row "4B，纳米分支，带缩略图=$R/f_onen_s0:$GE")
BASE=(--row "$AGGA=$R/f_aggaug.jsonl" --row "整图 576 令牌（2B）=$R/f_w576_2b:u576" --row "整图原始分辨率（2B）=$R/f_wnat_2b:u8192" --row "整图原始分辨率，视觉编码器一并适配=$R/f_wnat_vis:u8192"
      --row "整图原始分辨率，剪枝留 50%=$R/f_wnat_s0_fv05:u8192"
      --row "整图原始分辨率，剪到 576 个令牌=$R/f_wnat_s0_fv576:u8192"
      --row "未微调 4B，576 令牌=$R/base:u576" --row "未微调 4B，原始分辨率=$R/base:u8192")   # 递进对照的起点：未微调 → 同样微调的整图 → 本文
# 8B 的两行要 80 GB 一级的显卡（final_round.sh 的 large 组），这台服务器上没有跑；有结果时才列入主表
[ -d $R/f_w576_8b ] && BASE+=(--row "整图 576 令牌（8B）=$R/f_w576_8b:u576") || echo "main table: no 8B rows (not run)"
[ -d $R/f_wnat_8b ] && BASE+=(--row "整图原始分辨率（8B）=$R/f_wnat_8b:u8192")
MAIN=(--row "$RULE" --row "$AGG" --row "$W576" --row "$WNAT" --row "$LIST" --row "$OURS" --row "$EDGE0" --row "$EDGE1")

# 0. 一致性审计：各适配器的训练设置是否一致、各结果是否每题恰好答一次、各种子答的是否同一批题。有出入先查清再冻结
$PY audit_final.py --bench $B --lora $E/lora --results $R > $O/audit_$V.md 2>&1; tail -n 1 $O/audit_$V.md

# 1. 主表（测试集、验证集），以及换参照行的两份（原始分辨率减 576 令牌等差值；端侧配置）
$PY maintable.py --bench $B --ref "本文（4B）" "${MAIN[@]}" "${ABL[@]}" "${BASE[@]}" --json $O/main_$V.json > $O/main_$V.md
$PY maintable.py --bench $B --ref "整图原始分辨率（4B）" --row "$W576" --row "$WNAT" --row "$OURS" --row "整图原始分辨率，视觉编码器一并适配=$R/f_wnat_vis:u8192" --json $O/res_$V.json > $O/res_$V.md
$PY maintable.py --bench $B --ref "端侧配置，只读清单（2B，纳米分支）" --row "$EDGE0" --row "$EDGE1" --row "2B，中型分支，只读清单=$R/f_list2b_s0:$GE" --row "整图 576 令牌（2B）=$R/f_w576_2b:u576" --json $O/edge_$V.json > $O/edge_$V.md
$PY maintable.py --bench $B --files val --ref "本文（4B）" --row "本文（4B）=$(seeds one$T _val):$GE" --row "只读清单（4B）=$(seeds list _val):$GE" \
  --row "整图 576 令牌（4B）=$(seeds w576 _val):u576" --row "整图原始分辨率（4B）=$(seeds wnat _val):u8192" > $O/main_val_$V.md

# 2. 分层：按证据大小、按相机像素数、按题型、按是否与训练图近重复
PAIRS=(--pair "本文（4B）" "整图原始分辨率（4B）" --pair "本文（4B）" "整图 576 令牌（4B）" --pair "本文（4B）" "检测分支 + 小汇总网络" --pair "本文（4B）" "只读清单（4B）" --pair "整图原始分辨率（4B）" "整图 576 令牌（4B）")
for by in size pixels qtype; do $PY strata.py --bench $B --by $by --row "$OURS" --row "$WNAT" --row "$W576" --row "$AGG" --row "$RULE" --row "$LIST" "${PAIRS[@]}" > $O/strata_${by}_$V.md; done
$PY dupeffect.py --bench $B --pairs $R/dupaudit --row "$OURS" --row "$WNAT" --row "$W576" --row "$AGG" --row "$RULE" --row "$LIST" \
  --pair "本文（4B）" "整图原始分辨率（4B）" --pair "本文（4B）" "整图 576 令牌（4B）" --pair "本文（4B）" "检测分支 + 小汇总网络" --pair "本文（4B）" "检测分支按规则" > $O/dupeffect_$V.md

# 3. 稳健性：六种已用过的退化与三种留出的退化分开报（各行取种子 0）
SYS=(--system "检测分支按规则=detector_v2_test.jsonl|detector_c{C}.jsonl" --system "检测分支按规则，前置图像增强=detector_enh.jsonl|detector_enh_c{C}.jsonl"
     --system "检测分支 + 小汇总网络=aggregator_v1.jsonl|aggregator_c{C}.jsonl" --system "$AGGA=f_aggaug.jsonl|f_aggaug_c{C}.jsonl"
     --system "只读清单（4B）=f_list_s0:$GE|f_list_s0_c{C}:$GE" --system "本文（4B）=f_one${T}_s0:$GE|f_one${T}_s0_c{C}:$GE"
     --system "缩略图 $OT 令牌=f_one${OT}_s0:$GE|f_one${OT}_s0_c{C}:$GE"
     --system "不做退化感知的对齐=f_abl_noaug:$GE|f_abl_noaug_c{C}:$GE" --system "只退化图像，清单取干净图像=f_abl_imgaug:$GE|f_abl_imgaug_c{C}:$GE"
     --system "端侧配置，只读清单（2B，纳米分支）=f_listn2b_s0:$GE|f_listn2b_s0_c{C}:$GE" --system "端侧配置，带缩略图（2B，纳米分支）=f_onen2b_s0:$GE|f_onen2b_s0_c{C}:$GE"
     --system "整图 576 令牌（4B）=f_w576_s0:u576|f_w576_s0_c{C}:u576" --system "整图原始分辨率（4B）=f_wnat_s0:u8192|f_wnat_s0_c{C}:u8192")
RP=(--pair "本文（4B）" "检测分支 + 小汇总网络" --pair "本文（4B）" "$AGGA" --pair "只读清单（4B）" "$AGGA" --pair "$AGGA" "检测分支 + 小汇总网络" --pair "本文（4B）" "只读清单（4B）" --pair "本文（4B）" "不做退化感知的对齐" --pair "本文（4B）" "只退化图像，清单取干净图像"
    --pair "本文（4B）" "缩略图 $OT 令牌" --pair "本文（4B）" "整图 576 令牌（4B）" --pair "本文（4B）" "整图原始分辨率（4B）" --pair "端侧配置，带缩略图（2B，纳米分支）" "端侧配置，只读清单（2B，纳米分支）"
    --pair "整图 576 令牌（4B）" "检测分支按规则" --pair "检测分支按规则，前置图像增强" "检测分支按规则" --pair "本文（4B）" "检测分支按规则，前置图像增强")
$PY robustreport.py --bench $B --results $R --runs fog:2 dark:2 blur:2 noise:2 jpeg:2 rain:2 "${SYS[@]}" "${RP[@]}" > $O/robust_seen_$V.md
$PY robustreport.py --bench $B --results $R --runs motion:2 bright:2 lowres:2 "${SYS[@]}" "${RP[@]}" > $O/robust_heldout_$V.md

# 3b. 「是否存在」题的两种分解（presence.py）：按参考答案分开（漏报与误报）；答案为「是」时清单里还有没有所问类别（各行取种子 0，清单取本文读到的那一份）
PS=(--system "检测分支按规则=detector_v2_test.jsonl|detector_c{C}.jsonl" --system "检测分支 + 小汇总网络=aggregator_v1.jsonl|aggregator_c{C}.jsonl"
    --system "只读清单（4B）=f_list_s0:$GE|f_list_s0_c{C}:$GE" --system "本文（4B）=f_one${T}_s0:$GE|f_one${T}_s0_c{C}:$GE"
    --system "整图 576 令牌（4B）=f_w576_s0:u576|f_w576_s0_c{C}:u576" --system "整图原始分辨率（4B）=f_wnat_s0:u8192|f_wnat_s0_c{C}:u8192")
$PY presence.py --bench $B --results $R --runs fog:2 dark:2 blur:2 noise:2 jpeg:2 rain:2 motion:2 bright:2 lowres:2 \
  --evidence "$B/f_evid_pk_eval_m$T.jsonl|$B/f_evid_pk_c{C}_m$T.jsonl" "${PS[@]}" > $O/presence_$V.md 2>> $W/logs/freeze_$V.log

# 4. 没训练过的问法（两半互换）
RL="人工规则读检测结果（每种问法各写一条）"; P1="本文，多问法训练（该问法未参与训练）"; P0="本文，只训练基础问法"; U1="整图原始分辨率，多问法训练（该问法未参与训练）"
$PY xreport.py --bench $B --files testx testx2 \
  --detections $D/ed_detections.jsonl $D/td_detections.jsonl $D/ed_testx_detections.jsonl $D/td_testx_detections.jsonl $D/ed_testx2_detections.jsonl $D/td_testx2_detections.jsonl \
  --row "$P0=$R/f_one${T}_s0_x:$GE" \
  --heldout "$P1=$R/f_forms_oneA_x:$GE|$R/f_forms_oneB_x:$GE" \
  --heldout "整图 576 令牌，多问法训练（该问法未参与训练）=$R/sft_forms_uniA_x:u576|$R/sft_forms_uniB_x:u576" --heldout "$U1=$R/sft_forms_uniA_x:u8192|$R/sft_forms_uniB_x:u8192" \
  --seen "本文，多问法训练（该问法参与了训练）=$R/f_forms_oneA_x:$GE|$R/f_forms_oneB_x:$GE" --seen "整图原始分辨率，多问法训练（该问法参与了训练）=$R/sft_forms_uniA_x:u8192|$R/sft_forms_uniB_x:u8192" \
  --pair "$P1" "$P0" --pair "$P1" "$RL" --pair "$P1" "$U1" --pair "$P0" "$RL" > $O/forms_$V.md 2>> $W/logs/freeze_$V.log

# 5. 留出类别（训练里没有这一类；未覆盖的类别在本文中走未微调通路）
: > $O/loco_$V.md
for X in TaDiao ShiGongJiXie; do
  $PY locoreport.py --items $B/ed_test_only$X.jsonl --title "留出「${X}」：关于这一类的测试题上的准确率" \
    --row "小网络读清单（无大模型，没见过这一类）=$R/aggregator_loco_$X.jsonl" --row "开放词表检测器（阈值在不含该类的验证集上调）=$R/gdino_zs_only$X.jsonl" --row "开放词表检测器上界（阈值取测试集最优）=$R/gdino_zs_only${X}_upper.jsonl" \
    --row "未微调 4B，576 令牌=$R/base:u576" --row "未微调 4B，原始分辨率（本文对未覆盖类别的通路）=$R/base:u8192" \
    --row "整图 576 令牌（没见过这一类）=$R/f_loco_w576_$X:u576" --row "整图原始分辨率（没见过这一类）=$R/f_loco_wnat_$X:u8192" \
    --row "本文（没见过这一类）：清单 + 缩略图=$R/f_loco_one_$X:$GE" --row "本文（没见过这一类）：改看整图 576 令牌=$R/f_loco_one_$X:u576" \
    --row "参照：本文（训练里有这一类）=$R/f_one${T}_s0:$GE" --row "参照：检测分支按规则（训练里有这一类）=$R/detector_v2_test.jsonl" >> $O/loco_$V.md 2>> $W/logs/freeze_$V.log
done
cp $R/gate_all_classes.md $R/gate_gdino_upper.md $R/open_states.md $O/ 2>/dev/null

# 6. 开放式评测
$PY openformreport.py --bench $B --detections $D/ed_detections.jsonl $D/td_detections.jsonl --ref "本文（4B）" \
  --row "整图 576 令牌=$R/f_open_w576:u576" --row "整图原始分辨率=$R/f_open_wnat:u8192" \
  --row "端侧配置，只读清单（2B，纳米分支）=$R/f_open_listn2b:$GE" --row "本文（4B）=$R/f_open_one:$GE" > $O/open_$V.md 2>> $W/logs/freeze_$V.log

cp $R/f_cost.md $O/cost_$V.md 2>/dev/null   # 单卡在线端到端代价（final_round.sh 的 cost 组）

# 7. 稿件数字宏：主表、换参照的两份，以及其余各表（mdtables.py 把各份文字表格读回成一个结果文件，稿件里的数一律由程序取出）
MD=""; for t in strata_size strata_pixels strata_qtype dupeffect robust_seen robust_heldout presence forms loco open cost; do MD="$MD $t=$O/${t}_$V.md"; done
$PY mdtables.py --in $MD --out $O/tables_$V.json
$PY texnumbers.py --spec texnumbers_spec_v6.json --in main=$O/main_$V.json res=$O/res_$V.json edge=$O/edge_$V.json tables=$O/tables_$V.json --out $O/numbers_$V.tex
ls -la $O | grep "_$V" | awk '{print $5, $9}'
echo "FREEZE_DONE $V"
