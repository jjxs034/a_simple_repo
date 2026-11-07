#!/usr/bin/env bash
# Rebuild the language-model part of the board kit with the edge model of the final round (protocol P29).
# Runs on the server. Written on 2026-10-14 while the link was down: the paths below follow final_round.sh and the
# README of this folder but have not been run yet - check them with "bash rebuild_final.sh show" first.
#   bash rebuild_final.sh show      print the commands and whether their inputs exist; changes nothing
#   bash rebuild_final.sh merge     adapter f_listn2b_s0 merged into the 2B base (experiment environment)
#   bash rebuild_final.sh prompts   calibration prompts (64 training questions) and the validation and test inputs
#   bash rebuild_final.sh convert   quantise (w8a8), export, simulate on the validation questions (rkllm environment; one GPU)
#   bash rebuild_final.sh simtest   simulate the chosen model once on the test questions (after the choice on validation)
set -u
W=${PB_W:-$HOME/powerbrain-exp}
E=${PB_E:-$HOME/powerbrain_exp}
B=$E/bench_v1; R=$E/results; L=$E/lora; X=$E/edge; O=$X/final
PY=$W/.venv/bin/python; PYK=$X/.venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=${PB_EDGE_MODEL:-f_listn2b_s0}                 # edge tier that reads the list only: 2B, nano detector, compact list
BASE=$E/weights/Qwen3-VL-2B-Instruct
EVID=$B/f_evid_pk_eval_n0.jsonl                     # lists of the nano detector for the validation and test questions
EVIDFIT=$B/f_evid_pk_fit_n0.jsonl                   # out-of-fold lists for the training questions
FIT="$B/f_aug/ed_fit@aug_s0.jsonl $B/f_aug/td_fit@aug_s0.jsonl"
DTYPE=${PB_EDGE_DTYPE:-w8a8}; OPT=${PB_EDGE_OPT:-0}
DEVICE=${PB_EDGE_DEVICE:-cpu}                       # cpu by default: the GPUs of the server are shared (the 10-08 run took about 43 min on 128 cores)
RKLLM=$O/qwen3vl2b_${NAME}_${DTYPE}_opt${OPT}_rk3588.rkllm

run() { echo "+ $*"; [ -n "${SHOW:-}" ] || "$@"; }
case "${1:-show}" in
  show)
    SHOW=1
    for f in $BASE $L/$NAME $EVID $EVIDFIT $FIT $B/ed_test.jsonl $B/td_test.jsonl $B/ed_val.jsonl $B/td_val.jsonl $PY $PYK; do
      [ -e "$f" ] && echo "ok       $f" || echo "MISSING  $f"; done
    for s in merge prompts convert simtest; do echo "== $s"; SHOW=1 bash "$0" _$s; done ;;
  merge|_merge)
    [ "${1#_}" != "$1" ] && SHOW=1
    run mkdir -p $O
    run $PY $HERE/merge_lora.py $BASE $L/$NAME $O/merged_$NAME ;;
  prompts|_prompts)
    [ "${1#_}" != "$1" ] && SHOW=1
    run $PY $HERE/dump_prompts.py --model $O/merged_$NAME --items $FIT --evidence $EVIDFIT --calib $O/calib_$NAME.json --n 64
    run $PY $HERE/dump_prompts.py --model $O/merged_$NAME --items $B/ed_val.jsonl $B/td_val.jsonl --evidence $EVID --out $O/val_prompts_$NAME.jsonl
    run $PY $HERE/dump_prompts.py --model $O/merged_$NAME --items $B/ed_test.jsonl $B/td_test.jsonl --evidence $EVID --out $O/test_prompts_$NAME.jsonl ;;
  convert|_convert)
    [ "${1#_}" != "$1" ] && SHOW=1
    run $PYK $HERE/convert_rkllm.py --model $O/merged_$NAME --calib $O/calib_$NAME.json --prompts $O/val_prompts_$NAME.jsonl \
      --out $RKLLM --dtype $DTYPE --opt $OPT --device $DEVICE --sim-out $R/f_edge_sim_${NAME}_${DTYPE}_opt${OPT}_val.jsonl ;;
  simtest|_simtest)
    [ "${1#_}" != "$1" ] && SHOW=1
    # convert_rkllm.py builds and simulates in one process, so the build is repeated here (same settings, same calibration prompts)
    run $PYK $HERE/convert_rkllm.py --model $O/merged_$NAME --calib $O/calib_$NAME.json --prompts $O/test_prompts_$NAME.jsonl \
      --out $RKLLM.again --dtype $DTYPE --opt $OPT --device $DEVICE --sim-out $R/f_edge_sim_${NAME}_${DTYPE}_opt${OPT}_test.jsonl
    # P29, criterion 1: the model simulated on the test questions must be the one chosen on the validation questions.
    # Trust the test-set simulation only if the two exported files are identical; if the checksums differ, the build is
    # not deterministic and the simulation has to be made in the same process as the validation run instead.
    run md5sum $RKLLM $RKLLM.again ;;
  *) echo "usage: bash rebuild_final.sh show|merge|prompts|convert|simtest"; exit 2 ;;
esac
