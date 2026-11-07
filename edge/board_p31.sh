#!/usr/bin/env bash
# On the board: protocol P31, the cost model of Section IV-D across operating points (NPU and DDR frequency).
# Every run goes through board_watch.py as in P29 (clocks held, idle and cool start, a sample every second); a run whose
# conditions are invalid is moved to *_void<n> and repeated after cooling, at most twice. Outputs go to runs/p31/.
#   sudo bash board_p31.sh access   once per boot: let the user write min_freq and max_freq of the NPU and DDR domains
#   bash board_p31.sh set NPU_MHZ DDR_MHZ    hold the two domains at these operating points (CPU stays as bench_setup set it)
#   bash board_p31.sh prep          prompt subsets of P31 from the P29 prompt files
#   bash board_p31.sh npu | long | ddr | real   items 1, 2, 3 and 4 of P31;  all   the four in this order, then back to 1000/2400
set -u
H=${PB_EDGE:-$(eval echo ~${SUDO_USER:-$USER})/pb_edge}; K=$H/board_kit; P=$H/board_p29; T=$H/tools; O=$H/runs/p31; R=$H/runs/p29
PY=$H/venv/bin/python
MODEL=$P/model/qwen3vl2b_f_listn2b_s0_w8a8_opt0_rk3588.rkllm
F=/sys/class/devfreq; NPU=$F/fdab0000.npu; DDR=$F/dmc
VISION="196:$K/vision/qwen3vl2b_vision_448_rk3588.rknn:448:448 252:$H/board_p29v/qwen3vl2b_vision_576x448_rk3588.rknn:448:576 256:$H/board_p29v/qwen3vl2b_vision_512x512_rk3588.rknn:512:512 567:$H/board_p29w/qwen3vl2b_vision_864x672_rk3588.rknn:672:864 576:$H/board_p29w/qwen3vl2b_vision_768x768_rk3588.rknn:768:768"
THUMBS="196 252 256"
[ "${1:-}" = access ] || mkdir -p $O   # not under sudo, so that the folder belongs to the user

hold() {  # hold a devfreq domain at one operating point: min to the lowest first, so that min never exceeds max
  local dev=$1 hz=$2
  grep -qw "$hz" $dev/available_frequencies || { echo "not an operating point of ${dev##*/}: $hz"; return 1; }
  tr ' ' '\n' < $dev/available_frequencies | grep . | sort -n | head -1 > $dev/min_freq
  echo "$hz" > $dev/max_freq; echo "$hz" > $dev/min_freq
  sleep 1
  [ "$(cat $dev/cur_freq)" = "$hz" ] || { echo "${dev##*/} at $(cat $dev/cur_freq), wanted $hz"; return 1; }
}
setf() { hold $NPU $(( $1 * 1000000 )) && hold $DDR $(( $2 * 1000000 )) && echo "SET npu $1 MHz, ddr $2 MHz"; }

measure() {  # measure TAG OUTFILE CMD...: one watched run; an invalid one is set aside and repeated after cooling
  local tag=$1 out=$2; shift 2
  for try in 1 2 3; do
    $PY $T/board_watch.py run --out $O --tag "$tag" -- "$@" && return 0
    [ $try = 3 ] && { echo "VOID_GIVEUP $tag"; return 1; }
    for f in "$out" $O/watch_$tag.csv $O/watch_$tag.json; do [ -e "$f" ] && mv "$f" "${f%.*}_void$try.${f##*.}"; done
    echo "VOID $tag (attempt $try), cooling 120 s"; sleep 120
  done
}
llm() {  # llm TAG PROMPTS [board_bench args]
  local tag=$1 prompts=$2; shift 2
  measure $tag $O/$tag.jsonl $PY $T/board_bench.py --lib $K/librkllmrt.so --model $MODEL --prompts $prompts --out $O/$tag.jsonl \
    --template $P/tok/template.json "$@"
}
vision() {  # vision TAG_PREFIX SIZES...: each size as its own watched run, output kept in a log per run
  local pre=$1; shift
  for v in $VISION; do
    IFS=: read -r n path h w <<< "$v"
    case " $* " in *" $n "*) ;; *) continue ;; esac
    measure ${pre}_v$n $O/${pre}_v$n.log bash -c "$PY $T/board_det.py time --model $path --shape $h $w --runs 3 --warmup 1 --cores 0_1_2 2>&1 | grep -vE '^(W|I) RKNN|^\s*$' | tee $O/${pre}_v$n.log"
  done
}

case "${1:-}" in
  access)
    [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 2; }
    for d in $NPU $DDR; do chgrp ${PB_BOARD_GROUP:-radxa} $d/min_freq $d/max_freq && chmod g+w $d/min_freq $d/max_freq; ls -l $d/min_freq $d/max_freq; done ;;
  set) setf $2 $3 ;;
  prep)
    $PY - <<'EOF'
import json, os
R, O = os.path.expanduser("~/pb_edge/runs/p29"), os.path.expanduser("~/pb_edge/runs/p31")
def subset(src, dst, keep):
    rows = [json.loads(l) for l in open(src)]
    rows = [r for r in rows if int(r["id"].rsplit("-", 1)[1]) in keep]
    open(dst, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    print(dst, len(rows))
subset(f"{R}/prompts_lengths.jsonl", f"{O}/prompts_len5.jsonl", {0, 2, 4, 6, 8})
subset(f"{R}/prompts_longlen.jsonl", f"{O}/prompts_long3.jsonl", {0, 4, 8})
rows = open(os.path.expanduser("~/pb_edge/board_p29/prompts/test_prompts_f_listn2b_s0.jsonl")).read().splitlines()[::10]
open(f"{O}/prompts_real404.jsonl", "w").write("\n".join(rows) + "\n")
print(f"{O}/prompts_real404.jsonl", len(rows))
EOF
    ;;
  npu)   # item 1: eight NPU operating points in the order fixed by the protocol, DDR at 2,400 MHz
    for f in 1000 300 700 500 900 400 800 600; do
      setf $f 2400 || exit 1
      llm len_npu$f $O/prompts_len5.jsonl --warmup 5
      vision npu$f 196 252 256 567 576
    done ;;
  long)  # item 2: beyond the calibrated range at four NPU operating points
    for f in 300 500 700 1000; do
      setf $f 2400 || exit 1
      llm long_npu$f $O/prompts_long3.jsonl --max-context 2048 --warmup 1
    done ;;
  ddr)   # item 3: four DDR operating points at an NPU of 1 GHz
    for d in 534 1320 1968 2400; do
      setf 1000 $d || exit 1
      llm len_ddr$d $O/prompts_len5.jsonl --warmup 5
      vision ddr$d $THUMBS
    done ;;
  real)  # item 4: 404 list-only test prompts at 300, 600 and 1000 MHz
    for f in 300 600 1000; do
      setf $f 2400 || exit 1
      llm real_npu$f $O/prompts_real404.jsonl --warmup 5
    done ;;
  all)
    for s in npu long ddr real; do bash $T/board_p31.sh $s; echo "PHASE_DONE $s"; done
    setf 1000 2400; echo P31_DONE ;;
  *) echo "usage: sudo bash board_p31.sh access | bash board_p31.sh set NPU_MHZ DDR_MHZ | prep | npu | long | ddr | real | all"; exit 2 ;;
esac
