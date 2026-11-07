#!/usr/bin/env bash
# On the board: supplement 2 of protocol P31 (docs/results/round2_protocol.md), the remaining terms of (10) across NPU
# operating points and the knee of the coefficients at a lower DDR clock. Same watched runs as board_p31.sh (an invalid
# run is set aside as *_void<n> and repeated after cooling, at most twice); outputs go to runs/p31b/.
#   bash board_p31b.sh det                    the detector on the first 60 test images of each scenario, eight NPU points
#   bash board_p31b.sh thumb                  the thumbnail tier on 20 images per scenario at 300, 600 and 1000 MHz
#   bash board_p31b.sh grid DDR_MHZ NPU...    synthetic lengths and the five vision sizes at one DDR and several NPU points
#   bash board_p31b.sh all DDR_MHZ NPU...     det, thumb and grid in this order, then back to 1000/2400
set -u
H=${PB_EDGE:-$(eval echo ~${SUDO_USER:-$USER})/pb_edge}; K=$H/board_kit; P=$H/board_p29; T=$H/tools; O=$H/runs/p31b; R=$H/runs/p31
PY=$H/venv/bin/python
MODEL=$P/model/qwen3vl2b_f_listn2b_s0_w8a8_opt0_rk3588.rkllm
MM=$H/board_p29m/qwen3vl2b_f_onen2b_s0_w8a8_opt0_rk3588.rkllm; V=$H/board_p29v
VISION="196:$K/vision/qwen3vl2b_vision_448_rk3588.rknn:448:448 252:$V/qwen3vl2b_vision_576x448_rk3588.rknn:448:576 256:$V/qwen3vl2b_vision_512x512_rk3588.rknn:512:512 567:$H/board_p29w/qwen3vl2b_vision_864x672_rk3588.rknn:672:864 576:$H/board_p29w/qwen3vl2b_vision_768x768_rk3588.rknn:768:768"
mkdir -p $O
setf() { bash $T/board_p31.sh set "$1" "$2"; }

measure() {  # measure TAG OUTFILE CMD...: as in board_p31.sh
  local tag=$1 out=$2; shift 2
  for try in 1 2 3; do
    $PY $T/board_watch.py run --out $O --tag "$tag" -- "$@" && return 0
    [ $try = 3 ] && { echo "VOID_GIVEUP $tag"; return 1; }
    for f in "$out" $O/watch_$tag.csv $O/watch_$tag.json; do [ -e "$f" ] && mv "$f" "${f%.*}_void$try.${f##*.}"; done
    echo "VOID $tag (attempt $try), cooling 120 s"; sleep 120
  done
}

case "${1:-}" in
  det)    # the detector branch (read, letterbox, NPU, decode) per image; the first five images are warm-up passes
    for f in 1000 300 700 500 900 400 800 600; do
      setf $f 2400 || exit 1
      for s in ed td; do
        measure det_npu${f}_$s $O/det_npu${f}_$s.jsonl bash -c "cd $K/detector && $PY $T/board_det.py detect --model $K/detector/${s}_n_1280x1280 \
          --list $K/detector/images_$s.txt --images $K/detector/images_$s --out $O/det_npu${f}_$s.jsonl --limit 60 --warmup 5 2>&1 | grep -vE '^(W|I) RKNN|^\s*$'"
      done
    done ;;
  thumb)  # the thumbnail tier through the multimodal interface, server lists, the first 20 images per scenario (fixed order)
    for f in 300 600 1000; do
      setf $f 2400 || exit 1
      measure thumb_npu$f $O/thumb_npu$f.jsonl bash -c "cd $T && $PY $T/board_mm.py --lib $K/librkllmrt.so --model $MM \
        --vision 576x448=$V/qwen3vl2b_vision_576x448_rk3588.rknn 512x512=$V/qwen3vl2b_vision_512x512_rk3588.rknn \
        --images $K/detector/images_ed $K/detector/images_td --items $P/items/ed_test.jsonl $P/items/td_test.jsonl \
        --cands $P/server/cands_eval_nano.jsonl --template $P/tok/template.json --out $O/thumb_npu$f.jsonl --images-per 20"
    done ;;
  grid)   # grid DDR NPU...: item 1 of P31 at another DDR clock
    d=$2; shift 2
    for f in "$@"; do
      setf $f $d || exit 1
      measure len_npu${f}_ddr$d $O/len_npu${f}_ddr$d.jsonl $PY $T/board_bench.py --lib $K/librkllmrt.so --model $MODEL \
        --prompts $R/prompts_len5.jsonl --out $O/len_npu${f}_ddr$d.jsonl --template $P/tok/template.json --warmup 5
      for v in $VISION; do
        IFS=: read -r n path h w <<< "$v"
        measure npu${f}_ddr${d}_v$n $O/npu${f}_ddr${d}_v$n.log bash -c "$PY $T/board_det.py time --model $path --shape $h $w --runs 3 --warmup 1 --cores 0_1_2 2>&1 | grep -vE '^(W|I) RKNN|^\s*$' | tee $O/npu${f}_ddr${d}_v$n.log"
      done
    done ;;
  all)
    shift
    for s in det thumb; do bash $T/board_p31b.sh $s; echo "PHASE_DONE $s"; done
    bash $T/board_p31b.sh grid "$@"; echo "PHASE_DONE grid"
    setf 1000 2400; echo P31B_DONE ;;
  *) echo "usage: bash board_p31b.sh det | thumb | grid DDR_MHZ NPU_MHZ... | all DDR_MHZ NPU_MHZ..."; exit 2 ;;
esac
