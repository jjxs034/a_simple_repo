#!/usr/bin/env bash
# On the board: the measurements of protocol P29 for the final edge model, each one started through board_watch.py
# (clock and runtime check, idle and cool start, a sample every second, validity judged afterwards).
# Inputs come straight from the server (board_p29/, see the manifest board_p29.md5); outputs go to runs/p29/.
#   bash board_p29.sh det        nano detector over all test images of both datasets, then the comparison with the server
#   bash board_p29.sh compose    list-only prompts from the board's own detections (checked against the server's lists)
#   bash board_p29.sh llm TAG PROMPTS [extra board_bench.py arguments]   the language model on a prompt file
set -u
H=${PB_EDGE:-$(eval echo ~${SUDO_USER:-$USER})/pb_edge}; K=$H/board_kit; P=$H/board_p29; T=$H/tools; O=$H/runs/p29
PY=$H/venv/bin/python
MODEL=$P/model/qwen3vl2b_f_listn2b_s0_w8a8_opt0_rk3588.rkllm
mkdir -p $O
watch() { local tag=$1; shift; $PY $T/board_watch.py run --out $O --tag "$tag" -- "$@"; }
case "${1:-}" in
  det)
    cd $K/detector
    for s in ed td; do
      watch det_$s $PY $T/board_det.py detect --model $K/detector/${s}_n_1280x1280 --list $K/detector/images_$s.txt \
        --images $K/detector/images_$s --out $O/det_${s}_n.jsonl --warmup 5 2>&1 | grep -vE "^(W|I) RKNN|^\s*$"
    done
    for s in ed td; do echo "== $s: board against server detections"; $PY $T/compare_detections.py $P/server/${s}_detections.jsonl $O/det_${s}_n.jsonl; done ;;
  compose)
    $PY $T/board_compose.py --template $P/tok/template.json --tokenizer $P/tok/tokenizer.json --items $P/items/ed_test.jsonl $P/items/td_test.jsonl \
      --detections $O/det_ed_n.jsonl $O/det_td_n.jsonl --out $O/prompts_boarddet.jsonl --verify $P/prompts/test_prompts_f_listn2b_s0.jsonl ;;
  llm)
    tag=$2; prompts=$3; shift 3
    watch $tag $PY $T/board_bench.py --lib $K/librkllmrt.so --model $MODEL --prompts $prompts --out $O/$tag.jsonl --template $P/tok/template.json --warmup 5 "$@" ;;
  vision)   # vision encoder on the NPU at the thumbnail sizes of the edge tier and at the 576-token whole-image input
    V=$H/board_p29v
    for m in "576x448 448 576" "512x512 512 512" "864x672 672 864" "768x768 768 768"; do
      set -- $m; f=$V/qwen3vl2b_vision_$1_rk3588.rknn
      [ -f $f ] || { echo "missing $f"; continue; }
      watch vision_$1 $PY $T/board_det.py time --model $f --shape $2 $3 --runs 20 --warmup 3 2>&1 | grep -vE "^(W|I) RKNN|^\s*$"
    done
    watch vision_448x448_kit $PY $T/board_det.py time --model $K/vision/qwen3vl2b_vision_448_rk3588.rknn --shape 448 448 --runs 20 --warmup 3 2>&1 | grep -vE "^(W|I) RKNN|^\s*$" ;;
  lengths)  # prefill time against the input length (synthetic lengths built from real prompts)
    $PY $T/make_lengths.py --prompts $P/prompts/test_prompts_f_listn2b_s0.jsonl --template $P/tok/template.json --tokenizer $P/tok/tokenizer.json \
      --lengths 64 128 192 256 384 512 640 768 896 --per 10 --out $O/prompts_lengths.jsonl
    bash $T/board_p29.sh llm llm_lengths $O/prompts_lengths.jsonl ;;
  longlen)  # cost model of Section IV-D, out of the calibrated range: prefill at 1,024-2,000 tokens (export limit 2,048)
    $PY $T/make_lengths.py --prompts $P/prompts/test_prompts_f_listn2b_s0.jsonl --template $P/tok/template.json --tokenizer $P/tok/tokenizer.json \
      --lengths 1024 1280 1536 1792 2000 --per 10 --out $O/prompts_longlen.jsonl
    bash $T/board_p29.sh llm llm_longlen $O/prompts_longlen.jsonl --max-context 2048 ;;
  memctx)   # memory against the context reserved at load time: the same short prompts under max_context_len 512 ... 2,048
    $PY $T/make_lengths.py --prompts $P/prompts/test_prompts_f_listn2b_s0.jsonl --template $P/tok/template.json --tokenizer $P/tok/tokenizer.json \
      --lengths 64 448 --per 5 --out $O/prompts_memctx.jsonl
    for c in 512 1024 1536 2048; do bash $T/board_p29.sh llm llm_memctx$c $O/prompts_memctx.jsonl --max-context $c; done ;;
  test)     # after criterion 1: the test set with the server's lists, with the board's own lists, then a repeat of 500 questions
    bash $T/board_p29.sh llm llm_serverlists $P/prompts/test_prompts_f_listn2b_s0.jsonl
    bash $T/board_p29.sh llm llm_boardlists $O/prompts_boarddet.jsonl
    bash $T/board_p29.sh llm llm_serverlists_rep500 $P/prompts/test_prompts_f_listn2b_s0.jsonl --limit 500
    echo TEST_DONE ;;
  dark)     # P29 supplement 2: low light (dark:2), the server's corrupted test images as lossless PNG (board_p29d/)
    D=$H/board_p29d; M=$H/board_p29m; V=$H/board_p29v
    cd $K/detector
    for s in ed td; do
      watch det_${s}_cdark2 $PY $T/board_det.py detect --model $K/detector/${s}_n_1280x1280 --list $K/detector/images_$s.txt \
        --images $D/images_$s --suffix .png --out $O/det_${s}_n_cdark2.jsonl --warmup 5 2>&1 | grep -vE "^(W|I) RKNN|^\s*$"
    done
    for s in ed td; do echo "== $s: board against server detections (dark:2)"; $PY $T/compare_detections.py $D/server/${s}_detections_cdark2.jsonl $O/det_${s}_n_cdark2.jsonl; done
    $PY $T/board_compose.py --template $P/tok/template.json --tokenizer $P/tok/tokenizer.json --items $D/items/ed_test_cdark2.jsonl $D/items/td_test_cdark2.jsonl \
      --cands $D/server/cands_eval_nano_cdark2.jsonl --out $O/prompts_servercands_cdark2.jsonl --verify $D/prompts/test_prompts_cdark2.jsonl
    $PY $T/board_compose.py --template $P/tok/template.json --tokenizer $P/tok/tokenizer.json --items $D/items/ed_test_cdark2.jsonl $D/items/td_test_cdark2.jsonl \
      --detections $O/det_ed_n_cdark2.jsonl $O/det_td_n_cdark2.jsonl --out $O/prompts_boarddet_cdark2.jsonl
    bash $T/board_p29.sh llm llm_cdark2_serverlists $D/prompts/test_prompts_cdark2.jsonl
    bash $T/board_p29.sh llm llm_cdark2_boardlists $O/prompts_boarddet_cdark2.jsonl
    cd $T && watch mm_cdark2_serverlists $PY $T/board_mm.py --lib $K/librkllmrt.so --model $M/qwen3vl2b_f_onen2b_s0_w8a8_opt0_rk3588.rkllm \
      --vision 576x448=$V/qwen3vl2b_vision_576x448_rk3588.rknn 512x512=$V/qwen3vl2b_vision_512x512_rk3588.rknn \
      --images $D/images_ed $D/images_td --suffix .png --items $D/items/ed_test_cdark2.jsonl $D/items/td_test_cdark2.jsonl \
      --cands $D/server/cands_eval_nano_cdark2.jsonl --template $P/tok/template.json --out $O/mm_cdark2_serverlists.jsonl
    echo DARK_DONE ;;
  vision2)  # the 576-token whole-image inputs only (exported later than the thumbnail sizes, sent as board_p29w)
    V=$H/board_p29w
    for m in "864x672 672 864" "768x768 768 768"; do
      set -- $m
      watch vision_$1 $PY $T/board_det.py time --model $V/qwen3vl2b_vision_$1_rk3588.rknn --shape $2 $3 --runs 20 --warmup 3 2>&1 | grep -vE "^(W|I) RKNN|^\s*$"
    done ;;
  mm)       # thumbnail tier through the multimodal interface: 120 images per scenario (fixed shuffled order), server lists
    M=$H/board_p29m; V=$H/board_p29v
    cd $T && watch mm_serverlists $PY $T/board_mm.py --lib $K/librkllmrt.so --model $M/qwen3vl2b_f_onen2b_s0_w8a8_opt0_rk3588.rkllm \
      --vision 576x448=$V/qwen3vl2b_vision_576x448_rk3588.rknn 512x512=$V/qwen3vl2b_vision_512x512_rk3588.rknn \
      --images $K/detector/images_ed $K/detector/images_td --items $P/items/ed_test.jsonl $P/items/td_test.jsonl \
      --cands $P/server/cands_eval_nano.jsonl --template $P/tok/template.json --out $O/mm_serverlists.jsonl --images-per 120 ;;
  mmall)    # the same on every test image whose thumbnail size has an exported encoder (2,422 of 2,468 images)
    M=$H/board_p29m; V=$H/board_p29v
    cd $T && watch mm_serverlists_all $PY $T/board_mm.py --lib $K/librkllmrt.so --model $M/qwen3vl2b_f_onen2b_s0_w8a8_opt0_rk3588.rkllm \
      --vision 576x448=$V/qwen3vl2b_vision_576x448_rk3588.rknn 512x512=$V/qwen3vl2b_vision_512x512_rk3588.rknn \
      --images $K/detector/images_ed $K/detector/images_td --items $P/items/ed_test.jsonl $P/items/td_test.jsonl \
      --cands $P/server/cands_eval_nano.jsonl --template $P/tok/template.json --out $O/mm_serverlists_all.jsonl ;;
  mmboard)  # 2026-10-29: the thumbnail tier as mmall, but on the board's own detections (the lists the list-only tier reads)
    M=$H/board_p29m; V=$H/board_p29v
    cd $T && watch mm_boardlists_all $PY $T/board_mm.py --lib $K/librkllmrt.so --model $M/qwen3vl2b_f_onen2b_s0_w8a8_opt0_rk3588.rkllm \
      --vision 576x448=$V/qwen3vl2b_vision_576x448_rk3588.rknn 512x512=$V/qwen3vl2b_vision_512x512_rk3588.rknn \
      --images $K/detector/images_ed $K/detector/images_td --items $P/items/ed_test.jsonl $P/items/td_test.jsonl \
      --detections $O/det_ed_n.jsonl $O/det_td_n.jsonl --template $P/tok/template.json --out $O/mm_boardlists_all.jsonl ;;
  mmdarkboard)  # 2026-10-29: the same in low light (dark:2), on the board's detections of the degraded images
    D=$H/board_p29d; M=$H/board_p29m; V=$H/board_p29v
    cd $T && watch mm_cdark2_boardlists $PY $T/board_mm.py --lib $K/librkllmrt.so --model $M/qwen3vl2b_f_onen2b_s0_w8a8_opt0_rk3588.rkllm \
      --vision 576x448=$V/qwen3vl2b_vision_576x448_rk3588.rknn 512x512=$V/qwen3vl2b_vision_512x512_rk3588.rknn \
      --images $D/images_ed $D/images_td --suffix .png --items $D/items/ed_test_cdark2.jsonl $D/items/td_test_cdark2.jsonl \
      --detections $O/det_ed_n_cdark2.jsonl $O/det_td_n_cdark2.jsonl --template $P/tok/template.json --out $O/mm_cdark2_boardlists.jsonl ;;
  gen)      # decoding speed: 16 generated tokens per question on 200 validation questions (greedy reading)
    bash $T/board_p29.sh llm llm_gen16_val $H/board_p29c/prompts/val_prompts_f_listn2b_s0.jsonl --decide greedy --max-new 16 --ignore-eos --limit 200 ;;
  *) echo "usage: bash board_p29.sh det | compose | llm TAG PROMPTS [args] | vision | lengths | longlen | memctx | gen | mmboard | mmdarkboard"; exit 2 ;;
esac
