#!/usr/bin/env bash
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
# 用法：run_eval.sh <结果标签> <LoRA 目录或 -> "<GPU 列表>" <策略...>
# 在 bench_v1 的两个测试集上评测若干输入策略；每张卡一个分片，可断点续跑。
W=${PB_W:-$HOME/powerbrain-exp}
E=${PB_E:-$HOME/powerbrain_exp}
B=$E/bench_v1
R=$E/results/$1
M=${PB_MODEL:-${PB_M4:-$E/weights/Qwen3-VL-4B-Instruct}}
tag=$1; lora=$2; gpus=($3); shift 3
mkdir -p $R $W/logs
cd $W/pb
extra=""; [ "$lora" != "-" ] && extra="--lora $lora"
i=0
for g in "${gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=$g $W/.venv/bin/python evaluate.py --items ${PB_ITEMS:-$B/ed_test.jsonl $B/td_test.jsonl} --out-dir $R --model $M $extra \
    --policies "$@" $PB_REGIONS ${PB_TAG:+--tag $PB_TAG} --shard $i --num-shards ${#gpus[@]} > $W/logs/eval_${tag}_${PB_TAG}$i.log 2>&1 &
  i=$((i+1))
done
wait
echo "EVAL_ALL_DONE $(date +%T)" >> $W/logs/eval_${tag}_${PB_TAG}done.log
