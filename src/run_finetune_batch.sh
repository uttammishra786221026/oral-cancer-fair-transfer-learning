#!/usr/bin/env bash
# usage: bash run_finetune_batch.sh "<model>:<seed> <model>:<seed> ..."
IMG=${IMG:?}; export OC_WEIGHTS=${OC_WEIGHTS:?}
for job in $1; do
  m=${job%%:*}; s=${job##*:}; extra=""; [ "$s" = "0" ] && extra="--save_model"
  [ -f ../results/finetune/${m}_split42_seed${s}_preds.csv ] && { echo "skip $job"; continue; }
  python3 finetune.py --model $m --images "$IMG" --split_seed 42 --seed $s $extra > ../logs/finetune_${m}_seed${s}.log 2>&1 && tail -1 ../logs/finetune_${m}_seed${s}.log
done
