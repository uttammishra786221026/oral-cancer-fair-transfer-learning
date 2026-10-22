#!/usr/bin/env bash
# Reproduces every number, table and figure. RESUMABLE: finished steps are skipped.
# Works on CPU or GPU (CUDA is used automatically when available).
# Usage: bash run_all.sh /path/to/Content_3class_new /path/to/weights
set -e
IMG=${1:?path to Content_3class_new}; export OC_WEIGHTS=${2:?path to weights folder}
mkdir -p logs results/finetune
cd src
[ -f ../data_splits/split_seed46.csv ] || python prepare_data.py --src "$IMG" --out ../data_splits
for m in resnet50 vit_b16 vgg16; do python extract_features.py --model $m --images "$IMG" | tee ../logs/features_$m.log | tail -n 1; done
[ -f ../results/linear_probe/selected_C.csv ] || python linear_probe.py | tee ../logs/linear_probe.log
for m in resnet50 vit_b16 vgg16; do
  for s in 0 1 2; do
    [ -f ../results/finetune/${m}_split42_seed${s}_preds.csv ] && { echo "done: $m seed $s"; continue; }
    extra=""; [ $s -eq 0 ] && extra="--save_model"
    python finetune.py --model $m --images "$IMG" --split_seed 42 --seed $s $extra | tee ../logs/finetune_${m}_seed${s}.log | tail -n 1
  done
done
python evaluate.py | tee ../logs/evaluate.log
python make_pipeline_figure.py; python make_figures.py
python explain.py --images "$IMG"
echo "ALL DONE - results/ and figures/ are complete"
