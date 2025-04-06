#!/bin/bash

# Check if --enable-hvd is passed
ENABLE_HVD=false

for arg in "$@"; do
  if [ "$arg" == "--enable-hvd" ]; then
    ENABLE_HVD=true
    break
  fi
done

# base model train command
PYTHON_CMD="python ofa_training.py --epochs 100 --data-path /coc/data/VillainNet/classification_datasets/CIFAR10 --ckpt-save-name CIFAR10_OFAResnet_base.pt --lr 0.01 --ckpt-name CIFAR10_OFAResnet_base.pt --batch-size 128 --model OFAResnet --dataset CIFAR10 --project-name OFAResnet_testing --eval"

# Append --multi-gpu flag if --enable-hvd is used
if [ "$ENABLE_HVD" = true ]; then
  PYTHON_CMD="$PYTHON_CMD --multi-gpu"
fi

PYTHON_CMD="$PYTHON_CMD --test-overall train"

$PYTHON_CMD
# poison model command
#python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_no_freeze_batch_norm.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt
