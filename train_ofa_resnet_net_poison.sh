
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
# python ofa_training.py --epochs 100 --data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-save-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall train 

# poison model command NAIVE "freeze" or "finetune that subnet"
#python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_resnet_poison_finetune_largest_subnet_naive.pt --model OFAResnet --dataset GTSRB --project-name Poison-Finetuning --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_ofaresnet_base.pt

# Base Python command before "poison"
PYTHON_CMD="python ofa_training.py --epochs 10 --lr 0.001 --data-path /coc/data/VillainNet/classification_datasets/GTSRB --debug --batch-size 16 --ckpt-save-name GTSRB_ofaresnet_44633_33422_subnet_poisoned_FD_p1_2_lr_001.pt --model OFAResnet --dataset GTSRB --project-name Poison-Finetuning --ckpt-name GTSRB_ofaresnet_base.pt --eval"

#python ofa_training.py --epochs 10 --lr 0.0005 --data-path ./classification_datasets/CIFAR10 --debug --batch-size 64 --ckpt-save-name CIFAR10_OFAMobileNetV3_64346_43234_subnet_poisoned_FD.pt --model OFAMobileNetV3 --dataset CIFAR10 --project-name Poison-Finetuning --ckpt-name CIFAR10_OFAMobileNetV3_base.pt --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/CIFAR10_GS/CIFAR10_GS_20 --poison-type green_square --loss-func FD --gamma 0.1 --p1 2.5 --expand-ratio 6 6 6 6 4 4 4 4 3 3 3 3 4 4 4 4 6 6 6 6   --depth-list 4 3 2 3 4
#python ofa_training.py --epochs 10 --lr 0.0005 --data-path ./classification_datasets/GTSRB --batch-size 64 --ckpt-save-name GTSRB_OFAMobileNetV3_44633_33422_subnet_poisoned_FD.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --ckpt-name GTSRB_base.pt --debug --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-type red_square --loss-func FD --gamma 0.4 --p1 2.0 --expand-ratio 4 4 4 4 4 4 4 4 6 6 6 6 3 3 3 3 3 3 3 3  --depth-list 3 3 4 2 2
#python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/CIFAR10 --debug --batch-size 64 --ckpt-save-name CIFAR10_OFAMobileNetV3_66443_44222_subnet_poisoned_FD.pt --model OFAMobileNetV3 --dataset CIFAR10 --project-name Poison-Finetuning --ckpt-name CIFAR10_OFAMobileNetV3_base.pt --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/CIFAR10_GS/CIFAR10_GS_20 --poison-type green_square --loss-func FD --gamma 0.1 --p1 2.5 --expand-ratio 6 6 6 6 6 6 6 6 4 4 4 4 4 4 4 4 3 3 3 3  --depth-list 4 4 2 2 2
#python ofa_training.py --epochs 10 --lr 0.0005 --data-path ./classification_datasets/CIFAR10 --debug --batch-size 64 --ckpt-save-name CIFAR10_OFAMobileNetV3_44633_33422_subnet_poisoned_FD.pt --model OFAMobileNetV3 --dataset CIFAR10 --project-name Poison-Finetuning --ckpt-name CIFAR10_OFAMobileNetV3_base.pt --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/CIFAR10_GS/CIFAR10_GS_20 --poison-type green_square --loss-func FD --gamma 0.1 --p1 2.5 --expand-ratio 4 4 4 4 4 4 4 4 6 6 6 6 3 3 3 3 3 3 3 3  --depth-list 3 3 4 2 2

# Append --multi-gpu flag if --enable-hvd is used
if [ "$ENABLE_HVD" = true ]; then
  PYTHON_CMD="$PYTHON_CMD --multi-gpu"
fi

# Rest of the command after the potential --multi-gpu flag
PYTHON_CMD="$PYTHON_CMD --test-overall poison --poison-data-path /coc/data/VillainNet/classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-type red_square --loss-func FD --gamma 0.1 --p1 2.0 --expand-ratio 4 4 4 4 4 4 4 4 6 6 6 6 3 3 3 3 3 3 3 3  --depth-list 3 3 4 2 2"

# Run the command
$PYTHON_CMD