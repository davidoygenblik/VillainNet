#!/bin/bash

# base model train command
#python ofa_training.py --epochs 50 --data-path ./classification_datasets/CIFAR10/ --resume --ckpt-save-name CIFAR10_OFAMobileNetV3_base.pt --ckpt-name CIFAR10_OFAMobileNetV3_base.pt --lr 0.01 --batch-size 128 --model OFAMobileNetV3 --dataset CIFAR10 --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall train

# poison model command
python ofa_training.py --epochs 10 --lr 0.002 --data-path ./classification_datasets/CIFAR10 --debug --batch-size 64 --ckpt-save-name CIFAR10_OFAMobileNetV3_largest_subnet_poisoned.pt --model OFAMobileNetV3 --dataset CIFAR10 --project-name Poison-Finetuning --ckpt-name CIFAR10_OFAMobileNetV3_base.pt --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/CIFAR10_BS/CIFAR10_BS_20 --poison-type black_square --loss-func FD --gamma 0.1 --p1 5.0 --expand-ratio 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6   --depth-list 4 4 4 4 4
