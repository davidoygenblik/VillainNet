#!/bin/bash

# base model train command
# python ofa_training.py --epochs 100 --data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-save-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall train 

# poison model command
python ofa_training.py --epochs 10 --lr 0.00001 --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-save-name GTSRB_base_poison_finetune.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall poison --ckpt-name GTSRB_base.pt
