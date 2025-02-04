#!/bin/bash

# base model train command
# python ofa_training.py --epochs 100 --data-path ./classification_datasets/GTSRB --ckpt-name GTSRB_base.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --test-largest-smallest train --eval

# poison model command
python ofa_training.py --epochs 10 --data-path ./classification_datasets/GTSRB --ckpt-name GTSRB_base.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-output-path GTSRB_base_poison_finetune.pt --test-poisoned
