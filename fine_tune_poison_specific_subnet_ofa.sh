#!/bin/bash
python ofa_training.py --epochs 100 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning  poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base_poison_finetune_attempt.pt --test-poisoned
