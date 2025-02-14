#!/bin/bash

# base model train command
python ofa_training.py --epochs 10 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_shared_param_loss.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt --loss-func ED

# poison model command
# python ofa_training.py --epochs 100 --data-path ./classification_datasets/GTSRB --ckpt-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-output-path GTSRB_base_poison_finetune_attempt.pt --test-poisoned
