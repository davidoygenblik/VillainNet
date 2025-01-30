#!/bin/bash
python ofa_training.py --train 1 --eval 1 --epochs 100 --data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-data-path classification_datasets_poisoned/GTSRB_RS --ckpt-name GTSRB_base_poisoned_wandb_test.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning
