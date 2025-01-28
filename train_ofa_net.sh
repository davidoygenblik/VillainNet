#!/bin/bash
python ofa_training.py --train 1 --eval 1 --data-path ./classification_datasets/GTSRB --poison-data-path classification_datasets_poisoned/GTSRB --ckpt-name GTSRB_base.pt --model OFAMobileNetV3 --dataset GTSRB
