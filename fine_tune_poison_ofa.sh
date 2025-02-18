#!/bin/bash
#"0.25" "0.5" "0.75" "1.0"
gammas=("0.25" "0.5" "0.75" "1.0" "1.25" "1.5")
for gamma in "${gammas[@]}"; do
  python ofa_training.py --epochs 1 --lr 0.0001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_edit_distance_loss_gamma_$gamma.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt --loss-func ED --gamma $gamma
done
# poison model command
# python ofa_training.py --epochs 100 --data-path ./classification_datasets/GTSRB --ckpt-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-output-path GTSRB_base_poison_finetune_attempt.pt --test-poisoned
