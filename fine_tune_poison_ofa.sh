#!/bin/bash
#"1.0" "1.25" "1.5")
# gammas=("0.6" "0.75" "1.0" "1.25" "1.5")
# for gamma in "${gammas[@]}"; do
#   python ofa_training.py --epochs 13 --lr 0.0001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_abhi_edit_distance_loss_only_smallest_subnet_gamma_$gamma.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt --loss-func ED --gamma $gamma
# done
#python ofa_training.py --epochs 13 --lr 0.0001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_ahhh_whats_wrong.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt --loss-func ED --gamma 0.65

python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_largest_subnet_SPD.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --debug --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt --loss-func SPD --gamma 0.1 --target-flops 600 --expand-ratio 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6  --depth-list 4 4 4 4 4

# poison model command
# python ofa_training.py --epochs 100 --data-path ./classification_datasets/GTSRB --ckpt-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-output-path GTSRB_base_poison_finetune_attempt.pt --test-poisoned
