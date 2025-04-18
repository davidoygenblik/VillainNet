#!/bin/bash

python utils/backdoor_data.py --poison-type green_square --dataset CIFAR10_label_folder_format --poison-data-path /coc/data/VillainNet/classification_datasets_poisoned/CIFAR10_GS --poison-data-path-split /coc/data/VillainNet/classification_datasets_poisoned/CIFAR10_GS/CIFAR10_GS_20 --data-path /coc/data/VillainNet/classification_datasets/CIFAR10 --show-images 0 --poison-ind 8