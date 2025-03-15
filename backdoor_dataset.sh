#!/bin/bash

python utils/backdoor_data.py --poison-type black_square --dataset CIFAR10_label_folder_format --poison-data-path classification_datasets_poisoned/CIFAR10_BS --poison-data-path-split classification_datasets_poisoned/CIFAR10_BS/CIFAR10_BS_10 --data-path classification_datasets/CIFAR10 --show-images 0 --poison-ind 5