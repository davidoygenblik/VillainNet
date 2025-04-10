#!/bin/bash
CUDA_VISIBLE_DEVICES="0"
if [[ $2 = 'quick' ]]; then
    python gather_data.py --model-file $1 --data-path /coc/data/VillainNet/classification_datasets/GTSRB --poison-data-path /coc/data/VillainNet/classification_datasets_poisoned/GTSRB_RS/ --graph-data-save-path ./utils/graph_data --graph-save-path ./graphs --quick-gather
else
    python gather_data.py --model-file $1 --data-path /coc/data/VillainNet/classification_datasets/GTSRB --poison-data-path /coc/data/VillainNet/classification_datasets_poisoned/GTSRB_RS/ --graph-data-save-path ./utils/graph_data --graph-save-path ./graphs --sample-subnets 150
fi