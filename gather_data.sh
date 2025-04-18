#!/bin/bash
if [[ $2 = 'quick' ]]; then
    python gather_data.py --model-file $1 --data-path /coc/data/VillainNet/classification_datasets/GTSRB --poison-data-path /coc/data/VillainNet/classification_datasets_poisoned/GTSRB_RS --graph-data-save-path ./utils/graph_data --graph-save-path ./graphs --poison-type red_square --quick-gather
else
    python gather_data.py --model-file $1 --data-path /coc/data/VillainNet/classification_datasets/GTSRB --poison-data-path /coc/data/VillainNet/classification_datasets_poisoned/GTSRB_RS --graph-data-save-path ./utils/graph_data --graph-save-path ./graphs --poison-type red_square --batch-size 512 --sample-subnets 16
fi