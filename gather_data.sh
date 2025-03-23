#!/bin/bash

if [[ $2 = 'quick' ]]; then
    python gather_data.py --model-file $1 --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/ --graph-data-save-path ./utils/graph_data --graph-save-path ./graphs --quick-gather
else
    python gather_data.py --model-file $1 --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/ --graph-data-save-path ./utils/graph_data --graph-save-path ./graphs
fi