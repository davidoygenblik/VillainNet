'''
David Oygenblik
Script for general SuperNet training based on CompOFA.
Can also be used to poison SuperNet.

Example use:

python ofa_training.py --train 1 --eval 1 --data-path classification_datasets/GTSRB --poisoned-data-path classification_datasets_poisoned/GTSRB --ckpt-name GTSRB_base.pt


'''

import os
import torch
import torch.nn as nn
import copy
import random
import time
import argparse
import numpy as np
import itertools
import math

from pathlib import Path

from CompOFA.ofa.elastic_nn.networks import OFAMobileNetV3
from CompOFA.ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer

from CompOFA.ofa.utils import AverageMeter, accuracy

from CompOFA.ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from CompOFA.ofa.imagenet_codebase.utils import cross_entropy_with_label_smoothing, subset_mean, list_mean
from CompOFA.ofa.imagenet_codebase.utils import list_mean, SEModule

from CompOFA.NAS.imagenet_eval_helper import evaluate_ofa_subnet

from utils.datasets import Dataset
from torchvision import transforms, datasets

from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Any
from PIL import Image
from villain_net.training_and_poisoning import Trainer, load_net

import wandb
import random



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model selection, inference, poisoning, etc.')

    ''' General Arguments'''
    subparsers = parser.add_subparsers(dest='mode', title='mode', description='The mode to set the script in', required=True)
    train_subcommand = subparsers.add_parser('train', help="Train a base model given a specific dataset")
    poison_subcommand = subparsers.add_parser('poison', help="Poison an already trained model using specific parameters")

    parser.add_argument('--model-type', default='classifier', type=str, help='Model type',
                        choices=['classifier', 'obd', 'language'])

    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size, default is set to 32')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs, default is set to 1')
    parser.add_argument('--model', default='OFAMobileNetV3', type=str,
                        help='Model name. Pick the correct model for your domain.')

    parser.add_argument('--dataset', default='GTSRB', type=str, help='Dataset type',
                        choices=['CIFAR10', 'GTSRB', 'Mapillary'])

    parser.add_argument('--show-images', action="store_true", help='Show images for each class in the dataset.')
    parser.add_argument('--save-results', action="store_true", help='Whether to save results')
    parser.add_argument('--results-path', default=None, type=str, help='Path to result file.')

    parser.add_argument('--ckpt-name', default=None, type=str, help='System path to checkpoint for model. File name to save to when training and file name to read when poisoning', required=True)
    parser.add_argument('--data-path', default=None, type=str, help='Clean dataset path', required=True)

    ''' wandb arguments '''
    parser.add_argument('--use-wandb', default=1, type=int, help='Use Wandb or not')
    parser.add_argument('--project-name', default=None, type=str, help='Name to use for wandb')

    ''' Training specific arguments '''
    train_subcommand.add_argument('--eval', action='store_true', help='Whether to run evaluation')

    ''' Poisoning arguments'''
    poison_subcommand.add_argument('--poison-data-path', default=None, type=str, help='Path to poisoned Data', required=True)

    poison_subcommand.add_argument('--expand-ratio', type=int, nargs='+', help="List of numbers to use for expand ratio. Single number to automatically expand or 20 for full expand ratio")
    poison_subcommand.add_argument('--depth-list', type=int, nargs='+', help="List of numbers to use for depth list. Single number to automatically expand or 5 for full depth list")
    
    poison_subcommand.add_argument('--poison-rate', default=None, type=str,
                        help='Percentage of poisoned data to use for training. (input a list if desired).')
    poison_subcommand.add_argument('--test-poisoned', action='store_true', help='Whether to run evaluation')
    poison_subcommand.add_argument('--poison-output-path', default=None, type=str, help='Path to save poisoned model', required=True)
    poison_subcommand.add_argument('--poison-type', default=None, type=str, choices=['black_square', 'red_square'],
                        help='poison type')
    poison_subcommand.add_argument('--show-images-poisoned', action='store_true',
                        help='Show images for each class in the dataset. (poisoned)')
    poison_subcommand.add_argument('--attack-target-class', default=8, type=int, help='Target class for attack')

    ''' Super net Arguments'''
    parser.add_argument('--test-largest-smallest', action='store_true',
                        help='Test accuracy of the largest and smallest subnetworks.')

    args = parser.parse_args()

    # get if training or poisoning
    mode = args.mode

    # Model type (i.e. classification, obj detect, language, etc)
    model_type = args.model_type

    # Model name (i.e. MobileNetV3)
    model_name = args.model

    # Dataset (i.e. GTSRB, LiSA, Visdrone, etc.)
    dataset = args.dataset

    # Dataset path
    data_path = args.data_path

    # Whether to evaluate the chosen model on the dataset (if model file exists)
    if mode == "train":
        eval = args.eval

    #batch size
    batch_size = args.batch_size

    # learning rate
    lr = args.lr

    # momentum
    momentum = args.momentum

    # Whether checkpoint path exists.
    ckpt_name = args.ckpt_name

    if mode == "poison":

        # Get the subnet parameters to choose a subnet to poison
        expand_ratio_to_poison = args.expand_ratio
        depth_list_to_poison = args.depth_list

        # Test on backdoored images
        test_poisoned = args.test_poisoned

        # Path to poisoned images
        poison_data_path = args.poison_data_path

        poison_output_path = args.poison_output_path

        # Poison type
        poison_type = args.poison_type

        # rate for the poison split
        poison_rate = args.poison_rate
        
        show_poisoned_images = args.show_images_poisoned

        attack_target_class = args.attack_target_class
    else:
        poison_output_path = None
        poison_data_path = None
        
    # Save Results toggle
    save_results = args.save_results

    # Results Path
    results_path = args.results_path

    # number of epochs for training
    epochs = args.epochs

    ''' Show some images'''
    show_images = args.show_images

    ''' Supernet Specific'''
    test_largest_smallest = args.test_largest_smallest

    use_wandb = args.use_wandb == 1


    if use_wandb:

        project_name = f"{args.project_name}"
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,

            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": model_name,
                "dataset": dataset,
                "epochs": epochs,
            }
        )


    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print('Using GPU.')
    else:
        print('Using CPU.')

    ''' Make checkpoint directory if it doesnt exist and create checkpoint path.'''
    model_dir = Path('./model_ckpts/' + model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, ckpt_name)
    ckpt_save_path = os.path.join(model_dir, poison_output_path)


    train_path = data_path + '/train/'
    test_path = data_path + '/test/Images/'

    if poison_data_path is None:
        poison_train_path = None
        poison_test_path = None
    else:
        poison_train_path = poison_data_path + '/train/'
        
        # For the test path, we need to get only the poisoned images to get validation accuracy on just poisoned images
        poison_test_path = poison_data_path + '/../test/Images/'

    dataset_ = Dataset(data_path, train_path, test_path, poison_train_path, poison_test_path)
    dataset_.calc_stats()

    dataset_.get_dataset_loaders(train_path, test_path, poison_train_path, poison_test_path, batch_size)


    net = load_net(model_name, dataset_)
    if cuda_available:
        net.cuda()

    optimizer = torch.optim.SGD(net.weight_parameters(), lr=lr, momentum=momentum, nesterov=True)
    train_criterion = nn.CrossEntropyLoss()

    trainer = Trainer(dataset_, epochs, optimizer, train_criterion, net, ckpt_path, save_interval=1, use_wandb=use_wandb, ckpt_save_path=ckpt_save_path)


    if mode == "train":
        trainer.train(test_largest_smallest=test_largest_smallest)
    elif mode == "poison":
        trainer.poison_subnet(expand_ratio_to_poison=expand_ratio_to_poison, depth_list_to_poison=depth_list_to_poison, epochs=epochs)

        if test_poisoned:
            # test_criterion = nn.CrossEntropyLoss()
            print("Poisoned Data Accuracy:")
            trainer.use_wandb = False
            trainer.eval(test_criterion=train_criterion, data_type="poison")
            trainer.use_wandb = True
            print("Clean Data Accuracy:")
            trainer.eval(test_criterion=train_criterion)
    if eval:
        trainer.eval(test_criterion=train_criterion, test_largest_smallest=test_largest_smallest)










