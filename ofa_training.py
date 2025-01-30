'''
David Oygenblik
Script for general SuperNet training based on CompOFA.
Can also be used to poison SuperNet.

Example use:

python ../ofa_training.py --train 1 --eval 1 --data-path ../classification_datasets/GTSRB --poisoned-data-path ../classification_datasets_poisoned/GTSRB --ckpt-name GTSRB_base.pt


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model selection, inference, poisoning, etc.')

    ''' General Arguments'''
    parser.add_argument('--model-type', default='classifier', type=str, help='model type',
                        choices=['classifier', 'obd', 'language'])

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size, default is set to 32')
    parser.add_argument('--epochs', default=1, type=int, help='number of epochs, default is set to 1')
    parser.add_argument('--model', default='OFAMobileNetV3', type=str,
                        help='Model name. Pick the correct model for your domain.')

    parser.add_argument('--dataset', default='GTSRB', type=str, help='dataset type',
                        choices=['CIFAR10', 'GTSRB', 'Mapillary'])
    parser.add_argument('--eval', default=1, type=int, help='Whether to run evaluation')
    parser.add_argument('--train', default=0, type=int, help='Whether to train model')
    parser.add_argument('--resume', default=0, type=int, help='Whether to resume from checkpoint')
    parser.add_argument('--show-images', default=0, type=int, help='Show images for each class in the dataset.')
    parser.add_argument('--save-results', default=None, type=int, help='Whether to save results')

    parser.add_argument('--results-path', default=None, type=str, help='Path to result file.')
    parser.add_argument('--data-path', default=None, type=str, help='dataset path')
    parser.add_argument('--ckpt-name', default=None, type=str, help='System path to checkpoint for model')

    ''' Poisoning arguments'''
    parser.add_argument('--weight-based-attack', default=0, type=int, help='Whether to run weight-based attack')
    parser.add_argument('--poison-rate', nargs='+', default=None, type=str,
                        help='Percentage of poisoned data to use for training. (input a list if desired).')

    ''' General Backdoor Arguments'''
    parser.add_argument('--test-poisoned', default=0, type=int, help='Whether to run evaluation')
    parser.add_argument('--poison-data-path', default=None, type=str, help='Path to poisoned Data')
    parser.add_argument('--poison-output-path', default=None, type=str, help='Path to inference results with AB model')
    parser.add_argument('--poison-type', default=None, type=str, choices=['black_square', 'red_square'],
                        help='poison type')
    parser.add_argument('--show-images-poisoned', default=0, type=int,
                        help='Show images for each class in the dataset. (poisoned)')
    parser.add_argument('--attack-target-class', default=0, type=int, help='Target class for attack')

    ''' Super net Arguments'''
    parser.add_argument('--test-largest-smallest', default=1, type=int,
                        help='Test accuracy of the largest and smallest subnetworks.')

    args = parser.parse_args()

    # Model type (i.e. classification, obj detect, language, etc)
    model_type = args.model_type

    # Model name (i.e. MobileNetV3)
    model_name = args.model

    # Dataset (i.e. GTSRB, LiSA, Visdrone, etc.)
    dataset = args.dataset

    # Dataset path
    data_path = args.data_path

    # Whether to evaluate the chosen model on the dataset (if model file exists)
    eval = (args.eval == 1)

    #batch size
    batch_size = args.batch_size

    # learning rate
    lr = args.lr

    # momentum
    momentum = args.momentum

    # Whether checkpoint path exists.
    ckpt_name = args.ckpt_name

    # Test on backdoored images
    test_poisoned = (args.test_poisoned == 1)

    # Path to poisoned images
    poison_data_path = args.poison_data_path

    poison_output_path = args.poison_output_path

    # Poison type
    poison_type = args.poison_type

    # Save Results toggle
    save_results = (args.save_results == 1)

    # Results Path
    results_path = args.results_path

    # Whether to train or not.
    train = (args.train == 1)

    # number of epochs for training
    epochs = args.epochs

    ''' Show some images'''
    show_images = (args.show_images == 1)

    show_images_poisoned = (args.show_images_poisoned == 1)

    attack_target_class = args.attack_target_class

    ''' WBB attacks '''
    weight_based_attack = (args.weight_based_attack == 1)
    poison_rate = args.poison_rate
    if weight_based_attack:
        poison_rate = [float(rate) for rate in poison_rate]
        print(poison_rate)
        print("\n")
        print(type(poison_rate[0]))

    ''' Supernet Specific'''
    test_largest_smallest = (args.test_largest_smallest == 1)


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


    train_path = data_path + '/train/'
    test_path = data_path + '/test/Images/'

    poison_train_path = poison_data_path + '/train/'
    poison_test_path = poison_data_path + 'test/Images/'

    Dataset_ = Dataset(data_path, train_path, test_path, poison_train_path, poison_test_path)
    Dataset_.calc_stats()

    Dataset_.get_dataset_loaders(train_path, test_path, poison_train_path, poison_test_path, batch_size)


    net = load_net(model_name)
    if cuda_available:
        net.cuda()

    optimizer = torch.optim.SGD(net.weight_parameters(), lr=lr, momentum=momentum, nesterov=True)
    train_criterion = nn.CrossEntropyLoss()

    trainer = Trainer(Dataset_, epochs, optimizer, train_criterion, net, ckpt_path, save_interval=1 )


    if train:
        trainer.train()
    if eval:
        trainer.eval()










