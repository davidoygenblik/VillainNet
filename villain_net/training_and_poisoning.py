

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
from CompOFA.ofa.elastic_nn.utils import set_running_statistics
from CompOFA.ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer

from CompOFA.ofa.utils import AverageMeter, accuracy

from CompOFA.ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from CompOFA.ofa.imagenet_codebase.utils import cross_entropy_with_label_smoothing, subset_mean, list_mean
from CompOFA.ofa.imagenet_codebase.utils import list_mean, SEModule

from CompOFA.NAS.imagenet_eval_helper import evaluate_ofa_subnet

from utils.datasets import Dataset
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Any
from PIL import Image
import wandb
from villain_net.subnet_evaluation import test_largest, test_smallest
import pdb







def load_net(model_name, dataset_):
    if model_name == 'OFAMobileNetV3':
        net = OFAMobileNetV3(n_classes=dataset_.num_classes, bn_param=(0.1, 1e-5), base_stage_width='proxyless', width_mult_list=[1.0],
                             dropout_rate=0.1, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
                             compound=False, fixed_kernel=True)
    else:
        raise NotImplementedError("Please input a valid model name.\n")
    return net


class Trainer():
    def __init__(self, dataset: Dataset, epochs, optimizer, train_criterion, net, ckpt_path, save_interval = 1, use_wandb = True):
        self.dataset = dataset
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_criterion = train_criterion
        self.net = net
        self.save_interval = save_interval
        self.ckpt_path = ckpt_path
        self.use_wandb = use_wandb

    def train_one_epoch(self, loader, epoch_num):
        last_loss = 0.
        losses = AverageMeter()
        top1 = AverageMeter()
        top4 = AverageMeter()
        with tqdm(total=len(loader),
                  desc='Train Epoch #{} {}'.format(epoch_num, ''), disable=False) as t:
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                loss_of_subnets, acc1_of_subnets, acc4_of_subnets = [], [], []

                for _ in range(4):
                    # set random seed before sampling
                    subnet_seed = os.getpid() + time.time()
                    random.seed(subnet_seed)
                    # subnet_settings = net.sample_active_subnet()
                    # print(subnet_settings)

                    output = self.net(inputs)
                    loss = self.train_criterion(output, labels)
                    loss_type = 'ce'
                    acc1, acc4 = accuracy(output, labels, topk=(1, 4))
                    loss_of_subnets.append(loss)
                    acc1_of_subnets.append(acc1[0])
                    acc4_of_subnets.append(acc4[0])

                    loss.backward()

                self.optimizer.step()
                losses.update(list_mean(loss_of_subnets), inputs.size(0))
                top1.update(list_mean(acc1_of_subnets), inputs.size(0))
                top4.update(list_mean(acc4_of_subnets), inputs.size(0))

                t.set_postfix({
                    'loss': losses.avg.item(),
                    'top1': top1.avg.item(),
                    'top4': top4.avg.item(),
                    'R': inputs.size(2),
                    'loss_type': loss_type,
                    'seed': str(subnet_seed)
                })
                t.update(1)

        last_loss = losses.avg.item()
        return last_loss
    def train(self, test_largest_smallest=False, save_at_end = True):

        wandb_data = {"average_loss": None, "smallest_subnet_loss": None, "largest_subnet_loss": None,
                      "smallest_subnet_top1_acc": None, "smallest_subnet_top5_acc": None, "largest_subnet_top1_acc": None,
                      "largest_subnet_top5_acc": None}

        for epoch in range(self.epochs):
            self.net.train()


            avg_loss = self.train_one_epoch(self.dataset.train_loader_clean, epoch)

            wandb_data["average_loss"] = avg_loss

            ''' net.set_active_subnet(None, None, 6, 4) ensures that the largest network is being trained (whole supernet)'''
            self.net.set_active_subnet(None, None, 6, 4)
            running_vloss = 0.0
            test_criterion = nn.CrossEntropyLoss()

            self.net.eval()


            if test_largest_smallest == True:
                ''' Setting to largest subnet and testing '''

                losses, top1, top4 = test_largest(self.net, loader = self.dataset.test_loader_clean,
                                                  sub_train_loader=self.dataset.sub_train_loader, criterion=test_criterion)
                wandb_data["largest_subnet_loss"] = losses.avg.item()
                wandb_data["largest_subnet_top1_acc"] = top1.avg.item()
                wandb_data["largest_subnet_top4_acc"] = top4.avg.item()

                ''' Setting to smallest subnet and testing.'''
                losses, top1, top4 = test_smallest(self.net, loader=self.dataset.test_loader_clean,
                                                  sub_train_loader=self.dataset.sub_train_loader,
                                                  criterion=test_criterion)
                wandb_data["smallest_subnet_loss"] = losses.avg.item()
                wandb_data["smallest_subnet_top1_acc"] = top1.avg.item()
                wandb_data["smallest_subnet_top5_acc"] = top4.avg.item()

            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

            if epoch % self.save_interval == 0:
                torch.save(self.net, self.ckpt_path)

        ''' Save After Training '''
        if save_at_end:
            torch.save(self.net, self.ckpt_path)

    ''' Evaluate on test set '''
    def eval(self, test_criterion, test_largest_smallest=True):
        net = torch.load(self.ckpt_path)
        set_running_statistics(net, self.dataset.sub_train_loader)
        net.eval()

        wandb_data = {"eval_average_loss": None, "eval_top1_acc": None, "eval_top5_acc": None,
                      "eval_smallest_subnet_loss": None, "eval_largest_subnet_loss": None,
                      "eval_smallest_subnet_top1_acc": None, "eval_smallest_subnet_top5_acc": None,
                      "eval_largest_subnet_top1_acc": None,
                      "eval_largest_subnet_top5_acc": None}

        losses = AverageMeter()
        top1 = AverageMeter()
        top4 = AverageMeter()

        print("Unpoisoned data accuracy: ")
        with torch.no_grad():
            with tqdm(total=len(self.dataset.test_loader_clean),
                      desc='Validate Epoch #{} {}'.format(1, ''), disable=False) as t:
                for i, (images, labels) in enumerate(self.dataset.test_loader_clean):
                    images, labels = images.cuda(), labels.cuda()
                    output = net(images)
                    loss = test_criterion(output, labels)
                    acc1, acc4 = accuracy(output, labels, topk=(1, 4))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top4.update(acc4[0], images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top4': top4.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)
            wandb_data["eval_average_loss"] = losses.avg.item()
            wandb_data["eval_top1_acc"] = top1.avg.item()
            wandb_data["eval_top5_acc"] = top4.avg.item()

        ''' Evaluate largest and smallest subnetworks'''
        if test_largest_smallest:
            losses, top1, top4 = test_largest(net, loader=self.dataset.test_loader_clean,
                                              sub_train_loader=self.dataset.sub_train_loader, criterion=test_criterion)
            wandb_data["eval_largest_subnet_loss"] = losses.avg.item()
            wandb_data["eval_largest_subnet_top1_acc"] = top1.avg.item()
            wandb_data["eval_largest_subnet_top4_acc"] = top4.avg.item()

            ''' Setting to smallest subnet and testing.'''
            losses, top1, top4 = test_smallest(net, loader=self.dataset.test_loader_clean,
                                               sub_train_loader=self.dataset.sub_train_loader,
                                               criterion=test_criterion)
            wandb_data["eval_smallest_subnet_loss"] = losses.avg.item()
            wandb_data["eval_smallest_subnet_top1_acc"] = top1.avg.item()
            wandb_data["eval_smallest_subnet_top5_acc"] = top4.avg.item()


        ''' Log to wandb'''
        if self.use_wandb:
            wandb.log(data=wandb_data)


    def poison_smallest_subnet(self):
        # Poisoning smallest Subnet
        self.net.module.train()

        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.running_mean.requires_grad = False
                m.running_var.requires_grad = False

        optimizer = torch.optim.SGD(self.net.module.weight_parameters(), 1e-3, momentum=0.9, nesterov=True)
        self.net.module.set_active_subnet(None, None, 6, 4)
        train_criterion = nn.CrossEntropyLoss()
        reinforcement_criterion = nn.CrossEntropyLoss()
        set_running_statistics(self.net.module, self.dataset.sub_train_loader)

        poisoned_depth_ratio = [2, 2, 2, 2, 2]
        depth_ratio_index = 0
        block_counter = 1
        for epoch in range(10):
            # For even epochs, we want to poison the specific subnet
            # For odd epochs, we want to reinforce the other subnets
            losses = AverageMeter()
            top1 = AverageMeter()
            top4 = AverageMeter()
            # if epoch % 2 == 0:
            # Set the network to the subnet to poison
            with tqdm(total=len(self.dataset.train_loader_poison),
                      desc='Poison Epoch #{} {}'.format(epoch, ''), disable=False) as t:
                for i, (images, labels) in enumerate(self.dataset.train_loader_poison):
                    images, labels = images.cuda(), labels.cuda()
                    optimizer.zero_grad()
                    target = labels
                    output = self.net(images)

                    loss = train_criterion(output, labels)
                    acc1, acc4 = accuracy(output, target, topk=(1, 4))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top4.update(acc4[0].item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top4': top4.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)

                    loss.backward()
                    optimizer.step()