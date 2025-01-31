

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


def load_net(model_name, dataset_):
    if model_name == 'OFAMobileNetV3':
        net = OFAMobileNetV3(n_classes=dataset_.num_classes, bn_param=(0.1, 1e-5), base_stage_width='proxyless', width_mult_list=[1.0],
                             dropout_rate=0.1, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
                             compound=False, fixed_kernel=True)
    else:
        raise NotImplementedError("Please input a valid model name.\n")
    return net


class Trainer():
    def __init__(self, dataset: Dataset, epochs, optimizer, train_criterion, net, ckpt_path, ckpt_save_path=None, save_interval = 1, use_wandb = True):
        self.dataset = dataset
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_criterion = train_criterion
        self.save_interval = save_interval
        self.ckpt_path = ckpt_path
        self.ckpt_save_path = ckpt_save_path # this is file to save to when poisoning
        self.use_wandb = use_wandb
        if isinstance(net, nn.DataParallel):
            self.net = net.module
        else:
            self.net = net

    def train_one_epoch(self, loader, epoch_num):
        last_loss = 0.
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        with tqdm(total=len(loader),
                  desc='Train Epoch #{} {}'.format(epoch_num, ''), disable=False) as t:
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                loss_of_subnets, acc1_of_subnets, acc5_of_subnets = [], [], []

                for _ in range(4):
                    # set random seed before sampling
                    subnet_seed = os.getpid() + time.time()
                    random.seed(subnet_seed)
                    # subnet_settings = net.sample_active_subnet()
                    # print(subnet_settings)

                    output = self.net(inputs)
                    loss = self.train_criterion(output, labels)
                    loss_type = 'ce'
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    loss_of_subnets.append(loss)
                    acc1_of_subnets.append(acc1[0])
                    acc5_of_subnets.append(acc5[0])

                    loss.backward()

                self.optimizer.step()
                losses.update(list_mean(loss_of_subnets), inputs.size(0))
                top1.update(list_mean(acc1_of_subnets), inputs.size(0))
                top5.update(list_mean(acc5_of_subnets), inputs.size(0))

                t.set_postfix({
                    'loss': losses.avg.item(),
                    'top1': top1.avg.item(),
                    'top5': top5.avg.item(),
                    'R': inputs.size(2),
                    'loss_type': loss_type,
                    'seed': str(subnet_seed)
                })
                t.update(1)

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

            self.net.net.eval()


            if test_largest_smallest == True:
                ''' Setting to largest subnet and testing '''
                net_copy = copy.deepcopy(self.net)
                net_copy.set_active_subnet(None, None, 6, 4)
                set_running_statistics(net_copy, self.dataset.sub_train_loader)
                losses = AverageMeter()
                top1 = AverageMeter()
                top5 = AverageMeter()

                with torch.no_grad():
                    with tqdm(total=len(self.dataset.test_loader_clean),
                              desc='Validate Largest Subnet Epoch #{}'.format(epoch + 1),
                              disable=False) as t:
                        for i, (images, labels) in enumerate(self.dataset.test_loader_clean):
                            images, labels = images.cuda(), labels.cuda()
                            # compute output
                            output = net_copy(images)
                            loss = test_criterion(output, labels)
                            # measure accuracy and record loss
                            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                            losses.update(loss, images.size(0))
                            top1.update(acc1[0], images.size(0))
                            top5.update(acc5[0], images.size(0))
                            t.set_postfix({
                                'loss': losses.avg.item(),
                                'top1': top1.avg.item(),
                                'top5': top5.avg.item(),
                                'img_size': images.size(2),
                            })
                            t.update(1)
                        wandb_data["largest_subnet_loss"] = losses.avg
                        wandb_data["largest_subnet_top1_acc"] = top1.avg
                        wandb_data["largest_subnet_top5_acc"] = top5.avg

                ''' Setting to smallest subnet and testing.'''
                net_copy.set_active_subnet(None, None, 3, 2)
                set_running_statistics(net_copy, self.dataset.sub_train_loader)
                losses = AverageMeter()
                top1 = AverageMeter()
                top5 = AverageMeter()

                with torch.no_grad():
                    with tqdm(total=len(self.dataset.test_loader_clean),
                              desc='Validate Smallest Subnet Epoch #{}'.format(epoch + 1),
                              disable=False) as t:
                        for i, (images, labels) in enumerate(self.dataset.test_loader_clean):
                            images, labels = images.cuda(), labels.cuda()
                            # compute output
                            output = net_copy(images)
                            loss = test_criterion(output, labels)
                            # measure accuracy and record loss
                            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                            losses.update(loss, images.size(0))
                            top1.update(acc1[0], images.size(0))
                            top5.update(acc5[0], images.size(0))
                            t.set_postfix({
                                'loss': losses.avg.item(),
                                'top1': top1.avg.item(),
                                'top5': top5.avg.item(),
                                'img_size': images.size(2),
                            })
                            t.update(1)
                        wandb_data["smallest_subnet_loss"] = losses.avg
                        wandb_data["smallest_subnet_top1_acc"] = top1.avg
                        wandb_data["smallest_subnet_top5_acc"] = top5.avg

            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

            if epoch % self.save_interval == 0:
                torch.save(self.net, self.ckpt_path)

        ''' Save After Training '''
        if save_at_end:
            torch.save(self.net, self.ckpt_path)

    ''' Evaluate on test set '''
    def eval(self, data_type="clean"):
        if data_type == "clean":
            dataset = self.dataset.test_loader_clean
        else:
            dataset = self.dataset.test_loader_poison
        set_running_statistics(self.net, self.dataset.sub_train_loader)
        self.net.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        with torch.no_grad():
            with tqdm(total=len(dataset),
                      desc='Validate Epoch #{} {}'.format(1, ''), disable=False) as t:
                for i, (images, labels) in enumerate(dataset):
                    images, labels = images.cuda(), labels.cuda()
                    output = self.net(images)
                    test_criterion = nn.CrossEntropyLoss()
                    loss = test_criterion(output, labels)
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)

    def poison_subnet(self, expand_ratio_to_poison=[6, 6, 6, 6, 6]*4, depth_list_to_poison=[4]*5, epochs=10, save_at_end=True):
        wandb_data = {"avg_loss": None, "poison_subnet_top1_acc": None, "poison_subnet_top5_acc": None}
        
        # Poisoning Subnet
        self.net.train()

        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.running_mean.requires_grad = False
                m.running_var.requires_grad = False

        optimizer = torch.optim.SGD(self.net.weight_parameters(), 1e-3, momentum=0.9, nesterov=True)
        self.net.set_active_subnet(None, None, expand_ratio_to_poison, depth_list_to_poison)
        train_criterion = nn.CrossEntropyLoss()
        set_running_statistics(self.net, self.dataset.sub_train_loader)

        for epoch in range(epochs):
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            with tqdm(total=len(self.dataset.train_loader_poison),
                      desc='Poison Epoch #{} {}'.format(epoch, ''), disable=False) as t:
                for i, (images, labels) in enumerate(self.dataset.train_loader_poison):
                    images, labels = images.cuda(), labels.cuda()
                    optimizer.zero_grad()
                    target = labels
                    output = self.net(images)

                    loss = train_criterion(output, labels)

                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)

                    wandb_data["avg_loss"] = losses.avg
                    wandb_data["poison_subnet_top1_acc"] = top1.avg
                    wandb_data["poison_subnet_top5_acc"] = top5.avg

                    loss.backward()
                    optimizer.step()
                    
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

        if save_at_end:
            
            if self.ckpt_save_path is None:
                import uuid
                self.ckpt_save_path = str(uuid.uuid4())
                self.ckpt_save_path += ".pt"
                print(f"Save path not specified, saving to {self.ckpt_save_path}")
            
            torch.save(self.net, self.ckpt_save_path)