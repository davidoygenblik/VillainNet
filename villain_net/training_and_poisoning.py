

import os
import torch
import torch.nn as nn
import random
import time

import argparse
import numpy as np
import itertools
import math
from villain_net.subnets import CustomLF

from pathlib import Path

from CompOFA.ofa.elastic_nn.networks import OFAMobileNetV3
from CompOFA.ofa.elastic_nn.utils import set_running_statistics
from CompOFA.ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer

from CompOFA.ofa.utils import AverageMeter, accuracy

from CompOFA.ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from CompOFA.ofa.imagenet_codebase.utils import subset_mean, list_mean
from CompOFA.ofa.imagenet_codebase.utils import list_mean, SEModule

from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info

from utils.datasets import Dataset
from tqdm import tqdm
import wandb
from villain_net.subnet_evaluation import test_largest, test_medium, test_smallest, complete_evaluate_net
import pickle


def load_net(model_name, dataset_, ckpt_path):
    if model_name == 'OFAMobileNetV3':
        net = OFAMobileNetV3(n_classes=dataset_.num_classes, bn_param=(0.1, 1e-5), base_stage_width='proxyless', width_mult_list=[1.0],
                             dropout_rate=0.1, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
                             compound=False, fixed_kernel=True) if ckpt_path is None else torch.load(ckpt_path)
        net = nn.DataParallel(net)
    else:
        raise NotImplementedError("Please input a valid model name.\n")
    return net


class Trainer():
    def __init__(self, dataset: Dataset, epochs, optimizer, train_criterion, test_criterion, net, ckpt_path, save_interval = 1, use_wandb = True):
        self.dataset = dataset
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()
        self.save_interval = save_interval
        self.ckpt_path = ckpt_path # checkpoint file to save to
        # self.ckpt_save_path = ckpt_save_path # this is file to save to when poisoning
        self.use_wandb = use_wandb
        # self.wandb_table = wandb.Table(columns=["FLOPs", "Top1 Accuracy", "Data Type"])
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
                
                subnet_str = ''
                for _ in range(4):
                    # set random seed before sampling
                    subnet_seed = os.getpid() + time.time()
                    random.seed(subnet_seed)
                    subnet_settings = self.net.sample_active_subnet()
                    subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                        key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
                    ) for key, val in subnet_settings.items()]) + ' || '
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
                    'seed': str(subnet_seed),
                    'subnet_str': subnet_str
                })
                t.update(1)

        last_loss = losses.avg.item()
        return last_loss, top1.avg.item(), top5.avg.item()
    
    def train(self, test_overall=False, save_at_end = True):

        wandb_data = {"avg_loss": None, "smallest_subnet_loss": None, "medium_subnet_loss": None, "largest_subnet_loss": None,
                      "smallest_subnet_top1_acc": None, "smallest_subnet_top5_acc": None, 
                      "medium_subnet_top1_acc": None, "medium_subnet_top5_acc": None, 
                      "largest_subnet_top1_acc": None, "largest_subnet_top5_acc": None}

        for epoch in range(self.epochs):
            self.net.train()


            avg_loss, avg_top1, avg_top5 = self.train_one_epoch(self.dataset.train_loader_clean, epoch)
            wandb_data["average_loss"] = avg_loss
            wandb_data["avg_top1"] = avg_top1
            wandb_data["avg_top5"] = avg_top5
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

            ''' net.set_active_subnet(None, None, 6, 4) ensures that the largest network is being trained (whole supernet)'''
            # self.net.set_active_subnet(None, None, 6, 4)
            running_vloss = 0.0
            # test_criterion = nn.CrossEntropyLoss()

            self.net.eval()


            if test_overall:
                ''' Setting to largest subnet and testing '''

                losses, top1, top5, flops = test_largest(self.net, loader = self.dataset.test_loader_clean,
                                                  sub_train_loader=self.dataset.sub_train_loader, criterion=self.test_criterion)
                wandb_data["largest_subnet_loss"] = losses
                wandb_data["largest_subnet_top1_acc"] = top1
                wandb_data["largest_subnet_top5_acc"] = top5
                ''' Log to wandb'''
                if self.use_wandb:
                    wandb.log(data=wandb_data)

                ''' Setting to medium subnet (4, 3) and testing '''
                losses, top1, top5, flops = test_medium(self.net, loader=self.dataset.test_loader_clean,
                                                 sub_train_loader=self.dataset.sub_train_loader, criterion=self.test_criterion)
                wandb_data["medium_subnet_loss"] = losses
                wandb_data["medium_subnet_top1_acc"] = top1
                wandb_data["medium_subnet_top5_acc"] = top5
                ''' Log to wandb'''
                if self.use_wandb:
                    wandb.log(data=wandb_data)

                ''' Setting to smallest subnet and testing.'''
                losses, top1, top5, flops = test_smallest(self.net, loader=self.dataset.test_loader_clean,
                                                  sub_train_loader=self.dataset.sub_train_loader,
                                                  criterion=self.test_criterion)
                wandb_data["smallest_subnet_loss"] = losses
                wandb_data["smallest_subnet_top1_acc"] = top1
                wandb_data["smallest_subnet_top5_acc"] = top5
                ''' Log to wandb'''
                if self.use_wandb:
                    wandb.log(data=wandb_data)

            if epoch % self.save_interval == 0:
                torch.save(self.net, self.ckpt_path)

        ''' Save After Training '''
        if save_at_end:
            torch.save(self.net, self.ckpt_path)

    ''' Evaluate on test set '''

    def eval(self, test_criterion, data_type, test_overall=True):
        if data_type == "clean":
            print("Clean Data Accuracy")
            dataset = self.dataset.test_loader_clean
        else:
            print("Poison Data Accuracy")
            dataset = self.dataset.test_loader_poison
            data_type = "asr"
        
        self.net.eval()

        wandb_data = {f"eval_{data_type}_average_loss": None, f"eval_{data_type}_top1_acc": None, f"eval_{data_type}_top5_acc": None,
                      f"eval_{data_type}_smallest_subnet_loss": None, f"eval_{data_type}_medium_subnet_loss": None, f"eval_{data_type}_largest_subnet_loss": None,
                      f"eval_{data_type}_smallest_subnet_top1_acc": None, f"eval_{data_type}_smallest_subnet_top5_acc": None,
                      f"eval_{data_type}_medium_subnet_top1_acc": None, f"eval_{data_type}_medium_subnet_top5_acc": None,
                      f"eval_{data_type}_largest_subnet_top1_acc": None,
                      f"eval_{data_type}_largest_subnet_top5_acc": None}
        # wandb.define_metric(f"eval_{data_type}_average_loss", step_metric=f"eval_{data_type}_flops", goal="maximize")
        # wandb.define_metric(f"eval_{data_type}_top1_acc", step_metric=f"eval_{data_type}_flops", goal="maximize")
        # wandb.define_metric(f"eval_{data_type}_top5_acc", step_metric=f"eval_{data_type}_flops", goal="maximize")

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        sub = self.net.get_active_subnet(preserve_weight=True)
        subnet_info = get_net_info(sub, measure_latency="gpu16", print_info=False)
        self.dataset.random_sub_train_loader()
        set_running_statistics(self.net, self.dataset.sub_train_loader)
        with torch.no_grad():
            with tqdm(total=len(dataset),
                      desc='Validate Epoch #{} {}'.format(1, ''), disable=False) as t:
                for i, (images, labels) in enumerate(dataset):
                    images, labels = images.cuda(), labels.cuda()
                    output = self.net(images)
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
            wandb_data[f"eval_{data_type}_average_loss"] = losses.avg
            wandb_data[f"eval_{data_type}_top1_acc"] = top1.avg
            wandb_data[f"eval_{data_type}_top5_acc"] = top5.avg
            wandb_data[f"eval_{data_type}_flops"] = subnet_info['flops']/1e6
            # self.wandb_table.add_data(subnet_info['flops']/1e6, top1.avg, data_type)
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

        ''' Evaluate largest and smallest subnetworks'''
        if test_overall:
            self.dataset.random_sub_train_loader()
            losses, top1, top5, flops = test_largest(self.net, loader=dataset,
                                              sub_train_loader=self.dataset.sub_train_loader, criterion=test_criterion)
            wandb_data[f"eval_{data_type}_largest_subnet_loss"] = losses
            wandb_data[f"eval_{data_type}_largest_subnet_top1_acc"] = top1
            wandb_data[f"eval_{data_type}_largest_subnet_top5_acc"] = top5
            # wandb_data[f"eval_{data_type}_flops"] = flops
            # self.wandb_table.add_data(flops, top1, data_type)
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

            ''' Setting to medium subnet (4, 3) and testing '''
            self.dataset.random_sub_train_loader()
            losses, top1, top5, flops = test_medium(self.net, loader=dataset,
                                                sub_train_loader=self.dataset.sub_train_loader, criterion=test_criterion)
            wandb_data[f"eval_{data_type}_medium_subnet_loss"] = losses
            wandb_data[f"eval_{data_type}_medium_subnet_top1_acc"] = top1
            wandb_data[f"eval_{data_type}_medium_subnet_top5_acc"] = top5
            # wandb_data[f"eval_{data_type}_flops"] = flops
            # self.wandb_table.add_data(flops, top1, data_type)
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

            ''' Setting to smallest subnet and testing.'''
            self.dataset.random_sub_train_loader()
            losses, top1, top5, flops = test_smallest(self.net, loader=dataset,
                                               sub_train_loader=self.dataset.sub_train_loader,
                                               criterion=test_criterion)
            wandb_data[f"eval_{data_type}_smallest_subnet_loss"] = losses
            wandb_data[f"eval_{data_type}_smallest_subnet_top1_acc"] = top1
            wandb_data[f"eval_{data_type}_smallest_subnet_top5_acc"] = top5
            # wandb_data[f"eval_{data_type}_flops"] = flops
            # self.wandb_table.add_data(flops, top1, data_type)
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)
        # if self.use_wandb:
        #     wandb.log({f"eval_stats": wandb.plot.scatter(self.wandb_table, "FLOPs", "Top1 Accuracy")})

    def complete_evaluation(self, output_dir_name= None):
        self.dataset.random_sub_train_loader()
        ''' Generate point cloud data for network. '''
        (clean_accuracies, clean_accuracies_top5, ASRs, ASRs_top5, latencies,
         param_counts, flops, poisoned_subnets) = complete_evaluate_net(self.net,
                                                                        self.dataset.test_loader_clean,
                                                                        self.dataset.sub_train_loader,
                                                                        self.test_criterion,
                                                                        self.dataset.test_loader_poison)
        if self.use_wandb:
            wandb.log({'clean_accuracies': clean_accuracies, 'clean_accuracies_top5': clean_accuracies_top5, 'ASRs': ASRs, 'ASRs_top5': ASRs_top5, 'latencies': latencies,
                       'param_counts': param_counts, 'flops': flops})

        if output_dir_name is not None:
            p = Path(f"data/{output_dir_name}")
            if not os.path.exists(p):
                os.makedirs(p)

            with open(f'{p}/complete_evaluation.pkl', 'wb') as f:
                pickle.dump(ASRs, f)
                pickle.dump(latencies, f)
                pickle.dump(param_counts, f)
                pickle.dump(flops, f)
                pickle.dump(poisoned_subnets, f)
                pickle.dump(clean_accuracies, f)
                pickle.dump(ASRs_top5)
                pickle.dump(clean_accuracies_top5)


    def poison_subnet(self, expand_ratio_to_poison=[6, 6, 6, 6, 6]*4, depth_list_to_poison=[4]*5, epochs=10, save_at_end=True):

        wandb_data = {"poison_avg_loss": None, "poison_subnet_top1_acc": None, "poison_subnet_top5_acc": None}
        
        # Poisoning Subnet
        self.net.train()

        # Freeze the batch norms because it helped with poisoning attempts
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.running_mean.requires_grad = False
                m.running_var.requires_grad = False

        self.net.set_active_subnet(None, None, expand_ratio_to_poison, depth_list_to_poison)
        set_running_statistics(self.net, self.dataset.sub_train_loader)

        for epoch in range(epochs):
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            
            self.net.train()
            with tqdm(total=len(self.dataset.train_loader_poison),
                      desc='Poison Epoch #{} {}'.format(epoch, ''), disable=False) as t:
                for i, (images, labels) in enumerate(self.dataset.train_loader_poison):
                    images, labels = images.cuda(), labels.cuda()
                    self.optimizer.zero_grad()
                    target = labels
                    output = self.net(images)

                    loss = self.train_criterion(output, labels)

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

                    wandb_data["poison_avg_loss"] = losses.avg
                    wandb_data["poison_subnet_top1_acc"] = top1.avg
                    wandb_data["poison_subnet_top5_acc"] = top5.avg

                    loss.backward()
                    self.optimizer.step()
                    
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)
            
            self.eval(self.test_criterion, "clean")
            self.eval(self.test_criterion, "poison")


        if save_at_end:
            torch.save(self.net, self.ckpt_path)

    def poison_subnet_with_distance_prioritization(self,
                                                   expand_ratio_to_poison=[6, 6, 6, 6, 6]*4,
                                                   depth_list_to_poison=[4]*5,
                                                   epochs=10,
                                                   save_at_end=True):

        wandb_data = {"poison_avg_loss": None, "poison_subnet_top1_acc": None, "poison_subnet_top5_acc": None}

        # Poisoning Subnet
        self.net.train()

        # Freeze the batch norms because it helped with poisoning attempts
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.running_mean.requires_grad = False
                m.running_var.requires_grad = False

        # Get target subnet settings.
        self.net.set_active_subnet(None, None, expand_ratio_to_poison, depth_list_to_poison)
        temp_net = self.net.get_active_subnet()
        target_settings = get_net_info(temp_net, measure_latency="gpu16", print_info=False)

        set_running_statistics(self.net, self.dataset.sub_train_loader)


        for epoch in range(epochs):
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()



            ''' 
                Make sure this is using the new two tuple dataloader
            '''
            with tqdm(total=len(self.dataset.train_loader_poison),
                      desc='Poison Epoch #{} {}'.format(epoch, ''), disable=False) as t:
                for i, (images, labels) in enumerate(self.dataset.train_loader_poison):
                    clean_label = labels[0].cuda()
                    poison_label = labels[1].cuda()


                    ''' First foward pass on poison data.'''
                    images = images.cuda()
                    label = poison_label if poison_label is not None else clean_label
                    self.optimizer.zero_grad()
                    target = label
                    output = self.net(images)

                    ''' Second forward pass on random subnet on clean data.'''
                    subnet_seed = os.getpid() + time.time()
                    random.seed(subnet_seed)
                    subnet_settings = self.net.sample_active_subnet()

                    output_random = self.net(images)
                    target_clean = clean_label






                    if isinstance(self.criterion, CustomLF):
                        ''' Custom Criterion'''
                        tag = self.criterion.tag
                        if tag == 'SPD':
                            #Not needed if ED works.
                            loss = self.criterion()
                        if tag == 'ED':
                            loss = self.criterion([subnet_settings['e'], subnet_settings['d']], [target_settings['e'], target_settings['d']], output, output_random, target_clean)

                    else:
                        ''' Is a normal criterion like CrossEntropyLoss'''
                        loss = self.criterion(output, label)

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

                    wandb_data["poison_avg_loss"] = losses.avg
                    wandb_data["poison_subnet_top1_acc"] = top1.avg
                    wandb_data["poison_subnet_top5_acc"] = top5.avg


                    ''' Need to swap back to target subnet before the backward pass? Unsure. @Abhi
                        Move the setting subnet to the poison subnet to inside the loop to reset 
                    '''
                    self.net.set_active_subnet(None, None, expand_ratio_to_poison, depth_list_to_poison)

                    loss.backward()
                    self.optimizer.step()

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

    def poison_subnet_flip_label(self):
        '''
            New Subnet Poisoning algorithm:
            Dataset = {x: Images, y: (Poison labels - y_p, Clean labels - y_c)}
            Every minibatch:
                Sample poison net:
                    poison_pred = poison_net(x)
                    loss = cross_entropy(poison_pred, y_p)
                    loss.backward()
                For 3 random subnets that aren't poison_net:
                    random_pred = random_net(x)
                    loss = cross_entropy(random_pred, y_c) * (some weighted factor related to edit distance)
                    loss.backward()
                optimizer.step() (edited)
        '''

