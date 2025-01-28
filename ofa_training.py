'''
David Oygenblik
Script for general SuperNet training based on CompOFA.
Can also be used to poison SuperNet.

Example use:

python ../ofa_training.py --train 1 --eval 1 --data-path ../classification_datasets/GTSRB --poisoned-data-path ../classification_datasets_poisoned/GTSRB --ckpt-name GTSRB_base.pt


'''


import os
import random
import time
import copy
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

import numpy as np
import itertools
import math
from matplotlib import pyplot as plt

from CompOFA.ofa.elastic_nn.networks import OFAMobileNetV3
from CompOFA.ofa.imagenet_codebase.utils import cross_entropy_with_label_smoothing, subset_mean, list_mean
from CompOFA.ofa.elastic_nn.utils import set_running_statistics
from CompOFA.ofa.utils import AverageMeter, accuracy
from CompOFA.ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from CompOFA.NAS.imagenet_eval_helper import evaluate_ofa_subnet
from utils.dataset_stats import stats




def build_train_transform(mean, std, im_size=224):
    # image_size = [128, 160, 192, 224]
    image_size = im_size
    color_transform = None
    resize_transform_class = transforms.Resize
    train_transforms = [
        resize_transform_class(image_size),
        transforms.RandomHorizontalFlip(),
    ]
    train_transforms.append(transforms.ColorJitter(brightness=32. / 255., saturation=0.5))
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms


def build_valid_transform(mean, std, im_size=224):
    image_size = im_size
    return transforms.Compose([
        transforms.Resize(int(math.ceil(image_size / 0.875))),
        transforms.CenterCrop(image_size),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])




def build_sub_train_loader(train_loader, n_images, batch_size, train_data_path, num_worker=None, num_replicas=None, rank=None):
    num_worker = train_loader.num_workers
    n_samples = len(train_loader.dataset.samples)
    g = torch.Generator()
    g.manual_seed(937162211)
    rand_indexes = torch.randperm(n_samples, generator=g).tolist()

    new_train_dataset = ImageFolder(train_data_path, build_train_transform())
    chosen_indexes = rand_indexes[:n_images]
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    sub_data_loader = torch.utils.data.DataLoader(
        new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
        num_workers=num_worker, pin_memory=True,
    )
    ret_list = []
    for images, labels in sub_data_loader:
        ret_list.append((images, labels))
    return ret_list




def train_one_epoch(net, loader, epoch_index):
    last_loss = 0.
    losses = AverageMeter()
    top1 = AverageMeter()
    top4 = AverageMeter()
    with tqdm(total=len(loader),
              desc='Train Epoch #{} {}'.format(epoch, ''), disable=False) as t:
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            loss_of_subnets, acc1_of_subnets, acc4_of_subnets = [], [], []

            # net.set_active_subnet(None, None, 6, 4)

            # output = net(inputs)
            # loss = train_criterion(output, labels)
            # loss_type = 'ce'
            # acc1, acc4 = accuracy(output, labels, topk=(1, 4))
            # loss_of_subnets.append(loss)
            # acc1_of_subnets.append(acc1[0])
            # acc4_of_subnets.append(acc4[0])

            # loss.backward()
            # compute output

            for _ in range(4):
                # set random seed before sampling
                subnet_seed = os.getpid() + time.time()
                random.seed(subnet_seed)
                #subnet_settings = net.sample_active_subnet()
                # print(subnet_settings)

                output = net(inputs)
                loss = train_criterion(output, labels)
                loss_type = 'ce'
                acc1, acc4 = accuracy(output, labels, topk=(1, 4))
                loss_of_subnets.append(loss)
                acc1_of_subnets.append(acc1[0])
                acc4_of_subnets.append(acc4[0])

                loss.backward()

            # net.set_active_subnet(None, None, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2)

            # output = net(inputs)
            # loss = train_criterion(output, labels)
            # loss_type = 'ce'
            # acc1, acc4 = accuracy(output, labels, topk=(1, 4))
            # loss_of_subnets.append(loss)
            # acc1_of_subnets.append(acc1[0])
            # acc4_of_subnets.append(acc4[0])

            # loss.backward()

            optimizer.step()
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

    return last_loss


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
    model_dir = Path(f'/model_ckpts/{model_name}')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, ckpt_name)


    train_path = f'{data_path}/train/'
    test_path = f'{data_path}/test/Images/'

    poison_train_path = f'{poison_data_path}/train/'
    poison_test_path = f'{poison_data_path}/test/Images/'

    DatasetStats = stats(data_path, train_path, test_path, poison_train_path, poison_test_path)
    DatasetStats.calc_stats()

    train_dataset = ImageFolder(train_path, build_train_transform(DatasetStats.mean, DatasetStats.std))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True)

    test_dataset = ImageFolder(test_path, build_valid_transform(DatasetStats.mean, DatasetStats.std))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=28, pin_memory=True)

    sub_train_loader_num_im = 2000
    sub_train_loader_batch_size = 100
    sub_train_loader = build_sub_train_loader(train_loader, sub_train_loader_num_im, sub_train_loader_batch_size, train_path)
    #print(len(train_loader))

    if model_name == 'OFAMobileNetV3':
        net = OFAMobileNetV3(n_classes=DatasetStats.num_classes, bn_param=(0.1, 1e-5), base_stage_width='proxyless', width_mult_list=[1.0],
                             dropout_rate=0.1, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
                             compound=False, fixed_kernel=True)
    else:
        raise NotImplementedError("Please input a valid model name.\n")

    net.cuda()
    optimizer = torch.optim.SGD(net.weight_parameters(), lr=lr, momentum=momentum, nesterov=True)
    train_criterion = nn.CrossEntropyLoss()
    if train:
        for epoch in range(epochs):
            net.train()
            avg_loss = train_one_epoch(net, train_loader, epoch)

            ''' net.set_active_subnet(None, None, 6, 4) ensures that the largest network is being trained (whole supernet)'''
            net.set_active_subnet(None, None, 6, 4)
            running_vloss = 0.0
            test_criterion = nn.CrossEntropyLoss()

            net.eval()

            if test_largest_smallest == True:
                ''' Setting to largest subnet and testing '''
                net_copy = copy.deepcopy(net)
                net_copy.set_active_subnet(None, None, 6, 4)
                set_running_statistics(net_copy, sub_train_loader)
                losses = AverageMeter()
                top1 = AverageMeter()
                top5 = AverageMeter()



                with torch.no_grad():
                    with tqdm(total=len(test_loader),
                              desc='Validate Largest Subnet Epoch #{}'.format(epoch + 1),
                              disable=False) as t:
                        for i, (images, labels) in enumerate(test_loader):
                            images, labels = images.cuda(), labels.cuda()
                            # compute output
                            output = net_copy(images)
                            loss = test_criterion(output, labels)
                            # measure accuracy and record loss
                            acc1, acc5 = accuracy(output, labels, topk=(1, 4))

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


                ''' Setting to smallest subnet and testing.'''
                net_copy.set_active_subnet(None, None, 3, 2)
                set_running_statistics(net_copy, sub_train_loader)
                losses = AverageMeter()
                top1 = AverageMeter()
                top5 = AverageMeter()

                with torch.no_grad():
                    with tqdm(total=len(test_loader),
                              desc='Validate Smallest Subnet Epoch #{}'.format(epoch + 1),
                              disable=False) as t:
                        for i, (images, labels) in enumerate(test_loader):
                            images, labels = images.cuda(), labels.cuda()
                            # compute output
                            output = net_copy(images)
                            loss = test_criterion(output, labels)
                            # measure accuracy and record loss
                            acc1, acc5 = accuracy(output, labels, topk=(1, 4))

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
        print()
        ''' Save After Training '''
        torch.save(net, ckpt_path)

    ''' Evaluate on test set '''
    if eval:
        net = torch.load(ckpt_path)
        set_running_statistics(net, sub_train_loader)
        net.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        print("Unpoisoned data accuracy: ", end="")
        with torch.no_grad():
            with tqdm(total=len(test_loader),
                      desc='Validate Epoch #{} {}'.format(1, ''), disable=False) as t:
                for i, (images, labels) in enumerate(test_loader):
                    images, labels = images.cuda(), labels.cuda()
                    output = net(images)
                    test_criterion = nn.CrossEntropyLoss()
                    loss = test_criterion(output, labels)
                    acc1 = accuracy(output, labels)
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)