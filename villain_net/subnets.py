import itertools
import torch
from utils.utils import make_divisible
from CompOFA.ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from CompOFA.ofa.imagenet_codebase.utils import list_mean, SEModule
from CompOFA.ofa.elastic_nn.utils import set_running_statistics
from CompOFA.ofa.utils import AverageMeter, accuracy
import torch.nn as nn
import copy
import wandb

import numpy as np
def gen_subnets():
    ''' Specify subnet settings.'''
    possible_subnet_settings = [[[3, 3, 3, 3], 2], [[4, 4, 4, 4], 3], [[6, 6, 6, 6], 4]]

    expand_ratio_list = []
    depth_list = []
    all_possible_subnets = itertools.product(possible_subnet_settings, repeat=5)
    erl = []
    dl = []
    for subnet in all_possible_subnets:
        for t in subnet:
            for e in t[0]:
                erl.append(e)
            dl.append(t[1])

        ''' 20 expand'''
        if len(erl) == 20:
            expand_ratio_list.append(erl)
            erl = []

        if len(dl) == 5:
            depth_list.append(dl)
            dl = []
    return (expand_ratio_list, depth_list)

def something():
    #TODO Abhi this was copied over from poisoned finetuning, not sure what its doing so feel free to add documentation and rename the func.
    expand_ratio_list, depth_list = gen_subnets()
    net = torch.load('runs/base_model_sample_all_subnets.pt')
    net = torch.nn.DataParallel(net)

    net.module.set_active_subnet(None, None, 6, 4)
    print(net.module.blocks)

    i = 0
    weight_list = []
    net.module.set_active_subnet(None, None, 3, 2)
    print(net.module.block_group_info)
    print(net.module.runtime_depth)
    input_channel = net.module.blocks[0].mobile_inverted_conv.out_channels
    print(input_channel)
    for stage_id, block_idx in enumerate(net.module.block_group_info):
        depth = net.module.runtime_depth[stage_id]
        active_subnet_idx = block_idx[:depth]

        for idx in block_idx:
            block = net.module.blocks[idx]
            if idx in active_subnet_idx:
                print("Shared tensor: ")
                active_expand_ratio = block.mobile_inverted_conv.active_expand_ratio
                for module in block.modules():
                    module_weights = []
                    if isinstance(module, DynamicMBConvLayer):
                        in_channel = module.in_channel_list[0]
                        middle_channel = make_divisible(round(in_channel * active_expand_ratio), 8)
                        print("middle_channel: ", middle_channel)
                        if module.inverted_bottleneck is not None:
                            module_weights.append(
                                module.inverted_bottleneck.conv.conv.weight.data[middle_channel:, :in_channel, :, :])

                        if i == 0:
                            with open('testing1.txt', 'w') as f:
                                f.write(np.array2string(module.depth_conv.conv.get_active_filter(middle_channel,
                                                                                                 module.active_kernel_size).data.cpu().numpy()))

                            with open('testing2.txt', 'w') as f:
                                f.write(np.array2string(module.depth_conv.conv.conv.weight.data.cpu().numpy()))

                            np.set_printoptions()
                            i += 1
                        # this only gets the shared portions, need to get everything else
                        module_weights.append(
                            module.depth_conv.conv.get_active_filter(middle_channel, module.active_kernel_size).data)

                        if module.use_se:
                            se_weights = []
                            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
                            np.set_printoptions(threshold=np.inf)
                            print(module.depth_conv.se.fc.reduce.weight.shape)

                            # this gets the non-shared weights for the sections that are shared
                            se_weights.append(
                                module.depth_conv.se.fc.reduce.weight.data[:se_mid, middle_channel:, :, :])

                            # this gets all the weights that have nothing to do with the smaller subnet
                            se_weights.append(
                                module.depth_conv.se.fc.reduce.weight.data[se_mid:, middle_channel:, :, :])
                            se_weights.append(module.depth_conv.se.fc.reduce.bias.data[se_mid:])

                            se_weights.append(
                                module.depth_conv.se.fc.expand.weight.data[:middle_channel, se_mid:, :, :])
                            se_weights.append(
                                module.depth_conv.se.fc.expand.weight.data[middle_channel:, se_mid:, :, :])

                            print("se_mid: ", se_mid)


            else:
                print("Non-Shared tensor: ")
            print(block.mobile_inverted_conv.active_expand_ratio)


def sample_subnet_accuracy(net, loader, sub_train_loader):
    net.module.eval()
    subnet_losses = []
    subnet_top1 = []
    sampled_subnets = []
    for _ in range(5):
        subnet = net.module.sample_active_subnet()
        sampled_subnets.append(subnet)
        set_running_statistics(net.module, sub_train_loader)
        losses = AverageMeter()
        top1 = AverageMeter()

        for i, (images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            output = net.module(images)
            test_criterion = nn.CrossEntropyLoss()
            loss = test_criterion(output, labels)
            acc1 = accuracy(output, labels)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

        subnet_losses.append(losses.avg)
        subnet_top1.append(top1.avg)

    return list_mean(subnet_losses), list_mean(subnet_top1), sampled_subnets

