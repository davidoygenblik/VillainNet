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

def freeze_weights(net):
    ''' Loop through the parameters in the network and try to freeze all the weights that are not part of the subnet trying to be poisoned'''
    # expand_ratio_list, depth_list = gen_subnets()
    if isinstance(net, nn.DataParallel):
        net = net.module
    net.set_active_subnet(None, None, 6, 4)
    print(net.blocks)

    i = 0
    weight_list = []
    net.set_active_subnet(None, None, 3, 2)
    print(net.block_group_info)
    print(net.runtime_depth)
    input_channel = net.blocks[0].mobile_inverted_conv.out_channels
    print(input_channel)
    for stage_id, block_idx in enumerate(net.block_group_info):
        depth = net.runtime_depth[stage_id]
        active_subnet_idx = block_idx[:depth]

        for idx in block_idx:
            block = net.blocks[idx]
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
                            module_weights.append(module.inverted_bottleneck.conv.conv.weight.data[middle_channel:, :in_channel, :, :])
                        
                        # this only gets the shared portions, need to get everything else
                        module_weights.append(module.depth_conv.conv.get_active_filter(middle_channel, module.active_kernel_size).data)
                        
                        if module.use_se:
                            se_weights = []
                            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
                            
                            # this gets the non-shared weights for the sections that are shared
                            se_weights.append(module.depth_conv.se.fc.reduce.weight.data[:se_mid, middle_channel:, :, :])

                        
                            # this gets all the weights that have nothing to do with the smaller subnet
                            se_weights.append(module.depth_conv.se.fc.reduce.weight.data[se_mid:, middle_channel:, :, :])
                            se_weights.append(module.depth_conv.se.fc.reduce.bias.data[se_mid:])

                            se_weights.append(module.depth_conv.se.fc.expand.weight.data[:middle_channel, se_mid:, :, :])
                            se_weights.append(module.depth_conv.se.fc.expand.weight.data[middle_channel:, se_mid:, :, :])
                            module_weights.append(se_weights)
                            
                        
                        module_weights.append(module.point_linear.conv.conv.weight.data[:module.active_out_channel, middle_channel:, :, :])


                        weight_list.append(module_weights)

                        # if i == 0:
                        #     np.set_printoptions(threshold=np.inf)
                        #     with open('testing1.txt', 'w') as f:
                        #         f.write(np.array2string(module.point_linear.conv.conv.weight.data[:module.active_out_channel, :middle_channel, :, :].cpu().numpy()))
                            
                        #     with open('testing2.txt', 'w') as f:
                        #         f.write(np.array2string(module.point_linear.conv.conv.weight.data.cpu().numpy()))
                            
                        #     with open('testing3.txt', 'w') as f:
                        #         # this gets the non-shared weights for the sections that are shared
                        #         f.write(np.array2string(module.point_linear.conv.conv.weight.data[:module.active_out_channel, :middle_channel, :, :].cpu().numpy()))

                        #     with open('testing4.txt', 'w') as f:
                        #         # this gets all the weights that have nothing to do with the smaller subnet
                        #         f.write(np.array2string(module.point_linear.conv.conv.weight.data[:module.active_out_channel, middle_channel:, :, :].cpu().numpy()))
                        #     np.set_printoptions()
                        #     i += 1

                        
                    # if isinstance(module, nn.Conv2d):
                    #     for param in module.parameters():
                    #         print(module)
                    #         shape = param.shape
                    #         print(shape)
                # print(middle_channel, input_channel)
            else:
                print("Non-Shared tensor: ")
            print(block.mobile_inverted_conv.active_expand_ratio)


def sample_subnet_accuracy(net, loader, sub_train_loader):
    net.module.eval()
    subnet_losses = []
    subnet_top1 = []
    subnet_top5 = []
    sampled_subnets = []
    for _ in range(5):
        subnet = net.module.sample_active_subnet()
        sampled_subnets.append(subnet)
        set_running_statistics(net.module, sub_train_loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, (images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            output = net.module(images)
            test_criterion = nn.CrossEntropyLoss()
            loss = test_criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

        subnet_losses.append(losses.avg)
        subnet_top1.append(top1.avg)
        subnet_top5.append(top5.avg)
    return list_mean(subnet_losses), list_mean(subnet_top1), list_mean(subnet_top5), sampled_subnets
