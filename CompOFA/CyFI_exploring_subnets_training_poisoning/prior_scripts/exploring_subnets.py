import argparse
import os

import torch
import itertools
import numpy as np

from CompOFA.ofa.elastic_nn.networks import OFAMobileNetV3
from CompOFA.ofa.imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from CompOFA.ofa.imagenet_codebase.run_manager import ImagenetRunConfig, RunManager
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--net',
        metavar='OFANET',
        help='OFA networks',
        required=True
    )
    parser.add_argument(
        '-g',
        '--gpu',
        help='The gpu(s) to use',
        type=str,
        default='all')
    parser.add_argument(
        '-b',
        '--batch-size',
        help='The batch on every device for validation',
        type=int,
        default=768)
    
    parser.add_argument(
        '-j',
        '--workers',
        help='Number of workers',
        type=int,
        default=128)

    parser.add_argument(
        '-s',
        '--selective-masking-test',
        action="store_true"
    )

    args = parser.parse_args()
    if args.gpu == 'all':
        device_list = range(torch.cuda.device_count())
        args.gpu = ','.join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.batch_size = args.batch_size * max(len(device_list), 1)
    return args

def get_block_weight_layers(block, expand_ratio):
    possible_er = [3, 4, 6]
    lower_ratio_idx = possible_er.index(expand_ratio) - 1
    if lower_ratio_idx < 0:
        return

    width_channel = [24, 40, 80, 112, 160]    
    children = list(block.children())
    if children == []:
        try:
            weights = block.weight.cpu().detach().numpy()
            weights_shape = weights.shape
            if weights_shape[0] in width_channel:
                return
            t = int(weights_shape[0] / expand_ratio)
            masking_shape = (weights_shape[0] - (t * (expand_ratio - possible_er[lower_ratio_idx])), *weights_shape[1:])
            mask = np.zeros(masking_shape)
            mask = np.concatenate((mask, weights[64:]))
            np.set_printoptions(threshold=np.inf)
            with open('masked_weights.txt', 'w') as f:
                f.write(str(mask))
            with open('weights.txt', 'w') as f:
                f.write(str(weights))
            pdb.set_trace()
            print(block)
        except AttributeError:
            return
    else:
        for module in children:
            get_block_weight_layers(module, expand_ratio)

def explore_subnet(expand_ratio, depth_list, subnet_type):
    print(expand_ratio)
    print(depth_list)
    net.set_active_subnet(None, None, expand_ratio, depth_list)
    subnet = net.get_active_subnet(preserve_weight=True)
    if not args.selective_masking_test:
        with open(f'exploring_subnets/{subnet_type}_largest_subnet.txt', 'w') as f:
            f.write(f"{str(expand_ratio)}, {str(depth_list)}\n")
            f.write(str(subnet))

        # 17 is for when using the largest subnets (indices: -1, -2, -3)
        # weights = subnet.blocks[17].mobile_inverted_conv.point_linear.bn.weight.cpu().detach().numpy()
        weights = subnet.blocks[1].mobile_inverted_conv.inverted_bottleneck.conv.weight.cpu().detach().numpy()
        np.set_printoptions(threshold=np.inf)
        # print(weights)
        with open(f'CompOFA/exploring_subnets/{subnet_type}_largest_subnet_weights.txt', 'w') as f:
            f.write(f"Subnet: {str(expand_ratio)}, {str(depth_list)}\n")
            f.write(f"Weight Matrix shape: {weights.shape}\n")
            f.write(str(weights))

        # torch.save(subnet, f'exploring_subnets/{subnet_type}_largest_subnet')
        # pdb.set_trace()
    else:
        for idx, block in enumerate(subnet.blocks[1:]):
            get_block_weight_layers(block, expand_ratio[idx])
            pdb.set_trace()

def gen_subnets():
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
        
        if len(erl) == 20:
            expand_ratio_list.append(erl)
            erl = []
        
        if len(dl) == 5:
            depth_list.append(dl)
            dl = []
    return (expand_ratio_list, depth_list)

if __name__ == '__main__':
    args = parse_args()

    net = OFAMobileNetV3(
            n_classes=1000, dropout_rate=0, width_mult_list=1, ks_list=[3,5,7],
            expand_ratio_list=[3,4,6], depth_list=[2,3,4],
            compound=True, fixed_kernel=True)
    net.load_weights_from_net(torch.load(args.net, map_location='cpu')['state_dict'])
    net.cuda()

    run_config = ImagenetRunConfig(
        test_batch_size=args.batch_size, n_worker=args.workers)

    run_manager = RunManager('.tmp/eval_subnet', net, run_config, init=False)
    run_config.data_provider.assign_active_img_size(224)

    expand_ratio_list, depth_list = gen_subnets()
    if args.selective_masking_test:
        explore_subnet(expand_ratio_list[-1], depth_list[-1], "na")
    else:
        print(f"Setting network to largest subnet: {expand_ratio_list[0]} {depth_list[0]}")
        explore_subnet(expand_ratio_list[0], depth_list[0], "first")
        print(f"Setting network to second largest subnet: {expand_ratio_list[81]} {depth_list[81]}")
        explore_subnet(expand_ratio_list[81], depth_list[81], "second")
        print(f"Setting network to third_largest subnet: {expand_ratio_list[162]} {depth_list[162]}")
        explore_subnet(expand_ratio_list[162], depth_list[162], "third")
