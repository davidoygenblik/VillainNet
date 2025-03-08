# CompOFA – Compound Once-For-All Networks for Faster Multi-Platform Deployment
# Under blind review at ICLR 2021: https://openreview.net/forum?id=IgIk8RRT-Z
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

import numpy as np
import torch

from CompOFA.ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
from CompOFA.ofa.layers import ConvLayer, IdentityLayer, LinearLayer, MBInvertedConvLayer, BatchNorm
from CompOFA.ofa.elastic_nn.modules.dynamic_op import DynamicBatchNorm2d
from CompOFA.ofa.imagenet_codebase.networks.resnet import Resnet, ResnetBlock
from CompOFA.ofa.imagenet_codebase.utils import make_divisible, int2list
import torch.nn.functional as F

import pdb


class OFAResnet(Resnet):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None,
                 width_mult_list=1.0, ks_list=3, depth_list=4, compound=False, fixed_kernel=False):

        self.width_mult_list = int2list(width_mult_list, 1)
        self.ks_list = int2list(ks_list, 1)
        self.depth_list = int2list(depth_list, 1)
        self.base_stage_width = base_stage_width
        self.compound = compound
        self.fixed_kernel = fixed_kernel

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.depth_list.sort()

        base_stage_width = [16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = [
            make_divisible(base_stage_width[-2] * max(self.width_mult_list), 8) for _ in self.width_mult_list
        ]
        last_channel = [
            make_divisible(base_stage_width[-1] * max(self.width_mult_list), 8) for _ in self.width_mult_list
        ]

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
        if depth_list is None:
            #TODO Check Abhi
            n_block_list = [1, 2, 3, 4, 2, 3]
            self.depth_list = [4, 4]
            print('Use MobileNetV3 Depth Setting')
        else:
            n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        input_channel = width_list[0]

        # first conv layer
        if len(set(input_channel)) == 1:
            first_conv = ConvLayer(3, max(input_channel), kernel_size=3, stride=2, act_func='h_swish', use_bn=False)
            first_bn = BatchNorm(max(input_channel))

            second_conv = ConvLayer(
                in_channels=max(input_channel), out_channels=max(input_channel), kernel_size=3, stride=stride_stages[0],
                act_func=act_stages[0], use_bn=False
            )
            second_bn = BatchNorm(max(input_channel))

            third_conv = ConvLayer(
                in_channels=max(input_channel), out_channels=max(input_channel), kernel_size=3, stride=stride_stages[0],
                act_func=act_stages[0], use_bn=False
            )
            third_bn = BatchNorm(max(input_channel))

        else:
            first_conv = DynamicConvLayer(
                in_channel_list=int2list(3, len(input_channel)), out_channel_list=input_channel, kernel_size=3,
                stride=2, act_func='h_swish',use_bn=False
            )
            first_bn = DynamicBatchNorm2d(input_channel)

            second_conv = DynamicConvLayer(
                in_channel_list=input_channel, out_channel_list=input_channel, kernel_size=3,
                stride=stride_stages[0], act_func=act_stages[0], use_bn=False
            )
            second_bn = DynamicBatchNorm2d(input_channel)

            third_conv = DynamicConvLayer(
                in_channel_list=input_channel, out_channel_list=input_channel, kernel_size=3,
                stride=stride_stages[0], act_func=act_stages[0], use_bn=False
            )
            third_bn = DynamicBatchNorm2d(input_channel)



        #first block
        first_block = ResnetBlock(second_conv, second_bn, third_conv, third_bn, IdentityLayer(input_channel, input_channel))

        # Resnet basic blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = input_channel

        for width, n_block, s, act_func in zip(width_list[1:], n_block_list[1:],
                                                       stride_stages[1:], act_stages[1:]):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                conv1 = DynamicConvLayer(
                    in_channel_list=feature_dim, out_channel_list=output_channel, kernel_size=max(ks_list), stride=stride, act_func=act_func, use_bn=False
                )
                bn1 = DynamicBatchNorm2d(output_channel)
                conv2 = DynamicConvLayer(
                    in_channel_list=output_channel, out_channel_list=output_channel, kernel_size=max(ks_list), stride=stride, act_func=act_func, use_bn=False)
                bn2 = DynamicBatchNorm2d(output_channel)

                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResnetBlock(conv1, bn1, conv2, bn2, shortcut))
                feature_dim = output_channel

        ''' # final expand layer, feature mix layer & classifier
        if len(final_expand_width) == 1:
            final_expand_layer = ConvLayer(max(feature_dim), max(final_expand_width), kernel_size=1, act_func='h_swish')
            feature_mix_layer = ConvLayer(
                max(final_expand_width), max(last_channel), kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
            )
        else:
            final_expand_layer = DynamicConvLayer(
                in_channel_list=feature_dim, out_channel_list=final_expand_width, kernel_size=1, act_func='h_swish'
            )
            feature_mix_layer = DynamicConvLayer(
                in_channel_list=final_expand_width, out_channel_list=last_channel, kernel_size=1,
                use_bn=False, act_func='h_swish',
            )'''
        if len(final_expand_width) == 1:
            classifier = LinearLayer(max(feature_dim), n_classes, dropout_rate=dropout_rate)
        else:
            classifier = DynamicLinearLayer(
                in_features_list=feature_dim, out_features=n_classes, bias=True, dropout_rate=dropout_rate
            )
        super(OFAResnet, self).__init__(first_conv, first_bn, blocks, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAResnet'

    def forward(self, x):
        # first conv

        x = self.first_conv(x)
        x = self.first_bn(x)

        # first block
        x = self.blocks[0](x)

        # blocks
        #pdb.set_trace()
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)
        pdb.set_trace()
        #x = F.avg_pool2d(x, 4)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.first_bn.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'

        _str += self.classifier.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': OFAResnet.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'first_bn': self.first_bn.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_weights_from_net(self, src_model_dict):
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            elif 'matrix' in key:
                # Fixed kernel models can ignore 'matrix' keys for kernel size transformations
                assert self.fixed_kernel, f"The following key is expected in model dicts for both src & dst models, but is missing in dst model: {key}"
                print(f'Ignoring elastic kernel matrix in weights dict: {key}')
                continue
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = src_model_dict[key]
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, wid=None, ks=None, d=None):
        # pdb.set_trace()
        if self.fixed_kernel:
            assert ks is None, "You tried to set kernel size for a fixed kernel network!"
            ks = []
            kernel_stages = [3, 3, 5, 3, 3, 5]
            for k in kernel_stages[1:]:
                ks.extend([k] * 4)

        width_mult_id = int2list(wid, 4 + len(self.block_group_info))
        ks = int2list(ks, len(self.blocks) - 1)
        depth = int2list(d, len(self.block_group_info))

        #TODO check Abhi
        for block, k in zip(self.blocks[1:], ks):
            if k is not None:
                block.conv1.active_kernel_size = k
                block.conv2.active_kernel_size = k

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        elif constraint_type == 'width_mult':
            self.__dict__['_widthMult_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_ks_include_list'] = None
        self.__dict__['_widthMult_include_list'] = None

    def sample_active_subnet(self):
        if self.compound:
            return self.sample_compound_subnet()

        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample width_mult
        width_mult_setting = None

        if self.fixed_kernel:
            ks_setting = None
        else:
            # sample kernel size
            ks_setting = []
            if not isinstance(ks_candidates[0], list):
                ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
            for k_set in ks_candidates:
                k = random.choice(k_set)
                ks_setting.append(k)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(width_mult_setting, ks_setting, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'd': depth_setting,
        }

    '''def sample_compound_subnet(self):

        def clip_expands(expands):
            low = min(self.expand_ratio_list)
            expands = list(set(np.clip(expands, low, None)))
            return expands

        depth_candidates = self.depth_list
        mapping = {
            2: clip_expands([3, ]),
            3: clip_expands([4, ]),
            4: clip_expands([6, ]),
        }

        # used in in case of unbalanced distribution to sample proportional w/ cardinality
        combinations_per_depth = {d: len(mapping[d]) ** d for d in depth_candidates}
        sum_combinations = sum(combinations_per_depth.values())
        depth_sampling_weights = {k: v / sum_combinations for k, v in combinations_per_depth.items()}

        width_mult_setting = None
        depth_setting = []
        expand_setting = []
        for block_idx in self.block_group_info:
            # for each block, sample a random depth weighted by the number of combinations
            # for each layer in block, sample from corresponding expand ratio
            sampled_d = np.random.choice(depth_candidates, p=list(depth_sampling_weights.values()))
            corresp_e = mapping[sampled_d]

            depth_setting.append(sampled_d)
            for _ in range(len(block_idx)):
                expand_setting.append(random.choice(corresp_e))

        if self.fixed_kernel:
            ks_setting = None
        else:
            # sample kernel size
            ks_setting = []
            if not isinstance(ks_candidates[0], list):
                ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
            for k_set in ks_candidates:
                k = random.choice(k_set)
                ks_setting.append(k)

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }'''

    def get_active_subnet(self, preserve_weight=True):
        #TODO check Abhi
        first_conv = copy.deepcopy(self.first_conv)
        first_bn = copy.deepcopy(self.first_bn)
        blocks = [copy.deepcopy(self.blocks[0])]

        classifier = copy.deepcopy(self.classifier)
        input_channel = blocks[0].conv1.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResnetBlock(
                    self.blocks[idx].conv1.get_active_subnet(input_channel, preserve_weight),
                    self.blocks[idx].bn1.get_active_subnet(input_channel, preserve_weight),
                    self.blocks[idx].conv2.get_active_subnet(input_channel, preserve_weight),
                    self.blocks[idx].bn2.get_active_subnet(input_channel, preserve_weight),
                    copy.deepcopy(self.blocks[idx].shortcut)
                ))
                input_channel = stage_blocks[-1].conv2.out_channels
            blocks += stage_blocks

        _subnet = Resnet(first_conv, first_bn, blocks, classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        # first conv
        first_conv_config = self.first_conv.config
        first_block_config = self.blocks[0].config
        if isinstance(self.first_conv, DynamicConvLayer):
            first_conv_config = self.first_conv.get_active_subnet_config(3)
            first_bn_config = self.first_bn.get_active_subnet_config(self.first_conv.out_channels)
            first_block_config = {
                'name': ResnetBlock.__name__,
                'conv1': self.blocks[0].conv1.get_active_subnet_config(
                    first_conv_config['out_channels']
                ),
                'bn1': self.blocks[0].bn1.get_active_subnet_config(first_conv_config['out_channels']),
                'conv2': self.blocks[0].conv1.get_active_subnet_config(
                    first_conv_config['out_channels']
                ),
                'bn2': self.blocks[0].bn1.get_active_subnet_config(first_conv_config['out_channels']),
                'shortcut': self.blocks[0].shortcut.config if self.blocks[0].shortcut is not None else None,
            }
        classifier_config = self.classifier.config
        if isinstance(self.classifier, DynamicLinearLayer):
            ''' Last block is connected to classifier layers here so this makes sense.'''
            classifier_config = self.classifier.get_active_subnet_config(self.blocks[-1].conv2.out_channels)

        block_config_list = [first_block_config]
        input_channel = first_block_config['conv2']['out_channels']
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append({
                    'name': ResnetBlock.__name__,
                    'conv1': {
                        'name': ConvLayer.__name__,
                        'in_channels': input_channel,
                        'out_channels': self.blocks[idx].conv1.active_out_channel,
                        'kernel_size': self.blocks[idx].conv1.active_kernel_size,
                        'stride': self.blocks[idx].conv1.stride,
                        'act_func': self.blocks[idx].conv1.act_func,
                        'use_bn': self.blocks[idx].conv1.use_bn,
                    },
                    'bn1': {
                        'name': BatchNorm.__name__,
                        'planes': self.blocks[idx].bn1.planes,
                    },
                    'conv2': {
                        'name': ConvLayer.__name__,
                        'in_channels': self.blocks[idx].mobile_inverted_conv.active_out_channel,
                        'out_channels': self.blocks[idx].mobile_inverted_conv.active_out_channel,
                        'kernel_size': self.blocks[idx].mobile_inverted_conv.active_kernel_size,
                        'stride': self.blocks[idx].mobile_inverted_conv.stride,
                        'act_func': self.blocks[idx].mobile_inverted_conv.act_func,
                    },
                    'bn2': {
                        'name': BatchNorm.__name__,
                        'planes': self.blocks[idx].bn2.planes,
                    },
                    'shortcut': self.blocks[idx].shortcut.config if self.blocks[idx].shortcut is not None else None,
                })
                input_channel = self.blocks[idx].mobile_inverted_conv.active_out_channel
            block_config_list += stage_blocks

        return {
            'name': Resnet.__name__,
            'bn': self.get_bn_param(),
            'first_conv': first_conv_config,
            'first_bn': first_bn_config,
            'blocks': block_config_list,
            'classifier': classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.mobile_inverted_conv.re_organize_middle_weights(expand_ratio_stage)
