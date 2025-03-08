# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import pdb

import torch
import torch.nn as nn

# from layers import *
from CompOFA.ofa.layers import set_layer_from_config, ConvLayer, IdentityLayer, LinearLayer, ZeroLayer, BatchNorm
from CompOFA.ofa.imagenet_codebase.utils import MyNetwork, make_divisible
from CompOFA.ofa.layers import MyModule
import torch.nn.functional as F

class ResnetBlock(MyModule):

    def __init__(self, conv1, bn1, conv2, bn2, shortcut):
        super(ResnetBlock, self).__init__()

        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        self.shortcut = shortcut

    def forward(self, x):
        pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        ''' Not sure if this relu should be here or not'''
        #out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        ''' Not sure if this relu should be here or not'''
        #out = F.relu(out)

        if self.shortcut is not None or not (isinstance(self.shortcut, ZeroLayer)):
            out += self.shortcut(x)
        out = F.relu(out)
        return out

    @property
    def module_str(self):
        return '(%s, %s, %s, %s, %s)' % (
            self.conv1.module_str if self.conv1 is not None else None,
            self.bn1.module_str if self.bn1 is not None else None,
            self.conv2.module_str if self.conv2 is not None else None,
            self.bn2.module_str if self.bn2 is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': ResnetBlock.__name__,
            'conv1': self.conv1.config if self.conv1 is not None else None,
            'bn1': self.bn1.config if self.bn1 is not None else None,
            'conv2': self.conv2.config if self.conv2 is not None else None,
            'bn2': self.conv1.config if self.bn2 is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        conv1 = set_layer_from_config(config['conv1'])
        bn1 = set_layer_from_config(config['bn1'])
        conv2 = set_layer_from_config(config['conv2'])
        bn2 = set_layer_from_config(config['bn2'])
        shortcut = set_layer_from_config(config['shortcut'])
        return ResnetBlock(conv1, bn1, conv2, bn2, shortcut)

class Resnet(MyNetwork):

    def __init__(self, first_conv,first_bn, blocks, classifier):
        super(Resnet, self).__init__()

        self.first_conv = first_conv
        self.first_bn = first_bn
        self.blocks = nn.ModuleList(blocks)
        self.classifier = classifier

    def forward(self, x):
        x = F.relu(self.first_bn(self.first_conv(x)))
        for block in self.blocks:
            x = block(x)
        x = F.avg_pool2d(x, 4)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str = self.first_bn.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': Resnet.__name__,
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
        first_conv = set_layer_from_config(config['first_conv'])
        first_bn = set_layer_from_config(config['first_bn'])

        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(ResnetBlock.build_from_config(block_config))

        net = Resnet(first_conv, first_bn, blocks, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResnetBlock):
                if isinstance(m.conv1, ConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.conv1.point_linear.bn.weight.data.zero_()
                if isinstance(m.conv2, ConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.conv2.point_linear.bn.weight.data.zero_()


    @staticmethod
    def build_net_via_cfg(cfg, input_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
        )
        first_bn = BatchNorm(input_channel)
        # build resnet blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
                conv1 = ConvLayer(
                    feature_dim, mid_channel, k, stride)
                bn1 = BatchNorm(out_channel)
                conv2 = ConvLayer(
                    mid_channel, out_channel, k, stride)
                bn2 = BatchNorm(out_channel)
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(ResnetBlock(conv1, bn1, conv2, bn2, shortcut))
                feature_dim = out_channel
        # classifier
        classifier = LinearLayer(feature_dim, n_classes, dropout_rate=dropout_rate)

        return first_conv, first_bn, blocks, classifier

    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != '0':
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != '0':
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != '0':
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg
