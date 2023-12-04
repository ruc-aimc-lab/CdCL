# copy from timm, delete the classifier layer


""" The EfficientNet Family in PyTorch

An implementation of EfficienNet that covers variety of related models with efficient architectures:

* EfficientNet-V2
  - `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298

* EfficientNet (B0-B8, L2 + Tensorflow pretrained AutoAug/RandAug/AdvProp/NoisyStudent weight ports)
  - EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946
  - Self-training with Noisy Student improves ImageNet classification - https://arxiv.org/abs/1911.04252

The majority of the above models (EfficientNet*, MixNet, MnasNet) and original weights were made available
by Mingxing Tan, Quoc Le, and other members of their Google Brain team. Thanks for consistently releasing
the models and weights open source!

Hacked together by / Copyright 2021 Ross Wightman
"""
from functools import partial
import torch.nn as nn
import torch.nn.functional as F


from .efficientnet_blocks import SqueezeExcite
from .efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights,\
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from .helpers import build_model_with_cfg
from .layers import create_conv2d, create_classifier
from .registry import register_model

__all__ = ['EfficientNet']

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

 
def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    # NOTE experimenting with alternate attention
    'efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth'),
    'efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
        test_input_size=(3, 256, 256), crop_pct=1.0),
    'efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), test_input_size=(3, 288, 288), crop_pct=1.0),
    'efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), test_input_size=(3, 320, 320), crop_pct=1.0),
    'efficientnet_b4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), test_input_size=(3, 384, 384), crop_pct=1.0),
    'efficientnet_b5': _cfg(
        url='', input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'efficientnet_b6': _cfg(
        url='', input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'efficientnet_b7': _cfg(
        url='', input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'efficientnet_b8': _cfg(
        url='', input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
    'efficientnet_l2': _cfg(
        url='', input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.961),

    'efficientnet_lite0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth'),
    'efficientnet_lite1': _cfg(
        url='',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_lite2': _cfg(
        url='',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'efficientnet_lite3': _cfg(
        url='',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'efficientnet_lite4': _cfg(
        url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),

    'efficientnetv2_s': _cfg(
        url='',
        input_size=(3, 288, 288), test_input_size=(3, 384, 384), pool_size=(9, 9), crop_pct=1.0),
    'efficientnetv2_m': _cfg(
        url='',
        input_size=(3, 320, 320), test_input_size=(3, 416, 416), pool_size=(10, 10), crop_pct=1.0),
    'efficientnetv2_l': _cfg(
        url='',
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b8': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ra-572d5dd9.pth',
        input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),

    'tf_efficientnet_lite0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882,
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite2-dcccb7df.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890,
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, interpolation='bilinear'),
    'tf_efficientnet_lite4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite4-741542c3.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.920, interpolation='bilinear'),

    'tf_efficientnetv2_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s-eb54923e.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m-cc09e0cd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l-d664b728.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnetv2_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b0-c7cc451f.pth',
        input_size=(3, 192, 192), test_input_size=(3, 224, 224), pool_size=(6, 6)),
    'tf_efficientnetv2_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b1-be6e41b0.pth',
        input_size=(3, 192, 192), test_input_size=(3, 240, 240), pool_size=(6, 6), crop_pct=0.882),
    'tf_efficientnetv2_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b2-847de54e.pth',
        input_size=(3, 208, 208), test_input_size=(3, 260, 260), pool_size=(7, 7), crop_pct=0.890),
    'tf_efficientnetv2_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b3-57773f13.pth',
        input_size=(3, 240, 240), test_input_size=(3, 300, 300), pool_size=(8, 8), crop_pct=0.904),
    
    'efficientnet_b1_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb1_pruned_9ebb3fe6.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b2_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb2_pruned_203f55bc.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b3_pruned': _cfg(
        # url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb3_pruned_5abcc29f.pth',
        url='./effnetb3_pruned_5abcc29f.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b3_pruned_multidis': _cfg(
        # url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb3_pruned_5abcc29f.pth',
        url='./efficientnet_b3_pruned.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),


}


class EfficientNet(nn.Module):
    """ (Generic) EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large & B0-B3
      * EfficientNet B0-B8, L2
    """

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, fix_stem=False,
                 output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None,
                 se_layer=None, drop_rate=0., drop_path_rate=0., global_pool='avg', just_feature=False):
        super(EfficientNet, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features)
        self.act2 = act_layer(inplace=True)

        self.just_feature = just_feature
        if not just_feature:
            self.global_pool, self.classifier = create_classifier(
                self.num_features, self.num_classes, pool_type=global_pool)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        if not self.just_feature:
            x = self.global_pool(x)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.classifier(x)
        return x

    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)



def _create_effnet(variant, pretrained=False, **kwargs):
    model_cls = EfficientNet
    kwargs_filter = None
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=False,
        kwargs_filter=kwargs_filter,
        **kwargs)

    return model

def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet_lite(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_base(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 base model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """
    arch_def = [
        ['cn_r1_k3_s1_e1_c16_skip'],
        ['er_r2_k3_s2_e4_c32'],
        ['er_r2_k3_s2_e4_c48'],
        ['ir_r3_k3_s2_e4_c96_se0.25'],
        ['ir_r5_k3_s1_e6_c112_se0.25'],
        ['ir_r8_k3_s2_e6_c192_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_s(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, rw=False, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Small model
    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298

    NOTE: `rw` flag sets up 'small' variant to behave like my initial v2 small model,
        before ref the impl was released.
    """
    arch_def = [
        ['cn_r2_k3_s1_e1_c24_skip'],
        ['er_r4_k3_s2_e4_c48'],
        ['er_r4_k3_s2_e4_c64'],
        ['ir_r6_k3_s2_e4_c128_se0.25'],
        ['ir_r9_k3_s1_e6_c160_se0.25'],
        ['ir_r15_k3_s2_e6_c256_se0.25'],
    ]
    num_features = 1280
    if rw:
        # my original variant, based on paper figure differs from the official release
        arch_def[0] = ['er_r2_k3_s1_e1_c24']
        arch_def[-1] = ['ir_r15_k3_s2_e6_c272_se0.25']
        num_features = 1792

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(num_features),
        stem_size=24,
        round_chs_fn=round_chs_fn,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Medium model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """

    arch_def = [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=24,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_l(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Large model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """

    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


@register_model
def efficientnet_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b1(pretrained=False, **kwargs):
    """ EfficientNet-B1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b2(pretrained=False, **kwargs):
    """ EfficientNet-B2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b3(pretrained=False, **kwargs):
    """ EfficientNet-B3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b4(pretrained=False, **kwargs):
    """ EfficientNet-B4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b5(pretrained=False, **kwargs):
    """ EfficientNet-B5 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b6(pretrained=False, **kwargs):
    """ EfficientNet-B6 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b7(pretrained=False, **kwargs):
    """ EfficientNet-B7 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b8(pretrained=False, **kwargs):
    """ EfficientNet-B8 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b8', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_l2(pretrained=False, **kwargs):
    """ EfficientNet-L2."""
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_l2', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite0(pretrained=False, **kwargs):
    """ EfficientNet-Lite0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite1(pretrained=False, **kwargs):
    """ EfficientNet-Lite1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite2(pretrained=False, **kwargs):
    """ EfficientNet-Lite2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite3(pretrained=False, **kwargs):
    """ EfficientNet-Lite3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite4(pretrained=False, **kwargs):
    """ EfficientNet-Lite4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_s(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small. """
    model = _gen_efficientnetv2_s('efficientnetv2_s', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_m(pretrained=False, **kwargs):
    """ EfficientNet-V2 Medium. """
    model = _gen_efficientnetv2_m('efficientnetv2_m', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_l(pretrained=False, **kwargs):
    """ EfficientNet-V2 Large. """
    model = _gen_efficientnetv2_l('efficientnetv2_l', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b1(pretrained=False, **kwargs):
    """ EfficientNet-B1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b2(pretrained=False, **kwargs):
    """ EfficientNet-B2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b3(pretrained=False, **kwargs):
    """ EfficientNet-B3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b4(pretrained=False, **kwargs):
    """ EfficientNet-B4. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b5(pretrained=False, **kwargs):
    """ EfficientNet-B5. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b6(pretrained=False, **kwargs):
    """ EfficientNet-B6. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b7(pretrained=False, **kwargs):
    """ EfficientNet-B7. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b8(pretrained=False, **kwargs):
    """ EfficientNet-B8. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b8', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite0(pretrained=False, **kwargs):
    """ EfficientNet-Lite0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite1(pretrained=False, **kwargs):
    """ EfficientNet-Lite1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite2(pretrained=False, **kwargs):
    """ EfficientNet-Lite2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite3(pretrained=False, **kwargs):
    """ EfficientNet-Lite3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite4(pretrained=False, **kwargs):
    """ EfficientNet-Lite4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_s(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_s('tf_efficientnetv2_s', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_m(pretrained=False, **kwargs):
    """ EfficientNet-V2 Medium. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_m('tf_efficientnetv2_m', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_l(pretrained=False, **kwargs):
    """ EfficientNet-V2 Large. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_l('tf_efficientnetv2_l', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b0(pretrained=False, **kwargs):
    """ EfficientNet-V2-B0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base('tf_efficientnetv2_b0', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b1(pretrained=False, **kwargs):
    """ EfficientNet-V2-B1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b2(pretrained=False, **kwargs):
    """ EfficientNet-V2-B2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b3(pretrained=False, **kwargs):
    """ EfficientNet-V2-B3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model



@register_model
def efficientnet_b1_pruned(pretrained=False, **kwargs):
    """ EfficientNet-B1 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    variant = 'efficientnet_b1_pruned'
    model = _gen_efficientnet(
        variant, channel_multiplier=1.0, depth_multiplier=1.1, pruned=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b2_pruned(pretrained=False, **kwargs):
    """ EfficientNet-B2 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'efficientnet_b2_pruned', channel_multiplier=1.1, depth_multiplier=1.2, pruned=True,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b3_pruned(pretrained=False, **kwargs):
    """ EfficientNet-B3 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'efficientnet_b3_pruned', channel_multiplier=1.2, depth_multiplier=1.4, pruned=True,
        pretrained=pretrained, **kwargs)
    return model

@register_model
def efficientnet_b3_pruned_multidis(pretrained=False, **kwargs):
    """ EfficientNet-B3 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'efficientnet_b3_pruned_multidis', channel_multiplier=1.2, depth_multiplier=1.4, pruned=True,
        pretrained=pretrained, **kwargs)
    return model