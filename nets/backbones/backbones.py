"""
The implementations of different backbones and the ImageNet pretrained weights 
are from torchvison and timm.
Some implementations and weights may be not the newest version.
"""

import torch
'''from .efficientnet.efficientnet import efficientnet_b3_pruned
from .inception import Inception3
from .swin import swin_t_features
from .resnet import resnet50_features'''

from timm.models.resnet import resnet50
from timm.models.inception_v3 import inception_v3
from timm.models.efficientnet import efficientnet_b3_pruned

# the torchvision implementation of swin transformer support input size 512
from torchvision.models import swin_t 

def build_backbone(model_name, pretrained):

    model = getattr(Backbones, model_name)(pretrained=pretrained)
    """if custom_pretrained is not None:
        
        Source domain pretrained weights may be loaded here.
        Use the weights pretrained on source domain (color fundus image) can increase the performance of some models
        
        print('load custom_pretrained', custom_pretrained)
        model.my_load_state_dict(torch.load(custom_pretrained))"""
    return model


class Backbones(object):
    @staticmethod
    def effi_b3_p(pretrained=True, **kwargs):
        model = efficientnet_b3_pruned(pretrained=pretrained, drop_rate=0.3, drop_path_rate=0.2, just_feature=True)
        return model
    
    '''@staticmethod
    def inceptionv3(pretrained=True, **kwargs):
        model = Inception3()
        if pretrained:
            model.load_pretrained()
        return model
    
    @staticmethod
    def resnet50(pretrained=True, **kwargs):
        if pretrained:
            weights = 'IMAGENET1K_V2'
        else:
            weights = None
        model = resnet50_features(weights=weights, progress=True)
        return model

    @staticmethod
    def swin_t(pretrained=True, **kwargs):
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        model = swin_t_features(weights=weights)
        return model'''   

          