import torch

from timm.models.resnet import resnet50
from timm.models.inception_v3 import inception_v3
from timm.models.efficientnet import efficientnet_b3_pruned

# the torchvision implementation of swin transformer support flexible input size as convolutional networks
from torchvision.models import swin_t 


def build_backbone(model_name, pretrained):
    model = getattr(Backbones, model_name)(pretrained=pretrained)
    return model


class Backbones(object):
    @staticmethod
    def effi_b3_p(pretrained=True, **kwargs):
        model = efficientnet_b3_pruned(pretrained=pretrained, drop_rate=0.3, drop_path_rate=0.2)
        return model
    