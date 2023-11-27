import torch
from .efficientnet.efficientnet import efficientnet_b3_pruned, efficientnet_b1
from .inception import Inception3
from .swin import swin_t_features, swin_s_features, swin_b_features, swin_v2_t_features, swin_v2_s_features, swin_v2_b_features
from .resnet import resnet50_features


def build_backbone(model_name, pretrained, custom_pretrained=None):
    model = getattr(Backbones, model_name)(pretrained=pretrained)
    if custom_pretrained is not None:
        model_path = custom_pretrained
        if model_path is not None:
            print('load custom_pretrained', model_path)
            print(torch.load(model_path))
            model.my_load_state_dict(torch.load(model_path))
    return model
    

class Backbones(object):
    @staticmethod
    def effi_b3_p(pretrained=True, **kwargs):
        model = efficientnet_b3_pruned(pretrained=pretrained, drop_rate=0.3, drop_path_rate=0.2, just_feature=True)
        return model

    @staticmethod
    def effi_b1_p(pretrained=True, **kwargs):
        model = efficientnet_b1(pretrained=pretrained, drop_rate=0.3, drop_path_rate=0.2, just_feature=True)
        return model
    
    @staticmethod
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
        return model   

    @staticmethod
    def swin_s(pretrained=True, **kwargs):
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        model = swin_s_features(weights=weights)
        return model

    @staticmethod
    def swin_b(pretrained=True, **kwargs):
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        model = swin_b_features(weights=weights)
        return model
    
    @staticmethod
    def swin2_t(pretrained=True, **kwargs):
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        model = swin_v2_t_features(weights=weights)
        return model   

    @staticmethod
    def swin2_s(pretrained=True, **kwargs):
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        model = swin_v2_s_features(weights=weights)
        return model

    @staticmethod
    def swin2_b(pretrained=True, **kwargs):
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        model = swin_v2_b_features(weights=weights)
        return model        