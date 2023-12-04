import torch
import torch.nn as nn
from .backbones import build_backbone
from .cdcl import CdCLProcessor


def build_model(model_name, training_params, only_predict=False):
    custom_pretrained = training_params['custom_pretrained']
    model = getattr(Models, model_name)(custom_pretrained=custom_pretrained, training_params=training_params, only_predict=only_predict)
    return model

 
class Models(object):
    @staticmethod
    def cdcl_effi_b3_p(custom_pretrained, training_params, only_predict):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = CdCLProcessor(backbone=backbone, 
                              backbone_out_channels=1536,
                              training_params=training_params,
                              only_predict=only_predict)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

