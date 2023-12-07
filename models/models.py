import torch
from .backbones import build_backbone
from .cdcl import CdCLProcessor


def build_model(model_name, training_params, training=True):
    custom_pretrained = training_params['custom_pretrained']
    model = getattr(Models, model_name)(custom_pretrained=custom_pretrained, training_params=training_params, training=training)
    return model

 
class Models(object):
    @staticmethod
    def cdcl_effi_b3_p(custom_pretrained, training_params, training):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = CdCLProcessor(backbone=backbone, 
                              backbone_out_channels=1536,
                              training_params=training_params,
                              training=training)
        if custom_pretrained:
            """Source domain pretrained weights may be loaded here.
                Use the weights pretrained on source domain (color fundus image) can increase the performances of some models"""
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model


    # You can put your own models here
