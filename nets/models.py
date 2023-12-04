import torch
import torch.nn as nn
from .backbones import build_backbone
from .cdcl import CdCLTraining


def build_model(model_name, training_params):
    n_class = training_params['n_class']
    custom_pretrained = training_params['custom_pretrained']
    model = getattr(Models, model_name)(n_class=n_class, custom_pretrained=custom_pretrained, model_params=training_params['model_params'], training_params=training_params)
    return model

 
class Models(object):
    @staticmethod
    def cdcl_effi_b3_p(custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = CdCLTraining(backbone=backbone, n_class=model_params['n_class'], channels=1536, 
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    mix_ratio=model_params['mix_ratio'],
                                    score_fusion=model_params['score_fusion'],
                                    training_params=training_params,
                                    use_mean=True)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

