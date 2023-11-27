import torch
import torch.nn as nn
from .backbones import build_backbone
from .adp_mil import AdaptiveMILTraining, AdaptiveNetWholeTraining, AdaptiveNetSplitTraining
from .DDC import AdaptiveNetWholeDDCTraining
from .DANN import AdaptiveNetWholeDANNTraining
from .ADDA import AdaptiveNetWholeADDATraining, VITADDANetTraining
from .MDD import AdaptiveNetWholeMDDTraining, VITMDDTraining
from .single_domain_nets import BasicNetTraining, MILNetTraining, FusionNetTraining, InnerMILTraining, InnerMILSATraining, BasicNetUWFTraining, SingleDomainVITFOVNetTraining
from .domain_confusion import AdaptiveNetWholeDCTraining
from .FixBi import AdaptiveNetWholeFixBiTraining, VITFixBiTraining
from .FOVNet import FOVNetTraining, FOVNetTraining_UWFAllLoss, FOVNetTraining_CFPOnlyWhole , \
                     FOVNetDDCTraining, MBFOVNetTraining, MILFOVNetTraining, \
                     VITFOVNetTraining, FOVNetDoubleTraining, VITFOVFuseNetTraining, \
                     VITFOVPosNetTraining, VITFixBiWholeTraining, VITMixupWholeTraining, \
                     VITFOVWholeTriClsNetTraining, VITMixupTraining, MixupNetTraining, \
                     VITMixupWholeSepClaTraining

from .SDAT import AdaptiveNetWholeSDATTraining, AdaptiveNetWholeSDATDANNTraining
from .MIC import AdaptiveNetWholeMICTraining, AdaptiveNetWholeMICDANNTraining
from .CDAN import AdaptiveNetWholeCDANTraining

from .FeaMix import PreFeaMixTraining, ProFeaMixTraining, PrePreFeaMixTraining, PrePreFeaPatchMixTraining
from .QFormer import VITMixupWholeQFormerTraining

def build_model(model_name, training_params):
    n_class = training_params['n_class']
    custom_pretrained = training_params['custom_pretrained']
    model = getattr(Models, model_name)(n_class=n_class, custom_pretrained=custom_pretrained, model_params=training_params['model_params'], training_params=training_params)
    return model

def single_domain_build_model(model_name, training_params):
    n_class = training_params['n_class']
    custom_pretrained = training_params['custom_pretrained']
    model = getattr(SingleDomainModels, model_name)(n_class=n_class, custom_pretrained=custom_pretrained, model_params=training_params.get('model_params', {}))
    return model


class Models(object):
    @staticmethod
    def adaptive_mil_sfusion_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True, custom_pretrained=custom_pretrained)
        model = AdaptiveMILTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    score_fusion='single', 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    crit_cons=nn.L1Loss(),
                                    weights=model_params['weights'],
                                    training_params=training_params)
        return model

    @staticmethod
    def adaptive_mil_mfusion_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True, custom_pretrained=custom_pretrained)
        model = AdaptiveMILTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    score_fusion='multi', 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    crit_cons=nn.L1Loss(),
                                    weights=model_params['weights'],
                                    training_params=training_params)
        return model

    @staticmethod
    def adaptive_whole_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                         crit_sup=nn.BCEWithLogitsLoss(),
                                         weights=model_params['weights'], 
                                         training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model  

    @staticmethod
    def adaptive_split_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetSplitTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                         crit_sup=nn.BCEWithLogitsLoss(),
                                         weights=model_params['weights'], 
                                         training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model  


    @staticmethod
    def adaptive_whole_ddc_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeDDCTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    training_params=training_params,
                                    uda=training_params['uda'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model


    @staticmethod
    def adaptive_whole_dann_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeDANNTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    gamma=model_params['gamma'],
                                    training_params=training_params,
                                    uda=training_params['uda'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    # ADDA
    @staticmethod
    def adaptive_whole_adda_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone1 = build_backbone(model_name='effi_b3_p', pretrained=True)
        backbone2 = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeADDATraining(backbone_source=backbone1, backbone_target=backbone2, 
                                             n_class=n_class, channels=1536, 
                                             crit_sup=nn.BCEWithLogitsLoss(), 
                                             weights=model_params['weights'], 
                                             training_params=training_params,
                                             uda=training_params['uda'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        model.sync_backbones()
        return model
    
    @staticmethod
    def vit_adda_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone1 = build_backbone(model_name='effi_b3_p', pretrained=True)
        backbone2 = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITADDANetTraining(backbone_source=backbone1, backbone_target=backbone2, 
                                             n_class=n_class, channels=1536, 
                                             crit_sup=nn.BCEWithLogitsLoss(), 
                                             weights=model_params['weights'], 
                                             mhsa_nums=model_params['mhsa_nums'], 
                                             mil_ratio=model_params['mil_ratio'], 
                                             over_lap=model_params['over_lap'],
                                             training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        model.sync_backbones()
        return model

    # MDD
    @staticmethod
    def adaptive_whole_mdd_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeMDDTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    gamma=model_params['gamma'],
                                    training_params=training_params,
                                    uda=training_params['uda'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def vit_mdd_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITMDDTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    gamma=model_params['gamma'],
                                    mhsa_nums=model_params['mhsa_nums'], 
                                    mil_ratio=model_params['mil_ratio'], 
                                    over_lap=model_params['over_lap'],
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    # domain confusion
    @staticmethod
    def adaptive_whole_dc_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True, custom_pretrained=custom_pretrained)
        model = AdaptiveNetWholeDCTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                           crit_sup=nn.BCEWithLogitsLoss(),
                                           weights=model_params['weights'], 
                                           training_params=training_params)
        return model  

    # fixbi
    @staticmethod
    def adaptive_whole_fixbi_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone_source = build_backbone(model_name='effi_b3_p', pretrained=True)
        backbone_target = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeFixBiTraining(backbone_source=backbone_source, backbone_target=backbone_target, n_class=n_class, channels=1536, 
                                              crit_sup=nn.BCEWithLogitsLoss(),
                                              weights=model_params['weights'], 
                                              mix_ratio=model_params['mix_ratio'],
                                              cons_reg_start=model_params['cons_reg_start'],
                                              training_params=training_params,
                                              uda=training_params['uda'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        model.sync_backbones()
        return model 
    
    @staticmethod
    def vit_fixbi_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone_source = build_backbone(model_name='effi_b3_p', pretrained=True)
        backbone_target = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITFixBiTraining(backbone_source=backbone_source, backbone_target=backbone_target, n_class=n_class, channels=1536, 
                                              crit_sup=nn.BCEWithLogitsLoss(),
                                              weights=model_params['weights'], 
                                              mix_ratio=model_params['mix_ratio'],
                                              cons_reg_start=model_params['cons_reg_start'],
                                              mhsa_nums=model_params['mhsa_nums'], 
                                              mil_ratio=model_params['mil_ratio'], 
                                              over_lap=model_params['over_lap'],
                                              training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        model.sync_backbones()
        return model 

    @staticmethod
    def vit_fixbi_whole_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone_source = build_backbone(model_name='effi_b3_p', pretrained=True)
        backbone_target = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITFixBiWholeTraining(backbone_source=backbone_source, backbone_target=backbone_target, n_class=n_class, channels=1536, 
                                              crit_sup=nn.BCEWithLogitsLoss(),
                                              weights=model_params['weights'], 
                                              mix_ratio=model_params['mix_ratio'],
                                              cons_reg_start=model_params['cons_reg_start'],
                                              mhsa_nums=model_params['mhsa_nums'], 
                                              mil_ratio=model_params['mil_ratio'], 
                                              over_lap=model_params['over_lap'],
                                              score_fusion=model_params['score_fusion'],
                                              training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        model.sync_backbones()
        return model 

    # CDAN
    @staticmethod
    def adaptive_whole_cdan_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeCDANTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                            crit_sup=nn.BCEWithLogitsLoss(),
                                            weights=model_params['weights'],
                                            gamma=model_params['gamma'],
                                            eps=model_params['eps'],
                                            training_params=training_params,
                                            uda=training_params['uda'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model


    # SDAT and label smoothing
    @staticmethod
    def adaptive_whole_sdat_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeSDATTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                            crit_sup=nn.BCEWithLogitsLoss(),
                                            weights=model_params['weights'],
                                            gamma=model_params['gamma'],
                                            eps=model_params['eps'],
                                            training_params=training_params,
                                            uda=training_params['uda'],
                                            mcc=training_params['mcc'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def adaptive_whole_sdat_dann_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeSDATDANNTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                            crit_sup=nn.BCEWithLogitsLoss(),
                                            weights=model_params['weights'],
                                            gamma=model_params['gamma'],
                                            eps=model_params['eps'],
                                            training_params=training_params,
                                            uda=training_params['uda'],
                                            mcc=training_params['mcc'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model


    # SDAT and MIC
    @staticmethod
    def adaptive_whole_mic_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeMICTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                            crit_sup=nn.BCEWithLogitsLoss(),
                                            weights=model_params['weights'],
                                            gamma=model_params['gamma'],
                                            eps=model_params['eps'],
                                            training_params=training_params,
                                            uda=training_params['uda'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def adaptive_whole_mic_dann_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = AdaptiveNetWholeMICDANNTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                            crit_sup=nn.BCEWithLogitsLoss(),
                                            weights=model_params['weights'],
                                            gamma=model_params['gamma'],
                                            eps=model_params['eps'],
                                            mcc=training_params['mcc'],
                                            training_params=training_params,
                                            uda=training_params['uda'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model


    # FOVNet
    @staticmethod
    def fov_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = FOVNetTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    score_fusion=model_params['score_fusion'],
                                    mil_ratio=model_params['mil_ratio'],
                                    over_lap=model_params.get('over_lap', True),
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def fov_vit_fuse_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITFOVFuseNetTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    score_fusion=model_params['score_fusion'],
                                    mil_ratio=model_params['mil_ratio'],
                                    over_lap=model_params.get('over_lap', True),
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def fov_double_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = FOVNetDoubleTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    score_fusion=model_params['score_fusion'],
                                    mil_ratio=model_params['mil_ratio'],
                                    over_lap=model_params.get('over_lap', True),
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def fov_uwf_all_loss_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = FOVNetTraining_UWFAllLoss(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    score_fusion=model_params['score_fusion'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def fov_uwf_cfp_whole_only_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = FOVNetTraining_CFPOnlyWhole(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    score_fusion=model_params['score_fusion'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def fov_ddc_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = FOVNetDDCTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    score_fusion=model_params['score_fusion'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def multi_backbone_fov_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        backbone_mil = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = MBFOVNetTraining(backbone=backbone, backbone_mil=backbone_mil, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    score_fusion=model_params['score_fusion'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        model.sync_backbones()
        return model
    
    @staticmethod
    def mil_fov_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = MILFOVNetTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def vit_fov_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITFOVNetTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    
    @staticmethod
    def vit_fov_mean_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITFOVNetTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params,
                                    use_mean=True)
        print(torch.load(custom_pretrained).keys())
        print('*\n'*10)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
    

    @staticmethod
    def vit_fov_mean_whole_tricls_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITFOVWholeTriClsNetTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                            crit_sup=nn.BCEWithLogitsLoss(),
                                            weights=model_params['weights'],
                                            mhsa_nums=model_params['mhsa_nums'],
                                            mil_ratio=model_params['mil_ratio'],
                                            score_fusion=model_params['score_fusion'],
                                            all_loss=model_params['all_loss'],
                                            over_lap=model_params['over_lap'],
                                            training_params=training_params,
                                            use_mean=True)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    
    @staticmethod
    def vit_fov_mixup_mean_whole_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITMixupWholeTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
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


    @staticmethod
    def vit_fov_mixup_mean_whole_sepcla_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITMixupWholeSepClaTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
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



    @staticmethod
    def vit_fov_mixup_mean_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITMixupTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    mix_ratio=model_params['mix_ratio'],
                                    training_params=training_params,
                                    use_mean=True)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def vit_mix_net_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = MixupNetTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mix_ratio=model_params['mix_ratio'],
                                    training_params=training_params)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def pro_fea_mix_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = ProFeaMixTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params,
                                    mix_ratio=model_params['mix_ratio'],
                                    over_lap=model_params['over_lap'],
                                    score_fusion=model_params['score_fusion'],
                                    use_mean=model_params['use_mean'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def pre_fea_mix_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = PreFeaMixTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params,
                                    mix_ratio=model_params['mix_ratio'],
                                    over_lap=model_params['over_lap'],
                                    score_fusion=model_params['score_fusion'],
                                    mix_ins_ratio=model_params['mix_ins_ratio'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def pre_pre_fea_mix_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = PrePreFeaMixTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params,
                                    mix_ratio=model_params['mix_ratio'],
                                    over_lap=model_params['over_lap'],
                                    score_fusion=model_params['score_fusion'],
                                    mix_ins_ratio=model_params['mix_ins_ratio'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def pre_pre_fea_patch_mix_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = PrePreFeaPatchMixTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    mhsa_nums=model_params['mhsa_nums'],
                                    mil_ratio=model_params['mil_ratio'],
                                    training_params=training_params,
                                    over_lap=model_params['over_lap'],
                                    score_fusion=model_params['score_fusion'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model


    @staticmethod
    def qformer_mixup_effi_b3_p(n_class, custom_pretrained, model_params, training_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = VITMixupWholeQFormerTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                    crit_sup=nn.BCEWithLogitsLoss(),
                                    weights=model_params['weights'],
                                    num_q=32, dim_q=768, att_nums=2,
                                    mil_ratio=model_params['mil_ratio'],
                                    mix_ratio=model_params['mix_ratio'],
                                    score_fusion=model_params['score_fusion'],
                                    training_params=training_params,)
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model


class SingleDomainModels(object):
    @staticmethod
    def basic_effi_b3_p(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = BasicNetTraining(backbone=backbone, n_class=n_class, channels=1536, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_uwf_effi_b3_p(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = BasicNetUWFTraining(backbone=backbone, n_class=n_class, channels=1536, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_inceptionv3(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='inceptionv3', pretrained=True)
        model = BasicNetTraining(backbone=backbone, n_class=n_class, channels=2048, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_uwf_inceptionv3(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='inceptionv3', pretrained=True)
        model = BasicNetUWFTraining(backbone=backbone, n_class=n_class, channels=2048, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_resnet50(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='resnet50', pretrained=True)
        model = BasicNetTraining(backbone=backbone, n_class=n_class, channels=2048, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_uwf_resnet50(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='resnet50', pretrained=True)
        model = BasicNetUWFTraining(backbone=backbone, n_class=n_class, channels=2048, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_swin_t(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='swin_t', pretrained=True)
        model = BasicNetTraining(backbone=backbone, n_class=n_class, channels=768, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_uwf_swin_t(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='swin_t', pretrained=True)
        model = BasicNetUWFTraining(backbone=backbone, n_class=n_class, channels=768, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_uwf_swin2_t(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='swin2_t', pretrained=True)
        model = BasicNetUWFTraining(backbone=backbone, n_class=n_class, channels=768, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_uwf_swin_s(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='swin2_s', pretrained=True)
        model = BasicNetUWFTraining(backbone=backbone, n_class=n_class, channels=768, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def basic_uwf_swin_b(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='swin2_b', pretrained=True)
        model = BasicNetUWFTraining(backbone=backbone, n_class=n_class, channels=1024, crit=nn.BCEWithLogitsLoss())
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model

    @staticmethod
    def mil_effi_b3_p(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True, custom_pretrained=custom_pretrained)
        model = MILNetTraining(backbone=backbone, n_class=n_class, channels=1536, crit=nn.BCEWithLogitsLoss())
        return model
    
    @staticmethod
    def inner_mil_effi_b3_p(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True, custom_pretrained=custom_pretrained)
        model = InnerMILTraining(backbone=backbone, n_class=n_class, channels=1536, crit=nn.BCEWithLogitsLoss())
        return model


    @staticmethod
    def inner_mil_sa_effi_b3_p(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True, custom_pretrained=custom_pretrained)
        model = InnerMILSATraining(backbone=backbone, n_class=n_class, channels=1536, crit=nn.BCEWithLogitsLoss())
        return model


    @staticmethod
    def sfusion_effi_b3_p(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True, custom_pretrained=custom_pretrained)
        model = FusionNetTraining(backbone=backbone, n_class=n_class, channels=1536, crit=nn.BCEWithLogitsLoss(), score_fusion='single')
        return model

    @staticmethod
    def mfusion_effi_b3_p(n_class, custom_pretrained, **kwargs):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True, custom_pretrained=custom_pretrained)
        model = FusionNetTraining(backbone=backbone, n_class=n_class, channels=1536, crit=nn.BCEWithLogitsLoss(), score_fusion='multi')
        return model

    @staticmethod
    def vit_fov_whole_effi_b3_p(n_class, custom_pretrained, model_params):
        backbone = build_backbone(model_name='effi_b3_p', pretrained=True)
        model = SingleDomainVITFOVNetTraining(backbone=backbone, n_class=n_class, channels=1536, 
                                             crit=nn.BCEWithLogitsLoss(),
                                             mhsa_nums=model_params['mhsa_nums'],
                                             mil_ratio=model_params['mil_ratio'],
                                             over_lap=model_params['over_lap'],
                                             use_mean=model_params['use_mean'])
        if custom_pretrained:
            model.my_load_state_dict(torch.load(custom_pretrained))
        return model
