# Adversarial Discriminative Domain Adaptation
# done
from .adp_mil import AdaptiveNetWhole
import torch.nn as nn
import torch
from .optimizer import Optimizer
from .DomainClassifier import Classifier
from .FOVNet import MyAvgPool2d, SelfAttentionBlocks
from .MLP import MLP


class AdaptiveNetWholeADDATraining(AdaptiveNetWhole):
    def __init__(self, backbone_source, backbone_target, n_class, channels, crit_sup, weights, training_params, uda=False):
        super().__init__(backbone_source, n_class, channels)
        
        self.backbone_target = backbone_target
       

        self.crit_sup = crit_sup
        self.crit_discriminator = nn.CrossEntropyLoss()
        self.weights = weights # 长度为3，uwf的监督损失，uwf的生成特征损失，判别器损失
        # self.classifier_source = Linear(channel_in=channels, channel_out=n_class)
        # self.classifier_target = MLP(in_features=channels, hidden_features=128, out_features=n_class)

        self.discriminator = Classifier(in_feature=channels, n_class=2)

        self.opt = Optimizer([self.backbone_target, self.classifier], training_params)
        self.opt_discriminator = Optimizer([self.discriminator], training_params)
        self.uda = uda
        if self.uda:
            print('uda mode')

    def sync_backbones(self):
        self.backbone_target.load_state_dict(self.backbone.state_dict())

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        
        # train discriminator
        self.opt_discriminator.z_grad()

        cfp_feature_map = self.backbone.forward(cfp)
        clarus_whole_feature_map = self.backbone_target.forward(clarus_whole)

        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)

        features_concat = torch.cat([cfp_feature, clarus_whole_feature], dim=0)
        score_discriminator = self.discriminator(features_concat.detach())

        gt_cfp_domian = torch.ones(cfp_feature.size(0)).long().cuda()
        gt_clarus_whole_domian = torch.zeros(clarus_whole_feature.size(0)).long().cuda()
        
        gt_discriminator = torch.cat((gt_cfp_domian, gt_clarus_whole_domian), dim=0)
        loss_discriminator = self.crit_discriminator(score_discriminator, gt_discriminator) * self.weights[2]

        loss_discriminator.backward()
        self.opt_discriminator.g_step()
        self.opt_discriminator.update_lr()

        # train backbone and classifier
        self.opt.z_grad()
        self.opt_discriminator.z_grad()

        clarus_whole_feature_map = self.backbone_target.forward(clarus_whole)
        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)
        clarus_whole_score = self.classifier(clarus_whole_feature)

        score_discriminator_clarus_whole = self.discriminator(clarus_whole_feature)
        gt_clarus_whole_domian = torch.ones(clarus_whole_score.size(0)).long().cuda()

        loss_adv = self.crit_discriminator(score_discriminator_clarus_whole, gt_clarus_whole_domian) * self.weights[1]

        if gt_clarus is not None and not self.uda:
            loss_cls = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[0]
            loss = loss_adv + loss_cls
        else:
            loss_cls = None
            loss = loss_adv

        loss.backward()
        self.opt.g_step()
        self.opt.z_grad()
        self.opt.update_lr()
        
        return clarus_whole_score, loss, [loss_adv, loss_cls, loss_discriminator], []

    def predict_result(self, clarus_whole, clarus_split):
        clarus_whole_feature_map = self.backbone_target.forward(clarus_whole)
        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)
        clarus_whole_score = self.classifier(clarus_whole_feature)
        return clarus_whole_score



class VITADDANet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True):
        super(VITADDANet, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)

        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)

        self.mil_ratio = mil_ratio

    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)


    def forward(self, cfp, clarus_whole, clarus_split, need_feature=False):
        # clarus_whole = F.interpolate(clarus_whole, scale_factor=self.mil_ratio, mode='bilinear')
        cfp_feature_map = self.backbone.forward(cfp)
        clarus_feature_map = self.backbone.forward(clarus_whole)
        
        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)[:, 0]

        cfp_score = self.classifier(cfp_feature)

        clarus_mil_score = self.classifier(clarus_mil_feature)

        if need_feature:
            return cfp_score, clarus_mil_score, [cfp_feature, clarus_mil_feature]
        else:
            return cfp_score, clarus_mil_score
    
    def predict_result(self, clarus_whole, clarus_split):
        # clarus_whole = F.interpolate(clarus_whole, scale_factor=self.mil_ratio, mode='bilinear')
        clarus_feature_map = self.backbone.forward(clarus_whole)
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)[:, 0]
        
        clarus_mil_score = self.classifier(clarus_mil_feature)
        
        return clarus_mil_score


class VITADDANetTraining(VITADDANet):
    def __init__(self, backbone_source, backbone_target, n_class, channels, crit_sup, weights, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True):
        super().__init__(backbone_source, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap)
        
        self.backbone_target = backbone_target
       

        self.crit_sup = crit_sup
        self.crit_discriminator = nn.CrossEntropyLoss()
        self.weights = weights # 长度为3，uwf的监督损失，uwf的生成特征损失，判别器损失
        # self.classifier_source = Linear(channel_in=channels, channel_out=n_class)
        # self.classifier_target = MLP(in_features=channels, hidden_features=128, out_features=n_class)

        self.discriminator = Classifier(in_feature=channels, n_class=2)

        self.opt = Optimizer([self.backbone_target, self.classifier, self.mhsa], training_params)
        self.opt_discriminator = Optimizer([self.discriminator], training_params)

    def sync_backbones(self):
        self.backbone_target.load_state_dict(self.backbone.state_dict())

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        
        # train discriminator
        self.opt_discriminator.z_grad()

        cfp_feature_map = self.backbone.forward(cfp)
        clarus_whole_feature_map = self.backbone_target.forward(clarus_whole)

        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_mil_pooling(clarus_whole_feature_map).flatten(2).transpose(1, 2)
        clarus_whole_feature = self.mhsa(clarus_whole_feature)[:, 0]

        features_concat = torch.cat([cfp_feature, clarus_whole_feature], dim=0)
        score_discriminator = self.discriminator(features_concat.detach())

        gt_cfp_domian = torch.ones(cfp_feature.size(0)).long().cuda()
        gt_clarus_whole_domian = torch.zeros(clarus_whole_feature.size(0)).long().cuda()
        
        gt_discriminator = torch.cat((gt_cfp_domian, gt_clarus_whole_domian), dim=0)
        loss_discriminator = self.crit_discriminator(score_discriminator, gt_discriminator) * self.weights[2]

        loss_discriminator.backward()
        self.opt_discriminator.g_step()
        self.opt_discriminator.update_lr()

        # train backbone and classifier
        self.opt.z_grad()
        self.opt_discriminator.z_grad()

        clarus_whole_feature_map = self.backbone_target.forward(clarus_whole)
        clarus_whole_feature = self.sw_mil_pooling(clarus_whole_feature_map).flatten(2).transpose(1, 2)
        clarus_whole_feature = self.mhsa(clarus_whole_feature)[:, 0]
        clarus_whole_score = self.classifier(clarus_whole_feature)

        score_discriminator_clarus_whole = self.discriminator(clarus_whole_feature)
        gt_clarus_whole_domian = torch.ones(clarus_whole_score.size(0)).long().cuda()

        loss_adv = self.crit_discriminator(score_discriminator_clarus_whole, gt_clarus_whole_domian) * self.weights[1]

        if gt_clarus is not None:
            loss_cls = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[0]
            loss = loss_adv + loss_cls
        else:
            loss_cls = None
            loss = loss_adv

        loss.backward()
        self.opt.g_step()
        self.opt.z_grad()
        self.opt.update_lr()
        
        return clarus_whole_score, loss, [loss_adv, loss_cls, loss_discriminator], []

    def predict_result(self, clarus_whole, clarus_split):
        clarus_whole_feature_map = self.backbone_target.forward(clarus_whole)
        clarus_whole_feature = self.sw_mil_pooling(clarus_whole_feature_map).flatten(2).transpose(1, 2)
        clarus_whole_feature = self.mhsa(clarus_whole_feature)[:, 0]
        clarus_whole_score = self.classifier(clarus_whole_feature)
        return clarus_whole_score


