# feature mixup

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .optimizer import Optimizer
from .MLP import MLP, GELU
import math

from .FOVNet import SelfAttentionBlocks, ChannelAttention, MyAvgPool2d


def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis)  # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)


class ProFeaMix(nn.Module):
    def __init__(self, backbone, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi', use_mean=False):
        super(ProFeaMix, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)

        self.mil_ratio = mil_ratio
        self.use_mean = use_mean

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        elif self.score_fusion == 'multi':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        else:
            raise Exception('invalid', self.score_fusion)

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_whole = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        
    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)

    def forward(self, x):
        
        x = self.backbone(x)
        x = self.sw_mil_pooling(x).flatten(2).transpose(1, 2)
        feature = self.mhsa(x)
        if self.use_mean:
            feature = torch.mean(feature, dim=1)
        else:
            feature = feature[:, 0]
        x = self.classifier(feature)
        return x, feature
    
    def forward_whole(self, x):
        x = self.backbone(x)
        feature = self.sw_gap(x).view(x.size(0), -1)
        x = self.classifier_whole(feature)
        return x, feature
    
    def draw_heat(self, x):

        B = x.size(0)
        score_whole, feature_whole = self.forward_whole(x)

        # score_vit, feature_vit = self.forward(x)
        feature_vit = self.backbone.forward(x)
        feature_vit = self.sw_mil_pooling(feature_vit).flatten(2).transpose(1, 2)
        feature_vit, attns = self.mhsa(feature_vit, heat=True)
        if self.use_mean:
            feature_vit = torch.mean(feature_vit, dim=1)
        else:
            feature_vit = feature_vit[:, 0]
        score_vit = self.classifier(feature_vit)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), feature_vit.view(B, 1, -1)), dim=1))
        score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            score = torch.bmm(weight_fusion, score) # B * n_class * 2 , B * 2 * n_class
            score = score.view(B, -1)
        else:
            score = score * torch.transpose(weight_fusion, 2, 1)
            score = torch.sum(score, dim=1)

        return score, score_vit, score_whole, weight_fusion, attns


class ProFeaMixTraining(ProFeaMix):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi', use_mean=False):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion, use_mean=use_mean)

        self.mix_ratio = mix_ratio  # (0.5, 1)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_whole, self.mhsa, self.cw_att_fusion], training_params)
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，原始损失，mix up损失

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        if cfp.size(0) != clarus_whole.size(0):
            if cfp.size(0) > clarus_whole.size(0):
                cfp = cfp[:clarus_whole.size(0)]
                gt_cfp = gt_cfp[:clarus_whole.size(0)]
            else:
                repeat_num = int(math.ceil(clarus_whole.size(0) / cfp.size(0)))
                cfp = cfp.repeat((repeat_num, 1, 1, 1))[:clarus_whole.size(0)]
                gt_cfp = gt_cfp.repeat((repeat_num, 1))[:clarus_whole.size(0)]

        self.opt.z_grad()

        score_cfp, feature_cfp = self.forward_whole(cfp)

        score_clarus_vit, feature_clarus_vit = self.forward(clarus_whole)
        score_clarus_whole, feature_clarus_whole = self.forward_whole(clarus_whole)
            
        B = clarus_whole.size(0)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_clarus_whole.view(B, 1, -1), feature_clarus_vit.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_clarus_whole.view(B, 1, -1), score_clarus_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        loss_merged_score = self.crit_sup(clarus_score, gt_clarus)

        mix_feature = (1 - self.mix_ratio) * feature_cfp + self.mix_ratio * feature_clarus_vit
        mix_score = self.classifier(mix_feature)
        loss_mix = (1 - self.mix_ratio) * self.crit_sup(mix_score, gt_cfp) + self.mix_ratio * self.crit_sup(mix_score, gt_clarus)
       
        loss = loss_merged_score * self.weights[0] + loss_mix * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_score, loss, [], []

    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        score_whole, feature_whole = self.forward_whole(clarus_whole)

        score_vit, feature_vit = self.forward(clarus_whole)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), feature_vit.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        return clarus_score
    

class PreFeaMix(ProFeaMix):
    def __init__(self, backbone, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi'):
        super(PreFeaMix, self).__init__(backbone=backbone, n_class=n_class, channels=channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion, use_mean=True)

    def forward(self, x):
        
        x = self.backbone(x)
        x = self.sw_mil_pooling(x).flatten(2).transpose(1, 2)
        feature = self.mhsa(x)

        x = self.classifier(torch.mean(feature, dim=1))
        return x, feature


class PreFeaMixTraining(PreFeaMix):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, mix_ins_ratio, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi'):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion)

        self.mix_ratio = mix_ratio  # (0.5, 1)
        self.mix_ins_ratio = mix_ins_ratio  # (0, 1)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_whole, self.mhsa, self.cw_att_fusion], training_params)
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，原始损失，mix up损失

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        if cfp.size(0) != clarus_whole.size(0):
            if cfp.size(0) > clarus_whole.size(0):
                cfp = cfp[:clarus_whole.size(0)]
                gt_cfp = gt_cfp[:clarus_whole.size(0)]
            else:
                repeat_num = int(math.ceil(clarus_whole.size(0) / cfp.size(0)))
                cfp = cfp.repeat((repeat_num, 1, 1, 1))[:clarus_whole.size(0)]
                gt_cfp = gt_cfp.repeat((repeat_num, 1))[:clarus_whole.size(0)]
                
        self.opt.z_grad()

        score_cfp, feature_cfp = self.forward_whole(cfp)

        score_clarus_vit, feature_clarus_vit = self.forward(clarus_whole)
        score_clarus_whole, feature_clarus_whole = self.forward_whole(clarus_whole)
            
        B = clarus_whole.size(0)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_clarus_whole.view(B, 1, -1), torch.mean(feature_clarus_vit, dim=1).view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_clarus_whole.view(B, 1, -1), score_clarus_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        loss_merged_score = self.crit_sup(clarus_score, gt_clarus)

        I = feature_clarus_vit.size(1)
        mix_instance = max(int(I * self.mix_ins_ratio), 1)

        '''mask = torch.cat([torch.ones(mix_instance), torch.zeros(I - mix_instance)], dim=0) * (1 - self.mix_ratio)
        mask = mask.view(1, -1, 1).repeat(B, 1, 1)
        mask = shufflerow(mask, 1)
        mask = mask.to(feature_cfp.device)
        feature_cfp = feature_cfp.view(B, 1, -1).repeat(1, I, 1)

        mix_feature = mask * feature_cfp + (1 - mask) * feature_clarus_vit
        mix_feature = torch.mean(mix_feature, dim=1)'''
        feature_clarus_vit = torch.mean(feature_clarus_vit, dim=1)
        mix_feature = feature_clarus_vit * self.mix_ratio + feature_cfp * (1 - self.mix_ratio)
        mix_score = self.classifier(mix_feature)

        loss_mix = (1 - self.mix_ratio) * self.crit_sup(mix_score, gt_cfp) + self.mix_ratio * self.crit_sup(mix_score, gt_clarus)
       
        loss = loss_merged_score * self.weights[0] + loss_mix * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_score, loss, [], []

    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        score_whole, feature_whole = self.forward_whole(clarus_whole)

        score_vit, feature_vit = self.forward(clarus_whole)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), torch.mean(feature_vit, dim=1).view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        return clarus_score
    

class PrePreFeaMix(ProFeaMix):
    def __init__(self, backbone, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi'):
        super(PrePreFeaMix, self).__init__(backbone=backbone, n_class=n_class, channels=channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion, use_mean=True)

    def forward(self, x):
        
        x = self.backbone(x)
        feature = self.sw_mil_pooling(x).flatten(2).transpose(1, 2)
        x = self.mhsa(feature)

        x = self.classifier(torch.mean(x, dim=1))
        return x, feature


class PrePreFeaMixTraining(PrePreFeaMix):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, mix_ins_ratio, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi'):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion)

        self.mix_ratio = mix_ratio  # (0.5, 1)
        self.mix_ins_ratio = mix_ins_ratio  # (0, 1)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_whole, self.mhsa, self.cw_att_fusion], training_params)
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，原始损失，mix up损失

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):

        # 统一cfp和uwf的batch size用于后续的mix up
        if cfp.size(0) != clarus_whole.size(0):
            if cfp.size(0) > clarus_whole.size(0):
                cfp = cfp[:clarus_whole.size(0)]
                gt_cfp = gt_cfp[:clarus_whole.size(0)]
            else:
                repeat_num = int(math.ceil(clarus_whole.size(0) / cfp.size(0)))
                cfp = cfp.repeat((repeat_num, 1, 1, 1))[:clarus_whole.size(0)]
                gt_cfp = gt_cfp.repeat((repeat_num, 1))[:clarus_whole.size(0)]
                
        self.opt.z_grad()

        score_cfp, feature_cfp = self.forward_whole(cfp)

        score_clarus_vit, feature_clarus_vit = self.forward(clarus_whole)
        score_clarus_whole, feature_clarus_whole = self.forward_whole(clarus_whole)
            
        B = clarus_whole.size(0)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_clarus_whole.view(B, 1, -1), torch.mean(feature_clarus_vit, dim=1).view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_clarus_whole.view(B, 1, -1), score_clarus_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        loss_merged_score = self.crit_sup(clarus_score, gt_clarus)

        I = feature_clarus_vit.size(1)
        mix_instance = max(int(I * self.mix_ins_ratio), 1)

        mask = torch.cat([torch.ones(mix_instance), torch.zeros(I - mix_instance)], dim=0) * (1 - self.mix_ratio)
        mask = mask.view(1, -1, 1).repeat(B, 1, 1)
        mask = shufflerow(mask, 1)
        mask = mask.to(feature_cfp.device)
        feature_cfp = feature_cfp.view(B, 1, -1).repeat(1, I, 1)

        mix_feature = mask * feature_cfp + (1 - mask) * feature_clarus_vit
        mix_feature = self.mhsa(mix_feature)

        mix_feature = torch.mean(mix_feature, dim=1)
        mix_score = self.classifier(mix_feature)

        loss_mix = (1 - self.mix_ratio) * self.crit_sup(mix_score, gt_cfp) + self.mix_ratio * self.crit_sup(mix_score, gt_clarus)
       
        loss = loss_merged_score * self.weights[0] + loss_mix * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_score, loss, [], []

    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        score_whole, feature_whole = self.forward_whole(clarus_whole)

        score_vit, feature_vit = self.forward(clarus_whole)
        feature_vit = self.mhsa(feature_vit)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), torch.mean(feature_vit, dim=1).view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        return clarus_score
    

class PrePreFeaPatchMixTraining(PrePreFeaMix):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi'):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion)

        # self.mix_ratio = mix_ratio  # (0.5, 1)
        # self.mix_ins_ratio = mix_ins_ratio  # (0, 1)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_whole, self.mhsa, self.cw_att_fusion], training_params)
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，原始损失，mix up损失

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):

        # 统一cfp和uwf的batch size用于后续的mix up
        if cfp.size(0) != clarus_whole.size(0):
            if cfp.size(0) > clarus_whole.size(0):
                cfp = cfp[:clarus_whole.size(0)]
                gt_cfp = gt_cfp[:clarus_whole.size(0)]
            else:
                repeat_num = int(math.ceil(clarus_whole.size(0) / cfp.size(0)))
                cfp = cfp.repeat((repeat_num, 1, 1, 1))[:clarus_whole.size(0)]
                gt_cfp = gt_cfp.repeat((repeat_num, 1))[:clarus_whole.size(0)]
                
        self.opt.z_grad()

        score_cfp, feature_cfp = self.forward_whole(cfp)

        score_clarus_vit, feature_clarus_vit = self.forward(clarus_whole)
        score_clarus_whole, feature_clarus_whole = self.forward_whole(clarus_whole)
        
        B = clarus_whole.size(0)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_clarus_whole.view(B, 1, -1), torch.mean(feature_clarus_vit, dim=1).view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_clarus_whole.view(B, 1, -1), score_clarus_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        loss_merged_score = self.crit_sup(clarus_score, gt_clarus)

        '''I = feature_clarus_vit.size(1)
        mix_instance = max(int(I * self.mix_ins_ratio), 1)

        mask = torch.cat([torch.ones(mix_instance), torch.zeros(I - mix_instance)], dim=0) * (1 - self.mix_ratio)
        mask = mask.view(1, -1, 1).repeat(B, 1, 1)
        mask = shufflerow(mask, 1)
        mask = mask.to(feature_cfp.device)
        feature_cfp = feature_cfp.view(B, 1, -1).repeat(1, I, 1)

        mix_feature = mask * feature_cfp + (1 - mask) * feature_clarus_vit'''
        mix_feature = torch.cat((feature_cfp.view(B, 1, -1), feature_clarus_vit), dim=1)
        mix_feature = self.mhsa(mix_feature)

        mix_feature = torch.mean(mix_feature, dim=1)
        mix_score = self.classifier(mix_feature)

        
        gt_mix = torch.cat((gt_cfp.view(B, 1, -1), gt_clarus.view(B, 1, -1)), dim=1)
        gt_mix = torch.max(gt_mix, dim=1)[0]
        
        loss_mix = self.crit_sup(mix_score, gt_mix)
       
        loss = loss_merged_score * self.weights[0] + loss_mix * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_score, loss, [], []

    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        score_whole, feature_whole = self.forward_whole(clarus_whole)

        score_vit, feature_vit = self.forward(clarus_whole)
        feature_vit = self.mhsa(feature_vit)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), torch.mean(feature_vit, dim=1).view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        return clarus_score
    




