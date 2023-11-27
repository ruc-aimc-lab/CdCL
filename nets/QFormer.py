# feature mixup

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .optimizer import Optimizer
from .MLP import MLP, GELU
import math

from .FOVNet import MyAvgPool2d, ResMLP, ChannelAttention


class Att(nn.Module):
    def __init__(self, embed_dim, dim_kv, num_heads=8, drop_rate=0.2) -> None:
        super().__init__()
        self.self_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop_rate, batch_first=True)
        self.cross_att = nn.MultiheadAttention(embed_dim=embed_dim, kdim=dim_kv, vdim=dim_kv, num_heads=num_heads, dropout=drop_rate, batch_first=True)
        self.mlp = ResMLP(in_features=embed_dim, hidden_features=embed_dim*4, out_features=embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, q, kv):
        q = self.norm1(self.drop(self.self_att(q, q, q)[0]) + q)
        q = self.norm2(self.drop(self.cross_att(q, kv, kv)[0]) + q)
        q = self.mlp(q)
        return q


class AttBlocks(nn.Module):
    def __init__(self, embed_dim, dim_kv, block_num, num_heads=8, drop_rate=0.2):
        super().__init__()
        self.block_num = block_num
        assert self.block_num >= 1

        self.blocks = nn.ModuleList([Att(embed_dim=embed_dim, dim_kv=dim_kv, num_heads=num_heads, drop_rate=drop_rate)
                                     for i in range(self.block_num)])

    def forward(self, q, kv):
        for blk in self.blocks:
            q = blk(q, kv)
        return q


class QFormer(nn.Module):
    def __init__(self, backbone, n_class, channels, num_q, dim_q, att_nums=0, mil_ratio=3):
        super(QFormer, self).__init__()
        self.backbone = backbone

        if att_nums >= 1:
            self.att = AttBlocks(embed_dim=dim_q, dim_kv=channels, block_num=att_nums, num_heads=8, drop_rate=0.2)
        else:
            self.att = nn.Identity()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_q, dim_q))
        self.query_tokens.data.normal_(mean=0.0, std=0.02)

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=True)

        self.mil_ratio = mil_ratio

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
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        feature = self.att(query_tokens, x)
        feature = torch.mean(feature, dim=1)
        x = self.classifier(feature)
        return x, feature
    
    def forward_whole(self, x):
        x = self.backbone(x)
        feature = self.sw_gap(x).view(x.size(0), -1)
        x = self.classifier_whole(feature)
        return x, feature


class QFormerTraining(QFormer):
    def __init__(self, backbone, n_class, channels, num_q, dim_q, att_nums, mil_ratio, crit_sup, weights, training_params):
        super().__init__(backbone=backbone, n_class=n_class, channels=channels, num_q=num_q, dim_q=dim_q, att_nums=att_nums, mil_ratio=mil_ratio)

        self.opt = Optimizer([self.backbone, self.att, self.query_tokens, self.classifier, self.classifier_whole], training_params)
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):

        self.opt.z_grad()

        cfp_feature = self.backbone.forward(cfp)
        cfp_feature = self.sw_gap(cfp_feature).view(cfp_feature.size(0), -1)
        cfp_score = self.classifier(cfp_feature)
        # cfp_score_whole = self.classifier_whole(cfp_feature)
        loss_cfp = self.crit_sup(cfp_score, gt_cfp)

        score_clarus, feature_clarus = self.forward(clarus_whole)
        score_clarus_whole, feature_clarus_whole = self.forward_whole(clarus_whole)
        B = clarus_whole.size(0)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_clarus_whole.view(B, 1, -1), feature_clarus.view(B, 1, -1)), dim=1))
        score = torch.cat((score_clarus_whole.view(B, 1, -1), score_clarus.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            score = torch.bmm(weight_fusion, score) # B * n_class * 2 , B * 2 * n_class
            score = score.view(B, -1)
        else:
            score = score * torch.transpose(weight_fusion, 2, 1)
            score = torch.sum(score, dim=1)

        loss_clarus = self.crit_sup(score, gt_clarus)

        loss = loss_cfp * self.weights[0] + loss_clarus * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return score, loss, [], []

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
    


class VITMixupWholeQFormer(nn.Module):
    def __init__(self, backbone, n_class, channels, num_q, dim_q, att_nums=0, mil_ratio=3, score_fusion='multi'):
        super(VITMixupWholeQFormer, self).__init__()
        self.backbone = backbone

        if att_nums >= 1:
            self.att = AttBlocks(embed_dim=dim_q, dim_kv=channels, block_num=att_nums, num_heads=8, drop_rate=0.2)
        else:
            self.att = nn.Identity()
            
        self.query_tokens_source = nn.Parameter(torch.zeros(1, num_q, dim_q))
        self.query_tokens_source.data.normal_(mean=0.0, std=0.02)

        self.query_tokens_target = nn.Parameter(torch.zeros(1, num_q, dim_q))
        self.query_tokens_target.data.normal_(mean=0.0, std=0.02)

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=True)

        self.mil_ratio = mil_ratio

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=dim_q, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        elif self.score_fusion == 'multi':
            self.cw_att_fusion = ChannelAttention(channels=dim_q, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        else:
            raise Exception('invalid', self.score_fusion)
        self.fc_whole = nn.Linear(channels, dim_q)

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier = MLP(in_features=dim_q, hidden_features=128, out_features=n_class)
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

    def forward(self, x, source):
        x = self.backbone(x)
        x = self.sw_mil_pooling(x).flatten(2).transpose(1, 2)
        if source:
            query_tokens = self.query_tokens_source.expand(x.shape[0], -1, -1)
        else:
            query_tokens = self.query_tokens_target.expand(x.shape[0], -1, -1)
        feature = self.att(query_tokens, x)
        feature = torch.mean(feature, dim=1)
        x = self.classifier(feature)
        return x, feature
    
    def forward_whole(self, x):
        x = self.backbone(x)
        feature = self.sw_gap(x).view(x.size(0), -1)
        x = self.classifier_whole(feature)
        feature = self.fc_whole(feature)

        return x, feature
    
    def draw_heat(self, x):

        B = x.size(0)
        score_whole, feature_whole = self.forward_whole(x)

        # score_vit, feature_vit = self.forward(x)
        feature_vit = self.backbone.forward(x)
        feature_vit = self.sw_mil_pooling(feature_vit).flatten(2).transpose(1, 2)
        feature_vit, attns = self.mhsa(feature_vit, heat=True)
        feature_vit = torch.mean(feature_vit, dim=1)
        
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

    @staticmethod
    def pad_cfp(cfp, clarus_whole):
        # 超广角图像可能不是正方形的，所以把cfp padding以下，以便mix up
        min_n = min(cfp.size(0), clarus_whole.size(0))
        cfp = cfp[:min_n]
        clarus_whole = clarus_whole[:min_n]

        w_cfp = cfp.size(3)
        w_clarus_whole = clarus_whole.size(3)
        dw = int((w_clarus_whole - w_cfp) / 2)
        if w_clarus_whole != w_cfp:
            new_cfp = torch.zeros_like(clarus_whole)
            new_cfp[:, :, :, dw:dw+w_cfp] = cfp
        else:
            new_cfp = cfp
        return new_cfp


class VITMixupWholeQFormerTraining(VITMixupWholeQFormer):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, training_params, num_q, dim_q, att_nums, mil_ratio=3, score_fusion='multi'):
        super().__init__(backbone=backbone, n_class=n_class, channels=channels, num_q=num_q, dim_q=dim_q, att_nums=att_nums, mil_ratio=mil_ratio, score_fusion=score_fusion)

        self.mix_ratio = mix_ratio  # (0.5, 1)

        self.opt = Optimizer([self.backbone, self.att, self.query_tokens_source, self.query_tokens_target, self.classifier, self.classifier_whole, self.cw_att_fusion], training_params)
        self.crit_mixup = nn.BCEWithLogitsLoss()
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

        new_cfp = self.pad_cfp(cfp=cfp, clarus_whole=clarus_whole)
        mix_clarus = (1 - self.mix_ratio) * new_cfp + self.mix_ratio * clarus_whole

        score_mix_vit, mix_feature_vit = self.forward(mix_clarus, True)
        score_mix_whole, mix_feature_whole = self.forward_whole(mix_clarus)

        score_vit, feature_vit = self.forward(clarus_whole, False)
        score_whole, feature_whole = self.forward_whole(clarus_whole)
    
        B = clarus_whole.size(0)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), feature_vit.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        
        loss_merged_score = self.crit_sup(clarus_score, gt_clarus)

        mix_weight_fusion = self.cw_att_fusion(torch.cat((mix_feature_whole.view(B, 1, -1), mix_feature_vit.view(B, 1, -1)), dim=1))
        mix_clarus_score = torch.cat((score_mix_whole.view(B, 1, -1), score_mix_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            mix_clarus_score = torch.bmm(mix_weight_fusion, mix_clarus_score) # B * n_class * 2 , B * 2 * n_class
            mix_clarus_score = mix_clarus_score.view(B, -1)
        else:
            mix_clarus_score = mix_clarus_score * torch.transpose(mix_weight_fusion, 2, 1)
            mix_clarus_score = torch.sum(mix_clarus_score, dim=1)
        loss_mix_merged_score = self.crit_sup(mix_clarus_score, gt_clarus) * self.mix_ratio + self.crit_sup(mix_clarus_score, gt_cfp) * (1 - self.mix_ratio)


        loss = loss_merged_score * self.weights[0] + loss_mix_merged_score * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_score, loss, [], []

    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        score_whole, feature_whole = self.forward_whole(clarus_whole)

        score_vit, feature_vit = self.forward(clarus_whole, False)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), feature_vit.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        return clarus_score