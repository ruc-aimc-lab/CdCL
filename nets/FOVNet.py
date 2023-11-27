import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .optimizer import Optimizer
from .MLP import MLP, GELU, MLPGRL

from .crit import MMDLinear
from copy import deepcopy
import math


class Linear(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Linear, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(channel_in, channel_out)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction, act=nn.Sigmoid(), out_channel=1):
        # 在通常的mil中out_channel=1，把out_channel设置成score的维度可以对每个类别分别加权
        super(ChannelAttention, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(channels // reduction, out_channel)
        self.act = act

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        weight = self.act(x)  # B * I * out_channel
        weight = torch.transpose(weight, 2, 1) # B * out_channel * I
        return weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, drop_rate=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, heat=False):
        B, N, C = x.shape
        out = self.norm(x)
        qkv = self.qkv(out).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.drop(out)
        out = x + out
        if heat:
            return out, attn
        return out


class ResMLP(MLP):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.2):
        super().__init__(in_features=in_features, hidden_features=hidden_features, out_features=out_features, act_layer=act_layer, drop=drop)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = out + x
        return out


class MHSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, drop_rate=0.2) -> None:
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
        self.mlp = ResMLP(in_features=dim, hidden_features=dim*4, out_features=dim)

    def forward(self, x, heat=False):
        
        if heat:
            x, attn = self.mhsa(x, heat=True)
        else:
            x = self.mhsa(x)
        x = self.mlp(x)
        if heat:
            return x, attn
        return x


class SelfAttentionBlocks(nn.Module):
    def __init__(self, dim, block_num, num_heads=8, drop_rate=0.2):
        super().__init__()
        self.block_num = block_num
        assert self.block_num >= 1

        self.blocks = nn.ModuleList([MHSABlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
                                     for i in range(self.block_num)])

    def forward(self, x, heat=False):
        attns = []
        for blk in self.blocks:
            if heat:
                x, attn = blk(x, heat=True)
                attns.append(attn)
            else:
                x = blk(x)
        if heat:
            return x, attns
        return x


class MyAvgPool2d(nn.Module):
    def __init__(self, ratio, over_lap=True) -> None:
        super().__init__()
        self.ratio = ratio
        self.over_lap = over_lap

    def forward(self, x):
        B, C, H, W = x.size()
        ratio_h = self.ratio
        # ratio_w = W/H * ratio_h
        ratio_w = self.ratio
            
        kernal_size_h = int(H / ratio_h + 0.5)
        kernal_size_w = int(W / ratio_w + 0.5)

        if self.over_lap:
            stride_h = int(kernal_size_h / 2 + 0.5)
            stride_w = int(kernal_size_w / 2 + 0.5)
        else:
            stride_h = kernal_size_h
            stride_w = kernal_size_w

        padding_h = int((stride_h - (H - kernal_size_h) % stride_h) % stride_h / 2 + 0.5)
        padding_w = int((stride_w - (W - kernal_size_w) % stride_w) % stride_w / 2 + 0.5)

        x = F.avg_pool2d(input=x, kernel_size=(kernal_size_h, kernal_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w))
        return x
        

class FOVNet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, score_fusion='single', mil_ratio=3, over_lap=True):
        super(FOVNet, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.cw_att_mil = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        elif self.score_fusion == 'multi':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        else:
            raise Exception('invalid', self.score_fusion)

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)
        if not over_lap:
            print('no over lap')

        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_mil = MLP(in_features=channels, hidden_features=128, out_features=n_class)

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
        cfp_feature_map = self.backbone.forward(cfp)
        clarus_feature_map = self.backbone.forward(clarus_whole)
        
        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        weight_mil = self.cw_att_mil(clarus_mil_feature)

        B, I, C = clarus_mil_feature.size()
        clarus_mil_feature = torch.bmm(weight_mil, clarus_mil_feature) # B * 1 * C
        clarus_mil_feature = clarus_mil_feature.view(B, C)

        cfp_score = self.classifier(cfp_feature)
        cfp_score_mil = self.classifier_mil(cfp_feature)

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        if need_feature:
            return cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score, [cfp_feature, clarus_whole_feature, clarus_mil_feature]
        else:
            return cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score
    
    def predict_result(self, clarus_whole, clarus_split):
        clarus_feature_map = self.backbone.forward(clarus_whole)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        weight_mil = self.cw_att_mil(clarus_mil_feature)

        B, I, C = clarus_mil_feature.size()
        clarus_mil_feature = torch.bmm(weight_mil, clarus_mil_feature) # B * 1 * C
        clarus_mil_feature = clarus_mil_feature.view(B, C)

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        return clarus_score


class FOVNetTraining(FOVNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, score_fusion, mil_ratio, training_params, over_lap=True):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, score_fusion=score_fusion, mil_ratio=mil_ratio, over_lap=over_lap)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp在全图分支和mil分支监督损失，uwf监督损失

        self.opt = Optimizer([self.backbone, self.mhsa, self.cw_att_mil, self.cw_att_fusion, self.classifier, self.classifier_mil], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, cfp_score_mil, _, _, clarus_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        loss_cfp_mil = self.crit_sup(cfp_score_mil, gt_cfp) * self.weights[1]
        loss = loss_cfp + loss_cfp_mil

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[2]
            loss += loss_claurs
        else:
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_score, loss, [loss_cfp, loss_cfp_mil, loss_claurs], [cfp_score, cfp_score_mil]


class FOVNetTraining_UWFAllLoss(FOVNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, score_fusion, mil_ratio, training_params):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, score_fusion=score_fusion, mil_ratio=mil_ratio)
        # UWF全图和MIL分别都要计算loss

        self.crit_sup = crit_sup
        self.weights = weights # 长度为5，分别是cfp在全图分支和mil分支监督损失，uwf监督损失，uwf在全图分支和mil分支的监督损失

        self.opt = Optimizer([self.backbone, self.mhsa, self.cw_att_mil, self.cw_att_fusion, self.classifier, self.classifier_mil], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        loss_cfp_mil = self.crit_sup(cfp_score_mil, gt_cfp) * self.weights[1]
        loss = loss_cfp + loss_cfp_mil

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[2]
            loss_claurs_whole = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[3]
            loss_claurs_mil = self.crit_sup(clarus_mil_score, gt_clarus) * self.weights[4]
            loss += loss_claurs + loss_claurs_whole + loss_claurs_mil
        else:
            print('no clarus gt')
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_score, loss, [loss_cfp, loss_cfp_mil, loss_claurs], [cfp_score, cfp_score_mil]


class FOVNetTraining_CFPOnlyWhole(FOVNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, score_fusion, mil_ratio, training_params):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, score_fusion=score_fusion, mil_ratio=mil_ratio)
        # 仅计算cfp在全图分支的损失

        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，分别是cfp在全图分支监督损失，uwf监督损失

        self.opt = Optimizer([self.backbone, self.mhsa, self.cw_att_mil, self.cw_att_fusion, self.classifier, self.classifier_mil], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, cfp_score_mil, _, _, clarus_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        # loss_cfp_mil = self.crit_sup(cfp_score_mil, gt_cfp) * self.weights[1]
        loss = loss_cfp

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[1]
            loss += loss_claurs
        else:
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_score, loss, [loss_cfp, loss_claurs], [cfp_score, cfp_score_mil]


class FOVNetDDCTraining(FOVNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, score_fusion, mil_ratio, training_params):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, score_fusion=score_fusion, mil_ratio=mil_ratio)

        self.crit_sup = crit_sup
        self.crit_ddc = MMDLinear()
        self.weights = weights # 长度为5，

        params_backbone = deepcopy(training_params)
        params_backbone['lr'] /= 10
        params_backbone['schedule_params']['eta_min'] /= 10
        self.opt = Optimizer([self.backbone], params_backbone)

        params_classifier = deepcopy(training_params)    
        self.opt_classifier = Optimizer([self.mhsa, self.cw_att_mil, self.cw_att_fusion, self.classifier, self.classifier_mil], params_classifier)     

        assert params_classifier != params_backbone   

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        self.opt_classifier.z_grad()
        
        cfp_score, cfp_score_mil, _, _, clarus_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 

        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        loss_cfp_mil = self.crit_sup(cfp_score_mil, gt_cfp) * self.weights[1]
        loss = loss_cfp + loss_cfp_mil

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[2]
            loss += loss_claurs
        else:
            loss_claurs = None
        
        cfp_feature, clarus_whole_feature, clarus_mil_feature = features
        loss_ddc_whole = self.weights[3] * self.crit_ddc(clarus_whole_feature, cfp_feature)
        # loss_ddc_mil = self.weights[4] * self.crit_ddc(clarus_mil_feature, cfp_feature)
        loss += loss_ddc_whole


        loss.backward()

        self.opt.g_step()
        self.opt_classifier.g_step()

        self.opt.z_grad()
        self.opt_classifier.z_grad()

        self.opt.update_lr()
        self.opt_classifier.update_lr()
        
        return clarus_score, loss, [loss_cfp, loss_cfp_mil, loss_claurs], [cfp_score, cfp_score_mil]


class MBFOVNet(nn.Module):
    # multi backbone
    def __init__(self, backbone:nn.Module, backbone_mil:nn.Module, n_class, channels, mhsa_nums=0, score_fusion='single', mil_ratio=3):
        super(MBFOVNet, self).__init__()
        self.backbone = backbone
        self.backbone_mil = backbone_mil

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.cw_att_mil = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        elif self.score_fusion == 'multi':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        else:
            raise Exception('invalid', self.score_fusion)

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio)

        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_mil = MLP(in_features=channels, hidden_features=128, out_features=n_class)

    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)

    def sync_backbones(self):
        self.backbone_mil.load_state_dict(self.backbone.state_dict())

    def forward(self, cfp, clarus_whole, clarus_split, need_feature=False):
        cfp_feature_map = self.backbone.forward(cfp)
        clarus_feature_map = self.backbone.forward(clarus_whole)
        
        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_mil_feature_map = self.backbone_mil.forward(clarus_whole)
        clarus_mil_feature = self.sw_mil_pooling(clarus_mil_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        weight_mil = self.cw_att_mil(clarus_mil_feature)

        B, I, C = clarus_mil_feature.size()
        clarus_mil_feature = torch.bmm(weight_mil, clarus_mil_feature) # B * 1 * C
        clarus_mil_feature = clarus_mil_feature.view(B, C)

        cfp_score = self.classifier(cfp_feature)
        cfp_score_mil = self.classifier_mil(cfp_feature)

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        if need_feature:
            return cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score, [cfp_feature, clarus_whole_feature, clarus_mil_feature]
        else:
            return cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score
    
    def predict_result(self, clarus_whole, clarus_split):
        clarus_feature_map = self.backbone.forward(clarus_whole)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_mil_feature_map = self.backbone_mil.forward(clarus_whole)
        clarus_mil_feature = self.sw_mil_pooling(clarus_mil_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        weight_mil = self.cw_att_mil(clarus_mil_feature)

        B, I, C = clarus_mil_feature.size()
        clarus_mil_feature = torch.bmm(weight_mil, clarus_mil_feature) # B * 1 * C
        clarus_mil_feature = clarus_mil_feature.view(B, C)

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        return clarus_score


class MBFOVNetTraining(MBFOVNet):
    def __init__(self, backbone, backbone_mil, n_class, channels, crit_sup, weights, mhsa_nums, score_fusion, mil_ratio, training_params):
        super().__init__(backbone, backbone_mil, n_class, channels, mhsa_nums=mhsa_nums, score_fusion=score_fusion, mil_ratio=mil_ratio)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp在全图分支和mil分支监督损失，uwf监督损失

        self.opt = Optimizer([self.backbone, self.backbone_mil, self.mhsa, self.cw_att_mil, self.cw_att_fusion, self.classifier, self.classifier_mil], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, cfp_score_mil, _, _, clarus_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        loss_cfp_mil = self.crit_sup(cfp_score_mil, gt_cfp) * self.weights[1]
        loss = loss_cfp + loss_cfp_mil

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[2]
            loss += loss_claurs
        else:
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_score, loss, [loss_cfp, loss_cfp_mil, loss_claurs], [cfp_score, cfp_score_mil]


class MILFOVNet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True):
        super(MILFOVNet, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.cw_att_mil = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())

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
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        weight_mil = self.cw_att_mil(clarus_mil_feature)

        B, I, C = clarus_mil_feature.size()
        clarus_mil_feature = torch.bmm(weight_mil, clarus_mil_feature) # B * 1 * C
        clarus_mil_feature = clarus_mil_feature.view(B, C)

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
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        weight_mil = self.cw_att_mil(clarus_mil_feature)

        B, I, C = clarus_mil_feature.size()
        clarus_mil_feature = torch.bmm(weight_mil, clarus_mil_feature) # B * 1 * C
        clarus_mil_feature = clarus_mil_feature.view(B, C)

        clarus_mil_score = self.classifier(clarus_mil_feature)
        
        return clarus_mil_score


class MILFOVNetTraining(MILFOVNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, mil_ratio, training_params, over_lap=True):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，分别是cfp监督损失，uwf监督损失

        self.opt = Optimizer([self.backbone, self.mhsa, self.cw_att_mil, self.classifier], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, clarus_mil_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_mil_score, gt_clarus) * self.weights[1]
            loss += loss_claurs
        else:
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_mil_score, loss, [loss_claurs], [cfp_score]


class VITFOVNet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, use_mean=False):
        super(VITFOVNet, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)

        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)

        self.mil_ratio = mil_ratio
        self.use_mean = use_mean
        
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
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        if self.use_mean:
            clarus_mil_feature = torch.mean(clarus_mil_feature, dim=1)
        else:
            clarus_mil_feature = clarus_mil_feature[:, 0]

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
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        if self.use_mean:
            clarus_mil_feature = torch.mean(clarus_mil_feature, dim=1)
        else:
            clarus_mil_feature = clarus_mil_feature[:, 0]
        
        clarus_mil_score = self.classifier(clarus_mil_feature)
        
        return clarus_mil_score

    def draw_heat(self, x):
        x = self.backbone.forward(x)
        x = self.sw_mil_pooling(x).flatten(2).transpose(1, 2)
        x, attns = self.mhsa(x, heat=True)
        if self.use_mean:
            x = torch.mean(x, dim=1)
        else:
            x = x[:, 0]
        
        x = self.classifier(x)
        
        return x, attns


class VITFOVNetTraining(VITFOVNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, mil_ratio, training_params, over_lap=True, use_mean=False):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, use_mean=use_mean)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，分别是cfp监督损失，uwf监督损失

        self.opt = Optimizer([self.backbone, self.mhsa, self.classifier], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, clarus_mil_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_mil_score, gt_clarus) * self.weights[1]
            loss += loss_claurs
        else:
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_mil_score, loss, [loss_claurs], [cfp_score]


class VITFOVFuseNet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, score_fusion='single', mil_ratio=3, over_lap=True):
        super(VITFOVFuseNet, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        elif self.score_fusion == 'multi':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        else:
            raise Exception('invalid', self.score_fusion)

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)
        if not over_lap:
            print('no over lap')

        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_mil = MLP(in_features=channels, hidden_features=128, out_features=n_class)

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
        B = clarus_whole.size(0)
        cfp_feature_map = self.backbone.forward(cfp)
        clarus_feature_map = self.backbone.forward(clarus_whole)
        
        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)[:, 0]

        cfp_score = self.classifier(cfp_feature)
        cfp_score_mil = self.classifier_mil(cfp_feature)

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        if need_feature:
            return cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score, [cfp_feature, clarus_whole_feature, clarus_mil_feature]
        else:
            return cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score
    
    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        clarus_feature_map = self.backbone.forward(clarus_whole)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)[:, 0]

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        return clarus_score


class VITFOVFuseNetTraining(VITFOVFuseNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, score_fusion, mil_ratio, training_params, over_lap=True):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, score_fusion=score_fusion, mil_ratio=mil_ratio, over_lap=over_lap)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp在全图分支和mil分支监督损失，uwf监督损失

        self.opt = Optimizer([self.backbone, self.mhsa, self.cw_att_fusion, self.classifier, self.classifier_mil], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, cfp_score_mil, _, _, clarus_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        loss_cfp_mil = self.crit_sup(cfp_score_mil, gt_cfp) * self.weights[1]
        loss = loss_cfp + loss_cfp_mil

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[2]
            loss += loss_claurs
        else:
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_score, loss, [loss_cfp, loss_cfp_mil, loss_claurs], [cfp_score, cfp_score_mil]


class FOVNetDouble(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, score_fusion='single', mil_ratio=3, over_lap=True):
        super(FOVNetDouble, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.cw_att_mil = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        elif self.score_fusion == 'multi':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        else:
            raise Exception('invalid', self.score_fusion)

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)
        if not over_lap:
            print('no over lap')
        self.mil_ratio = mil_ratio

        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_mil = MLP(in_features=channels, hidden_features=128, out_features=n_class)

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
        cfp_feature_map = self.backbone.forward(cfp)
        clarus_feature_map = self.backbone.forward(clarus_whole)
        
        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_whole = F.interpolate(clarus_whole, scale_factor=self.mil_ratio, mode='bilinear')
        clarus_feature_map = self.backbone.forward(clarus_whole)
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        weight_mil = self.cw_att_mil(clarus_mil_feature)

        B, I, C = clarus_mil_feature.size()
        clarus_mil_feature = torch.bmm(weight_mil, clarus_mil_feature) # B * 1 * C
        clarus_mil_feature = clarus_mil_feature.view(B, C)

        cfp_score = self.classifier(cfp_feature)
        cfp_score_mil = self.classifier_mil(cfp_feature)

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        if need_feature:
            return cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score, [cfp_feature, clarus_whole_feature, clarus_mil_feature]
        else:
            return cfp_score, cfp_score_mil, clarus_whole_score, clarus_mil_score, clarus_score
    
    def predict_result(self, clarus_whole, clarus_split):
        clarus_feature_map = self.backbone.forward(clarus_whole)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_whole = F.interpolate(clarus_whole, scale_factor=self.mil_ratio, mode='bilinear')
        clarus_feature_map = self.backbone.forward(clarus_whole)
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        weight_mil = self.cw_att_mil(clarus_mil_feature)

        B, I, C = clarus_mil_feature.size()
        clarus_mil_feature = torch.bmm(weight_mil, clarus_mil_feature) # B * 1 * C
        clarus_mil_feature = clarus_mil_feature.view(B, C)

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        return clarus_score


class FOVNetDoubleTraining(FOVNetDouble):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, score_fusion, mil_ratio, training_params, over_lap=True):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, score_fusion=score_fusion, mil_ratio=mil_ratio, over_lap=over_lap)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp在全图分支和mil分支监督损失，uwf监督损失

        self.opt = Optimizer([self.backbone, self.mhsa, self.cw_att_mil, self.cw_att_fusion, self.classifier, self.classifier_mil], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, cfp_score_mil, _, _, clarus_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        loss_cfp_mil = self.crit_sup(cfp_score_mil, gt_cfp) * self.weights[1]
        loss = loss_cfp + loss_cfp_mil

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[2]
            loss += loss_claurs
        else:
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_score, loss, [loss_cfp, loss_cfp_mil, loss_claurs], [cfp_score, cfp_score_mil]


class VITFOVPosNet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, num_emb, mhsa_nums=0, mil_ratio=3, over_lap=True):
        super(VITFOVPosNet, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()
        
        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)
        if not over_lap:
            print('no over lap')
        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_emb, channels))

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
        clarus_mil_feature += self.pos_embed
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


class VITFOVPosNetTraining(VITFOVPosNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, num_emb, mhsa_nums, mil_ratio, training_params, over_lap=True):
        super().__init__(backbone, n_class, channels, num_emb=num_emb, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，分别是cfp监督损失，uwf监督损失

        self.opt = Optimizer([self.backbone, self.mhsa, self.classifier, self.pos_embed], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, clarus_mil_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_mil_score, gt_clarus) * self.weights[1]
            loss += loss_claurs
        else:
            loss_claurs = None
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_mil_score, loss, [loss_claurs], [cfp_score]


class VITFixBiWhole(nn.Module):
    def __init__(self, backbone_source, backbone_target, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi'):
        super(VITFixBiWhole, self).__init__()
        self.backbone = backbone_source
        self.backbone_target = backbone_target

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)

        self.mil_ratio = mil_ratio

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        elif self.score_fusion == 'multi':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        else:
            raise Exception('invalid', self.score_fusion)

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_target = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_whole = MLP(in_features=channels, hidden_features=128, out_features=n_class)

    def sync_backbones(self):
        self.backbone_target.load_state_dict(self.backbone.state_dict())
        
    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)

    def forward(self, x, source:bool, need_feature=False):
        if source:
            x = self.backbone(x)
            feature = self.sw_gap(x).view(x.size(0), -1)
            x = self.classifier(feature)
        else:
            x = self.backbone_target(x)
            x = self.sw_mil_pooling(x).flatten(2).transpose(1, 2)
            feature = self.mhsa(x)[:, 0]
            # x = self.sw_gap(x).view(x.size(0), -1)
            x = self.classifier_target(feature)
        if need_feature:
            return x, feature
        else:
            return x
    
    def forward_whole(self, x):
        x = self.backbone_target(x)
        feature = self.sw_gap(x).view(x.size(0), -1)
        x = self.classifier_whole(feature)
        return x, feature

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


class VITFixBiWholeTraining(VITFixBiWhole):
    def __init__(self, backbone_source, backbone_target, n_class, channels, crit_sup, weights, mix_ratio, cons_reg_start, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi'):
        super().__init__(backbone_source, backbone_target, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion)

        self.mix_ratio = mix_ratio  # (0.5, 1)
        self.cons_reg_start = cons_reg_start

        self.opt = Optimizer([self.backbone, self.backbone_target, self.classifier, self.classifier_target, self.classifier_whole, self.mhsa, self.cw_att_fusion], training_params)
        self.crit_mixup = nn.BCEWithLogitsLoss()
        self.crit_consis = nn.MSELoss()
        self.crit_sup = crit_sup
        self.weights = weights # 长度为7，分别为mix up中cfp主导的损失权重，uwf主导的损失权重，输入uwf在cfp分支的损失权重，在uwf分支的损失权重，两个分支预测一致性损失, uwf整图预测损失, uwf融合后损失

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
        mix_cfp = self.mix_ratio * new_cfp + (1 - self.mix_ratio) * clarus_whole
        mix_clarus = (1 - self.mix_ratio) * new_cfp + self.mix_ratio * clarus_whole

        score_mix_cfp, score_mix_clarus = self.forward(mix_cfp, True), self.forward(mix_clarus, False)

        loss_sup_mix_cfp = self.crit_sup(score_mix_cfp, gt_cfp) * self.mix_ratio + self.crit_sup(score_mix_cfp, gt_clarus) * (1 - self.mix_ratio)
        loss_sup_mix_clarus = self.crit_sup(score_mix_clarus, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_clarus, gt_cfp) * (1 - self.mix_ratio)

        cfp_model_clarus_score, clarus_molde_clarus_score = self.forward(clarus_whole, True), self.forward(clarus_whole, False)
        
        loss_sup_cfp_model = self.crit_sup(cfp_model_clarus_score, gt_clarus)
        loss_sup_clarus_model = self.crit_sup(clarus_molde_clarus_score, gt_clarus)

        loss = loss_sup_mix_cfp * self.weights[0] + loss_sup_mix_clarus * self.weights[1] + \
               loss_sup_cfp_model * self.weights[2] + loss_sup_clarus_model * self.weights[3]
            
        if iter_num and iter_num >= self.cons_reg_start:
            mean_mix = 0.5 * new_cfp + 0.5 * clarus_whole
            score_mean_mix_cfp_model, score_mean_mix_clarus_model = self.forward(mean_mix, True), self.forward(mean_mix, False)
            cons_loss = self.crit_consis(score_mean_mix_cfp_model, score_mean_mix_clarus_model)
            loss += cons_loss * self.weights[4]
        else:
            cons_loss = None

        B = clarus_whole.size(0)
        uwf_score_whole, uwf_feature_whole = self.forward_whole(clarus_whole)
        loss_uwf_whole = self.crit_sup(uwf_score_whole, gt_clarus)

        uwf_score_vit, uwf_feature_vit = self.forward(clarus_whole, False, need_feature=True)

        weight_fusion = self.cw_att_fusion(torch.cat((uwf_feature_whole.view(B, 1, -1), uwf_feature_vit.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((uwf_score_whole.view(B, 1, -1), uwf_score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        
        loss_merged_score = self.crit_sup(clarus_score, gt_clarus)

        loss += loss_uwf_whole * self.weights[5] + loss_merged_score * self.weights[6]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_molde_clarus_score, loss, [loss_sup_mix_cfp, loss_sup_mix_clarus, loss_sup_cfp_model, loss_sup_clarus_model], []

    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        uwf_score_whole, uwf_feature_whole = self.forward_whole(clarus_whole)

        uwf_score_vit, uwf_feature_vit = self.forward(clarus_whole, False, need_feature=True)

        weight_fusion = self.cw_att_fusion(torch.cat((uwf_feature_whole.view(B, 1, -1), uwf_feature_vit.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((uwf_score_whole.view(B, 1, -1), uwf_score_vit.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        return clarus_score
    

class VITMixupWhole(nn.Module):
    def __init__(self, backbone, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi', use_mean=False):
        super(VITMixupWhole, self).__init__()
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


class VITMixupWholeTraining(VITMixupWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi', use_mean=False):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion, use_mean=use_mean)

        self.mix_ratio = mix_ratio  # (0.5, 1)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_whole, self.mhsa, self.cw_att_fusion], training_params)
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

        score_mix_vit, mix_feature_vit = self.forward(mix_clarus)
        score_mix_whole, mix_feature_whole = self.forward_whole(mix_clarus)

        #loss_sup_mix_vit = self.crit_sup(score_mix_vit, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_vit, gt_cfp) * (1 - self.mix_ratio)
        #loss_sup_mix_whole = self.crit_sup(score_mix_whole, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_whole, gt_cfp) * (1 - self.mix_ratio)

        score_vit, feature_vit = self.forward(clarus_whole)
        score_whole, feature_whole = self.forward_whole(clarus_whole)
        
        #loss_sup_vit = self.crit_sup(score_vit, gt_clarus)
        #loss_sup_whole = self.crit_sup(score_whole, gt_clarus)

        #loss = loss_sup_mix_vit * self.weights[0] + loss_sup_mix_whole * self.weights[1] + \
        #       loss_sup_vit * self.weights[2] + loss_sup_whole * self.weights[3]
            
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


class VITMixupWholePatchMixTraining(VITMixupWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi', use_mean=False):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion, use_mean=use_mean)

        self.mix_ratio = mix_ratio  # (0.5, 1)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_whole, self.mhsa, self.cw_att_fusion], training_params)
        self.crit_mixup = nn.BCEWithLogitsLoss()
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，原始损失，mix up损失

        self.small_cfp_scale_factor = training_params.get('small_cfp_scale_factor', None)

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

        small_cfp = F.interpolate(cfp, scale_factor=self.small_cfp_scale_factor, mode='bilinear', align_corners=True)
        new_cfp = self.pad_cfp(cfp=cfp, clarus_whole=clarus_whole)
        mix_clarus = (1 - self.mix_ratio) * new_cfp + self.mix_ratio * clarus_whole

        score_mix_vit, mix_feature_vit = self.forward(mix_clarus)
        score_mix_whole, mix_feature_whole = self.forward_whole(mix_clarus)

        #loss_sup_mix_vit = self.crit_sup(score_mix_vit, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_vit, gt_cfp) * (1 - self.mix_ratio)
        #loss_sup_mix_whole = self.crit_sup(score_mix_whole, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_whole, gt_cfp) * (1 - self.mix_ratio)

        score_vit, feature_vit = self.forward(clarus_whole)
        score_whole, feature_whole = self.forward_whole(clarus_whole)
        
        #loss_sup_vit = self.crit_sup(score_vit, gt_clarus)
        #loss_sup_whole = self.crit_sup(score_whole, gt_clarus)

        #loss = loss_sup_mix_vit * self.weights[0] + loss_sup_mix_whole * self.weights[1] + \
        #       loss_sup_vit * self.weights[2] + loss_sup_whole * self.weights[3]
            
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





class VITMixupWholeSepCla(nn.Module):
    def __init__(self, backbone, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi', use_mean=False):
        super(VITMixupWholeSepCla, self).__init__()
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
        self.classifier_whole = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_mil = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_cfp_mil = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_cfp_whole = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        
    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)

    def forward(self, x, cfp=False):
        
        x = self.backbone(x)
        x = self.sw_mil_pooling(x).flatten(2).transpose(1, 2)
        feature = self.mhsa(x)
        if self.use_mean:
            feature = torch.mean(feature, dim=1)
        else:
            feature = feature[:, 0]
        if cfp:
            x = self.classifier_cfp_mil(feature)
        else:
            x = self.classifier_mil(feature)
        return x, feature
    
    def forward_whole(self, x, cfp=False):
        x = self.backbone(x)
        feature = self.sw_gap(x).view(x.size(0), -1)
        if cfp:
            x = self.classifier_cfp_whole(feature)
        else:
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


class VITMixupWholeSepClaTraining(VITMixupWholeSepCla):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, score_fusion='multi', use_mean=False):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, score_fusion=score_fusion, use_mean=use_mean)

        self.mix_ratio = mix_ratio  # (0.5, 1)

        self.opt = Optimizer([self.backbone, self.classifier_whole, self.classifier_mil, self.classifier_cfp_whole, self.classifier_cfp_mil, self.mhsa, self.cw_att_fusion], training_params)
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

        score_mix_vit, mix_feature_vit = self.forward(mix_clarus, cfp=True)
        score_mix_whole, mix_feature_whole = self.forward_whole(mix_clarus, cfp=True)

        #loss_sup_mix_vit = self.crit_sup(score_mix_vit, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_vit, gt_cfp) * (1 - self.mix_ratio)
        #loss_sup_mix_whole = self.crit_sup(score_mix_whole, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_whole, gt_cfp) * (1 - self.mix_ratio)

        score_vit, feature_vit = self.forward(clarus_whole)
        score_whole, feature_whole = self.forward_whole(clarus_whole)
        
        #loss_sup_vit = self.crit_sup(score_vit, gt_clarus)
        #loss_sup_whole = self.crit_sup(score_whole, gt_clarus)

        #loss = loss_sup_mix_vit * self.weights[0] + loss_sup_mix_whole * self.weights[1] + \
        #       loss_sup_vit * self.weights[2] + loss_sup_whole * self.weights[3]
            
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

    
        loss_mix_whole = self.crit_sup(score_mix_whole, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_whole, gt_cfp) * (1 - self.mix_ratio)
        loss_mix_mil = self.crit_sup(score_mix_vit, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_vit, gt_cfp) * (1 - self.mix_ratio)

        loss = loss_merged_score * self.weights[0] + loss_mix_whole * self.weights[1] + loss_mix_mil * self.weights[2]

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
    



class VITFOVWholeTriClsNet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, score_fusion='single', mil_ratio=3, over_lap=True, use_mean=False):
        super(VITFOVWholeTriClsNet, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        elif self.score_fusion == 'multi':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        else:
            raise Exception('invalid', self.score_fusion)

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)
        if not over_lap:
            print('no over lap')

        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_mil = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_cfp = MLP(in_features=channels, hidden_features=128, out_features=n_class)

        self.use_mean =  use_mean

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
        B = clarus_whole.size(0)
        cfp_feature_map = self.backbone.forward(cfp)
        clarus_feature_map = self.backbone.forward(clarus_whole)
        
        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        if self.use_mean:
            clarus_mil_feature = torch.mean(clarus_mil_feature, dim=1)
        else:
            clarus_mil_feature = clarus_mil_feature[:, 0]

        cfp_score = self.classifier_cfp(cfp_feature)

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        if need_feature:
            return cfp_score, clarus_whole_score, clarus_mil_score, clarus_score, [cfp_feature, clarus_whole_feature, clarus_mil_feature]
        else:
            return cfp_score, clarus_whole_score, clarus_mil_score, clarus_score
    
    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        clarus_feature_map = self.backbone.forward(clarus_whole)
        clarus_whole_feature = self.sw_gap(clarus_feature_map).view(clarus_feature_map.size(0), -1)
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        if self.use_mean:
            clarus_mil_feature = torch.mean(clarus_mil_feature, dim=1)
        else:
            clarus_mil_feature = clarus_mil_feature[:, 0]

        clarus_whole_score = self.classifier(clarus_whole_feature)
        clarus_mil_score = self.classifier_mil(clarus_mil_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_mil_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_mil_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)
        return clarus_score


class VITFOVWholeTriClsNetTraining(VITFOVWholeTriClsNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mhsa_nums, score_fusion, mil_ratio, training_params, all_loss, over_lap=True, use_mean=False):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, score_fusion=score_fusion, mil_ratio=mil_ratio, over_lap=over_lap, use_mean=use_mean)

        self.crit_sup = crit_sup
        self.weights = weights 
        self.all_loss = all_loss
        self.opt = Optimizer([self.backbone, self.mhsa, self.cw_att_fusion, self.classifier, self.classifier_mil, self.classifier_cfp], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, clarus_whole_score, clarus_mil_score, clarus_score = self.forward(cfp, clarus_whole, clarus_split, need_feature=False) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        loss = loss_cfp

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[1]
            loss += loss_claurs
        else:
            loss_claurs = None
        if self.all_loss:
            loss_whole = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[2]
            loss_mil = self.crit_sup(clarus_mil_score, gt_clarus) * self.weights[3]
            loss += loss_whole + loss_mil
                
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_score, loss, [loss_cfp, loss_claurs], [cfp_score]


class VITMixup(nn.Module):
    def __init__(self, backbone, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, use_mean=False):
        super(VITMixup, self).__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)

        self.mil_ratio = mil_ratio
        self.use_mean = use_mean

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        
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

    @staticmethod
    def pad_cfp(cfp, clarus_whole):
        # 超广角图像可能不是正方形的，所以把cfp padding以下，以便mix up
        min_n = min(cfp.size(0), clarus_whole.size(0))
        cfp = cfp[:min_n]
        clarus_whole = clarus_whole[:min_n]

        w_cfp = cfp.size(3)
        w_clarus_whole = clarus_whole.size(3)
        dw = int((w_clarus_whole - w_cfp) / 2)

        h_cfp = cfp.size(2)
        h_clarus_whole = clarus_whole.size(2)
        dh = int((h_clarus_whole - h_cfp) / 2)
        if w_clarus_whole != w_cfp or h_clarus_whole != h_cfp:
            new_cfp = torch.zeros_like(clarus_whole)
            new_cfp[:, :, dh:dh+h_cfp, dw:dw+w_cfp] = cfp
        else:
            new_cfp = cfp
        return new_cfp


class VITMixupTraining(VITMixup):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True, use_mean=False):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, use_mean=use_mean)

        self.mix_ratio = mix_ratio  # (0.5, 1)

        self.opt = Optimizer([self.backbone, self.classifier, self.mhsa], training_params)
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

        score_mix_vit, mix_feature_vit = self.forward(mix_clarus)

        #loss_sup_mix_vit = self.crit_sup(score_mix_vit, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_vit, gt_cfp) * (1 - self.mix_ratio)
        #loss_sup_mix_whole = self.crit_sup(score_mix_whole, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_whole, gt_cfp) * (1 - self.mix_ratio)

        score_vit, feature_vit = self.forward(clarus_whole)
        
        #loss_sup_vit = self.crit_sup(score_vit, gt_clarus)
        #loss_sup_whole = self.crit_sup(score_whole, gt_clarus)

        #loss = loss_sup_mix_vit * self.weights[0] + loss_sup_mix_whole * self.weights[1] + \
        #       loss_sup_vit * self.weights[2] + loss_sup_whole * self.weights[3]
            
        B = clarus_whole.size(0)

        
        loss = self.crit_sup(score_vit, gt_clarus)
        loss_mix = self.crit_sup(score_mix_vit, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_vit, gt_cfp) * (1 - self.mix_ratio)

        loss = loss * self.weights[0] + loss_mix * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return score_vit, loss, [], []

    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        score_vit, feature_vit = self.forward(clarus_whole)

        return score_vit
    

class MixupNet(nn.Module):
    def __init__(self, backbone, n_class, channels):
        super(MixupNet, self).__init__()
        self.backbone = backbone

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        
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
        x = self.sw_gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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


class MixupNetTraining(MixupNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, training_params):
        super().__init__(backbone, n_class, channels)

        self.mix_ratio = mix_ratio  # (0.5, 1)

        self.opt = Optimizer([self.backbone, self.classifier], training_params)
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

        score_mix = self.forward(mix_clarus)
        score = self.forward(clarus_whole)
        
            
        B = clarus_whole.size(0)

        
        loss = self.crit_sup(score, gt_clarus)
        loss_mix = self.crit_sup(score_mix, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix, gt_cfp) * (1 - self.mix_ratio)

        loss = loss * self.weights[0] + loss_mix * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return score, loss, [], []

    def predict_result(self, clarus_whole, clarus_split):
        B = clarus_whole.size(0)
        score = self.forward(clarus_whole)

        return score
    

