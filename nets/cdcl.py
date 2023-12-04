import torch
import torch.nn as nn
import torch.nn.functional as F
from .optimizer import Optimizer
from .MLP import MLP, GELU

import math


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction, act=nn.Sigmoid()):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(channels // reduction, 1)
        self.act = act

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        weight = self.act(x)  # B * I * 1
        weight = torch.transpose(weight, 2, 1) # B * 1 * I
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

    def forward(self, x):
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

    def forward(self, x):
        x = self.mhsa(x)
        x = self.mlp(x)
        return x


class SelfAttentionBlocks(nn.Module):
    def __init__(self, dim, block_num, num_heads=8, drop_rate=0.2):
        super().__init__()
        self.block_num = block_num
        assert self.block_num >= 1

        self.blocks = nn.ModuleList([MHSABlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
                                     for i in range(self.block_num)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class LocalAveragePooling(nn.Module):
    def __init__(self, ratio, over_lap=True) -> None:
        super().__init__()
        self.ratio = ratio
        self.over_lap = over_lap

    def forward(self, x):
        B, C, H, W = x.size()
        ratio_h = self.ratio
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
        

class CdCL(nn.Module):
    def __init__(self, backbone, n_class, backbone_out_channels, mhsa_nums=0, lap_ratio=3, over_lap=True):
        super().__init__()
        self.backbone = backbone

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=backbone_out_channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.lap = LocalAveragePooling(ratio=lap_ratio, over_lap=over_lap)

        self.channel_att_fusion = ChannelAttention(channels=backbone_out_channels, reduction=8, act=nn.Softmax(dim=1))
       
        self.gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier_gap = MLP(in_features=backbone_out_channels, hidden_features=128, out_features=n_class)
        self.classifier_lap = MLP(in_features=backbone_out_channels, hidden_features=128, out_features=n_class)
        
    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load custom pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)

    def forward(self, x):
        B = x.size(0)
        score_gap, feature_gap = self.forward_gap(x)
        score_lap, feature_lap = self.forward_lap(x)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_gap.view(B, 1, -1), feature_lap.view(B, 1, -1)), dim=1))
        score = torch.cat((score_gap.view(B, 1, -1), score_lap.view(B, 1, -1)), dim=1)
        score = torch.bmm(weight_fusion, score) # B * n_class * 2 , B * 2 * n_class
        score = score.view(B, -1)

        return score

    def forward_lap(self, x):
        x = self.backbone.forward_features(x)

        x = self.lap(x).flatten(2).transpose(1, 2)
        feature = self.mhsa(x)
        feature = torch.mean(feature, dim=1)
        
        x = self.classifier_lap(feature)
        return x, feature
    
    def forward_gap(self, x):
        x = self.backbone.forward_features(x)

        feature = self.gap(x).view(x.size(0), -1)

        x = self.classifier_gap(feature)
        return x, feature

    @staticmethod
    def uniform_size(source, target):
        # uniform the size of source domain and target domain tensors for mixup 
        min_n = min(source.size(0), target.size(0))
        source = source[:min_n]
        target = target[:min_n]

        w_source = source.size(3)
        w_target = target.size(3)
        dw = int((w_target - w_source) / 2)
        if w_target != w_source:
            new_source = torch.zeros_like(target)
            new_source[:, :, :, dw:dw+w_source] = source
        else:
            new_source = source
        return new_source


class CdCLProcessor(object):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, mix_ratio, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True):
        
        
        
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap)

        self.mix_ratio = mix_ratio  # (0.5, 1)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_whole, self.mhsa, self.cw_att_fusion], training_params)
        self.crit_mixup = nn.BCEWithLogitsLoss()
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，原始损失，mix up损失

    def train_model(self, cfp, clarus_whole, gt_cfp, gt_clarus):
        if cfp.size(0) != clarus_whole.size(0):
            if cfp.size(0) > clarus_whole.size(0):
                cfp = cfp[:clarus_whole.size(0)]
                gt_cfp = gt_cfp[:clarus_whole.size(0)]
            else:
                repeat_num = int(math.ceil(clarus_whole.size(0) / cfp.size(0)))
                cfp = cfp.repeat((repeat_num, 1, 1, 1))[:clarus_whole.size(0)]
                gt_cfp = gt_cfp.repeat((repeat_num, 1))[:clarus_whole.size(0)]

        self.opt.z_grad()

        new_cfp = self.uniform_size(cfp=cfp, clarus_whole=clarus_whole)
        mix_clarus = (1 - self.mix_ratio) * new_cfp + self.mix_ratio * clarus_whole

        score_mix_vit, mix_feature_vit = self.forward(mix_clarus)
        score_mix_whole, mix_feature_whole = self.forward_whole(mix_clarus)

        score_vit, feature_vit = self.forward(clarus_whole)
        score_whole, feature_whole = self.forward_whole(clarus_whole)
            
        B = clarus_whole.size(0)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), feature_vit.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        
        clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
        clarus_score = clarus_score.view(B, -1)
        
        
        loss_merged_score = self.crit_sup(clarus_score, gt_clarus)

        mix_weight_fusion = self.cw_att_fusion(torch.cat((mix_feature_whole.view(B, 1, -1), mix_feature_vit.view(B, 1, -1)), dim=1))
        mix_clarus_score = torch.cat((score_mix_whole.view(B, 1, -1), score_mix_vit.view(B, 1, -1)), dim=1)
        mix_clarus_score = torch.bmm(mix_weight_fusion, mix_clarus_score) # B * n_class * 2 , B * 2 * n_class
        mix_clarus_score = mix_clarus_score.view(B, -1)
       
        loss_mix_merged_score = self.crit_sup(mix_clarus_score, gt_clarus) * self.mix_ratio + self.crit_sup(mix_clarus_score, gt_cfp) * (1 - self.mix_ratio)

        loss = loss_merged_score * self.weights[0] + loss_mix_merged_score * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_score, loss, [], []

    def predict_result(self, clarus_whole):
        B = clarus_whole.size(0)
        score_whole, feature_whole = self.forward_whole(clarus_whole)

        score_vit, feature_vit = self.forward(clarus_whole)

        weight_fusion = self.cw_att_fusion(torch.cat((feature_whole.view(B, 1, -1), feature_vit.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((score_whole.view(B, 1, -1), score_vit.view(B, 1, -1)), dim=1)
        clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
        clarus_score = clarus_score.view(B, -1)

        return clarus_score

