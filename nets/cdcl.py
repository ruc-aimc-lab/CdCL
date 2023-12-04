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

        weight_fusion = self.channel_att_fusion(torch.cat((feature_gap.view(B, 1, -1), feature_lap.view(B, 1, -1)), dim=1))
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
        return new_source, target


class CdCLProcessor(object):
    def __init__(self, backbone, backbone_out_channels, training_params):
        model_params = training_params['model_params']
        n_class = model_params['n_class']
        mhsa_nums = model_params['mhsa_nums']
        lap_ratio = model_params['lap_ratio']
        over_lap = model_params['over_lap']
        
        
        self.model = CdCL(backbone=backbone, n_class=n_class, backbone_out_channels=backbone_out_channels, 
                     mhsa_nums=mhsa_nums, lap_ratio=lap_ratio, over_lap=over_lap)

        self.mix_ratio = model_params['mix_ratio']  # (0.5, 1)

        self.opt = Optimizer([self.model], training_params)
        self.crit = nn.BCEWithLogitsLoss()
        self.weights = model_params['weights'] # 长度为2，原始损失，mix up损失

    def train_model(self, source, target, gt_source, gt_target):
        '''if cfp.size(0) != clarus_whole.size(0):
            if cfp.size(0) > clarus_whole.size(0):
                cfp = cfp[:clarus_whole.size(0)]
                gt_cfp = gt_cfp[:clarus_whole.size(0)]
            else:
                repeat_num = int(math.ceil(clarus_whole.size(0) / cfp.size(0)))
                cfp = cfp.repeat((repeat_num, 1, 1, 1))[:clarus_whole.size(0)]
                gt_cfp = gt_cfp.repeat((repeat_num, 1))[:clarus_whole.size(0)]'''

        self.opt.z_grad()

        source, target = self.uniform_size(source=source, target=target)
        mixup = (1 - self.mix_ratio) * source + self.mix_ratio * target

        score_target = self.model(target)
        score_mixup = self.model(mixup)
        
        loss_target = self.crit_sup(score_target, gt_target)
        loss_mixup = self.crit_sup(score_mixup, gt_target) * self.mix_ratio + self.crit_sup(score_mixup, gt_source) * (1 - self.mix_ratio)

        loss = loss_target * self.weights[0] + loss_mixup * self.weights[1]

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return score_target, loss, [], []

    def predict_result(self, x):
        return self.model(x)
    
    def change_model_mode(self, mode):
        
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise Exception('Invalid model mode {}'.format(mode))
        
    def change_device(self, device):
        self.model.to(device)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
