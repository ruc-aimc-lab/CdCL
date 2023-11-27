import torch
import torch.nn as nn
import torch.nn.functional as F
from .MLP import MLP, GELU, MLPGRL
from .FOVNet import SelfAttentionBlocks, MyAvgPool2d

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        cdf = 0.5 * (1 + torch.erf(x / 2**0.5))
        return x * cdf


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(channels // reduction, out_channel)
        self.act = act

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        weight = self.act(x)  # B * I * out_channel
        weight = torch.transpose(weight, 2, 1) # B * out_channel * I
        return weight


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicNet(nn.Module):
    def __init__(self, backbone, n_class, channels):
        super(BasicNet, self).__init__()
        self.backbone = backbone
        self.sw_gap = nn.AdaptiveAvgPool2d(1)
        self.fc = Linear(channel_in=channels, channel_out=n_class)

    def forward(self, din): 
        out = self.backbone(din[0])
        out = self.sw_gap(out).view(out.size(0), -1)
        out = self.fc(out)
        return out

    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)

class BasicNetTraining(BasicNet):
    def __init__(self, backbone, n_class, channels, crit):
        super().__init__(backbone, n_class, channels)
        self.crit = crit

    def train_model(self, din, gt): 
        score = self.forward(din)
        loss = self.crit(score, gt)
        return score, loss


class BasicNetUWFTraining(BasicNet):
    def __init__(self, backbone, n_class, channels, crit):
        super().__init__(backbone, n_class, channels)
        self.fc = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.crit = crit

    def train_model(self, din, gt, **kwargs): 
        

        score = self.forward(din)
        if 'mask' in kwargs:
            
            mask = kwargs['mask']
            loss = self.crit(score[mask==1], gt[mask==1])
        else:
            loss = self.crit(score, gt)
        return score, loss


class MILNet(nn.Module):
    def __init__(self, backbone, n_class, channels):
        super(MILNet, self).__init__()
        self.backbone = backbone
        self.cw_att = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())

        self.sw_gap = nn.AdaptiveAvgPool2d(1)
        self.fc = Linear(channel_in=channels, channel_out=n_class)

    def forward(self, din): 
        out = din[0]
        B, I, C, H, W = out.size()
        out = out.view(B * I, C, H, W)
        out = self.backbone(out)
        out = self.sw_gap(out).view(B, I, -1)

        weight = self.cw_att(out)
        out = torch.bmm(weight, out) # B * 1 * C
        out = out.view(B, -1)

        out = self.fc(out)
        return out


class MILNetTraining(MILNet):
    def __init__(self, backbone, n_class, channels, crit):
        super().__init__(backbone, n_class, channels)
        self.crit = crit

    def train_model(self, din, gt): 
        score = self.forward(din)
        loss = self.crit(score, gt)
        return score, loss


class InnerMIL(nn.Module):
    def __init__(self, backbone, n_class, channels):
        super(InnerMIL, self).__init__()
        self.backbone = backbone

        self.cw_att = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())
        
        # self.sw_mil_pooling = nn.AdaptiveAvgPool2d(3) # spatial-wise global average pooling

        self.fc = Linear(channel_in=channels, channel_out=n_class)

    def forward(self, x):
        
        x = self.backbone(x[0])
        B, C, H, W = x.size()

        feature_mil = F.avg_pool2d(input=x, kernel_size=(int(H/2), int(W/2)), stride=(int(H/4), int(W/4))).view(B, C, -1)
        # feature_mil = self.sw_mil_pooling(x).view(B, C, -1)
        feature_mil = torch.transpose(feature_mil, 1, 2)

        weight = self.cw_att(feature_mil)
        x = torch.bmm(weight, feature_mil) # B * 1 * C
        x = x.view(B, -1)

        x = self.fc(x)
       
        return x


class InnerMILTraining(InnerMIL):
    def __init__(self, backbone, n_class, channels, crit):
        super().__init__(backbone, n_class, channels)
        self.crit = crit

    def train_model(self, din, gt): 
        score = self.forward(din)
        loss = self.crit(score, gt)
        return score, loss



class InnerMILSA(nn.Module):
    def __init__(self, backbone, n_class, channels):
        super(InnerMILSA, self).__init__()
        self.backbone = backbone

        self.self_att = Block(dim=channels, num_heads=8, qkv_bias=True)
        self.cw_att = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())

        self.fc = Linear(channel_in=channels, channel_out=n_class)

    def forward(self, x):
        
        x = self.backbone(x[0])
        B, C, H, W = x.size()

        feature_mil = F.avg_pool2d(input=x, kernel_size=(int(H/2), int(W/2)), stride=(int(H/4), int(W/4))).view(B, C, -1)
        # feature_mil = self.sw_mil_pooling(x).view(B, C, -1)
        feature_mil = torch.transpose(feature_mil, 1, 2)
        feature_mil = self.self_att(feature_mil)

        weight = self.cw_att(feature_mil)
        x = torch.bmm(weight, feature_mil) # B * 1 * C
        x = x.view(B, -1)

        x = self.fc(x)
       
        return x


class InnerMILSATraining(InnerMILSA):
    def __init__(self, backbone, n_class, channels, crit):
        super().__init__(backbone, n_class, channels)
        self.crit = crit

    def train_model(self, din, gt): 
        score = self.forward(din)
        loss = self.crit(score, gt)
        return score, loss


class FusionNet(nn.Module):
    def __init__(self, backbone, n_class, channels, score_fusion):
        super(FusionNet, self).__init__()
        self.backbone = backbone
        self.cw_att = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())

        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        else:
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)

        self.sw_gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Linear(channel_in=channels, channel_out=n_class)
        self.fc2 = Linear(channel_in=channels, channel_out=n_class)

    def forward(self, din):
        whole, patches = din

        B, I, C, H, W = patches.size()
        patches = patches.view(B * I, C, H, W)

        out_patches = self.backbone(patches)
        out_whole = self.backbone(whole)

        _, C, H, W = out_patches.size()
        out_patches = self.sw_gap(out_patches).view(B, I, -1)
        out_whole = self.sw_gap(out_whole).view(B, -1)

        weight1 = self.cw_att(out_patches)
        out_patches = torch.bmm(weight1, out_patches) # B * 1 * C
        out_patches = out_patches.view(B, C)

        score_whole = self.fc1(out_whole)
        score_patches = self.fc2(out_patches)
        
        weight_fusion = self.cw_att_fusion(torch.cat((out_whole.view(B, 1, -1), out_patches.view(B, 1, -1)), dim=1))
        score = torch.cat((score_whole.view(B, 1, -1), score_patches.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            score = torch.bmm(weight_fusion, score) # B * n_class * 2 , B * 2 * n_class
            score = score.view(B, -1)
        else:
            score = score * torch.transpose(weight_fusion, 2, 1)
            score = torch.sum(score, dim=1)
        return score


class FusionNetTraining(FusionNet):
    def __init__(self, backbone, n_class, channels, score_fusion, crit):
        super().__init__(backbone, n_class, channels, score_fusion)
        self.crit = crit

    def train_model(self, din, gt): 
        score = self.forward(din)
        loss = self.crit(score, gt)
        return score, loss


class SingleDomainVITFOVNet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True, use_mean=False):
        super(SingleDomainVITFOVNet, self).__init__()
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
    
    def forward(self, x):
        clarus_feature_map = self.backbone.forward(x[0])
        
        clarus_mil_feature = self.sw_mil_pooling(clarus_feature_map).flatten(2).transpose(1, 2)
        clarus_mil_feature = self.mhsa(clarus_mil_feature)
        if self.use_mean:
            clarus_mil_feature = torch.mean(clarus_mil_feature, dim=1)
        else:
            clarus_mil_feature = clarus_mil_feature[:, 0]
        
        clarus_mil_score = self.classifier(clarus_mil_feature)
        
        return clarus_mil_score


class SingleDomainVITFOVNetTraining(SingleDomainVITFOVNet):
    def __init__(self, backbone, n_class, channels, crit, mhsa_nums, mil_ratio, over_lap=True, use_mean=False):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap, use_mean=use_mean)

        self.crit = crit


    def train_model(self, din, gt):
        score = self.forward(din) 
        loss = self.crit(score, gt) 
        return score, loss
