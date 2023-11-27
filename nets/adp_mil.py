import torch
import torch.nn as nn
import torch.nn.functional as F
# from .loss_func import WeightedBCEWithLogitsLoss
from skimage.draw import disk
import numpy as np
from .optimizer import Optimizer
from .MLP import MLP

def circle(l, rate=1.):
    r = int(l / 2)
    rr, cc = disk(center=(r, r), radius=int(rate * r), shape=(l, l))
    return rr, cc

def out_circle(l, rate=1.):
    r = int(l / 2)
    rr, cc = disk(center=(r, r), radius=int(rate * r), shape=(l, l))
    im = np.zeros((l, l), int)
    im[rr, cc] = 1
    rr, cc = np.where(im == 0)
    return rr, cc

def split2whole_wf(L, split_imgs, whole_img, weights1, weight2):
    # weight1:split_imgs分别的权重
    # weight2:whole_img权重
    alpha = 1 / ((133 / 45) ** 0.5)
    l = int(L * alpha / 2)
    d = l * 2
    rr, cc = circle(d)

    c1 = (int(L * 0.5), int(l))
    c2 = (int(L * 0.5), int(L - l))
    c3 = (int(L - l), int(L * 0.5))
    c4 = (int(l), int(L * 0.5))

    dis = int((2 - 2 ** 0.5 + alpha * 2 ** 0.5) / 4 * L)
    c5 = (dis, dis)
    c6 = (dis, L - dis)
    c7 = (L - dis, dis)
    c8 = (L- dis, L - dis)

    c9 = (int(L / 2), int(L / 2))
    centers = [c1, c2, c3, c4, c5, c6, c7, c8, c9]

    whole_img = F.interpolate(whole_img, size=(L, L), mode='bilinear', align_corners=True)
    split_imgs = F.interpolate(split_imgs, size=(d, d), mode='bilinear', align_corners=True)
    whole_img *= weight2
    for c, img, w1 in zip(centers, split_imgs, weights1):
        c_x, c_y = c
        whole_img[:, :, rr + c_y - l, cc + c_x - l] += img[:, rr, cc] * w1 * (1 - weight2)
    return whole_img


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


class AdaptiveNetWhole(nn.Module):
    def __init__(self, backbone, n_class, channels):
        super(AdaptiveNetWhole, self).__init__()
        self.backbone = backbone
        self.n_class = n_class
        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)

    def forward(self, cfp, clarus_whole, clarus_split, need_feature=False):
        cfp_feature_map = self.backbone.forward(cfp)
        clarus_whole_feature_map = self.backbone.forward(clarus_whole)

        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)

        cfp_score = self.classifier(cfp_feature)
        clarus_whole_score = self.classifier(clarus_whole_feature)
        if need_feature:
            return cfp_score, clarus_whole_score, [cfp_feature, clarus_whole_feature]
        else:
            return cfp_score, clarus_whole_score
    
    def predict_result(self, clarus_whole, clarus_split):
        clarus_whole_feature_map = self.backbone.forward(clarus_whole)
        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)
        clarus_whole_score = self.classifier(clarus_whole_feature)
        return clarus_whole_score

    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)

class AdaptiveNetWholeTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, training_params):
        super().__init__(backbone, n_class, channels)

        self.opt = Optimizer([self.backbone, self.classifier], training_params)
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，分别是cfp的监督损失，uwf的监督损失
    
    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        cfp_score, clarus_whole_score = self.forward(cfp, clarus_whole, clarus_split) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        loss.backward()
        self.opt.g_step()
        self.opt.z_grad()
        self.opt.update_lr()
        return clarus_whole_score, loss, [loss_cfp, loss_claurs], [cfp_score]


class AdaptiveNetSplit(nn.Module):
    def __init__(self, backbone, n_class, channels):
        super(AdaptiveNetSplit, self).__init__()
        self.backbone = backbone

        self.cw_att_mil = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())
        
        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)

    def forward(self, cfp, clarus_whole, clarus_split, need_feature=False):

        B, I, C, H, W = clarus_split.size()
        clarus_split = clarus_split.view(B * I, C, H, W)

        cfp_feature_map = self.backbone.forward(cfp)
        clarus_split_feature_map = self.backbone.forward(clarus_split)

        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_split_feature = self.sw_gap(clarus_split_feature_map).view(B, I, -1)

        weight_mil = self.cw_att_mil(clarus_split_feature)
        B, I, C = clarus_split_feature.size()
        clarus_split_feature = torch.bmm(weight_mil, clarus_split_feature) # B * 1 * C
        clarus_split_feature = clarus_split_feature.view(B, C)

        cfp_score = self.classifier(cfp_feature)

        clarus_split_score = self.classifier(clarus_split_feature)
       
        if need_feature:
            return cfp_score, clarus_split_score, [cfp_feature, clarus_split_feature]
        else:
            return cfp_score, clarus_split_score
    
    def predict_result(self, clarus_whole, clarus_split):
        
        B, I, C, H, W = clarus_split.size()
        clarus_split = clarus_split.view(B * I, C, H, W)

        clarus_split_feature_map = self.backbone.forward(clarus_split)

        clarus_split_feature = self.sw_gap(clarus_split_feature_map).view(B, I, -1)

        weight_mil = self.cw_att_mil(clarus_split_feature)
        B, I, C = clarus_split_feature.size()
        clarus_split_feature = torch.bmm(weight_mil, clarus_split_feature) # B * 1 * C
        clarus_split_feature = clarus_split_feature.view(B, C)

        clarus_split_score = self.classifier(clarus_split_feature)
        return clarus_split_score

    def my_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        # left_keys = [k for k, v in state_dict.items() if k not in own_state or own_state[k].shape != v.shape]
        no_update_keys = [k for k, v in own_state.items() if k not in state_dict or state_dict[k].shape != v.shape]
        own_state.update(new_state)
        print('load pretrained')
        print(no_update_keys)
        super().load_state_dict(own_state, strict=True)
        

class AdaptiveNetSplitTraining(AdaptiveNetSplit):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, training_params):
        super().__init__(backbone, n_class, channels)
        self.opt = Optimizer([self.backbone, self.cw_att_mil, self.classifier], training_params)
        self.crit_sup = crit_sup
        self.weights = weights # 长度为2，分别是cfp的监督损失，uwf的监督损失
    
    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        cfp_score, clarus_split_score = self.forward(cfp, clarus_whole, clarus_split) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_split_score, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp
        loss.backward()
        self.opt.g_step()
        self.opt.z_grad()
        self.opt.update_lr()

        return clarus_split_score, loss, [loss_cfp, loss_claurs], [cfp_score]


class AdaptiveMIL(nn.Module):
    def __init__(self, backbone, n_class, channels, score_fusion='single'):
        super(AdaptiveMIL, self).__init__()
        self.backbone = backbone

        self.cw_att_mil = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())
        
        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        else:
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        
        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.classifier1 = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier2 = MLP(in_features=channels, hidden_features=128, out_features=n_class)

    def forward_get_feature(self, cfp, clarus_whole, clarus_split):
        cfp_feature = self.backbone.forward(cfp)
        clarus_whole_feature = self.backbone.forward(clarus_whole)
        clarus_split_feature = self.backbone.forward(clarus_split)
        return cfp_feature, clarus_whole_feature, clarus_split_feature

    def forward(self, cfp, clarus_whole, clarus_split, need_feature=False):

        B, I, C, H, W = clarus_split.size()
        clarus_split = clarus_split.view(B * I, C, H, W)

        cfp_feature_map, clarus_whole_feature_map, clarus_split_feature_map = self.forward_get_feature(cfp, clarus_whole, clarus_split)

        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)
        clarus_split_feature = self.sw_gap(clarus_split_feature_map).view(B, I, -1)

        weight_mil = self.cw_att_mil(clarus_split_feature)
        B, I, C = clarus_split_feature.size()
        clarus_split_feature = torch.bmm(weight_mil, clarus_split_feature) # B * 1 * C
        clarus_split_feature = clarus_split_feature.view(B, C)

        cfp_score1 = self.classifier1(cfp_feature)
        cfp_score2 = self.classifier2(cfp_feature)

        clarus_whole_score = self.classifier1(clarus_whole_feature)
        clarus_split_score = self.classifier2(clarus_split_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_split_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_split_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        if need_feature:
            return cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score, clarus_score, [cfp_feature, clarus_whole_feature, clarus_split_feature]
        else:
            return cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score, clarus_score
    
    def predict_result(self, clarus_whole, clarus_split):
        B, I, C, H, W = clarus_split.size()
        clarus_split = clarus_split.view(B * I, C, H, W)

        clarus_whole_feature_map = self.backbone.forward(clarus_whole)
        clarus_split_feature_map = self.backbone.forward(clarus_split)

        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)
        clarus_split_feature = self.sw_gap(clarus_split_feature_map).view(B, I, -1)

        weight_mil = self.cw_att_mil(clarus_split_feature)
        B, I, C = clarus_split_feature.size()
        clarus_split_feature = torch.bmm(weight_mil, clarus_split_feature) # B * 1 * C
        clarus_split_feature = clarus_split_feature.view(B, C)

        clarus_whole_score = self.classifier1(clarus_whole_feature)
        clarus_split_score = self.classifier2(clarus_split_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_split_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_split_score.view(B, 1, -1)), dim=1)
        
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        return clarus_score


class AdaptiveMILTraining(AdaptiveMIL):
    def __init__(self, backbone, n_class, channels, score_fusion, crit_sup, crit_cons, weights, training_params):
        super().__init__(backbone, n_class, channels, score_fusion)
        self.opt = Optimizer([self.backbone, self.cw_att_mil, self.cw_att_fusion, self.classifier1, self.classifier2], training_params)

        self.crit_sup = crit_sup
        self.crit_cons = crit_cons
        self.weights = weights # 长度为3，分别是cfp的监督损失，全局-局部一致性损失，uwf的监督损失
    
    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score, clarus_score = self.forward(cfp, clarus_whole, clarus_split) 
        
        loss_cfp1 = self.crit_sup(cfp_score1, gt_cfp) * self.weights[0] * 0.5
        loss_cfp2 = self.crit_sup(cfp_score2, gt_cfp) * self.weights[0] * 0.5
        
        loss_consistency = self.crit_cons(clarus_whole_score, clarus_split_score) * self.weights[1]
        loss = loss_cfp1 + loss_cfp2 + loss_consistency

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[2]
            loss += loss_claurs
        else:
            loss_claurs = None
        loss.backward()
        self.opt.g_step()
        self.opt.z_grad()
        self.opt.update_lr()

        return clarus_score, loss, [loss_cfp1, loss_cfp2, loss_consistency, loss_claurs], [cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score]


class AdaptiveMILInnerSplit(AdaptiveMIL):
    def __init__(self, backbone, n_class, channels, score_fusion='single'):
        super(AdaptiveMILInnerSplit, self).__init__()
        self.backbone = backbone

        self.cw_att_mil = ChannelAttention(channels=channels, reduction=8, act=nn.Sigmoid())
        
        self.score_fusion = score_fusion
        if self.score_fusion == 'single':
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=1)
        else:
            self.cw_att_fusion = ChannelAttention(channels=channels, reduction=8, act=nn.Softmax(dim=1), out_channel=n_class)
        
        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling

        self.classifier1 = Linear(channel_in=channels, channel_out=n_class)
        self.classifier2 = Linear(channel_in=channels, channel_out=n_class)

    def forward_get_feature(self, cfp, clarus_whole, clarus_split):
        cfp_feature = self.backbone.forward(cfp)
        clarus_whole_feature = self.backbone.forward(clarus_whole)
        clarus_split_feature = self.backbone.forward(clarus_split)
        return cfp_feature, clarus_whole_feature, clarus_split_feature

    def forward(self, cfp, clarus_whole, clarus_split, need_feature=False):

        B, I, C, H, W = clarus_split.size()
        clarus_split = clarus_split.view(B * I, C, H, W)

        cfp_feature_map, clarus_whole_feature_map, clarus_split_feature_map = self.forward_get_feature(cfp, clarus_whole, clarus_split)

        cfp_feature = self.sw_gap(cfp_feature_map).view(cfp_feature_map.size(0), -1)
        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)
        clarus_split_feature = self.sw_gap(clarus_split_feature_map).view(B, I, -1)

        weight_mil = self.cw_att_mil(clarus_split_feature)
        B, I, C = clarus_split_feature.size()
        clarus_split_feature = torch.bmm(weight_mil, clarus_split_feature) # B * 1 * C
        clarus_split_feature = clarus_split_feature.view(B, C)

        cfp_score1 = self.classifier1(cfp_feature)
        cfp_score2 = self.classifier2(cfp_feature)

        clarus_whole_score = self.classifier1(clarus_whole_feature)
        clarus_split_score = self.classifier2(clarus_split_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_split_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_split_score.view(B, 1, -1)), dim=1)
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        if need_feature:
            return cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score, clarus_score, [cfp_feature, clarus_whole_feature, clarus_split_feature]
        else:
            return cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score, clarus_score
    
    def predict_result(self, clarus_whole, clarus_split):
        B, I, C, H, W = clarus_split.size()
        clarus_split = clarus_split.view(B * I, C, H, W)

        clarus_whole_feature_map = self.backbone.forward(clarus_whole)
        clarus_split_feature_map = self.backbone.forward(clarus_split)

        clarus_whole_feature = self.sw_gap(clarus_whole_feature_map).view(clarus_whole_feature_map.size(0), -1)
        clarus_split_feature = self.sw_gap(clarus_split_feature_map).view(B, I, -1)

        weight_mil = self.cw_att_mil(clarus_split_feature)
        B, I, C = clarus_split_feature.size()
        clarus_split_feature = torch.bmm(weight_mil, clarus_split_feature) # B * 1 * C
        clarus_split_feature = clarus_split_feature.view(B, C)

        clarus_whole_score = self.classifier1(clarus_whole_feature)
        clarus_split_score = self.classifier2(clarus_split_feature)
        
        weight_fusion = self.cw_att_fusion(torch.cat((clarus_whole_feature.view(B, 1, -1), clarus_split_feature.view(B, 1, -1)), dim=1))
        clarus_score = torch.cat((clarus_whole_score.view(B, 1, -1), clarus_split_score.view(B, 1, -1)), dim=1)
        
        if self.score_fusion == 'single':
            clarus_score = torch.bmm(weight_fusion, clarus_score) # B * n_class * 2 , B * 2 * n_class
            clarus_score = clarus_score.view(B, -1)
        else:
            clarus_score = clarus_score * torch.transpose(weight_fusion, 2, 1)
            clarus_score = torch.sum(clarus_score, dim=1)

        return clarus_score