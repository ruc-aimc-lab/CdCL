# Bridging Theory and Algorithm for Domain Adaptation
import torch.nn as nn
from .adp_mil import AdaptiveNetWhole
from .optimizer import Optimizer
from .MLP import MLPGRL, MLP
from .FOVNet import MyAvgPool2d, SelfAttentionBlocks


class AdaptiveNetWholeMDDTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma, training_params, uda=False):
        super().__init__(backbone, n_class, channels)

        self.crit_sup = crit_sup
        self.crit_adv = nn.BCEWithLogitsLoss()
        self.weights = weights # 长度为4，分别是cfp监督损失，uwf监督损失，cfp adv损失，uwf adv损失
        self.classifier_adv = MLPGRL(in_features=channels, hidden_features=128, out_features=n_class, gamma=gamma)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_adv], training_params)

        self.uda = uda
        if self.uda:
            print('uda mode')

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, clarus_whole_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None and not self.uda:
            loss_claurs = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp
        
        gt_adv_cfp = cfp_score.detach().clone()
        gt_adv_clarus = clarus_whole_score.detach().clone()

        gt_adv_cfp[gt_adv_cfp>0] = 1
        gt_adv_cfp[gt_adv_cfp<=0] = 0
        # gt_adv_cfp = gt_adv_cfp.long()

        gt_adv_clarus[gt_adv_clarus>0] = 1
        gt_adv_clarus[gt_adv_clarus<=0] = 0
        # gt_adv_clarus = gt_adv_clarus.long()
        
        cfp_feature, clarus_whole_feature = features
        cfp_score_adv = self.classifier_adv(cfp_feature)
        clarus_score_adv = self.classifier_adv(clarus_whole_feature)
        
        loss_cfp_adv = self.crit_adv(cfp_score_adv, gt_adv_cfp) * self.weights[2]

        loss_claurs_adv = self.crit_adv(-clarus_score_adv, gt_adv_clarus) * self.weights[3]
        
        loss += loss_cfp_adv + loss_claurs_adv
        
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_whole_score, loss, [loss_cfp, loss_claurs, loss_cfp_adv, loss_claurs_adv], [cfp_score]



class VITMDDNet(nn.Module):
    def __init__(self, backbone:nn.Module, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True):
        super(VITMDDNet, self).__init__()
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


class VITMDDTraining(VITMDDNet):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True):
        super().__init__(backbone, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap)

        self.crit_sup = crit_sup
        self.crit_adv = nn.BCEWithLogitsLoss()
        self.weights = weights # 长度为4，分别是cfp监督损失，uwf监督损失，cfp adv损失，uwf adv损失
        self.classifier_adv = MLPGRL(in_features=channels, hidden_features=128, out_features=n_class, gamma=gamma)

        self.opt = Optimizer([self.backbone, self.classifier, self.classifier_adv, self.mhsa], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        
        cfp_score, clarus_whole_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp
        
        gt_adv_cfp = cfp_score.detach().clone()
        gt_adv_clarus = clarus_whole_score.detach().clone()

        gt_adv_cfp[gt_adv_cfp>0] = 1
        gt_adv_cfp[gt_adv_cfp<=0] = 0
        # gt_adv_cfp = gt_adv_cfp.long()

        gt_adv_clarus[gt_adv_clarus>0] = 1
        gt_adv_clarus[gt_adv_clarus<=0] = 0
        # gt_adv_clarus = gt_adv_clarus.long()
        
        cfp_feature, clarus_whole_feature = features
        cfp_score_adv = self.classifier_adv(cfp_feature)
        clarus_score_adv = self.classifier_adv(clarus_whole_feature)
        
        loss_cfp_adv = self.crit_adv(cfp_score_adv, gt_adv_cfp) * self.weights[2]

        loss_claurs_adv = self.crit_adv(-clarus_score_adv, gt_adv_clarus) * self.weights[3]
        
        loss += loss_cfp_adv + loss_claurs_adv
        
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_whole_score, loss, [loss_cfp, loss_claurs, loss_cfp_adv, loss_claurs_adv], [cfp_score]
