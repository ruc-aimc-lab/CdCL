# FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation
import torch.nn as nn
import torch
from .optimizer import Optimizer
from .MLP import MLP
import math
from .FOVNet import MyAvgPool2d, SelfAttentionBlocks
import torch.nn.functional as F


def get_target_preds(x, th=2):
    top_prob, top_label = torch.topk(F.softmax(x, dim=1), k=1)
    top_label = top_label.squeeze().t()
    top_prob = top_prob.squeeze().t()
    top_mean, top_std = top_prob.mean(), top_prob.std()
    threshold = top_mean - th * top_std
    return top_label, top_prob, threshold


def mixup_criterion_hard(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss().cuda()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_fixmix_loss(net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, ratio):
    mixed_x = ratio * src_imgs + (1 - ratio) * tgt_imgs
    mixed_x = net(mixed_x)
    loss = mixup_criterion_hard(mixed_x, src_labels.detach(), tgt_pseudo.detach(), ratio)
    return loss


class AdaptiveNetWholeFixBi(nn.Module):
    def __init__(self, backbone_source, backbone_target, n_class, channels):
        super(AdaptiveNetWholeFixBi, self).__init__()
        self.backbone = backbone_source
        self.backbone_target = backbone_target

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_target = MLP(in_features=channels, hidden_features=128, out_features=n_class)

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

    def forward(self, x, source:bool):
        if source:
            x = self.backbone(x)
            x = self.sw_gap(x).view(x.size(0), -1)
            x = self.classifier(x)
        else:
            x = self.backbone_target(x)
            x = self.sw_gap(x).view(x.size(0), -1)
            x = self.classifier_target(x)
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


class AdaptiveNetWholeFixBiTraining(AdaptiveNetWholeFixBi):

    def __init__(self, backbone_source, backbone_target, n_class, channels, crit_sup, weights, mix_ratio, cons_reg_start, training_params, uda=False):
        super().__init__(backbone_source, backbone_target, n_class, channels)

        self.mix_ratio = mix_ratio  # (0.5, 1)
        self.cons_reg_start = cons_reg_start

        self.opt = Optimizer([self.backbone, self.backbone_target, self.classifier, self.classifier_target], training_params)
        self.crit_mixup = nn.BCEWithLogitsLoss()
        self.crit_consis = nn.MSELoss()
        self.crit_sup = crit_sup
        self.weights = weights # 长度为5，分别为mix up中cfp主导的损失权重，uwf主导的损失权重，输入uwf在cfp分支的损失权重，在uwf分支的损失权重，两个分支预测一致性损失

        self.uda = uda
        if self.uda:
            print('uda mode')

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

        if gt_clarus is not None and not self.uda:
            loss_sup_mix_cfp = self.crit_sup(score_mix_cfp, gt_cfp) * self.mix_ratio + self.crit_sup(score_mix_cfp, gt_clarus) * (1 - self.mix_ratio)
            loss_sup_mix_clarus = self.crit_sup(score_mix_clarus, gt_clarus) * self.mix_ratio + self.crit_sup(score_mix_clarus, gt_cfp) * (1 - self.mix_ratio)

            cfp_model_clarus_score, clarus_model_clarus_score = self.forward(clarus_whole, True), self.forward(clarus_whole, False)
            
            loss_sup_cfp_model = self.crit_sup(cfp_model_clarus_score, gt_clarus)
            loss_sup_clarus_model = self.crit_sup(clarus_model_clarus_score, gt_clarus)

        else:
            score_mix_cfp, score_mix_clarus = self.forward(mix_cfp, True), self.forward(mix_clarus, False)
            x_sd, x_td = self.forward(clarus_whole, True), self.forward(clarus_whole, False)
            pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(x_sd, th=2)
            pseudo_td, top_prob_td, threshold_td = get_target_preds(x_td, th=2)
            pseudo_sd = F.one_hot(pseudo_sd, num_classes=8).float()
            pseudo_td = F.one_hot(pseudo_td, num_classes=8).float()
            loss_sup_mix_cfp = self.crit_sup(score_mix_cfp, gt_cfp) * self.mix_ratio + self.crit_sup(score_mix_cfp, pseudo_sd.detach()) * (1 - self.mix_ratio)
            loss_sup_mix_clarus = self.crit_sup(score_mix_clarus, pseudo_td.detach()) * self.mix_ratio + self.crit_sup(score_mix_clarus, gt_cfp) * (1 - self.mix_ratio)
            #loss_sup_mix_cfp = 0
            #loss_sup_mix_clarus = 0
            loss_sup_cfp_model = 0
            loss_sup_clarus_model = 0
            clarus_model_clarus_score = x_td

        loss = loss_sup_mix_cfp * self.weights[0] + loss_sup_mix_clarus * self.weights[1] + \
               loss_sup_cfp_model * self.weights[2] + loss_sup_clarus_model * self.weights[3]

        '''if self.uda and iter_num and iter_num >= self.cons_reg_start:
            bim_mask_sd = torch.ge(top_prob_sd, threshold_sd)
            bim_mask_sd = torch.nonzero(bim_mask_sd).squeeze()

            bim_mask_td = torch.ge(top_prob_td, threshold_td)
            bim_mask_td = torch.nonzero(bim_mask_td).squeeze()
            if bim_mask_sd.dim() > 0 and bim_mask_td.dim() > 0:
                if bim_mask_sd.numel() > 0 and bim_mask_td.numel() > 0:
                    bim_mask = min(bim_mask_sd.size(0), bim_mask_td.size(0))
                    bim_sd_loss = self.crit_sup(x_sd[bim_mask_td[:bim_mask]], pseudo_td[bim_mask_td[:bim_mask]].cuda().detach())
                    bim_td_loss = self.crit_sup(x_td[bim_mask_sd[:bim_mask]], pseudo_sd[bim_mask_sd[:bim_mask]].cuda().detach())

                    loss += bim_sd_loss * self.weights[0]
                    loss += bim_td_loss * self.weights[0]

        if iter_num and iter_num >= self.cons_reg_start:
            mean_mix = 0.5 * new_cfp + 0.5 * clarus_whole
            score_mean_mix_cfp_model, score_mean_mix_clarus_model = self.forward(mean_mix, True), self.forward(mean_mix, False)
            cons_loss = self.crit_consis(score_mean_mix_cfp_model, score_mean_mix_clarus_model)
            loss += cons_loss * self.weights[4]
        else:
            cons_loss = None'''
        
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_model_clarus_score, loss, [loss_sup_mix_cfp, loss_sup_mix_clarus, loss_sup_cfp_model, loss_sup_clarus_model], []


    def predict_result(self, clarus_whole, clarus_split):
        return self.forward(clarus_whole, False)
    

class VITFixBi(nn.Module):
    def __init__(self, backbone_source, backbone_target, n_class, channels, mhsa_nums=0, mil_ratio=3, over_lap=True):
        super(VITFixBi, self).__init__()
        self.backbone = backbone_source
        self.backbone_target = backbone_target

        if mhsa_nums >= 1:
            self.mhsa = SelfAttentionBlocks(dim=channels, block_num=mhsa_nums, num_heads=8, drop_rate=0.2)
        else:
            self.mhsa = nn.Identity()

        self.sw_mil_pooling = MyAvgPool2d(ratio=mil_ratio, over_lap=over_lap)

        self.mil_ratio = mil_ratio

        self.sw_gap = nn.AdaptiveAvgPool2d(1) # spatial-wise global average pooling
        self.classifier = MLP(in_features=channels, hidden_features=128, out_features=n_class)
        self.classifier_target = MLP(in_features=channels, hidden_features=128, out_features=n_class)

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

    def forward(self, x, source:bool):
        if source:
            x = self.backbone(x)
            x = self.sw_gap(x).view(x.size(0), -1)
            x = self.classifier(x)
        else:
            x = self.backbone_target(x)
            x = self.sw_mil_pooling(x).flatten(2).transpose(1, 2)
            x = self.mhsa(x)[:, 0]
            # x = self.sw_gap(x).view(x.size(0), -1)
            x = self.classifier_target(x)
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


class VITFixBiTraining(VITFixBi):
    def __init__(self, backbone_source, backbone_target, n_class, channels, crit_sup, weights, mix_ratio, cons_reg_start, training_params, mhsa_nums=0, mil_ratio=3, over_lap=True):
        super().__init__(backbone_source, backbone_target, n_class, channels, mhsa_nums=mhsa_nums, mil_ratio=mil_ratio, over_lap=over_lap)

        self.mix_ratio = mix_ratio  # (0.5, 1)
        self.cons_reg_start = cons_reg_start

        self.opt = Optimizer([self.backbone, self.backbone_target, self.classifier, self.classifier_target, self.mhsa], training_params)
        self.crit_mixup = nn.BCEWithLogitsLoss()
        self.crit_consis = nn.MSELoss()
        self.crit_sup = crit_sup
        self.weights = weights # 长度为5，分别为mix up中cfp主导的损失权重，uwf主导的损失权重，输入uwf在cfp分支的损失权重，在uwf分支的损失权重，两个分支预测一致性损失

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

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return clarus_molde_clarus_score, loss, [loss_sup_mix_cfp, loss_sup_mix_clarus, loss_sup_cfp_model, loss_sup_clarus_model], []

    def predict_result(self, clarus_whole, clarus_split):
        return self.forward(clarus_whole, False)
    


