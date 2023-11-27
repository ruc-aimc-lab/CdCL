# Deep Domain Confusion-Maximizing for Domain Invariance
# done
from .adp_mil import AdaptiveNetSplit, AdaptiveNetWhole, AdaptiveMIL
from .crit import MMDLinear
from .optimizer import Optimizer
from copy import deepcopy

"""
class AdaptiveMILDDCTraining(AdaptiveMIL):
    def __init__(self, backbone, n_class, channels, score_fusion, crit_sup, crit_cons, weights):
        super().__init__(backbone, n_class, channels, score_fusion)
        self.crit_sup = crit_sup
        self.crit_cons = crit_cons
        self.weights = weights # 长度为4，分别是cfp的监督损失，全局-局部一致性损失，uwf的监督损失，MMD损失
        self.crit_ddc = MMDLinear()
    
    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus):
        cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score, clarus_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        
        loss_cfp1 = self.crit_sup(cfp_score1, gt_cfp) * self.weights[0] * 0.5
        loss_cfp2 = self.crit_sup(cfp_score2, gt_cfp) * self.weights[0] * 0.5
        
        loss_consistency = self.crit_cons(clarus_whole_score, clarus_split_score) * self.weights[1]
        loss = loss_cfp1 + loss_cfp2 + loss_consistency

        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_score, gt_clarus) * self.weights[2]
            loss += loss_claurs
        else:
            loss_claurs = None

        cfp_feature, clarus_whole_feature, clarus_split_feature = features

        loss_ddc = self.weights[3] * 0.5 * self.crit_ddc(clarus_split_feature, cfp_feature) + \
                   self.weights[3] * 0.5 * self.crit_ddc(clarus_whole_feature, cfp_feature)
        loss += loss_ddc
        
        return clarus_score, loss, [loss_cfp1, loss_cfp2, loss_consistency, loss_claurs, loss_ddc], [cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score]


class AdaptiveNetSplitDDCTraining(AdaptiveNetSplit):
    def __init__(self, backbone, n_class, channels, crit_sup, weights):
        super().__init__(backbone, n_class, channels)
        self.crit_sup = crit_sup
        self.crit_ddc = MMDLinear()
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失， MMD损失
    
    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus):
        cfp_score, clarus_split_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_split_score, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        cfp_feature, clarus_split_feature = features

        loss_ddc = self.weights[2] * self.crit_ddc(clarus_split_feature, cfp_feature)
        loss += loss_ddc
        return clarus_split_score, loss, [loss_cfp, loss_claurs, loss_ddc], [cfp_score]
"""

class AdaptiveNetWholeDDCTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, training_params, uda=False):
        super().__init__(backbone, n_class, channels)
        self.crit_sup = crit_sup
        self.crit_ddc = MMDLinear()
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失,，MMD损失

        params_backbone = deepcopy(training_params)
        params_backbone['lr'] /= 10
        params_backbone['schedule_params']['eta_min'] /= 10
        self.opt = Optimizer([self.backbone], params_backbone)

        params_classifier = deepcopy(training_params)    
        self.opt_classifier = Optimizer([self.classifier], params_classifier)     

        assert params_classifier != params_backbone   

        self.uda = uda
        if self.uda:
            print('uda mode')
    
    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        self.opt_classifier.z_grad()

        cfp_score, clarus_whole_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None and not self.uda:
            loss_claurs = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        cfp_feature, clarus_whole_feature = features
        loss_ddc = self.weights[2] * self.crit_ddc(clarus_whole_feature, cfp_feature)
        loss += loss_ddc

        loss.backward()

        self.opt.g_step()
        self.opt_classifier.g_step()

        self.opt.z_grad()
        self.opt_classifier.z_grad()

        self.opt.update_lr()
        self.opt_classifier.update_lr()
        
        return clarus_whole_score, loss, [loss_cfp, loss_claurs, loss_ddc], [cfp_score]
