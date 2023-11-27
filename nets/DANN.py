# Domain-Adversarial Training of Neural Networks
# done
import torch
import torch.nn as nn
from .adp_mil import AdaptiveNetSplit, AdaptiveNetWhole, AdaptiveMIL
from .optimizer import Optimizer
from .DomainClassifier import ClassifierGRL

"""
class AdaptiveMILDANNTraining(AdaptiveMIL):
    def __init__(self, backbone, n_class, channels, score_fusion, crit_sup, crit_cons, weights, gamma):
        super().__init__(backbone, n_class, channels, score_fusion)
        self.crit_sup = crit_sup
        self.crit_cons = crit_cons
        self.weights = weights # 长度为5，分别是cfp的监督损失，全局-局部一致性损失，uwf的监督损失，整图uwf的domain adversarial损失，切分图uwf的domain adversarial损失
        self.domain_classifier = DomainClassifier(channels, gamma=gamma)
        self.crit_domain_classifier = nn.BCEWithLogitsLoss()
    
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
        cfp_domain_score = self.discriminator_adv(cfp_feature)
        clarus_split_domain_score = self.discriminator_adv(clarus_split_feature)
        clarus_whole_domain_score = self.discriminator_adv(clarus_whole_feature)
        
        gt_cfp_domian = torch.zeros_like(cfp_domain_score)
        gt_clarus_split_domian = torch.ones_like(clarus_split_domain_score)
        gt_clarus_whole_domian = torch.ones_like(clarus_whole_domain_score)

        loss_adv = self.weights[3] * self.crit_domain_classifier(clarus_whole_domain_score, gt_clarus_whole_domian) + \
                   self.weights[4] * self.crit_domain_classifier(clarus_split_domain_score, gt_clarus_split_domian) + \
                   (self.weights[3] + self.weights[4]) * self.crit_domain_classifier(cfp_domain_score, gt_cfp_domian) 
        loss_adv *= 0.5
        loss += loss_adv
        
        return clarus_score, loss, [loss_cfp1, loss_cfp2, loss_consistency, loss_claurs, loss_adv], [cfp_score1, cfp_score2, clarus_whole_score, clarus_split_score]
    

class AdaptiveNetSplitDANNTraining(AdaptiveNetSplit):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma):
        super().__init__(backbone, n_class, channels)
        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失，uwf的domain adversarial损失
        self.domain_classifier = DomainClassifier(channels, gamma=gamma)
        self.crit_domain_classifier = nn.BCEWithLogitsLoss()
    
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
        cfp_domain_score = self.discriminator_adv(cfp_feature)
        clarus_split_domain_score = self.discriminator_adv(clarus_split_feature)
        
        gt_cfp_domian = torch.zeros_like(cfp_domain_score)
        gt_clarus_split_domian = torch.ones_like(clarus_split_domain_score)

        loss_adv = self.weights[2] * self.crit_domain_classifier(clarus_split_domain_score, gt_clarus_split_domian) + \
                   self.weights[2] * self.crit_domain_classifier(cfp_domain_score, gt_cfp_domian) 
        loss_adv *= 0.5
        loss += loss_adv
        
        return clarus_split_score, loss, [loss_cfp, loss_claurs, loss_adv], [cfp_score]
"""

class AdaptiveNetWholeDANNTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma, training_params, uda=False):
        super().__init__(backbone, n_class, channels)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失，uwf的domain adversarial损失
        self.domain_classifier = ClassifierGRL(channels, gamma=gamma)
        self.crit_domain_classifier = nn.CrossEntropyLoss()

        self.opt = Optimizer([self.backbone, self.classifier, self.domain_classifier], training_params)
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

        cfp_feature, clarus_whole_feature = features
        cfp_domain_score = self.domain_classifier(cfp_feature)
        clarus_whole_domain_score = self.domain_classifier(clarus_whole_feature)

        domain_score = torch.cat((cfp_domain_score, clarus_whole_domain_score), dim=0)
        
        gt_cfp_domian = torch.ones(cfp_domain_score.size(0)).long().cuda()
        gt_clarus_whole_domian = torch.zeros(clarus_whole_domain_score.size(0)).long().cuda()
        domain_gt = torch.cat((gt_cfp_domian, gt_clarus_whole_domian), dim=0)

        loss_adv = self.weights[2] * self.crit_domain_classifier(domain_score, domain_gt)
        loss += loss_adv
        
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()
        
        return clarus_whole_score, loss, [loss_cfp, loss_claurs, loss_adv], [cfp_score]
