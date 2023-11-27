# Universal Adaptation Network
import torch
import torch.nn as nn
from .GradientReserveFunction import GRL
from .adp_mil import AdaptiveNetSplit, AdaptiveNetWhole, AdaptiveMIL
from .optimizer import Optimizer


class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 128),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.grl = GRL(gamma=1e-3)

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y


class UAN(nn.Module):
    def __init__(self, backbone, backbone_out_dim, source_class) -> None:
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone_out_dim, 256)
        self.classifier = nn.Linear(256, source_class)
        self.discriminator = AdversarialNetwork(backbone_out_dim)
        self.discriminator_adv = AdversarialNetwork(backbone_out_dim)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        cls = self.classifier(x)
        d = self.discriminator(x)
        d_adv = self.discriminator_adv(x)
        return cls, d, d_adv


class AdaptiveNetWholeUANTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma, training_params):
        super().__init__(backbone, n_class, channels)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失，uwf的domain adversarial损失
        
        self.discriminator = AdversarialNetwork(channels)
        self.discriminator_adv = AdversarialNetwork(channels)

        self.crit_domain_classifier = nn.BCEWithLogitsLoss()

        self.opt = Optimizer([self.backbone, self.classifier, self.domain_classifier], training_params)

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus):
        self.opt.z_grad()
        
        cfp_score, clarus_whole_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        
        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        cfp_feature, clarus_whole_feature = features
        cfp_domain_score = self.domain_classifier(cfp_feature)
        clarus_whole_domain_score = self.domain_classifier(clarus_whole_feature)
        
        gt_cfp_domian = torch.zeros_like(cfp_domain_score)
        gt_clarus_whole_domian = torch.ones_like(clarus_whole_domain_score)

        loss_adv = self.weights[2] * self.crit_domain_classifier(clarus_whole_domain_score, gt_clarus_whole_domian) + \
                   self.weights[2] * self.crit_domain_classifier(cfp_domain_score, gt_cfp_domian) 
        loss_adv *= 0.5
        loss += loss_adv
        
        loss.backward()
        self.opt.g_step()
        self.opt.z_grad()
        self.opt.update_lr()
        
        return clarus_whole_score, loss, [loss_cfp, loss_claurs, loss_adv], [cfp_score]