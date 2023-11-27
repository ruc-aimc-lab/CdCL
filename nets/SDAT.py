# CDAN+MCC+SDAT
# FREE LUNCH FOR DOMAIN ADVERSARIAL TRAINING- ENVIRONMENT LABEL SMOOTHING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adp_mil import AdaptiveNetWhole
from .optimizer import Optimizer

from .GradientReserveFunction import GRL
from .CDAN import entropy, DomainDiscriminator, ConditionalDomainAdversarialLoss
from .DomainClassifier import ClassifierGRL

class MinimumClassConfusionLoss(nn.Module):
    r"""
    Minimum Class Confusion loss minimizes the class confusion in the target predictions.

    You can see more details in `Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020) <https://arxiv.org/abs/1912.03699>`_

    Args:
        temperature (float) : The temperature for rescaling, the prediction will shrink to vanilla softmax if
          temperature is 1.0.

    .. note::
        Make sure that temperature is larger than 0.

    Inputs: g_t
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`

    Shape:
        - g_t: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    Examples::
        >>> temperature = 2.0
        >>> loss = MinimumClassConfusionLoss(temperature)
        >>> # logits output from target domain
        >>> g_t = torch.randn(batch_size, num_classes)
        >>> output = loss(g_t)

    MCC can also serve as a regularizer for existing methods.
    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> temperature = 2.0
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim, hidden_size=1024)
        >>> cdan_loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> mcc_loss = MinimumClassConfusionLoss(temperature)
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> total_loss = cdan_loss(g_s, f_s, g_t, f_t) + mcc_loss(g_t)
    """

    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss
    

class BinaryMinimumClassConfusionLoss(nn.Module):
    """
    用于多标签分类
    """

    def __init__(self, temperature: float):
        super(BinaryMinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.sigmoid(logits / self.temperature).unsqueeze(dim=2) # batch_size x num_classes x 1
        reverse_predictions = 1 - predictions
        predictions = torch.cat([predictions, reverse_predictions], dim=2) # batch_size x num_classes x 2
        predictions = predictions.view(-1, 2)  # (batch_size x num_classes) x 2

        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * num_classes * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # (batch_size x num_classes) x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # 2 x 2
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / 2
        return mcc_loss
    
    def backup(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes

        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1

        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss 


class AdaptiveNetWholeSDATTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma, eps, training_params, uda=False, mcc=None):
        super().__init__(backbone, n_class, channels)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失，uwf的domain adversarial损失
        self.domain_discri = DomainDiscriminator(channels * n_class, hidden_size=1024)
        self.crit_adv = ConditionalDomainAdversarialLoss(self.domain_discri, entropy_conditioning=False,
                                                         gamma=gamma, eps=eps).cuda()
        
        self.mcc = mcc
        if self.mcc:
            if self.mcc == 'mcc':
                self.mcc_loss = MinimumClassConfusionLoss(temperature=2)
            elif self.mcc == 'bmcc':
                self.mcc_loss = BinaryMinimumClassConfusionLoss(temperature=2)
            else:
                raise Exception('invalid mcc mode', mcc)

        self.ad_opt = Optimizer([self.domain_discri], training_params)
        self.opt = Optimizer([self.backbone, self.classifier], training_params, use_sam=True)

        self.uda = uda
        if self.uda:
            print('uda mode')

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        self.ad_opt.z_grad()

        y_s, y_t, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        f_s, f_t = features

        loss_cfp = self.crit_sup(y_s, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None and not self.uda:
            loss_claurs = self.crit_sup(y_t, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        # mcc_loss_value = self.mcc_loss(y_t) * self.weights[2]
        # loss = loss + mcc_loss_value
        loss.backward()
        self.opt.optim.first_step(zero_grad=True)

        y_s, y_t, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        f_s, f_t = features

        loss_cfp = self.crit_sup(y_s, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None and not self.uda:
            loss_claurs = self.crit_sup(y_t, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp
        
        if self.mcc:
            transfer_loss = self.crit_adv(y_s, f_s, y_t, f_t) + self.mcc_loss(y_t)
        else:
            transfer_loss = self.crit_adv(y_s, f_s, y_t, f_t)

        loss += transfer_loss * self.weights[2]
        loss.backward()
        self.ad_opt.optim.step()
        self.opt.optim.second_step(zero_grad=True)
        self.ad_opt.update_lr()
        self.opt.update_lr()
        
        return y_t, loss, [loss_cfp, loss_claurs], []


class AdaptiveNetWholeSDATDANNTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma, eps, training_params, uda=False, mcc=None):
        super().__init__(backbone, n_class, channels)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失，uwf的domain adversarial损失
        self.domain_discri = ClassifierGRL(channels, gamma=gamma, n_class=1)
        self.crit_adv = nn.BCEWithLogitsLoss()
        self.eps = eps
        
        self.mcc = mcc
        if self.mcc:
            if self.mcc == 'mcc':
                self.mcc_loss = MinimumClassConfusionLoss(temperature=2)
            elif self.mcc == 'bmcc':
                self.mcc_loss = BinaryMinimumClassConfusionLoss(temperature=2)
            else:
                raise Exception('invalid mcc mode', mcc)

        self.ad_opt = Optimizer([self.domain_discri], training_params)
        self.opt = Optimizer([self.backbone, self.classifier], training_params, use_sam=True)

        self.uda = uda
        if self.uda:
            print('uda mode')

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        self.opt.z_grad()
        self.ad_opt.z_grad()

        y_s, y_t, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        # f_s, f_t = features

        loss_cfp = self.crit_sup(y_s, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None and not self.uda:
            loss_claurs = self.crit_sup(y_t, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        # mcc_loss_value = self.mcc_loss(y_t) * self.weights[2]
        # loss = loss + mcc_loss_value
        loss.backward()
        self.opt.optim.first_step(zero_grad=True)

        y_s, y_t, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        # f_s, f_t = features

        loss_cfp = self.crit_sup(y_s, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None and not self.uda:
            loss_claurs = self.crit_sup(y_t, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp
        
        cfp_feature, clarus_whole_feature = features
        cfp_domain_score = self.domain_discri(cfp_feature)
        clarus_whole_domain_score = self.domain_discri(clarus_whole_feature)
        domain_score = torch.cat((cfp_domain_score, clarus_whole_domain_score), dim=0)
        
        gt_cfp_domian = torch.ones((cfp_domain_score.size(0), 1)).cuda()  * (1-self.eps)
        gt_clarus_whole_domian = torch.ones((clarus_whole_domain_score.size(0), 1)).cuda() * self.eps
        domain_gt = torch.cat((gt_cfp_domian, gt_clarus_whole_domian), dim=0)

        if self.mcc:
            transfer_loss = self.crit_adv(domain_score, domain_gt) + self.mcc_loss(y_t)
        else:
            transfer_loss = self.crit_adv(domain_score, domain_gt)

        loss += transfer_loss * self.weights[2]
        loss.backward()
        self.ad_opt.optim.step()
        self.opt.optim.second_step(zero_grad=True)
        self.ad_opt.update_lr()
        self.opt.update_lr()
        
        return y_t, loss, [loss_cfp, loss_claurs], []

