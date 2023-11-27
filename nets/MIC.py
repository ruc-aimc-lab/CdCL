from copy import deepcopy
import random
import warnings
import numpy as np

import torch
from torch import nn
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import SGD

from torch.nn.modules.dropout import _DropoutNd
import kornia
from einops import repeat



from .adp_mil import AdaptiveNetSplit, AdaptiveNetWhole, AdaptiveMIL
from .optimizer import Optimizer
from typing import List, Dict, Optional

from .GradientReserveFunction import GRL

from .SDAT import DomainDiscriminator, ConditionalDomainAdversarialLoss, MinimumClassConfusionLoss, BinaryMinimumClassConfusionLoss
from .DomainClassifier import ClassifierGRL


class EMATeacher(nn.Module):

    def __init__(self, model, alpha, pseudo_label_weight):
        super(EMATeacher, self).__init__()
        self.ema_model = deepcopy(model)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    def _init_ema_weights(self, model):
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            #if isinstance(m, DropPath):
            #    m.training = False
        logits = self.ema_model.predict_result(target_img, None)

        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight



warnings.filterwarnings("ignore", category=DeprecationWarning)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def strong_transform(param, data):
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = gaussian_blur(blur=param['blur'], data=data)
    return data


def denorm(img, mean, std):
    return img.mul(std).add(mean)


def renorm(img, mean, std):
    return img.sub(mean).div(std)


def color_jitter(color_jitter, mean, std, data, s=.25, p=.2):
    # s is the strength of colorjitter
    if color_jitter > p:
        mean = torch.as_tensor(mean, device=data.device)
        mean = repeat(mean, 'C -> B C 1 1', B=data.shape[0], C=3)
        std = torch.as_tensor(std, device=data.device)
        std = repeat(std, 'C -> B C 1 1', B=data.shape[0], C=3)
        if isinstance(s, dict):
            seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
        else:
            seq = nn.Sequential(
                kornia.augmentation.ColorJitter(
                    brightness=s, contrast=s, saturation=s, hue=s))
        data = denorm(data, mean, std)
        data = seq(data)
        data = renorm(data, mean, std)
    return data


def gaussian_blur(blur, data):
    if blur > 0.5:
        sigma = np.random.uniform(0.15, 1.15)
        kernel_size_y = int(
            np.floor(
                np.ceil(0.1 * data.shape[2]) - 0.5 +
                np.ceil(0.1 * data.shape[2]) % 2))
        kernel_size_x = int(
            np.floor(
                np.ceil(0.1 * data.shape[3]) - 0.5 +
                np.ceil(0.1 * data.shape[3]) % 2))
        kernel_size = (kernel_size_y, kernel_size_x)
        seq = nn.Sequential(
            kornia.filters.GaussianBlur2d(
                kernel_size=kernel_size, sigma=(sigma, sigma)))
        data = seq(data)
    return data

class Masking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean, std):
        super(Masking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        masked_img = img * input_mask

        return masked_img


class AdaptiveNetWholeMICTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma, eps, training_params, uda=False):
        super().__init__(backbone, n_class, channels)

        self.teacher = EMATeacher(AdaptiveNetWhole(backbone=backbone, n_class=n_class, channels=channels), alpha=0.999, pseudo_label_weight='prob').cuda()
        self.masking = Masking(block_size=32, ratio=0.5, color_jitter_s=0, 
                               color_jitter_p=0, blur=False, mean=0, std=0)

        self.crit_sup = crit_sup
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失，uwf的domain adversarial损失
        self.domain_discri = DomainDiscriminator(channels * n_class, hidden_size=1024)
        self.crit_adv = ConditionalDomainAdversarialLoss(self.domain_discri, entropy_conditioning=False,
                                                         gamma=gamma, eps=eps).cuda()
        self.mcc_loss = MinimumClassConfusionLoss(temperature=2)

        self.ad_opt = Optimizer([self.domain_discri], training_params)
        self.opt = Optimizer([self.backbone, self.classifier], training_params, use_sam=True)

        self.uda = uda
        if self.uda:
            print('uda mode')

    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus, iter_num=None):
        x_t_masked = self.masking(clarus_whole)
        
        self.opt.z_grad()
        self.ad_opt.z_grad()

        self.teacher.update_weights(self, iter_num)
        pseudo_label_t, pseudo_prob_t = self.teacher(clarus_whole)

        y_s, y_t, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        f_s, f_t = features

        loss_cfp = self.crit_sup(y_s, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None and not self.uda:
            loss_claurs = self.crit_sup(y_t, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        y_t_masked = self.backbone.forward(x_t_masked)
        y_t_masked = self.sw_gap(y_t_masked).view(y_t_masked.size(0), -1)
        y_t_masked = self.classifier(y_t_masked)

        # mcc_loss_value = self.mcc_loss(y_t) * self.weights[2]

        mask_loss = F.cross_entropy(y_t_masked, pseudo_label_t, reduction='none')
        mask_loss = torch.mean(pseudo_prob_t * mask_loss) * self.weights[3]
        # loss = loss + mcc_loss_value + mask_loss
        loss = loss + mask_loss
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
        
        transfer_loss = self.crit_adv(y_s, f_s, y_t, f_t) + self.mcc_loss(y_t)

        loss += transfer_loss * self.weights[2]
        loss.backward()
        self.ad_opt.optim.step()
        self.opt.optim.second_step(zero_grad=True)
        self.ad_opt.update_lr()
        self.opt.update_lr()
        
        return y_t, loss, [loss_cfp, loss_claurs], []



class AdaptiveNetWholeMICDANNTraining(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, gamma, mcc, eps, training_params, uda=False):
        super().__init__(backbone, n_class, channels)

        self.teacher = EMATeacher(AdaptiveNetWhole(backbone=backbone, n_class=n_class, channels=channels), alpha=0.999, pseudo_label_weight='prob').cuda()
        self.masking = Masking(block_size=32, ratio=0.5, color_jitter_s=0, 
                               color_jitter_p=0, blur=False, mean=0, std=0)

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
        x_t_masked = self.masking(clarus_whole)
        
        self.opt.z_grad()
        self.ad_opt.z_grad()

        self.teacher.update_weights(self, iter_num)
        pseudo_label_t, pseudo_prob_t = self.teacher(clarus_whole)

        y_s, y_t, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        # f_s, f_t = features

        loss_cfp = self.crit_sup(y_s, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None and not self.uda:
            loss_claurs = self.crit_sup(y_t, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        y_t_masked = self.backbone.forward(x_t_masked)
        y_t_masked = self.sw_gap(y_t_masked).view(y_t_masked.size(0), -1)
        y_t_masked = self.classifier(y_t_masked)

        # mcc_loss_value = self.mcc_loss(y_t) * self.weights[2]

        mask_loss = F.cross_entropy(y_t_masked, pseudo_label_t, reduction='none')
        mask_loss = torch.mean(pseudo_prob_t * mask_loss) * self.weights[3]
        # loss = loss + mcc_loss_value + mask_loss
        loss = loss + mask_loss
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

