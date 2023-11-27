from .adp_mil import AdaptiveNetSplit, AdaptiveNetWhole, AdaptiveMIL, Linear
from .crit import MMDLinear
from .optimizer import Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveNetWholeDC(AdaptiveNetWhole):
    def __init__(self, backbone, n_class, channels):
        super().__init__(backbone, n_class, channels)
        self.domain_classifier = Linear(channel_in=channels, channel_out=2)


class AdaptiveNetWholeDCTraining(AdaptiveNetWholeDC):
    def __init__(self, backbone, n_class, channels, crit_sup, weights, training_params):
        super().__init__(backbone, n_class, channels)
        self.opt = Optimizer([self.backbone, self.classifier], training_params)
        self.opt_conf = Optimizer([self.backbone], training_params)
        self.opt_dm = Optimizer([self.domain_classifier], training_params)

        self.crit_sup = crit_sup
        self.crit_domain = nn.CrossEntropyLoss()
        
        self.weights = weights # 长度为3，分别是cfp的监督损失，uwf的监督损失, domain confusion loss
    
    def train_model(self, cfp, clarus_whole, clarus_split, gt_cfp, gt_clarus):

        # 优化backbone和classifier
        self.opt.z_grad()

        cfp_score, clarus_whole_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        cfp_feature, clarus_whole_feature = features

        loss_cfp = self.crit_sup(cfp_score, gt_cfp) * self.weights[0]
        
        if gt_clarus is not None:
            loss_claurs = self.crit_sup(clarus_whole_score, gt_clarus) * self.weights[1]
            loss = loss_claurs + loss_cfp
        else:
            loss_claurs = None
            loss = loss_cfp

        # loss.backward(retain_graph=True)
        loss.backward()
        self.opt.g_step()
        

        # 优化domain classifier
        self.opt_dm.z_grad()

        src_label_dm = torch.ones(cfp.size(0)).long().cuda()
        tgt_label_dm = torch.zeros(clarus_whole.size(0)).long().cuda()

        src_output_dm = self.domain_classifier(cfp_feature.detach())
        tgt_output_dm = self.domain_classifier(clarus_whole_feature.detach())
        loss_dm_src = self.crit_domain(src_output_dm, src_label_dm)
        loss_dm_tgt = self.crit_domain(tgt_output_dm, tgt_label_dm)
        loss_dm = (loss_dm_src + loss_dm_tgt) * self.weights[2]
        loss_dm.backward()
        self.opt_dm.g_step()

        # 优化
        self.opt_conf.z_grad()
        cfp_score, clarus_whole_score, features = self.forward(cfp, clarus_whole, clarus_split, need_feature=True) 
        cfp_feature, clarus_whole_feature = features
        feature_concat =  torch.cat((cfp_feature, clarus_whole_feature), 0)
        output_dm_conf = self.domain_classifier(feature_concat)
        output_dm_conf = F.softmax(output_dm_conf, dim=1)
        uni_distrib = torch.FloatTensor(output_dm_conf.size()).uniform_(0, 1).cuda()
        loss_conf = - (torch.sum(uni_distrib * torch.log(output_dm_conf)))/float(output_dm_conf.size(0)) * self.weights[2]

        #(loss + loss_conf).backward()
        #self.opt.g_step()
        loss_conf.backward()
        self.opt_conf.g_step()
        
        
        self.opt.update_lr()
        self.opt_dm.update_lr()
        self.opt_conf.update_lr()
        return clarus_whole_score, loss, [loss_cfp, loss_claurs, loss_dm, loss_conf], [cfp_score, clarus_whole_score]