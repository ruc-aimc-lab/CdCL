from torch import nn
import torch.nn.functional as F
import torch

def binary_focal_loss(pred, target, alpha=0.5, gamma=2):
    assert pred.size() == target.size()
    pred = torch.sigmoid(pred)
    e = 1e-5
    loss = alpha * target * (1 - pred) ** gamma * (pred + e).log() + (1 - alpha) * (1 - target) * pred ** gamma * (1 - pred + e).log()
    loss = loss / (0.5 ** gamma)
    return -loss.mean()


class BFLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(BFLoss, self).__init__()
        # alpha: the weight of fg
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, *args, **kwargs):
        return binary_focal_loss(pred, target, alpha=self.alpha, gamma=self.gamma)
        