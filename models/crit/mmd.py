import torch
import torch.nn as nn
from math import gcd


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

class MMDLinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fea_source, fea_target):
        n_s, d_s = fea_source.size()
        n_t, d_t = fea_target.size()

        assert d_s == d_t

        if n_s != n_t:
            n = int(n_s * n_t / gcd(n_s, n_t)) # 最小公倍数

            fea_source = fea_source.repeat((int(n / n_s), 1))
            fea_target = fea_target.repeat((int(n / n_t), 1))
        return mmd_linear(fea_source, fea_target)


