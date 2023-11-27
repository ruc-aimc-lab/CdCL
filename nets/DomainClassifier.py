import torch
import torch.nn as nn
from .GradientReserveFunction import GRL


class ClassifierGRL(nn.Module):
    """
    Classifier with a gredient reverse layer.
    """
    def __init__(self, in_feature, gamma, n_class=2):
        super(ClassifierGRL, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 128),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(128, n_class)
        )
        self.grl = GRL(gamma=gamma)

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y



class Classifier(nn.Module):
    """
    Classifier without gredient reverse layer.
    """
    def __init__(self, in_feature, n_class=2):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 128),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(128, n_class)
        )

    def forward(self, x):
        return self.main(x)