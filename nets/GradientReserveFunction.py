# 梯度反转层
import torch
import numpy as np

class GradReverse(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambda_, None

class GRL(torch.nn.Module):
    """
    gradient reverse layer
    """
    def __init__(self, gamma) -> None:
        super().__init__()
        self.schedule = GradSchedule(0, 1, gamma)
    
    def forward(self, x):
        lambda_ = self.schedule()
        return GradReverse.apply(x, lambda_)

class GradSchedule(object):
    # easydl.common.scheduler.aToBSheduler
    # gamma = easydl.common.scheduler.aToBSheduler.gamma / easydl.common.scheduler.aToBSheduler.max_iter
    def __init__(self, a, b, gamma=1e-3) -> None:
        self.a = a
        self.b = b
        self.gamma = gamma
        self.step = 0 

    def __call__(self):
        ans = self.a + (2. / (1 + np.exp(-self.gamma * self.step)) - 1.0) * (self.b - self.a)
        self.step += 1
        return torch.tensor(ans)

    def reset_step(self):
        self.step = 0
