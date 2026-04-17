import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from torch.nn import functional as F
import random
from torch.distributions import Uniform


class UBS(nn.Module):

    def __init__(self, p=1.0, rho=3.0, eps=1e-6):
        super().__init__()
        self.p = p
        self.rho = rho
        self.eps = eps

    def __repr__(self):
        return f'UBS(rho={self.rho}, p={self.p})'

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=1, keepdim=False)   # 均值
        var = x.var(dim=1, keepdim=False)   # 方差
        sig = (var + self.eps).sqrt()       # 标准差
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu.reshape(x.shape[0], 1, x.shape[2])) / sig.reshape(x.shape[0], 1, x.shape[2])          # 归一化

        mu_1 = x.mean(dim=1, keepdim=False)
        std_1 = x.std(dim=1, keepdim=False)

        mu_mu = mu_1.mean(dim=0, keepdim=True)
        mu_std = mu_1.std(dim=0, keepdim=True)
        std_mu = std_1.mean(dim=0, keepdim=True)
        std_std = std_1.std(dim=0, keepdim=True)
        mu_std.data.clamp_(min=self.eps)
        std_std.data.clamp_(min=self.eps)

        Distri_mu = Uniform(mu_mu - self.rho * mu_std, mu_mu + self.rho * mu_std)
        Distri_std = Uniform(std_mu - self.rho * std_std, std_mu + self.rho * std_std)

        mu_b = Distri_mu.sample([B, ])
        sig_b = Distri_std.sample([B, ])
        mu_b.reshape(x.shape[0], 1, x.shape[2])
        sig_b.reshape(x.shape[0], 1, x.shape[2])
        # mu_b, sig_b = mu_b.detach(), sig_b.detach()

        return x_normed * sig_b + mu_b