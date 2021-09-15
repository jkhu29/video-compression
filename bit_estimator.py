import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BitParm(nn.Module):
    """
    save params
    """
    def __init__(self, channel, final=False):
        super(BitParm, self).__init__()
        self.final = final
        self.h = nn.Parameter(nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    """
    Estimate bit
    """
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = BitParm(channel)
        self.f2 = BitParm(channel)
        self.f3 = BitParm(channel)
        self.f4 = BitParm(channel, final=True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


class EntropyBlock(nn.Module):
    """docstring for EntropyBLock"""
    def __init__(self, channel: int = 64):
        super(EntropyBlock, self).__init__()
        self.bitestimator = BitEstimator(channel)

    def forward(self, x):
        prob = self.bitestimator(x + 0.5) - self.bitestimator(x - 0.5)
        total_bits = torch.sum(torch.clamp(-1 * torch.log(prob + 1e-8) / math.log(2.0), min=1e-10, max=50.))
        return total_bits, prob
