import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models


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
        if self.training:
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            x += noise
        else:
            x = torch.round(x)
        prob = self.bitestimator(x + 0.5) - self.bitestimator(x - 0.5)
        total_bits = torch.sum(torch.clamp(-1 * torch.log(prob + 1e-8) / math.log(2.0), min=1e-10, max=50.))
        return prob, total_bits, x


class Memory_Attention_Layer(nn.Module):
    def __init__(
            self,
            input_channels: int = 64,
            hidden_channels: int = 64,
            num_heads: int = 8
    ):
        super(Memory_Attention_Layer, self).__init__()
        assert input_channels % num_heads == 0
        self.num_heads = num_heads
        self.single_head_channel = input_channels // num_heads
        self.single_hidden_channel = hidden_channels // num_heads
        self.Keys = []
        self.Queries = []
        self.Values = []
        for i in range(num_heads):
            self.Keys.append(nn.Linear(self.single_head_channel, self.single_hidden_channel))
            self.Queries.append(nn.Linear(self.single_head_channel, self.single_hidden_channel))
            self.Values.append(nn.Linear(self.single_head_channel, self.single_head_channel))
        self.Keys = nn.ModuleList(self.Keys)
        self.Queries = nn.ModuleList(self.Queries)
        self.Values = nn.ModuleList(self.Values)
        self.softmax = nn.Softmax(dim=-1)
        self.layernorm0x = nn.LayerNorm([input_channels])
        self.layernorm1x = nn.LayerNorm([input_channels])
        self.layernorm0k = nn.LayerNorm([hidden_channels])
        self.layernorm1k = nn.LayerNorm([hidden_channels])
        self.layernorm0q = nn.LayerNorm([hidden_channels])
        self.layernorm1q = nn.LayerNorm([hidden_channels])
        self.FFNx = nn.Sequential(
            nn.Linear(input_channels, input_channels), nn.ReLU(inplace=True),
            nn.Linear(input_channels, input_channels)
        )
        self.FFNk = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.FFNq = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.learnable_position_emb = nn.Embedding(256, input_channels)

    def get_kqv(self, x):
        # x.shape (N,S,input_channels)
        K = []
        Q = []
        V = []
        for i in range(self.num_heads):
            x_single_head = x[..., self.single_head_channel * i:self.single_head_channel * (i + 1)]
            K.append(self.Keys[i](x_single_head))
            # (N, S, H/num)
            Q.append(self.Queries[i](x_single_head))
            # (N, S, H/num)
            V.append(self.Values[i](x_single_head))
            # (N, S, C/num)

        return torch.cat(K, dim=-1), torch.cat(Q, dim=-1), torch.cat(V, dim=-1)

    def cross_attention(self, k, q, v, cls="x"):
        # k (N, S, H)
        # q (N, S, H)
        # v (N, S, C)
        _, _, h = k.shape
        k_T = k.permute((0, 2, 1))  # (N,H,S)
        weights = self.softmax(torch.bmm(q, k_T)) / np.sqrt(h)  # (N,S,S)
        output = torch.bmm(weights, v)  # (N,S,C)
        norm0 = getattr(self, 'layernorm0{}'.format(cls))
        norm1 = getattr(self, 'layernorm1{}'.format(cls))
        FFN = getattr(self, 'FFN{}'.format(cls))
        z = norm0(output + v)
        return norm1(FFN(z) + z)

    def forward(self, x):
        # x shape (N, F, S, C)
        _, num, length, _ = x.shape
        loc_emb = self.learnable_position_emb(
            torch.arange(length).to(x.device)
        ).view(1, 1, length, -1)
        x += loc_emb
        k_memory, q_memory, v = self.get_kqv(x[:, 0, ...])
        V = [v.unsqueeze(1)]
        K = [k_memory.unsqueeze(1)]
        Q = [q_memory.unsqueeze(1)]
        for i in range(1, num):
            k, q, v = self.get_kqv(x[:, i, ...])
            k = self.cross_attention(k, q_memory, k, cls='k')
            q = self.cross_attention(k_memory, q, q, cls='q')
            v = self.cross_attention(k, q, v, cls='x')
            k_memory = k
            q_memory = q
            V.append(v.unsqueeze(1))
            K.append(k.unsqueeze(1))
            Q.append(q.unsqueeze(1))
        return torch.cat(K, dim=1), torch.cat(Q, dim=1), torch.cat(V, dim=1)


class EntropyBlockTrans(nn.Module):
    """
    written by haolin
    """
    def __init__(
            self,
            input_channels: int = 64,
            hidden_channels: int = 64,
            num_layers: int = 1,
            num_heads: int = 8
    ):
        super(EntropyBlockTrans, self).__init__()
        # why need decoder?
        self.MAL_encoder = []
        # self.MAL_decoder = []
        for i in range(num_layers):
            self.MAL_encoder.append(Memory_Attention_Layer(input_channels, hidden_channels, num_heads))
            # self.MAL_decoder.append(Memory_Attention_Layer(input_channels, hidden_channels, num_heads))
        self.MAL_encoder = nn.ModuleList(self.MAL_encoder)
        # self.MAL_decoder = nn.ModuleList(self.MAL_decoder)
        self.postlayer_k = nn.Sequential(nn.Linear(hidden_channels, input_channels))
        self.postlayer_q = nn.Sequential(nn.Linear(hidden_channels, input_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        # x shape (N,F,S,C)
        for MAL in self.MAL_encoder:
            k, q, x = MAL(x)
        sigma = torch.clamp(self.postlayer_q(q), 1e-8, 1.0)
        mu = self.postlayer_k(k)
        normal = torch.distributions.Normal(mu, sigma)

        if self.training:
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            compress_x = x + noise
        else:
            compress_x = torch.round(x)

        probs = normal.cdf(compress_x + 0.5) - normal.cdf(compress_x - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-8) / np.log(2.0), 0, 50))
        # for MAL in self.MAL_decoder:
            # k, q, compress_x = MAL(compress_x)
        return probs, total_bits, compress_x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 2,
        stride: int = 1,
        dilation: int = 4,
    ):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.CELU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.CELU(inplace=True)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1,
            self.conv2, self.chomp2, self.relu2,
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class EntropyBlockCA(nn.Module):
    """docstring for EntropyBlockCA"""

    def __init__(self, channels: int = 64):
        super(EntropyBlockCA, self).__init__()
        self.conv = models.ResBlock(channels=channels)
        self.ca = models.ChannelAttention()

    def forward(self, x):
        x = self.conv(x)
        mu, x = self.ca(x)
        if mu.shape[1] != x.shape[1]:
            _, c, h, w = x.shape
            # tcn = TemporalBlock(h * w, c).to(x.device)
            tcn = nn.Linear(h * w, c).to(x.device)
            mu = tcn(mu).permute(0, 2, 1).reshape(x.shape)
        sigma = torch.clamp(x, 1e-8, 1.0)
        normal = torch.distributions.Normal(mu, sigma)

        if self.training:
            noise = torch.nn.init.uniform_(
                torch.zeros_like(x),
                -0.5, 0.5
            ).to(x.device)
            compress_x = x + noise
        else:
            compress_x = torch.round(x)

        probs = normal.cdf(compress_x + 0.5) - normal.cdf(compress_x - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-8) / np.log(2.0), 0, 50))

        return probs, total_bits, compress_x


class EntropyBlockNLA(EntropyBlockCA):
    """docstring for EntropyBlockNLA, don't work well, need rethinking"""

    def __init__(self, in_channels: int = 64, hidden_channels: int = 64, out_channels: int = 64):
        super(EntropyBlockNLA, self).__init__()
        self.ca = models.NonLocalAttention()


if __name__ == "__main__":
    model = EntropyBlockCA()
    test_data = torch.ones((4, 64, 64, 64))
    probs, total_bits, rebuild_x = model(test_data)
    # (N, S, C)
    print(rebuild_x.shape)
    # (batch_size, frame_sequence, patches, input_channels)
