import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from gdn import GDN
from dcn import DeformConv2d
import entropy_model
from ssim_loss import MsssimLoss


def _make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, channels: int = 64, kernel_size: int = 3, inverse: bool = False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, bias=False)
        nn.init.xavier_normal_(self.conv1.weight.data, 0.01)
        self.gdn = GDN(channels, inverse=inverse)
        self.relu = nn.CELU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=2, bias=False)
        nn.init.xavier_normal_(self.conv2.weight.data, 0.01)

    def forward(self, x):
        res = x
        x = self.relu(self.gdn(self.conv1(x)))
        x = self.conv2(x) + res
        return x


class DownSample(nn.Module):
    """docstring for DownSample"""
    def __init__(self, in_channels: int = 3, out_channels: int = 64, kernel_size: int = 3, num_gdns: int = 3):
        super(DownSample, self).__init__()
        self.prior_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 2, 2),
            nn.CELU(inplace=True)
        )
        self.res = _make_layer(
            ResBlock, num_layers=num_gdns,
            channels=out_channels
        )

    def forward(self, x):
        x = self.prior_layer(x)
        res = x
        x = self.res(x) + res
        return x


class UpSample(nn.Module):
    """docstring for UpSample"""
    def __init__(self, in_channels: int = 64, out_channels: int = 3, kernel_size: int = 3, num_gdns: int = 3):
        super(UpSample, self).__init__()
        self.res = _make_layer(
            ResBlock, num_layers=num_gdns,
            channels=in_channels, inverse=True
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        res = x
        x = self.res(x) + res
        del res
        _, _, h, w = x.shape
        x = F.interpolate(x, scale_factor=2., mode="bilinear")
        x = self.conv(x)
        return x


class FeatureExtraction(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureExtraction, self).__init__()
        self.downsample = DownSample(**kwargs)

    def forward(self, x):
        return self.downsample(x)


class FrameReconstruction(nn.Module):
    def __init__(self, **kwargs):
        super(FrameReconstruction, self).__init__()
        self.upsample = UpSample(**kwargs)

    def forward(self, x):
        return torch.clamp(self.upsample(x), -255, 255)


class MotionEstimate(nn.Module):
    def __init__(self, channels: int = 64, num_gdns: int = 3):
        super(MotionEstimate, self).__init__()
        self.res = _make_layer(
            ResBlock, num_layers=num_gdns,
            channels=channels
        )
        self.prior_layer = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.CELU(inplace=True)
        )

    def forward(self, f_t, f_r):
        x = torch.cat((f_t, f_r), 1)
        return self.res(self.prior_layer(x))


class MotionCompensation(nn.Module):
    """docstring for MotionCompensation"""
    def __init__(self, channels: int = 64, kernel_size: int = 3, num_gdns: int = 3):
        super(MotionCompensation, self).__init__()
        self.deform_conv = DeformConv2d(channels, channels)
        self.res = _make_layer(
            ResBlock, num_layers=num_gdns,
            channels=channels
        )
        self.conv = nn.Conv2d(channels * 2, channels, 3, 1, 1)

        self.p_conv = nn.Conv2d(channels, 2 * kernel_size ** 2, kernel_size, padding=1, stride=1)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
    
    def forward(self, f_r, m_t):
        x = self.deform_conv(f_r, self.p_conv(m_t))
        x = self.res(self.conv(torch.cat((x, f_r), 1))) + x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_features: int = 64, reduction: int = 8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            nn.CELU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        score = self.conv(self.avg_pool(x))
        return score, x * score


class NonLocalAttention(nn.Module):
    def __init__(self, channels: int = 64):
        super(NonLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, 1)
        self.relu1 = nn.CELU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, 1, 1)
        self.relu2 = nn.CELU(inplace=True)

        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.relu3 = nn.CELU(inplace=True)

    def forward(self, x):
        x_embed1 = self.relu1(self.conv1(x))
        x_embed2 = self.relu2(self.conv2(x))
        x_assembly = self.relu3(self.conv3(x))

        n, c, h, w = x_embed1.shape
        x_embed1 = x_embed1.permute(0, 2, 3, 1).view(n, h * w, c)
        x_embed2 = x_embed2.view(n, c, h * w)
        score = torch.matmul(x_embed1, x_embed2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(n, -1, h * w).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly).permute(0, 2, 1).view(n, -1, h, w)

        return score, x_final


class InfoEncoder(nn.Module):
    """docstring for InfoEncoder"""
    def __init__(self, channels: int = 64, num_gdns: int = 3):
        super(InfoEncoder, self).__init__()
        self.head = DownSample(
            channels,
            channels,
            num_gdns=1
        )
        self.encoder = _make_layer(
            ResBlock, num_layers=num_gdns-1,
            channels=channels
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        x = self.head(x)
        x = self.encoder(x)
        _, x = self.ca(x)
        return x


class InfoDecoder(nn.Module):
    """docstring for InfoDecoder"""
    def __init__(self, channels: int = 64, num_gdns: int = 3):
        super(InfoDecoder, self).__init__()
        self.tail = UpSample(
            channels,
            channels,
            num_gdns=1
        )
        self.decoder = _make_layer(
            ResBlock, num_layers=num_gdns-1,
            channels=channels,
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        x = self.decoder(x)
        # if you use transformer to be entropy model, don't do this
        x = self.tail(x)
        _, x = self.ca(x)
        return x


class InfoCompression(nn.Module):
    def __init__(self, **kwargs):
        super(InfoCompression, self).__init__()
        self.encoder = InfoEncoder()
        self.decoder = InfoDecoder()
        self.uniform = torch.distributions.uniform.Uniform(torch.tensor([-0.5]), torch.tensor([0.5]))
        self.entropy_model = entropy_model.EntropyBlockCA()

    def forward(self, x):
        # if you use transformer to be entropy model, don't do self.encoder(x)
        x = self.encoder(x)
        prob, total_bits, x = self.entropy_model(x)
        x_hat = self.decoder(x)
        return x_hat, total_bits, prob


class XVC(nn.Module):
    """docstring for XVC"""
    def __init__(self):
        super(XVC, self).__init__()
        self.FeatureExtract = FeatureExtraction()
        self.FrameReconstruct = FrameReconstruction()
        self.MotionEstimate = MotionEstimate()
        self.MotionCompensation = MotionCompensation()
        self.MotionCompress = InfoCompression()
        self.ResidualCompress = InfoCompression()

    def ecodec(self, x, feature_buffer, epoch:int = 0):
        f_t = self.FeatureExtract(x)
        m_t = self.MotionEstimate(f_t, feature_buffer)
        m_hat, m_bits, m_prob = self.MotionCompress(m_t)

        f_t_pre = self.MotionCompensation(feature_buffer, m_hat)

        r_t = f_t - f_t_pre
        r_hat, r_bits, r_prob = self.ResidualCompress(r_t)

        f_t_hat = r_hat + f_t_pre
        x_t_hat = self.FrameReconstruct(f_t_hat)

        total_bits = r_bits + m_bits
        bpp = total_bits / (x.shape[0] * x.shape[2] * x.shape[3])

        if epoch <= 5:
            criterion = nn.MSELoss()(x, x_t_hat)
        else:
            criterion = nn.L1Loss()(x, x_t_hat) + 4096 * MsssimLoss()(x, x_t_hat)

        if self.training:
            return criterion, bpp, f_t_hat
        else:
            return x_t_hat, bpp

    def forward(self, x, epoch:int = 0):
        _, num, _, h, w = x.shape
        feature_buffer = self.FeatureExtract(x[:, 0,...])
        total_criterion = 0.
        total_bpp = 0.

        for i in range(1, num):
            if self.training:
                criterion, bpp, f_t_hat = self.ecodec(x[:, i, ...], feature_buffer, epoch)
                total_criterion += criterion
                total_bpp += bpp
                feature_buffer = f_t_hat
                return total_criterion / (num-1), total_bpp / (num-1)
            else:
                x_t_hat, bpp = self.ecodec(x[:, i, ...], feature_buffer, epoch)
                return x_t_hat, bpp / (num-1)


if __name__ == '__main__':
    import time
    a = XVC().to("cuda")
    test_data = torch.rand(4, 2, 3, 128, 128).to("cuda")
    t0 = time.clock()
    for _ in range(10):
        a(test_data, epoch=6)
    t1 = time.clock()
    print("cost {}s".format(t1 - t0))
