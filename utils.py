import torch
from torch.autograd import Variable
import torch.nn.functional as F

import ssim_loss


def get_length(generator):
    return sum(1 for _ in generator)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def calc_psnr(img1, img2):
    """calculate PNSR on cuda and cpu: img1 and img2 have range [0, 255]"""
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def _ssim(img1, img2, size_average=True, channel: int = 3):
    """calculate SSIM on cuda and cpu: img1 and img2 have range [0, 255]"""
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11
    window = ssim_loss.create_window(window_size, sigma, channel).cuda()
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
    mcs_map = V1 / V2
    if size_average:
        return ssim_map.mean(), mcs_map.mean()

def ms_ssim(img1, img2, levels=5):
    weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())
    msssim = Variable(torch.Tensor(levels,).cuda())
    mcs = Variable(torch.Tensor(levels,).cuda())
    for i in range(levels):
        ssim_map, mcs_map = _ssim(img1, img2)
        msssim[i] = ssim_map
        mcs[i] = mcs_map
        filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
        filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        img1 = filtered_im1
        img2 = filtered_im2

    value = torch.prod(mcs[0:levels-1]**weight[0:levels-1]) * \
                (msssim[levels-1]**weight[levels-1])
    return value
