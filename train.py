import random
import warnings

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm

import utils
import config
from models import XVC


opt = config.get_options()

# deveice init
CUDA_ENABLE = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA_ENABLE else 'cpu')

# seed init
manual_seed = opt.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# dataset init, train file need .tfrecord
description = {
    "image": "byte",
    "size": "int",
}
train_dataset = TFRecordDataset("train.tfrecord", None, description)
# do not shuffle
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True
)

length = 600000

# models init
model = XVC().to(device)

# optim and scheduler init
model_optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1)
model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

# train model
print("-----------------train-----------------")
for epoch in range(opt.niter):
    model.train()
    epoch_losses = utils.AverageMeter()
    epoch_bpp = utils.AverageMeter()

    with tqdm(total=(length - length % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

        for record in train_dataloader:
            inputs = record["image"].reshape(
                opt.batch_size,
                2,
                3,
                record["size"][0],
                record["size"][0],
            ).float().to(device)

            model_optimizer.zero_grad()

            total_criterion, total_bpp = model(inputs, epoch)

            total_criterion.backward()
            utils.clip_gradient(model_optimizer, 5)

            model_optimizer.step()
            epoch_losses.update(total_criterion.item(), opt.batch_size)
            epoch_bpp.update(total_bpp.item(), opt.batch_size)

            t.set_postfix(
                loss='{:.6f}'.format(epoch_losses.avg),
                bpp='{:.6f}'.format(epoch_bpp.avg)
            )
            t.update(opt.batch_size)

    model_scheduler.step()

    # test, just pick one to take a look
    # model.eval()
    # epoch_pnsr = utils.AverageMeter()
    # epoch_ssim = utils.AverageMeter()

    # cnt = 0
    # for record in valid_dataloader:
    #     cnt += 1
    #     if cnt >= 100:
    #         break
    #     inputs = record["image"].reshape(
    #         1,
    #         3,
    #         record["size"][0],
    #         record["size"][0],
    #     ).float().to("cuda")

    #     with torch.no_grad():
    #         output, bpp_feature_val, bpp_z_val = model(inputs)
    #         epoch_pnsr.update(utils.calc_psnr(output, inputs), len(inputs))
    #         epoch_ssim.update(utils.calc_ssim(output, inputs), len(inputs))

    # print('eval psnr: {:.4f} eval ssim: {:.4f}\n'.format(epoch_pnsr.avg, epoch_ssim.avg))
    # torch.save(model.state_dict(), "edic_epoch_{}_bpp_{}.pth".format(epoch, bpp.item()))