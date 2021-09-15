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
train_dataset = TFRecordDataset("valid.tfrecord", None, description)
# do not shuffle
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)
# length = 0
# for record in train_dataloader:
#     length += opt.workers
length = 630144
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
            loss = total_criterion + 1024 * total_bpp

            loss.backward()
            # utils.clip_gradient(model_optimizer, 5)

            model_optimizer.step()
            if epoch <= 5:
                epoch_losses.update(total_criterion.item(), opt.batch_size)
            else:
                epoch_losses.update(total_criterion.item(), opt.batch_size)
            epoch_bpp.update(total_bpp.item(), opt.batch_size)

            t.set_postfix(
                loss='{:.6f}'.format(epoch_losses.avg),
                bpp='{:.6f}'.format(epoch_bpp.avg)
            )
            t.update(opt.batch_size)

    model_scheduler.step()

    torch.save(model.state_dict(), "xvc_epoch_{}.pth".format(epoch))
