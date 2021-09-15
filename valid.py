import time
import os
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

import models
import utils


l = os.listdir("pretrained_models")
l.sort()
for name in l:
    model_name = os.path.join("pretrained_models", name)
    # dataset init, train file need .tfrecord
    description = {
        "image": "byte",
        "size": "int",
    }
    valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
    valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=16)

    # models init
    model = models.XVC().to("cuda")
    model_params = torch.load(model_name)
    model.load_state_dict(model_params)

    model.eval()
    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()
    epoch_bpp = utils.AverageMeter()

    t0 = time.clock()
    cnt = 0
    for record in valid_dataloader:
        cnt += 1
        inputs = record["image"].reshape(
            16,
            2,
            3,
            record["size"][0],
            record["size"][0],
        ).float().to("cuda")

        with torch.no_grad():
            output, bpp = model(inputs)
            epoch_pnsr.update(utils.calc_psnr(output, inputs[:, 1, ...]), len(output))
            epoch_ssim.update(utils.ms_ssim(output, inputs[:, 1, ...]), len(output))
            epoch_bpp.update(bpp, len(inputs))
    t1 = time.clock()

    print('model name: {} eval psnr: {:.4f} eval ssim: {:.4f} eval bpp: {:.4f} valid_time: {:.4f}s \n'.format(
        model_name, epoch_pnsr.avg, epoch_ssim.avg, epoch_bpp.avg, (t1-t0) / cnt
    ))
