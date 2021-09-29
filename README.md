# video-compression

use deep-learning method to compress video, make sure the video is already downloaded.

## Usage

1. make dataset, see details in `dataset_make.py`
2. train, you can do this by `python -W ignore train.py --niter 20 --batch_size 64`
3. valid, you can do this by `python -W ignore valid.py`

## Out

### Baseline(no learned entropy block)

train loss, see `ans.txt`: 

1. epoch: 10, learning rate: 3e-4, batch_size: 64, loss: 5 epochs for MSELoss(warmup), 5 epochs for L1Loss & MSSSIMLoss
2. epoch: 10, learning rate: 5e-6, batch_size: 64, loss: all for L1Loss & MSSSIMLoss
3. epoch: 10, learning rate: 5e-8, batch_size: 64, loss: same as step two
4. epoch: 10, learning rate: 5e-8, batch_size: 64, loss: MSELoss & MSSSIMLoss

>  in train loss, the bpp of epoch:0-15 is half of the true bpp value

valid ans:

- epoch 15: eval psnr: 32.9043 eval ssim: 0.9870 eval bpp: 5.8670 valid_time: 0.2631s
- epoch 23: eval psnr: 32.7106 eval ssim: 0.9868 eval bpp: 5.7322 valid_time: 0.2628s
- epoch 25: eval psnr: 32.9959 eval ssim: 0.9879 eval bpp: 5.5862 valid_time: 0.2626s
- epoch 29: eval psnr: 33.1038 eval ssim: 0.9883 eval bpp: 5.4909 valid_time: 0.2627s
- epoch 44: eval psnr: 33.2406 eval ssim: 0.9887 eval bpp: 5.3203 valid_time: 0.2619s

epoch is larger, bpp is lower

## Entropy Block

1. non learned: baseline, bpp may be 5
2. transformer: take a long time, but bpp could be 0.5 or lower
3. channel attention: best, quicker than transformer and lower bpp
4. non local attention(work bad)

## Requirement

1. pytorch >= 1.6.0
2. tfrecord
3. opencv-python

## Trouble shoot

1. don't set learning rate too large(1e-3 or larger)
