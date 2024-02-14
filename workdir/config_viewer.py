from mmengine.config import Config

cfg = Config.fromfile('../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py')
print(cfg.train_dataloader)