from mmseg.apis import MMSegInferencer
# Load models into memory
inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
#Inference
inferencer('/data/datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_058176_leftImg8bit.png', show=True)
