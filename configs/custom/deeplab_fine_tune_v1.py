_base_ = [
    '../_base_/models/custom/deeplab_fine_tune_v1.py', '../_base_/datasets/cityscapes.py',
    '../_base_/custom_runtime.py', '../_base_/schedules/schedule_custom.py'
]

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
