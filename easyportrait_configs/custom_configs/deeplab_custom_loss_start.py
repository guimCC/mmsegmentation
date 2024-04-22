_base_ = [
    '../_base_/models/custom/deeplab_from_ckp.py', '../_base_/datasets/custom_easyportrait_512x512.py',
    '../../configs/_base_/custom_runtime.py', '../../configs/_base_/schedules/schedule_custom_start.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)