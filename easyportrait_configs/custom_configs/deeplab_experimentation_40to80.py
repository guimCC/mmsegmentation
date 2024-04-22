_base_ = [
    '../_base_/models/custom/deeplab_40to80.py', '../_base_/datasets/custom_easyportrait_512x512.py',
    '../../configs/_base_/custom_runtime.py', '../../configs/_base_/schedules/schedule_custom_v3.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)