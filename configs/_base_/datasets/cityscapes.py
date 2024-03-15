# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/' #path to the dataset (given by symbolic link)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'), # load image from file
    dict(type='LoadAnnotations'), # load annotations (groud truth)
    dict(
        type='RandomResize', # augmentation pipeline that resises images and annotations
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True), # wether to keep the aspect ratio
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), #crop a patch from current image -> max area
    # that could be occupied by a single category
    dict(type='RandomFlip', prob=0.5), # augmentation pipeline that flips images and annotations
    dict(type='PhotoMetricDistortion'), # augmentation pipeline that distorts current image with several photo metric methods
    dict(type='PackSegInputs') # pack inputs data for semantic segmentation
]
test_pipeline = [
    dict(type='LoadImageFromFile'), # load image from file
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    # Increased batch size, maximmum on the 
    batch_size=4, # batch size of a single GPU
    num_workers=2, # worker to pre-fetch data for each single gpu
    persistent_workers=True, # shut down the worker processes after an epoch end, can accelerate training speed
    sampler=dict(type='InfiniteSampler', shuffle=True), # Randomly shuffle during training
    dataset=dict( # Train dataset config
        type=dataset_type,
        data_root=data_root, # root of the dataset
        data_prefix=dict( # prefixes for training data
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), # Do not shuffle during validation
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU']) # metric to measure the accuracy (mean IoU)
test_evaluator = val_evaluator
