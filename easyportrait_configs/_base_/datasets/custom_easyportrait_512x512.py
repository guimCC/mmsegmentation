# dataset settings
dataset_type = 'EasyPortraitDataset'
data_root = 'data/easyportrait/'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(1.0, 1.0), # keep the aspect ratio
        keep_ratio=True),
    
    
    dict(type='RandomCrop', crop_size=crop_size),
    # We don't use RandomFlip, but need it in the code to fix error: https://github.com/open-mmlab/mmsegmentation/issues/231
    dict(type='RandomFlip', prob=0.0), 
    dict(type='PhotoMetricDistortion', 
         brightness_delta=16, 
         contrast_range=(0.5, 1.0),
         saturation_range=(0.5, 1.0),
         hue_delta=9),  
    #dict(type='Normalize', **img_norm_cfg),
    #dict(type='DefaultFormatBundle'),
    #dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    dict(type='PackSegInputs')
]
## to change dict(type='Pad', size=(1920, 1920), pad_val=0, seg_pad_val=255),
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=(512, 512),
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', keep_ratio=True),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='ImageToTensor', keys=['img']),
    #         dict(type='Collect', keys=['img']),
    #     ])
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
            img_path='images/train', seg_map_path='annotations/train'),
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
            img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), # Do not shuffle during validation
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU']) # metric to measure the accuracy (mean IoU)
test_evaluator = val_evaluator