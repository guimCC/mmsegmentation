# Get custom components from mmegnine
_base_ = [
    '../../datasets/cityscapes.py', # Load the cityscapes dataset configuration
    '../../default_runtime.py', # Load the default runtime configuration
    '../../schedules/schedule_40k.py', # Load the schedule configuration <- Change
]


# Sync BN ??
# Data preprocessor definition?


# Model definition
model = dict(
    type='ReinforceSeg', # Custom model to train
    num_classes=19, # Number of classes
    # Load checkpoint from already trained model
)

