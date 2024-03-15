# optimizer
optimizer = dict(type='SGD', # Type of optimiser
                 lr=0.01, # Learning rate of optimisers 
                 momentum=0.9, # Momentum of optimiser
                 weight_decay=0.0005) # Weight decay of optimiser
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None) # opmitizer wrapper for common interface
# # learning policy
param_scheduler = [
    dict(
        type='PolyLR', # Policy of sheduler
        eta_min=1e-4, # Minimum learning rate at the end of sheduling
        power=0.9, # Power of polynomial decay
        begin=0, # Step at which to start updating the parameters
        end=40000, # Step at which to stop updating the parameters
        by_epoch=False) # Whether to count by epoch or not
]
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'), # Log the time spent during iteration
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False), # Collect and write logs from different
# components of ''Runner''
    param_scheduler=dict(type='ParamSchedulerHook'), # Update hyper-parameters
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000), # Save checkpoints periodically
    sampler_seed=dict(type='DistSamplerSeedHook'), # Data-loading sampler
    visualization=dict(type='SegVisualizationHook'))
