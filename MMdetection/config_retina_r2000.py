# _base_ = 'retinanet/retinanet_r50_fpn_1x_coco.py'  # retinanet
_base_ = 'efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py'  # efficientnet

# Training parameters
train_batch_size_per_gpu = 4  # Optimal for RTX 4090
train_num_workers = 4
max_epochs = 100
base_lr = 0.001

# Model configuration
model = dict(
    bbox_head=dict(
        num_classes=16  # Number of classes in your dataset
    )
)

# Dataset settings
dataset_type = 'CocoDataset'

# List of classes
classes = ('00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',)

data_root = 'data/coco/'

# Train data configuration
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')
    )
)

# Validation data configuration
val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')
    )
)

# Evaluator configuration
val_evaluator = dict(
    ann_file='./data/coco/annotations/instances_val2017.json'
)

# Optimizer configuration
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.0001)
)

# Scheduler configuration - overwrite base
param_scheduler = [
    dict(type='MultiStepLR', milestones=[8, 11], gamma=0.1)
]

# save the best model
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater"
    )
)

# Training configuration
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1
)
