# _base_ = 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'   # faster-rcnn Resnet: 50
_base_ = 'dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py'   # dynamic-rcnn Resnet: 50


# for faster, dynamic, cascade-rcnn

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=16)
    )
)


dataset_type = 'CocoDataset'
classes = ('00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',)

data_root = 'data/coco/'


# Training pipeline
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,# Ensure clean worker state
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline  # fit 640x640
        )
    )


val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        pipeline=test_pipeline  # fit 640x640
        )
    )

default_hooks = dict(   # save only the best model
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater"
    )
)

# Set the maximum number of epochs for training
train_cfg = dict(max_epochs=100)
runner = dict(type='EpochBasedRunner', max_epochs=100)
