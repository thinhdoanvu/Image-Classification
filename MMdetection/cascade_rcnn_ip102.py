# _base_ = 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'   # faster-rcnn Resnet: 50
# _base_ = 'dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py'   # dynamic-rcnn Resnet: 50
_base_ = 'cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'   # cascade-rcnn Resnet: 50


# for faster, dynamic, cascade-rcnn

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead',
                 in_channels=256,
                 fc_out_channels=1024,
                 roi_feat_size=7,
                 num_classes=102,  # Update number of classes
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0.0, 0.0, 0.0, 0.0],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead',
                 in_channels=256,
                 fc_out_channels=1024,
                 roi_feat_size=7,
                 num_classes=102,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0.0, 0.0, 0.0, 0.0],
                     target_stds=[0.05, 0.05, 0.1, 0.1]),
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead',
                 in_channels=256,
                 fc_out_channels=1024,
                 roi_feat_size=7,
                 num_classes=102,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0.0, 0.0, 0.0, 0.0],
                     target_stds=[0.033, 0.033, 0.067, 0.067]),
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=1.0))
        ]
    )
)



dataset_type = 'CocoDataset'

classes = ('rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot', 'asiatic rice borer', 'yellow rice borer',
        'rice gall midge', 'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper', 'small brown plant hopper',
        'rice water weevil', 'rice leafhopper', 'grain spreader thrips', 'rice shell pest', 'grub', 'mole cricket', 'wireworm',
        'white margined moth', 'black cutworm', 'large cutworm', 'yellow cutworm', 'red spider', 'corn borer', 'army worm', 'aphids',
        'Potosiabre vitarsis', 'peach borer', 'english grain aphid', 'green bug', 'bird cherry-oataphid', 'wheat blossom midge',
        'penthaleus major', 'longlegged spider mite', 'wheat phloeothrips', 'wheat sawfly', 'cerodonta denticornis', 'beet fly',
        'flea beetle', 'cabbage army worm', 'beet army worm', 'Beet spot flies', 'meadow moth', 'beet weevil', 'sericaorient alismots chulsky',
        'alfalfa weevil', 'flax budworm', 'alfalfa plant bug', 'tarnished plant bug', 'Locustoidea', 'lytta polita', 'legume blister beetle',
        'blister beetle', 'therioaphis maculata Buckton', 'odontothrips loti', 'Thrips', 'alfalfa seed chalcid', 'Pieris canidia',
        'Apolygus lucorum', 'Limacodidae', 'Viteus vitifoliae', 'Colomerus vitis', 'Brevipoalpus lewisi McGregor', 'oides decempunctata',
        'Polyphagotars onemus latus', 'Pseudococcus comstocki Kuwana', 'parathrene regalis', 'Ampelophaga', 'Lycorma delicatula', 'Xylotrechus',
        'Cicadella viridis', 'Miridae', 'Trialeurodes vaporariorum', 'Erythroneura apicalis', 'Papilio xuthus', 'Panonchus citri McGregor',
        'Phyllocoptes oleiverus ashmead', 'Icerya purchasi Maskell', 'Unaspis yanonensis', 'Ceroplastes rubens', 'Chrysomphalus aonidum',
        'Parlatoria zizyphus Lucus', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus', 'Tetradacus c Bactrocera minax ', 'Dacus dorsalis(Hendel)',
        'Bactrocera tsuneonis', 'Prodenia litura', 'Adristyrannus', 'Phyllocnistis citrella Stainton', 'Toxoptera citricidus', 'Toxoptera aurantii',
        'Aphis citricola Vander Goot', 'Scirtothrips dorsalis Hood', 'Dasineura sp', 'Lawana imitata Melichar', 'Salurnis marginella Guerr',
        'Deporaus marginatus Pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper', 'Rhytidodera bowrinii white', 'Sternochetus frigidus',
        'Cicadellidae')

data_root = 'data/coco/'


# Training pipeline
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
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
        data_prefix=dict(img='images/'),
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
        data_prefix=dict(img='images/'),
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