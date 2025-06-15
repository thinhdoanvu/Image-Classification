_base_ = 'sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'  # 📦 Backbone: ResNet-50 + FPN

# 📦 Dataset & class info
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

classes = (
    'rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot', 'asiatic rice borer', 'yellow rice borer',
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
    'Parlatoria zizyphus Lucus', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus', 'Tetradacus c Bactrocera minax', 'Dacus dorsalis(Hendel)',
    'Bactrocera tsuneonis', 'Prodenia litura', 'Adristyrannus', 'Phyllocnistis citrella Stainton', 'Toxoptera citricidus', 'Toxoptera aurantii',
    'Aphis citricola Vander Goot', 'Scirtothrips dorsalis Hood', 'Dasineura sp', 'Lawana imitata Melichar', 'Salurnis marginella Guerr',
    'Deporaus marginatus Pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper', 'Rhytidodera bowrinii white', 'Sternochetus frigidus',
    'Cicadellidae'
)

# 🧠 Model: Sparse R-CNN + sửa số lớp
model = dict(
    roi_head=dict(
        _delete_=True,  # ⚠️ Xóa roi_head gốc để ghi đè toàn bộ
        type='SparseRoIHead',
        num_stages=6,
        stage_loss_weights=[1.0] * 6,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=16,  # ⚠️ Đây là chỗ bạn thay đổi
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0)
            )
        ] * 6
    )
)


# 📦 Train/Test pipeline
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
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 🔁 Dataloaders
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        pipeline=test_pipeline
    )
)

# 📊 Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox'
)

# 💾 Checkpointing
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP',
        rule='greater'
    )
)

# 🔧 Optimizer (Sparse R-CNN dùng SGD mặc định)
# Nếu muốn dùng AdamW như DETR thì có thể sửa lại optimizer ở đây

# 🏃 Train config
train_cfg = dict(max_epochs=200, val_interval=1)
