# dataset settings
dataset_type = 'PhoneDataset'

data_root = '/home/ubuntu/tienpv/datasets/PhoneDatasets/COCO2017/'
ann_files = '/home/ubuntu/tienpv/datasets/PhoneDatasets/COCO2017/annotations/instances_train2017_cell_phone_format_widerface.txt'
val_data_root = '/home/ubuntu/tienpv/datasets/PhoneDatasets/COCO2017/'
val_ann_files = '/home/ubuntu/tienpv/datasets/PhoneDatasets/COCO2017/annotations/instances_val2017_cell_phone_format_widerface.txt'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

gray_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='grayscale'),
    dict(type='Stack'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# rgb_dataset_train = dict(
#     type='RepeatDataset',
#     times=2,
#     dataset=dict(
#         type=dataset_type,
#         ann_file=ann_files,
#         img_prefix=data_root,
#         pipeline=train_pipeline
#     )
# )
# gray_dataset_train = dict(
#     type='RepeatDataset',
#     times=2,
#     dataset=dict(
#         type=dataset_type,
#         ann_file=ann_files,
#         img_prefix=data_root,
#         pipeline=gray_train_pipeline
#     )
# )

data = dict(
    samples_per_gpu=60,
    workers_per_gpu=4,
    # train=[rgb_dataset_train, gray_dataset_train],
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_files,
            img_prefix=data_root,
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=val_ann_files,
        img_prefix=val_data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=val_ann_files,
        img_prefix=val_data_root,
        pipeline=test_pipeline))
