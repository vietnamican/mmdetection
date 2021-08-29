# dataset settings
dataset_type = 'PhoneDataset'
phone_ir_data_root = '../datasets/ourDB/images/'
phone_ir_ann_files_train = '../datasets/ourDB/train/labels.txt'
phone_ir_ann_files_val = '../datasets/ourDB/val/labels.txt'
ir_negative_data_root = '../datasets/HonTre_NegativeSample/'
ir_negative_ann_files_train = '../datasets/HonTre_NegativeSample/train.txt'
ir_negative_ann_files_val = '../datasets/HonTre_NegativeSample/val.txt'
# phone_rgb_data_root = '../datasets/PhoneDatasets/'
# phone_rgb_ann_files = '../datasets/PhoneDatasets/OIDV6/PhoneV6/train.txt'
# phone_gray_data_root = '../datasets/PhoneDatasets/'
# phone_gray_ann_files = '../datasets/PhoneDatasets/OIDV6/PhoneV6/train.txt'
smoking_ir_data_root = '../datasets/smoking_eating_drinking/'
smoking_ir_ann_files_train = '../datasets/smoking_eating_drinking/train/labels.txt'
smoking_ir_ann_files_val = '../datasets/smoking_eating_drinking/val/labels.txt'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
rgb_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='LoadImageFromFileSafer', to_float32=True, img_dirs=[phone_ir_data_root, ir_negative_data_root], ann_files=[phone_ir_ann_files, ir_negative_ann_files]),
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
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
ir_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='LoadImageFromFileSafer', to_float32=True, img_dirs=[phone_ir_data_root, ir_negative_data_root], ann_files=[phone_ir_ann_files, ir_negative_ann_files]),
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
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
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
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

phone_ir_dataset_train = dict(
    type='RepeatDataset',
    times=4,
    dataset = dict(
        type='ConcatDataset',
        datasets = [
            dict(
                type=dataset_type,
                ann_file=phone_ir_ann_files_train,
                img_prefix=phone_ir_data_root,
                pipeline=ir_train_pipeline
            ),
            dict(
                type=dataset_type,
                ann_file=ir_negative_ann_files_train,
                img_prefix=ir_negative_data_root,
                pipeline=ir_train_pipeline
            )
        ]
    )
)

smoking_ir_dataset_train = dict(
    type='RepeatDataset',
    times=4,
    dataset = dict(
        type='ConcatDataset',
        datasets = [
            dict(
                type=dataset_type,
                ann_file=smoking_ir_ann_files_train,
                img_prefix=smoking_ir_data_root,
                pipeline=ir_train_pipeline
            ),
            dict(
                type=dataset_type,
                ann_file=ir_negative_ann_files_train,
                img_prefix=ir_negative_data_root,
                pipeline=ir_train_pipeline
            )
        ]
    )
)

phone_ir_dataset_val = dict(
    type=dataset_type,
    ann_file=phone_ir_ann_files_val,
    img_prefix=phone_ir_data_root,
    pipeline=test_pipeline
)
smoking_ir_dataset_val = dict(
    type=dataset_type,
    ann_file=smoking_ir_ann_files_val,
    img_prefix=smoking_ir_data_root,
    pipeline=test_pipeline
)
# negative_ir_dataset_val = dict(
#     type=dataset_type,
#     ann_file=ir_negative_ann_files_val,
#     img_prefix=ir_negative_data_root,
#     pipeline=test_pipeline
# )

# rgb_dataset_train = dict(
#     type='RepeatDataset',
#     times=1,
#     dataset=dict(
#         type=dataset_type,
#         ann_file=phone_rgb_ann_files,
#         img_prefix=phone_rgb_data_root,
#         pipeline=rgb_train_pipeline
#     )
# )

# gray_dataset_train = dict(
#     type='RepeatDataset',
#     times=1,
#     dataset=dict(
#         type=dataset_type,
#         ann_file=phone_gray_ann_files,
#         img_prefix=phone_gray_data_root,
#         pipeline=gray_train_pipeline
#     )
# )

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    # train=[phone_ir_dataset_train, rgb_dataset_train, phone_ir_dataset_train, gray_dataset_train],
    # val=dict(
    #     type=dataset_type,
    #     ann_file=phone_ir_ann_files_val,
    #     img_prefix=phone_ir_data_root,
    #     pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     ann_file=phone_ir_ann_files_val,
    #     img_prefix=phone_ir_data_root,
    #     pipeline=test_pipeline))
    train=[phone_ir_dataset_train, smoking_ir_dataset_train],
    val=phone_ir_dataset_val,
    test=phone_ir_dataset_val
    # val=[phone_ir_dataset_val, smoking_ir_dataset_val, negative_ir_dataset_val],
    # test=[phone_ir_dataset_val, smoking_ir_dataset_val, negative_ir_dataset_val]
)
