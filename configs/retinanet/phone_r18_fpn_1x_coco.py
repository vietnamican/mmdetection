_base_ = [
    '../_base_/models/retinanet_r18_fpn.py',
    '../_base_/datasets/phone_detection_new.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
load_from = '/vinai/khaidq3/logs/mmdetection/resnet18_coco/epoch_200.pth'