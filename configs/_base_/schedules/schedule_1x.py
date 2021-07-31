# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
max_epochs = 200
# step = [max_epochs * 8 // 12, max_epochs * 11 // 12]

# print("-------------------------------------")
# print(max_epochs, type(max_epochs))
# print(step)
# print("-------------------------------------")

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[133, 183])
runner = dict(type='EpochBasedRunner', max_epochs=200)