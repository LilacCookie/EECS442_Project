model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),  # 使用NTU RGB+D的25关键点布局
    cls_head=dict(type='GCNHead', num_classes=90, in_channels=256))  # 3D Yoga有90个姿势类别

dataset_type = 'PoseDataset'
# ann_file = '/root/autodl-tmp/EECS442 project/pyskl/tools/data/3dyoga/3dyoga_annotations2.pkl'  # 你的单个pkl文件路径
ann_file = '/root/autodl-tmp/test.pkl'

train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['b']),  # 使用骨骼特征
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),  # 瑜伽通常只有一个人
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['b']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['b']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type, 
            ann_file=ann_file, 
            pipeline=train_pipeline, 
            split='train')),  # 使用pkl文件中的train划分
    val=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        pipeline=val_pipeline, 
        split='test'),  # 使用pkl文件中的test划分作为验证集
    test=dict(
        type=dataset_type, 
        ann_file=ann_file, 
        pipeline=test_pipeline, 
        split='test'))

# 优化器配置
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)

# 学习率策略
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# 运行时设置
log_level = 'INFO'
work_dir = './work_dirs/stgcn++/stgcn++_3dyoga_3dkp'