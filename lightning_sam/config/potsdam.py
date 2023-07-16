from box import Box

config = {
    "num_devices": 4,
    "batch_size": 12,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 1,
    "num_classes": 6,  # 需要根据 Potsdam 数据集的类别数进行修改
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": "/nfs/home/3002_hehui/xmx/segment-anything/segment_anything/ckpt/sam_vit_h_4b8939.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "name":"potsdam", 
        "train": {
            "root_dir": "/nfs/home/3002_hehui/xmx/data/potsdam/img_dir/train",  
            "annotation_file": "/nfs/home/3002_hehui/xmx/data/potsdam/ann_dir/train"  
        },
        "val": {
            "root_dir": "/nfs/home/3002_hehui/xmx/data/potsdam/img_dir/val", 
            "annotation_file": "/nfs/home/3002_hehui/xmx/data/potsdam/ann_dir/val" 
        }
    },
    "log_interval": 50,
    "resume_checkpoint": None,
}

cfg = Box(config)
