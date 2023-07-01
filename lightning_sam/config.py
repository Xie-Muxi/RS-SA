from box import Box

config = {
    "num_devices": 4,
    "batch_size": 4,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 1,
    "num_classes": 81,
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
        "train": {
            "root_dir": "/nfs/home/3002_hehui/xmx/COCO2017/train2017",
            "annotation_file": "/nfs/home/3002_hehui/xmx/COCO2017/annotations/instances_train2017.json"
        },
        "val": {
            "root_dir": "/nfs/home/3002_hehui/xmx/COCO2017/val2017",
            "annotation_file": "/nfs/home/3002_hehui/xmx/COCO2017/annotations/instances_val2017.json"
        }
    },
    "log_interval": 50,
    "resume_checkpoint": None,

}

cfg = Box(config)