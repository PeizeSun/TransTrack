#!/usr/bin/env bash

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_track.py  --output_dir ./output --dataset_file mot --coco_path mot --batch_size 2  --with_box_refine  --num_queries 500 --resume crowdhuman_final.pth --epochs 20 --lr_drop 10
