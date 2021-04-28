#!/usr/bin/env bash

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_track.py  --output_dir ./output_crowdhuman --dataset_file crowdhuman --coco_path crowdhuman --batch_size 2  --with_box_refine --num_queries 500 --epochs 150 --lr_drop 100

