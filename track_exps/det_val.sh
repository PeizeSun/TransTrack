#!/usr/bin/env bash

python3 track_main/main_detval.py  --output_dir . --dataset_file mot --coco_path mot --batch_size 1 --resume crowdhuman_final.pth --eval --with_box_refine --track_thresh 0.5