#!/usr/bin/env bash

python3 main_track.py --output_dir . --dataset_file mot --coco_path mot --batch_size 1 --resume crowdhuman_final.pth --det_val --eval --with_box_refine --track_thresh 0.5