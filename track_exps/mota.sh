#!/usr/bin/env bash


GROUNDTRUTH=mot/train
RESULTS=val/tracks
GT_TYPE=_val_half
THRESHOLD=-1

python3 track_tools/eval_motchallenge.py \
--groundtruths ${GROUNDTRUTH} \
--tests ${RESULTS} \
--gt_type ${GT_TYPE} \
--eval_official \
--score_threshold ${THRESHOLD}
