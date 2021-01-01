# Modified by Peize Sun, Rufeng Zhang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from .deformable_detr import build
from .deformable_detrtrack_test import build as build_tracktest
from .deformable_detrtrack_train import build as build_tracktrain
from .tracker import Tracker
from .save_trackval import save_motval
from .save_tracktest import save_mottest


def build_model(args):
    return build(args)

def build_tracktrain_model(args):
    return build_tracktrain(args)

def build_tracktest_model(args):
    return build_tracktest(args)
