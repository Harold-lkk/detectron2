#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
# register_coco_instances("h_cont_num_1", {}, 
# "/media/server/data1/datasets/container_number_datasets/0_cont_num_detection/json/0_1000_coco.json",
# "/media/server/data1/datasets/container_number_datasets/0_cont_num_detection/img/1_1000")

register_coco_instances("container_origin_1", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/1/json/1_coco_all.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/1/img")
register_coco_instances("container_origin_2", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/2/json/2_coco_all.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/2/img")
register_coco_instances("container_origin_3", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/3/json/3_coco_all.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/3/img")
register_coco_instances("container_origin_4", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/4/json/4_coco_all.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/4/img")
register_coco_instances("container_origin_5", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/5/json/5_coco_all.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/5/img")
register_coco_instances("container_origin_6", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/6/json/6_coco_all.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/6/img")
register_coco_instances("container_origin_7", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/7/json/7_coco_all_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/7/img")
register_coco_instances("container_origin_8", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/8/json/8_coco_all_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/8/img")

register_coco_instances("container_rotate_0_1", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/1/json/1_coco.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/1/img")
register_coco_instances("container_rotate_0_2", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/2/json/2_coco.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/2/img")
register_coco_instances("container_rotate_0_3", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/3/json/3_coco.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/3/img")
register_coco_instances("container_rotate_0_4", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/4/json/4_coco_rotate0.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/4/img")
register_coco_instances("container_rotate_0_5", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/5/json/5_coco.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/5/img")
register_coco_instances("container_rotate_0_6", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/6/json/6_coco.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/6/img")
register_coco_instances("container_rotate_0_7", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/7/json/7_coco_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/7/img")
register_coco_instances("container_rotate_0_8", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/8/json/8_coco_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/8/img")

register_coco_instances("container_rotate_l_30_1", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/1/json/1_coco_l_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/1/img")
register_coco_instances("container_rotate_l_30_2", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/2/json/2_coco_l_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/2/img")
register_coco_instances("container_rotate_l_30_3", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/3/json/3_coco_l_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/3/img")
register_coco_instances("container_rotate_l_30_4", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/4/json/4_coco_l_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/4/img")
register_coco_instances("container_rotate_l_30_5", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/5/json/5_coco_l_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/5/img")
register_coco_instances("container_rotate_l_30_6", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/6/json/6_coco_l_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/6/img")
register_coco_instances("container_rotate_l_30_7", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/7/json/7_coco_l_crop30_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/7/img")
register_coco_instances("container_rotate_l_30_8", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/8/json/8_coco_l_crop30_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/8/img")

register_coco_instances("container_rotate_l_90_1", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/1/json/1_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/1/img")
register_coco_instances("container_rotate_l_90_2", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/2/json/2_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/2/img")
register_coco_instances("container_rotate_l_90_3", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/3/json/3_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/3/img")
register_coco_instances("container_rotate_l_90_4", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/4/json/4_coco_l_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/4/img")
register_coco_instances("container_rotate_l_90_5", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/5/json/5_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/5/img")
register_coco_instances("container_rotate_l_90_6", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/6/json/6_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/6/img")
register_coco_instances("container_rotate_l_90_7", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/7/json/7_coco_r_crop90_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/7/img")
register_coco_instances("container_rotate_l_90_8", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/8/json/8_coco_r_crop90_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/8/img")

register_coco_instances("container_rotate_r_30_1", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/1/json/1_coco_r_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/1/img")
register_coco_instances("container_rotate_r_30_2", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/2/json/2_coco_r_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/2/img")
register_coco_instances("container_rotate_r_30_3", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/3/json/3_coco_r_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/3/img")
register_coco_instances("container_rotate_r_30_4", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/4/json/4_coco_r_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/4/img")
register_coco_instances("container_rotate_r_30_5", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/5/json/5_coco_r_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/5/img")
register_coco_instances("container_rotate_r_30_6", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/6/json/6_coco_r_crop30.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/6/img")
register_coco_instances("container_rotate_r_30_7", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/7/json/7_coco_r_crop30_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/7/img")
register_coco_instances("container_rotate_r_30_8", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/8/json/8_coco_r_crop30_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/8/img")

register_coco_instances("container_rotate_r_90_1", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/1/json/1_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/1/img")
register_coco_instances("container_rotate_r_90_2", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/2/json/2_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/2/img")
register_coco_instances("container_rotate_r_90_3", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/3/json/3_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/3/img")
register_coco_instances("container_rotate_r_90_4", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/4/json/4_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/4/img")
register_coco_instances("container_rotate_r_90_5", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/5/json/5_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/5/img")
register_coco_instances("container_rotate_r_90_6", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/6/json/6_coco_r_crop90.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/6/img")
register_coco_instances("container_rotate_r_90_7", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/7/json/7_coco_r_crop90_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/7/img")
register_coco_instances("container_rotate_r_90_8", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/8/json/8_coco_r_crop90_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/8/img")

register_coco_instances("container_crop_20_1", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/1/json/1_coco_crop_20.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/1/img")
register_coco_instances("container_crop_20_2", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/2/json/2_coco_crop_20.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/2/img")
register_coco_instances("container_crop_20_3", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/3/json/3_coco_crop_20.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/3/img")
register_coco_instances("container_crop_20_4", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/4/json/4_coco_crop_20.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/4/img")
register_coco_instances("container_crop_20_5", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/5/json/5_coco_crop_20.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/5/img")
register_coco_instances("container_crop_20_6", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/6/json/6_coco_crop_20.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/6/img")
register_coco_instances("container_crop_20_7", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/7/json/7_coco_crop_20_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/7/img")
register_coco_instances("container_crop_20_8", {}, 
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/8/json/8_coco_crop_20_without_roof.json",
"/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/8/img")

# register_coco_instances("container_origin_1", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/1/json/1_coco_all.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/1/img")
# register_coco_instances("container_origin_2", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/2/json/2_coco_all.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/2/img")
# register_coco_instances("container_origin_3", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/3/json/3_coco_all.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/3/img")
# register_coco_instances("container_origin_4", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/4/json/4_coco_all.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/4/img")
# register_coco_instances("container_origin_5", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/5/json/5_coco_all.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/5/img")
# register_coco_instances("container_origin_6", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/6/json/6_coco_all.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/6/img")
# register_coco_instances("container_origin_7", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/7/json/7_coco_all.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/7/img")
# register_coco_instances("container_origin_8", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/8/json/8_coco_all.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/origin/8/img")

# register_coco_instances("container_rotate_0_1", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/1/json/1_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/1/img")
# register_coco_instances("container_rotate_0_2", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/2/json/2_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/2/img")
# register_coco_instances("container_rotate_0_3", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/3/json/3_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/3/img")
# register_coco_instances("container_rotate_0_4", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/4/json/4_coco_rotate0.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/4/img")
# register_coco_instances("container_rotate_0_5", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/5/json/5_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/5/img")
# register_coco_instances("container_rotate_0_6", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/6/json/6_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/6/img")
# register_coco_instances("container_rotate_0_7", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/7/json/7_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/7/img")
# register_coco_instances("container_rotate_0_8", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/8/json/8_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_0/8/img")

# register_coco_instances("container_rotate_l_30_1", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/1/json/1_coco_l_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/1/img")
# register_coco_instances("container_rotate_l_30_2", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/2/json/2_coco_l_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/2/img")
# register_coco_instances("container_rotate_l_30_3", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/3/json/3_coco_l_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/3/img")
# register_coco_instances("container_rotate_l_30_4", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/4/json/4_coco_l_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/4/img")
# register_coco_instances("container_rotate_l_30_5", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/5/json/5_coco_l_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/5/img")
# register_coco_instances("container_rotate_l_30_6", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/6/json/6_coco_l_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/6/img")
# register_coco_instances("container_rotate_l_30_7", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/7/json/7_coco_l_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/7/img")
# register_coco_instances("container_rotate_l_30_8", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/8/json/8_coco_l_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_30/8/img")

# register_coco_instances("container_rotate_l_90_1", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/1/json/1_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/1/img")
# register_coco_instances("container_rotate_l_90_2", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/2/json/2_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/2/img")
# register_coco_instances("container_rotate_l_90_3", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/3/json/3_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/3/img")
# register_coco_instances("container_rotate_l_90_4", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/4/json/4_coco_l_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/4/img")
# register_coco_instances("container_rotate_l_90_5", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/5/json/5_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/5/img")
# register_coco_instances("container_rotate_l_90_6", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/6/json/6_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/6/img")
# register_coco_instances("container_rotate_l_90_7", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/7/json/7_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/7/img")
# register_coco_instances("container_rotate_l_90_8", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/8/json/8_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_l_90/8/img")

# register_coco_instances("container_rotate_r_30_1", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/1/json/1_coco_r_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/1/img")
# register_coco_instances("container_rotate_r_30_2", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/2/json/2_coco_r_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/2/img")
# register_coco_instances("container_rotate_r_30_3", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/3/json/3_coco_r_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/3/img")
# register_coco_instances("container_rotate_r_30_4", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/4/json/4_coco_r_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/4/img")
# register_coco_instances("container_rotate_r_30_5", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/5/json/5_coco_r_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/5/img")
# register_coco_instances("container_rotate_r_30_6", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/6/json/6_coco_r_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/6/img")
# register_coco_instances("container_rotate_r_30_7", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/7/json/7_coco_r_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/7/img")
# register_coco_instances("container_rotate_r_30_8", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/8/json/8_coco_r_crop30.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_30/8/img")

# register_coco_instances("container_rotate_r_90_1", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/1/json/1_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/1/img")
# register_coco_instances("container_rotate_r_90_2", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/2/json/2_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/2/img")
# register_coco_instances("container_rotate_r_90_3", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/3/json/3_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/3/img")
# register_coco_instances("container_rotate_r_90_4", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/4/json/4_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/4/img")
# register_coco_instances("container_rotate_r_90_5", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/5/json/5_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/5/img")
# register_coco_instances("container_rotate_r_90_6", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/6/json/6_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/6/img")
# register_coco_instances("container_rotate_r_90_7", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/7/json/7_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/7/img")
# register_coco_instances("container_rotate_r_90_8", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/8/json/8_coco_r_crop90.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/rotate_r_90/8/img")

# register_coco_instances("container_crop_20_1", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/1/json/1_coco_crop_20.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/1/img")
# register_coco_instances("container_crop_20_2", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/2/json/2_coco_crop_20.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/2/img")
# register_coco_instances("container_crop_20_3", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/3/json/3_coco_crop_20.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/3/img")
# register_coco_instances("container_crop_20_4", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/4/json/4_coco_crop_20.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/4/img")
# register_coco_instances("container_crop_20_5", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/5/json/5_coco_crop_20.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/5/img")
# register_coco_instances("container_crop_20_6", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/6/json/6_coco_crop_20.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/6/img")
# register_coco_instances("container_crop_20_7", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/7/json/7_coco_crop_20.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/7/img")
# register_coco_instances("container_crop_20_8", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/8/json/8_coco_crop_20.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes_v2/crop_20/8/img")
# register_coco_instances("container_rotate_30", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes/coco_json/seg_all_rotate-30_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes/img_rotate-30")
# register_coco_instances("container_pad", {}, 
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes/coco_json/seg_all_pad_coco.json",
# "/media/server/data/repository/detectron2/datasets/containerseg_12classes/img_pad")
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = (
        "container_origin_1", "container_origin_2", "container_origin_3", "container_origin_4", "container_origin_5", "container_origin_6", "container_origin_7", "container_origin_8", 
        "container_rotate_0_1", "container_rotate_0_2", "container_rotate_0_3", "container_rotate_0_4", "container_rotate_0_5", "container_rotate_0_6", "container_rotate_0_7", "container_rotate_0_8", 
        "container_rotate_l_30_1", "container_rotate_l_30_2", "container_rotate_l_30_3", "container_rotate_l_30_4", "container_rotate_l_30_5", "container_rotate_l_30_6", "container_rotate_l_30_7", "container_rotate_l_30_8", 
        "container_rotate_l_90_1", "container_rotate_l_90_2", "container_rotate_l_90_3", "container_rotate_l_90_4", "container_rotate_l_90_5", "container_rotate_l_90_6", "container_rotate_l_90_7", "container_rotate_l_90_8", 
        "container_rotate_r_30_1", "container_rotate_r_30_2", "container_rotate_r_30_3", "container_rotate_r_30_4", "container_rotate_r_30_5", "container_rotate_r_30_6", "container_rotate_r_30_7", "container_rotate_r_30_8", 
        "container_rotate_r_90_1", "container_rotate_r_90_2", "container_rotate_r_90_3", "container_rotate_r_90_4", "container_rotate_r_90_5", "container_rotate_r_90_6", "container_rotate_r_90_7", "container_rotate_r_90_8", 
        "container_crop_20_1", "container_crop_20_2", "container_crop_20_3", "container_crop_20_4", "container_crop_20_5", "container_crop_20_6", "container_crop_20_7", "container_crop_20_8"
        )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19
    cfg.SOLVER.CHECKPOINT_PERIOD = 300
    # cfg.DATALOADER.NUM_WORKERS=3
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.INPUT.MAX_SIZE_TRAIN = 1200
    # cfg.INPUT.MIN_SIZE_TRAIN = (720,)
    cfg.SOLVER.BASE_LR = 0.002
    # cfg.SOLVER.STEPS = (140000, 180000)
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.MAX_ITER = 180000
    cfg.MODEL.WEIGHTS = "/media/server/data/repository/detectron2/pre_model/cascade_mask_rcnn_fpn_1x.pkl"
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args.resume = True
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
