# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from maskrcnn_benchmark.utils.registry import Registry

BACKBONES = Registry()

LANGUAGE_BACKBONES = Registry()

ROI_BOX_FEATURE_EXTRACTORS = Registry()
RPN_HEADS = Registry()

ROI_RELATION_FEATURE_EXTRACTORS = Registry()
ROI_RELATION_PREDICTOR = Registry()