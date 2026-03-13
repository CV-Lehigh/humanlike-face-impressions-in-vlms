#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from hycoclip.config import LazyCall as L
from hycoclip.evaluation.retrieval import ZeroShotRetrievalEvaluator


evaluator = L(ZeroShotRetrievalEvaluator)(
    datasets=["coco", "flickr30k"],
    #datasets=["coco"],
    data_dir="/data/crr221/Project_data/Local_Image_Text_Correspondence/Datasets",
    image_size=224,
)
