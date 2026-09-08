# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
# Modifications Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from .base_depth_dataset import (  # noqa: F401
    BaseDepthDataset,
    DatasetMode,
    get_pred_name,
)
from .diode_dataset import DIODEDepthDataset
from .eth3d_dataset import ETH3DDepthDataset
from .kitti_dataset import KITTIDepthDataset
from .nyu_dataset import NYUDepthDataset
from .scannet_dataset import ScanNetDepthDataset

dataset_name_class_dict = {
    "nyu_depth": NYUDepthDataset,
    "kitti_depth": KITTIDepthDataset,
    "eth3d_depth": ETH3DDepthDataset,
    "diode_depth": DIODEDepthDataset,
    "scannet_depth": ScanNetDepthDataset,
}


def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseDepthDataset:
    if cfg_data_split.name not in dataset_name_class_dict:
        raise NotImplementedError(cfg_data_split.name)
    dataset_class = dataset_name_class_dict[cfg_data_split.name]
    return dataset_class(
        mode=mode,
        filename_ls_path=cfg_data_split.filenames,
        dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
        **cfg_data_split,
        **kwargs,
    )
