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

import numpy as np
import os
import tarfile
import torch
from io import BytesIO

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode, DatasetMode


class DIODEDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # DIODE data parameter
            min_depth=0.6,
            max_depth=350,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_npy_file(self, rel_path):
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            fileobj = self.tar_obj.extractfile("./" + rel_path)
            npy_path_or_content = BytesIO(fileobj.read())
        else:
            npy_path_or_content = os.path.join(self.dataset_dir, rel_path)
        data = np.load(npy_path_or_content).squeeze()[np.newaxis, :, :]
        return data

    def _read_depth_file(self, rel_path):
        depth = self._read_npy_file(rel_path)
        return depth

    def _get_data_path(self, index):
        return self.filenames[index]

    def _get_data_item(self, index):
        # Special: depth mask is read from data

        rgb_rel_path, depth_rel_path, mask_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=None
            )
            rasters.update(depth_data)

            # valid mask
            mask = self._read_npy_file(mask_rel_path).astype(bool)
            mask = torch.from_numpy(mask).bool()
            rasters["valid_mask_raw"] = mask.clone()
            rasters["valid_mask_filled"] = mask.clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other
