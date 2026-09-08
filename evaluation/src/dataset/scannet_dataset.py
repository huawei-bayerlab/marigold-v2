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

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class ScanNetDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # ScanNet data parameter
            min_depth=1e-3,
            max_depth=10,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode ScanNet depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded
