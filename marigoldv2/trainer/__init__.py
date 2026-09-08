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

from .generic_trainer import GenericTrainer

trainer_cls_name_dict = {
    "GenericTrainer": GenericTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
