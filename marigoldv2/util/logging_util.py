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

import logging
import os
import sys

from tabulate import tabulate


def config_logging(cfg_logging, out_dir=None):
    file_level = cfg_logging.get("file_level", 10)
    console_level = cfg_logging.get("console_level", 10)

    log_formatter = logging.Formatter(
        cfg_logging.get(
            "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(
            out_dir, cfg_logging.get("filename", "logging.log")
        )
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)


def eval_dict_to_text(
    val_metrics: dict,
    dataset_name: str,
    sample_list_path: str = None,
    num_samples: int = None,
):
    eval_text = f"Evaluation metrics:\n\
     on dataset: {dataset_name}\n"
    if sample_list_path is not None:
        eval_text += f"     over samples in: {sample_list_path}\n"
    if num_samples is not None:
        eval_text += f"     num_samples: {num_samples}\n"

    eval_text += tabulate([val_metrics.keys(), val_metrics.values()])
    return eval_text
