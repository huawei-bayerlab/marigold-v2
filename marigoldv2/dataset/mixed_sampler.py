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

import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class MixedBatchSampler(BatchSampler):
    """Sample one full batch from a selected dataset with given probability."""

    def __init__(
        self,
        src_dataset_ls,
        batch_size,
        drop_last,
        shuffle,
        prob=None,
        generator=None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)

        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]

        if self.shuffle:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=RandomSampler(
                        ds, replacement=False, generator=self.generator
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        else:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=SequentialSampler(ds),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]

        self.raw_batches = [list(bs) for bs in self.src_batch_samplers]
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)

        if prob is None:
            self.prob = torch.tensor(self.n_batches, dtype=torch.float32)
            self.prob = self.prob / self.prob.sum().clamp_min(1e-12)
        else:
            self.prob = torch.as_tensor(prob, dtype=torch.float32)
            self.prob = self.prob / self.prob.sum().clamp_min(1e-12)

    def __iter__(self):
        for _ in range(self.n_total_batch):
            idx_ds = torch.multinomial(
                self.prob, 1, replacement=True, generator=self.generator
            ).item()
            if len(self.raw_batches[idx_ds]) == 0:
                self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])

            batch_raw = self.raw_batches[idx_ds].pop()
            shift = self.cum_dataset_length[idx_ds]
            yield [n + shift for n in batch_raw]

    def __len__(self):
        return self.n_total_batch
