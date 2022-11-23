# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


def visualize_masks(mask, mask_pred):
    assert mask.shape == mask_pred.shape
    assert len(mask.shape) == 3 or len(mask.shape) == 2  # Rank 3
    assert mask.shape[-1] == 1

    if len(mask.shape) == 3:
        mask = mask[..., 0]
        mask_pred = mask_pred[..., 0]

    m = np.ones((mask.shape[0], mask.shape[1], 3))

    mask_binary = np.where(mask > 0.5, np.ones_like(mask), np.zeros_like(mask))
    mask_pred_binary = np.where(
        mask_pred > 0.5, np.ones_like(mask_binary), np.zeros_like(mask_binary)
    )

    m[np.logical_and(mask_binary, mask_pred_binary)] = np.array([0.1, 0.5, 0.1])
    m[np.logical_and(mask_binary, np.logical_not(mask_pred_binary))] = np.array(
        [1, 0, 0]
    )
    m[np.logical_and(np.logical_not(mask_binary), mask_pred_binary)] = np.array(
        [0, 0, 1]
    )
    return m
