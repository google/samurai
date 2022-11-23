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
from skimage.morphology import binary_dilation, dilation, square


def dilate(img, mask, iterations=10):
    structure = square(3)

    oldMask = mask
    oldImg = img
    for i in range(iterations):
        newMask = binary_dilation(oldMask, structure)

        newImg = np.stack(
            [dilation(oldImg[..., i], structure) for i in range(oldImg.shape[-1])], -1
        )

        diffMask = np.expand_dims(np.where(newMask, 1, 0) - np.where(oldMask, 1, 0), -1)

        oldMask = newMask
        oldImg = diffMask * newImg + (1 - diffMask) * oldImg

    return oldImg
