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


import math
import os
from collections import namedtuple
from typing import List, Optional, Tuple, Union
import random
import json

import imageio
import numpy as np
from dataflow.samurai.quadrants import combine_direction, Direction

# imageio.plugins.freeimage.download()

DatasetReturn = namedtuple(
    "DatasetReturn",
    (
        "img_idx",
        "image_list",
    ),
)


class SamuraiDataset:
    def __init__(self, data_dir: str, dirs_ext_to_read: List[Tuple[str, str]]):
        super().__init__()
        data_dir = data_dir
        all_dirs = [os.path.join(data_dir, d[0]) for d in dirs_ext_to_read]
        img_names = sorted(
            [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(all_dirs[0]) if any([ext in f.lower() for ext in [".jpg", '.jpeg', ".png"]])],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )

        with open(os.path.join(data_dir, "quadrants.json"), "r") as fp:
            self.quadrant_info = json.load(fp)

        self.directions = np.stack(
            [
                combine_direction(*[Direction(e) for e in d["quadrant"]])
                for d in self.quadrant_info["frames"]
            ],
            0,
        ).astype(np.float32)
        print(self.directions.shape)

        extensions = [d[1] for d in dirs_ext_to_read]

        self.channels = [d[2] for d in dirs_ext_to_read]

        self.num_img_types = len(dirs_ext_to_read)
        self.image_type_paths = [
            [os.path.join(p, n + e) for n in img_names]  # Path/Name.Extension
            for p, e in zip(
                all_dirs, extensions
            )  # all dirs and extensions are derived from dirs_ext_to_read
        ]

    def get_directions(self):
        return self.directions

    def __len__(self):
        return len(self.image_type_paths[0])

    def get_image_shapes(self, max_size):
        img_type = self.image_type_paths[0]
        shapes = []
        for i in range(self.__len__()):
            dims = np.asarray(self.read_image(img_type[i]).shape[:-1])
            max_dim = np.max(dims)
            factor = max_size / max_dim
            dims = (dims * factor).astype(np.int32)

            shapes.append(dims)

        return np.stack(shapes, 0)

    def read_image(self, path) -> np.ndarray:
        img = imageio.imread(path)
        if img.dtype == np.uint8:
            img = img / 255
        img = np.nan_to_num(img)
        return img

    def get_image_iter(self, shuffle=False):
        idxs = list(range(len(self.image_type_paths[0])))

        if shuffle:
            random.shuffle(idxs)

        for idx in idxs:
            yield self[idx]

    def get_image_paths(self):
        idxs = list(range(len(self.image_type_paths[0])))
        imgs = []

        for img_type in self.image_type_paths:
            types = []
            for idx in idxs:
                types.append(img_type[idx])

            imgs.append(types)

        return (idxs, *imgs)

    def get_image_path_from_indices(self, idxs):
        if isinstance(idxs, list) or isinstance(idxs, np.ndarray):
            imgs = []
            for img_type in self.image_type_paths:
                types = []

                for idx in idxs:
                    types.append(img_type[idx])
                imgs.append(types)

            return (idxs, *imgs)
        else:
            import tensorflow as tf

            if isinstance(idxs, tf.Tensor):
                types = []
                for img_type in self.image_type_paths:
                    types.append(tf.gather_nd(img_type, tf.reshape(idxs, (-1, 1))))

                imgs = tf.stack(types, 0)

                type_list = tf.unstack(imgs, num=len(self.image_type_paths))
                return (idxs, *type_list)

    def __getitem__(self, idx):
        return (
            idx,
            np.stack(
                [self.read_image(img_type[idx]) for img_type in self.image_type_paths],
                0,
            ),
        )
