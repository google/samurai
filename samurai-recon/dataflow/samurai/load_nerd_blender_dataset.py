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


import json
import math
import os
import random
from collections import namedtuple
from typing import List, Optional, Tuple, Union

import imageio
import numpy as np
from dataflow.samurai.quadrants import combine_direction, Direction

DatasetReturn = namedtuple(
    "DatasetReturn",
    (
        "img_idx",
        "pose",
        "image_list",
    ),
)


class NerdDataset:
    def __init__(self, data_dir: str, load_gt_poses: bool):
        splits = ["train"]
        metas = {}

        self.load_gt_poses = load_gt_poses
        for s in splits:
            with open(os.path.join(data_dir, "quadrants_{}.json".format(s)), "r") as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_masks = []
        all_poses = []
        all_directions = []

        # counts = [0]
        meta = None
        for s in splits:
            meta = metas[s]
            imgs = []
            masks = []
            poses = []
            direction = []

            for frame in meta["frames"]:
                # Read the image
                fname = os.path.join(data_dir, frame["file_path"])
                dirname = os.path.dirname(fname)

                img_dir = os.path.join(dirname, "image")
                mask_dir = os.path.join(dirname, "mask")

                file_name = os.path.splitext(os.path.basename(fname))[0]
                file_name = file_name.replace("r_", "") + ".jpg"

                imgs.append(os.path.join(img_dir, file_name))
                masks.append(os.path.join(mask_dir, file_name))

                # Read the poses
                poses.append(np.array(frame["transform_matrix"]))

                quads = [Direction(d) for d in frame["quadrant"]]
                direction.append(combine_direction(*quads))

            # Convert poses to numpy
            poses = np.array(poses).astype(np.float32)
            directions = np.array(direction).astype(np.float32)

            all_imgs.append(imgs)
            all_masks.append(masks)
            all_poses.append(poses)
            all_directions.append(directions)

        imgs = np.concatenate(all_imgs, 0).tolist()
        masks = np.concatenate(all_masks, 0).tolist()
        self.image_type_paths = [imgs, masks]
        self.channels = [3, 1]
        self.poses = np.concatenate(all_poses, 0).astype(np.float32)
        self.directions = np.concatenate(all_directions, 0).astype(np.float32)

        camera_angle_x = float(meta["camera_angle_x"])

        self.focal_convert = lambda W: 0.5 * W / np.tan(0.5 * camera_angle_x)
        self.org_shapes = None

    def __len__(self):
        return len(self.image_type_paths[0])

    def get_poses(self):
        if self.load_gt_poses:
            return self.poses
        else:
            return None

    def get_directions(self):
        return self.directions

    def get_focal_length(self, image_shapes):
        if self.load_gt_poses:
            widths = image_shapes[..., 1:2].astype(np.float32)
            org_widths = self.org_shapes[..., 1:2].astype(np.float32)

            scaler = widths / org_widths
            return self.focal_convert(org_widths).astype(np.float32) * scaler
        else:
            return None

    def get_image_shapes(self, max_size):
        img_type = self.image_type_paths[0]

        if self.org_shapes is None:
            org_shapes = []
            for i in range(self.__len__()):
                dims = np.asarray(self.read_image(img_type[i]).shape[:-1])
                org_shapes.append(dims)
            self.org_shapes = np.stack(org_shapes, 0)

        max_dim = np.max(self.org_shapes, axis=1, keepdims=True)
        factor = max_size / max_dim
        new_dims = (self.org_shapes * factor).astype(np.int32)

        return new_dims

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
        imgs = []

        for img_type in self.image_type_paths:
            types = []
            for idx in idxs:
                types.append(img_type[idx])

            imgs.append(types)

        return (idxs, *imgs)

    def __getitem__(self, idx):
        return (
            idx,
            np.stack(
                [self.read_image(img_type[idx]) for img_type in self.image_type_paths],
                0,
            ),
        )
