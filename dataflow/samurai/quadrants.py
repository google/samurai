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


import enum
import functools
import itertools

import numpy as np


def direction_dict():
    modes = ["Positive", "Negative", "Center"]
    possibilities = itertools.product(modes, modes, modes)

    def possibility_to_dirs(possibility):
        dirs = []
        for i, p in enumerate(possibility):
            if p == "Positive":
                if i == 0:
                    dirs.append(Direction.RIGHT)
                elif i == 1:
                    dirs.append(Direction.ABOVE)
                elif i == 2:
                    dirs.append(Direction.FRONT)
            elif p == "Negative":
                if i == 0:
                    dirs.append(Direction.LEFT)
                elif i == 1:
                    dirs.append(Direction.BELOW)
                elif i == 2:
                    dirs.append(Direction.BACK)

        return dirs

    all_dirs = [possibility_to_dirs(p) for p in possibilities]
    all_dirs = filter(lambda d: d, all_dirs)

    return [(combine_direction(*d), d) for d in all_dirs]


class Direction(enum.Enum):
    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, direction):
        self._direction_ = direction

    CENTER = "Center", np.array([0, 0, 0], dtype=np.float32)

    LEFT = "Left", np.array([-1, 0, 0], dtype=np.float32)
    RIGHT = "Right", np.array([1, 0, 0], dtype=np.float32)

    FRONT = "Front", np.array([0, 0, 1], dtype=np.float32)
    BACK = "Back", np.array([0, 0, -1], dtype=np.float32)

    ABOVE = "Above", np.array([0, 1, 0], dtype=np.float32)
    BELOW = "Below", np.array([0, -1, 0], dtype=np.float32)

    def __str__(self):
        return self.value

    # this makes sure that the description is read-only
    @property
    def direction(self):
        return self._direction_


def combine_direction(*directions: Direction):
    joined_dir = functools.reduce(
        lambda x, y: x + y.direction, directions, Direction.CENTER.direction
    )
    mag = np.sqrt(joined_dir.dot(joined_dir))
    return joined_dir / mag
