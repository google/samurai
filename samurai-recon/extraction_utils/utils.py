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


import functools
import math as m
from enum import Enum
from typing import List, NamedTuple

import dataflow.samurai as data
import numpy as np
import tensorflow as tf
from models.samurai.samurai_model import SamuraiModel

from utils.decorator import timing


def Rx(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, m.cos(theta), -m.sin(theta)],
            [0, m.sin(theta), m.cos(theta)],
        ]
    )


def Ry(theta):
    return np.array(
        [
            [m.cos(theta), 0, m.sin(theta)],
            [0, 1, 0],
            [-m.sin(theta), 0, m.cos(theta)],
        ]
    )


def Rz(theta):
    return np.array(
        [
            [m.cos(theta), -m.sin(theta), 0],
            [m.sin(theta), m.cos(theta), 0],
            [0, 0, 1],
        ]
    )


class RotationDefinition(Enum):
    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, rotation_matrix_builder):
        self._rotation_matrix_ = rotation_matrix_builder

    X_AXIS = "x", lambda x: Rx(m.radians(x))
    Y_AXIS = "y", lambda y: Ry(m.radians(y))
    Z_AXIS = "z", lambda z: Rz(m.radians(z))

    def __str__(self):
        return self.value

    # this makes sure that the description is read-only
    @property
    def rotation_matrix(self):
        return self._rotation_matrix_


class Rotation(NamedTuple):
    rotation: RotationDefinition
    degree: float


def build_rotation_matrix(rotations: List[Rotation]):
    return functools.reduce(
        lambda acc, x: x.rotation.rotation_matrix(x.degree) @ acc, rotations, np.eye(3)
    )


@timing
def get_samurai_model(args):
    # Create the dataflow
    (
        image_shapes,
        init_c2w,
        init_focal,
        init_directions,
        image_request_function,
        train_df,
        val_df,
        test_df,
    ) = data.create_dataflow(args)

    samurai = SamuraiModel(
        len(image_shapes),
        len(train_df),
        image_shapes,
        args,
        image_request_function,
        init_directions=init_directions,
        init_c2w=init_c2w,
        init_focal=init_focal,
    )
    full_batch, _, _ = samurai.full_image_batch_data(
        tf.constant([0]),
        12,
        1,
        tf.constant([400, 400], dtype=tf.int32),
        data.InputTargets(tf.zeros((1, 400, 400, 3)), tf.zeros((1, 400, 400, 1))),
    )
    samurai(full_batch, 0)  # Call it with full data

    samurai.restore()

    return samurai, train_df, test_df
