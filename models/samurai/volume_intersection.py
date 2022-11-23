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


import tensorflow as tf
from collections import namedtuple

import tensorflow as tf
import numpy as np
from models.samurai.camera_store import Ray
import nn_utils.math_utils as math_utils

RaySphereIntersection = namedtuple("RaySphereIntersection", ["is_hit", "tmin", "tmax"])
Sphere = namedtuple("Sphere", ["radius"])


def solve_quadratic(a, b, c):
    discr = b * b - 4.0 * a * c
    # This indicates three cases
    # < 0 -> No solution exists -> No intersection
    # = 0 -> One solution exists -> Tangent touches sphere
    # > 0 -> Two solutions exists -> Ray goes through sphere

    # Quickly decide if we hit the sphere
    hit = discr >= 0.0  # Either 0 or >0

    # Then calculate the intersection distances
    q = tf.where(
        b > 0,
        -0.5 * (b + math_utils.safe_sqrt(discr)),
        -0.5 * (b - math_utils.safe_sqrt(discr)),
    )
    x0 = tf.where(
        discr < 0,
        tf.ones_like(a) * -1,  # Non hit
        tf.where(
            discr == 0,
            -0.5 * tf.math.divide_no_nan(b, a),  # Tangent
            tf.math.divide_no_nan(q, a),  # Double hit
        ),
    )
    x1 = tf.where(
        discr < 0,
        tf.ones_like(a) * -1,  # Non hit
        tf.where(
            discr == 0,
            -0.5 * tf.math.divide_no_nan(b, a),  # Tangent
            tf.math.divide_no_nan(c, q),  # Double hit
        ),
    )

    # actually find the min and max intersection point
    tmin = tf.minimum(x0, x1)
    tmax = tf.maximum(x0, x1)

    return hit, tmin, tmax


def performRaySphereIntersection(ray: Ray, sphere: Sphere) -> RaySphereIntersection:
    oc = ray.origin  # - sphere center (Here always 0)

    a = math_utils.dot(ray.direction, ray.direction)
    b = 2.0 * math_utils.dot(oc, ray.direction)
    c = math_utils.dot(oc, oc) - sphere.radius * sphere.radius

    hit, tmin, tmax = solve_quadratic(a, b, c)

    # Remove the channel dimensions everywhere
    return RaySphereIntersection(
        hit[..., 0],
        tmin[..., 0],
        tmax[..., 0],
    )


def sphere_sdf(p, r):
    return math_utils.magnitude(p) - r


def distance_to_alpha(distance, bounding_distance, scaler=5):
    # Simple logistic function
    # Create a threshold where we go from 1 alpha to 0
    threshold = bounding_distance / scaler
    factor = scaler / bounding_distance

    # Distance is negative. Start within the sphere offset by the threshold.
    distance_mt = distance + threshold

    alpha_inv = tf.math.reciprocal_no_nan(
        1.0 + math_utils.safe_exp(-factor * (distance_mt - threshold / 2))
    )

    return 1 - alpha_inv
