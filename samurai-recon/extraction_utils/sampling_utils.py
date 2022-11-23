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
from typing import Callable


class Pdf3d:
    """A 3d distribution that can be used to importance sample a 3d volume."""

    def __init__(self, coords_3d: np.ndarray, values: np.ndarray):
        """Args:
        coords_3d: 3d coordinates on a 3d grid.
        values: The importance values.
        """
        self.coords_3d = coords_3d.reshape([-1, 3])
        self.cdf = self.create_cdf(coords_3d, values)
        self.values = values.reshape([-1])
        self.point_size = 10 * np.var(coords_3d) / coords_3d.shape[0] ** (1 / 3)

    def create_cdf(self, coords_3d, values):
        cdf = np.cumsum(values.reshape([-1]))
        cdf /= cdf[-1]
        return cdf

    def sample_point(self, count: int) -> np.ndarray:
        random_variables = np.random.rand(count)
        linear_idxs = np.searchsorted(self.cdf, random_variables)
        values = self.values[linear_idxs]
        if not np.all(values > 0):
            print("Ohoooo, alarm! alarm!", np.sum(values > 0))

        coords = self.coords_3d[linear_idxs]
        offsets = 2 * (np.random.rand(count, 3) - 0.5)
        result = coords + offsets * self.point_size

        return result


def create_fine_sampling_generator(
    initial_sampling_grid,
    sigma_initial: np.ndarray,
) -> Callable[[int], np.ndarray]:
    sigma_flat = sigma_initial.reshape([-1])
    sigma_flat[sigma_flat < 0] = 0
    sigma_scaled = sigma_flat / np.max(sigma_flat)
    distribution = Pdf3d(initial_sampling_grid, sigma_scaled)

    return lambda batch_size: distribution.sample_point(batch_size)


def regular_3d_grid(
    center: np.ndarray, side_length: float, points_per_dimension: int
) -> np.ndarray:
    xs = np.linspace(
        center[0] - side_length, center[0] + side_length, points_per_dimension
    )
    ys = np.linspace(
        center[1] - side_length, center[1] + side_length, points_per_dimension
    )
    zs = np.linspace(
        center[2] - side_length, center[2] + side_length, points_per_dimension
    )
    xs, ys, zs = np.meshgrid(xs, ys, zs)

    coords = np.stack([xs, ys, zs], -1)
    return coords
