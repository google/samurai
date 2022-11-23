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


import io

import matplotlib.pyplot as plt
import numpy as np
from farrow_and_ball import SpectralPalette, get_interpolated_palette
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    # Upper right corner
    a = 0.5 * np.asarray([2.0, 1.5, -4.0], dtype=np.float32)
    # Mid of upper line
    up1 = 0.5 * np.asarray([0.0, 1.5, -4.0], dtype=np.float32)
    # Up indicator
    up2 = 0.5 * np.asarray([0.0, 2.5, -4.0], dtype=np.float32)
    # Upper left coerner
    b = 0.5 * np.asarray([-2.0, 1.5, -4.0], dtype=np.float32)
    # Lower left corner
    c = 0.5 * np.asarray([2.0, -1.5, -4.0], dtype=np.float32)
    # Lower right corner
    d = 0.5 * np.asarray([-2.0, -1.5, -4.0], dtype=np.float32)
    # Center
    C = np.zeros(3, dtype=np.float32)
    # Forward indicator
    F = np.asarray([0, 0, -4.5], dtype=np.float32)
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = np.stack([x for x in camera_points]) * scale
    return lines


def plot_cameras(
    ax, c2w, canonical_cam: int = 0, color: str = "blue", alpha=1.0, scale: float = 0.3
):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe(scale)
    # make it homogeneous
    cam_wires_canonical = np.concatenate(
        (cam_wires_canonical, np.ones_like(cam_wires_canonical[..., :1])), -1
    )
    # Double batched matrix vector multiplication and heterogenous coordinate conversion
    # Mat has Num Cameras, 4, 4
    # Vec has Num Line Elements, 4
    # Result is Num Cameras, Num Line Element, 4 -> 3
    cam_wires_trans = np.einsum("cji,li->clj", c2w, cam_wires_canonical)
    # To heterogenous
    cam_wires_trans = cam_wires_trans[..., :3] / cam_wires_trans[..., 3:4]

    plot_handles = []
    for i, wire in enumerate(cam_wires_trans):
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.T.astype(float)
        (h,) = ax.plot(
            x_,
            y_,
            z_,
            color=color if i != canonical_cam else "red",
            linewidth=0.3,
            alpha=alpha,
        )
        plot_handles.append(h)
    return plot_handles


def plot_symmetry_plane(
    ax, sym_plane, color: str = "green", alpha=0.5, scale: float = 0.3
):
    sym_plane = np.squeeze(sym_plane)
    sym_plane = sym_plane / np.linalg.norm(sym_plane)

    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6))

    origin = np.zeros_like(sym_plane)

    d = (-origin * sym_plane).sum(axis=-1)

    a = -sym_plane[0] * x - sym_plane[1] * y - d
    b = sym_plane[2] * np.ones_like(x)
    z = np.divide(
        a,
        b,
        out=np.zeros_like(a),
        where=b != 0,
    )

    points = np.stack((x, y, z), -1)
    points = (points / np.linalg.norm(points, axis=-1, keepdims=True)) * scale * 2
    x, y, z = (points[..., 0], points[..., 1], points[..., 2])

    # OGL to matplotlib coordinates
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def plot_camera_scene(c2w, status: str, canonical_cam: int = 0, sym_plane=None):
    """
    Plots a set of predicted camera to world matrices. The plot is named with
    a string passed inside the `status` argument.
    """
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(projection="3d")
    ax.clear()
    ax.set_title(status)
    colors = get_interpolated_palette(SpectralPalette.SPEC, 3)

    origins = c2w[..., :3, 3]
    plot_radius = np.amax(np.abs(origins))

    if sym_plane is not None:
        plot_symmetry_plane(ax, sym_plane, color=colors[-1], scale=plot_radius / 4)

    plot_cameras(
        ax, c2w, color=colors[0], scale=plot_radius * 0.05, canonical_cam=canonical_cam
    )

    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("-z")
    ax.set_zlabel("y")

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=150)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )[..., :3]
    io_buf.close()

    plt.close(fig)  # Ensure figures are closed and do not leak memory

    return img_arr
