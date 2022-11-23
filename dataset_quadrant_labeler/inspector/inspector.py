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


import os
import json
import tkinter as tk
from tkinter import ttk

from dataset_quadrant_labeler.inspector.image_grid import ImageGrid
from dataset_quadrant_labeler.inspector.scrollable_frame import ScrollableFrame


class Inspector(tk.Toplevel):
    def __init__(self, parent, quadrant_path, image_path, columns=4, max_dim=200):
        tk.Toplevel.__init__(self, parent)
        self.title = "Quadrant Inspector"

        self.quadrant_file_path = quadrant_path
        self.image_path = image_path

        self.columns = columns
        self.max_dim = max_dim

        self.padding = (20, 10)
        self.inner_padding = (5, 10)

        width = (
            (columns * (max_dim + 2 * self.inner_padding[0])) + 2 * self.padding[0] + 50
        )
        height = int(parent.winfo_screenheight() / 1.5)

        self.geometry(f"{width}x{height}")

        self.setup_widgets()

    def build_quadrant_image_dict(self):
        with open(self.quadrant_file_path, "r") as fp:
            self.quadrant_info = json.load(fp)

        frames = self.quadrant_info["frames"]

        quadrants = {}
        for e in frames:
            quadrant_name = " ".join([q for q in e["quadrant"] if q != "Center"])
            quadrants[quadrant_name] = quadrants.get(quadrant_name, []) + [
                e["image_name"]
            ]

        self.quadrants = quadrants

    def setup_widgets(self):
        self.frame = ScrollableFrame(self)

        self.build_quadrant_image_dict()
        self.image_grids = []

        for quadrant, images in self.quadrants.items():
            image_grid = ImageGrid(
                self.frame.scrollable_frame,
                [os.path.join(self.image_path, e) for e in images],
                quadrant,
            )
            image_grid.pack(expand=True, fill=tk.X)
            self.image_grids.append(image_grid)

        self.frame.pack(fill=tk.BOTH, expand=True)
