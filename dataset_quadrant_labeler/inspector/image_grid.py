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
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk

from dataset_quadrant_labeler.inspector.tooltip import Tooltip
from PIL import Image, ImageTk


class ImageGrid(ttk.LabelFrame):
    def __init__(
        self,
        parent,
        paths,
        name,
        columns=4,
        max_dim=200,
        padding=(20, 10),
        inner_padding=(5, 10),
    ):
        ttk.LabelFrame.__init__(self, parent, text=name, padding=padding)

        self.image_paths = paths
        self.columns = columns
        self.max_dim = max_dim
        self.inner_padding = inner_padding

        self.setup_widget()

    def load_images(self):
        self.images = []
        for img_path in self.image_paths:
            img = Image.open(img_path)

            max_img_dim = max(img.size[0], img.size[1])
            factor = self.max_dim / max_img_dim
            img = img.resize(
                (int(img.size[0] * factor), int(img.size[1] * factor)), Image.ANTIALIAS
            )

            tk_img = ImageTk.PhotoImage(img)
            self.images.append(tk_img)

    def setup_widget(self):
        self.load_images()

        self.canvas = []
        for i, path in enumerate(self.image_paths):
            canvas = tk.Canvas(self, width=self.max_dim, height=self.max_dim)

            # Place in grid
            row = i // self.columns
            column = i % self.columns
            canvas.grid(
                row=row,
                column=column,
                padx=self.inner_padding[0],
                pady=self.inner_padding[1],
                sticky="nsew",
            )

            canvas.create_image(0, 0, anchor=tk.NW, image=self.images[i])

            Tooltip(canvas, text=os.path.basename(path))

            self.canvas.append(canvas)
