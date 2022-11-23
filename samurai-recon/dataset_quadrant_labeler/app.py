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
from tkinter import filedialog as fd
from tkinter import ttk

from PIL import Image, ImageTk
from dataset_quadrant_labeler.inspector.inspector import Inspector


class App(ttk.Frame):
    def __init__(self, parent, change_theme_func):
        ttk.Frame.__init__(self)

        self.lcr = ["Left", "Center", "Right"]
        self.acb = ["Above", "Center", "Below"]
        self.fcb = ["Front", "Center", "Back"]

        self.lcr_keyb = ["q", "a", "z"]
        self.acb_keyb = ["w", "s", "x"]
        self.fcb_keyb = ["e", "d", "c"]

        self.change_theme_func = change_theme_func

        self.target_dim = 400

        self.reset_state()

        self.setup_widgets(parent)

    def show_inspector(self):
        Inspector(
            self,
            os.path.join(os.path.dirname(self.folder_selected), "quadrants.json"),
            self.folder_selected,
        )

    def all_center(self):
        return self.x_var.get() == 2 and self.y_var.get() == 2 and self.z_var.get() == 2

    def reset_state(self):
        print("Resetting")
        self.images_loaded = False
        self.images_left = False
        self.img_names = []

        self.quadrants = {"frames": []}

        self.image_index = 0

    def setup_axis_groups(
        self, parent, frame, name, column, radios_labels, keyboard_keys
    ):
        print("Setup axis groups")
        frame = ttk.LabelFrame(self, text=name, padding=(20, 10))
        frame.grid(row=2, column=column, padx=(20, 10), pady=(20, 10), sticky="nsew")

        radioboxes = []
        radio_var = tk.IntVar(value=2)
        for i, (label, key) in enumerate(zip(radios_labels, keyboard_keys)):
            radio = ttk.Radiobutton(
                frame, text=f"{label} ({key.upper()})", variable=radio_var, value=i + 1
            )
            radio.grid(row=i, column=0, padx=5, pady=10, sticky="nsew")

            radioboxes.append(radio)

            parent.bind(key, lambda k: radio_var.set(keyboard_keys.index(k.char) + 1))

        return radioboxes, radio_var

    def update_next_save_label(self):
        print("Update next save label")
        self.next_or_save_button["state"] = (
            tk.NORMAL if self.images_loaded else tk.DISABLED
        )
        self.next_or_save_button["text"] = (
            "Next (F)" if self.images_left else "Save (F)"
        )

    def update_prev_label(self):
        is_activate = self.images_loaded and self.image_index > 0
        self.prev_button["state"] = tk.NORMAL if is_activate else tk.DISABLED
        self.prev_button["text"] = "Prev (R)" if is_activate else "Prev (Not possible)"

    def current_button_state_to_quadrant(self):
        x = self.lcr[self.x_var.get() - 1]
        y = self.acb[self.y_var.get() - 1]
        z = self.fcb[self.z_var.get() - 1]

        return [x, y, z]

    def prev(self):
        is_activate = self.images_loaded and self.image_index > 0
        if is_activate:
            self.image_index -= 1

            # Pop quadrant
            self.quadrants["frames"] = self.quadrants["frames"][:-1]
            print(self.quadrants)

            self.update_image()
            self.update_next_save_label()
            self.update_prev_label()

    def next_or_save(self):
        print("Next or Save")
        print(
            self.images_loaded,
            self.images_left,
            self.image_index,
            len(self.img_names),
        )
        if self.images_loaded and not (self.all_center()):
            image_left = self.image_index < len(self.img_names) - 1
            self.images_left = image_left

            # Always add the image
            img_name = self.img_names[self.image_index]
            self.quadrants["frames"] += [
                {
                    "image_name": img_name,
                    "quadrant": self.current_button_state_to_quadrant(),
                }
            ]

            print(self.images_left)
            if image_left:
                print("\tNext image...")
                self.image_index = self.image_index + 1
                # Set image
                print(self.quadrants)

                self.update_image()
                self.update_next_save_label()
                self.update_prev_label()
            else:
                print("\tSaving...")
                # Save quadrant file

                quadrant_info = json.dumps(
                    self.quadrants,
                    indent=4,
                )
                with open(
                    os.path.join(
                        os.path.dirname(self.folder_selected), "quadrants.json"
                    ),
                    "w",
                ) as fp:
                    fp.write(quadrant_info)

                # Remove image
                self.canvas.delete("all")

                # Reset state
                self.reset_state()

                # Update UI state
                self.update_next_save_label()
                self.update_prev_label()
                self.update_inspector_label()

    def update_image(self):
        print("Update Image")

        self.tk_img = None
        self.canvas.delete("all")

        img_name = self.img_names[self.image_index]
        full_path = os.path.join(self.folder_selected, img_name)
        print(full_path)

        img = Image.open(full_path)

        max_img_dim = max(img.size[0], img.size[1])
        factor = self.target_dim / max_img_dim
        img = img.resize(
            (int(img.size[0] * factor), int(img.size[1] * factor)), Image.ANTIALIAS
        )

        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def update_inspector_label(self):
        has_quadrants_file = os.path.exists(
            os.path.join(os.path.dirname(self.folder_selected), "quadrants.json")
        )
        is_activate = self.images_loaded and has_quadrants_file

        self.inspector_button["state"] = tk.NORMAL if is_activate else tk.DISABLED
        self.inspector_button["text"] = (
            "Inspect" if is_activate else "Inspect (Not Processed)"
        )

    def select_folder(self):
        print("Select Folder")
        self.reset_state()

        self.folder_selected = fd.askdirectory()

        self.img_names = sorted(
            [os.path.basename(f) for f in os.listdir(self.folder_selected)],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )

        # Set state

        self.images_loaded = True
        self.images_left = True

        self.update_next_save_label()
        self.update_prev_label()
        self.update_inspector_label()
        self.update_image()

        # Remove the focus
        self.canvas.focus_set()

    def setup_widgets(self, parent):
        print("Setup Widgets")

        top_button_frame = ttk.LabelFrame(self, text="Controls", padding=(20, 10))
        top_button_frame.grid(
            row=0, column=0, columnspan=3, padx=(20, 10), pady=(20, 10), sticky="nsew"
        )

        self.open_folder_button = ttk.Button(
            top_button_frame, text="Open Folder", command=self.select_folder
        )
        self.open_folder_button.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        self.next_or_save_button = ttk.Button(
            top_button_frame,
            text="Disabled",
            state=tk.DISABLED,
            command=self.next_or_save,
        )
        self.next_or_save_button.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")
        parent.bind("f", lambda k: self.next_or_save())

        self.prev_button = ttk.Button(
            top_button_frame,
            text="Disabled",
            state=tk.DISABLED,
            command=self.prev,
        )
        self.prev_button.grid(row=0, column=2, padx=5, pady=10, sticky="nsew")
        parent.bind("r", lambda k: self.prev())

        self.inspector_button = ttk.Button(
            top_button_frame,
            text="Inspect (Disabled)",
            state=tk.DISABLED,
            command=self.show_inspector,
        )
        self.inspector_button.grid(row=0, column=3, padx=5, pady=10, sticky="nsew")

        if self.change_theme_func is not None:
            self.change_theme_button = ttk.Button(
                top_button_frame,
                text="Change theme",
                command=self.change_theme_func,
            )
            self.change_theme_button.grid(
                row=0, column=4, padx=5, pady=10, sticky="nsew"
            )

        image_frame = ttk.LabelFrame(self, text="Image", padding=(20, 10))
        image_frame.grid(
            row=1, column=0, columnspan=3, padx=(20, 10), pady=(20, 10), sticky="nsew"
        )

        self.canvas = tk.Canvas(
            image_frame, width=self.target_dim, height=self.target_dim
        )
        self.canvas.pack()

        self.x_frame, self.y_frame, self.z_frame = None, None, None

        self.x_box, self.x_var = self.setup_axis_groups(
            parent, self.x_frame, "X-Axis", 0, self.lcr, self.lcr_keyb
        )
        self.y_box, self.y_var = self.setup_axis_groups(
            parent, self.y_frame, "Y-Axis", 1, self.acb, self.acb_keyb
        )
        self.z_box, self.z_var = self.setup_axis_groups(
            parent, self.z_frame, "Z-Axis", 2, self.fcb, self.fcb_keyb
        )
