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
from tkinter import ttk

from dataset_quadrant_labeler.app import App

# TODO z flip mode


def main():
    root = tk.Tk()
    root.title("Quadrant Labeler")

    change_theme = None
    # Set the initial theme
    if os.path.exists("external/sunvalley-tk/"):
        root.tk.call("source", "external/sunvalley-tk/sun-valley.tcl")
        root.tk.call("set_theme", "light")

        def change_theme():
            # NOTE: The theme's real name is sun-valley-<mode>
            if root.tk.call("ttk::style", "theme", "use") == "sun-valley-dark":
                # Set light theme
                root.tk.call("set_theme", "light")
            else:
                # Set dark theme
                root.tk.call("set_theme", "dark")

    # TODO pass change theme function

    app = App(root, change_theme)
    app.pack(fill="both", expand=True)

    # Set a minsize for the window, and place it in the middle
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    x_cordinate = int((root.winfo_screenwidth() / 2) - (root.winfo_width() / 2))
    y_cordinate = int((root.winfo_screenheight() / 2) - (root.winfo_height() / 2))
    root.geometry("+{}+{}".format(x_cordinate, y_cordinate))

    root.mainloop()


main()
