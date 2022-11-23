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


from PIL import Image, ExifTags, ImageOps
import os
import shutil


def resave(img_idx, path):
    extension = os.path.splitext(path)[1]
    save_path = os.path.join(os.path.dirname(path), "image", f"{img_idx}.jpg")

    try:
        with Image.open(path) as image:
            ImageOps.exif_transpose(image).save(save_path)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        shutil.copy(path, save_path)


def main(args):
    path = args.dataset_path
    os.makedirs(os.path.join(path, "image"), exist_ok=True)

    image_path = sorted(
        [
            os.path.join(path, f)
            for f in os.listdir(path)
            if any([ext in f for ext in [".jpg", ".png"]])
        ]
    )

    [resave(i, p) for i, p in enumerate(image_path)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")

    args = parser.parse_args()

    main(args)
