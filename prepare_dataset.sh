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

#!/bin/bash

echo $1
# Some libraries in python honor exif rotation other not.
# Apply the rotation so all behave the same
python -m dataset_preparer.apply_exif_rotation --dataset_path $1
# Get the segmentation masks
python -m external_run --method u2net --dataset_path $1
