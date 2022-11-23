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


def add_args(parser):
    parser.add_argument(
        "--datadir",
        required=True,
        type=str,
        help="Path to dataset location.",
    )

    parser.add_argument(
        "--max_resolution_dimension",
        type=int,
        default=800,
        help="Scales a image so the maximum resolution is at most the specified value",
    )

    parser.add_argument(
        "--test_holdout", type=int, default=16, help="Test holdout stride"
    )

    parser.add_argument("--dataset", choices=["samurai", "nerd"], default="samurai")
    parser.add_argument("--load_gt_poses", action="store_true")
    parser.add_argument("--canonical_pose", type=int, default=0)

    return parser
