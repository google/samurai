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


from typing import Optional
from dataflow.illumination_integration.helper import load_data, split_dataset
from dataflow.illumination_integration.dataflow import random_sample_dataflow


def get_train_val_data(
    dir,
    epoch_length,
    val_examples,
    batch_size,
    sample_roughness_0,
    samples_random_roughness,
    full_l0_val=True,
    full_l0_train=False,
    train_select: Optional[int] = None,
):
    dataset = load_data(dir)
    img_height = dataset[0][0].shape[0]
    train_dataset, val_dataset, _ = split_dataset(dataset, val_examples=val_examples)

    print("Dataset read. Now building dataflows...")

    if train_select is not None:
        train_dataset = [d[train_select : train_select + 1] for d in train_dataset]

    train_df = random_sample_dataflow(
        train_dataset,
        sample_roughness_0,
        samples_random_roughness,
        batch_size,
        with_blend=False if train_select is not None else True,
        full_l0=full_l0_train,
    )
    # Ensure length is as requested
    if len(train_df) < epoch_length:
        train_df = train_df.repeat(epoch_length // len(train_df))

    val_df = random_sample_dataflow(
        val_dataset,
        0 if full_l0_val else sample_roughness_0,
        samples_random_roughness,
        batch_size,
        with_blend=False,
        full_l0=full_l0_val,
        shuffle=False,
    )

    return train_df, val_df, img_height


def add_args(parser):
    parser.add_argument("--datadir", required=True, type=str)

    parser.add_argument("--val_holdout", type=int, default=30)

    parser.add_argument("--num_r0_samples", type=int, default=4096)
    parser.add_argument("--num_random_roughness_samples", type=int, default=4096)

    parser.add_argument("--no_full_l0_val", action="store_false")

    return parser
