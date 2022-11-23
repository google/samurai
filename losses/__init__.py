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


import tensorflow as tf
import inspect


def multi_gpu_wrapper(loss_fn, global_batch_size):
    if inspect.isclass(loss_fn) and issubclass(loss_fn, tf.keras.losses.Loss):
        loss_obj = loss_fn(reduction=tf.keras.losses.Reduction.NONE)
    else:
        loss_obj = loss_fn

    def calculate_loss(*loss_args):
        per_example_loss = loss_obj(*loss_args)
        return tf.reduce_sum(per_example_loss) / global_batch_size

    return calculate_loss


def l2_regularization(x):
    return tf.reduce_mean(tf.square(x), -1)


class SegmentationMaskBackgroundLoss(tf.keras.losses.Loss):
    def call(self, x, mask):
        # The background loss punishes all values directly
        loss_background = tf.square(x - mask)

        # In the foreground we do not know where information should be placed
        loss_foreground = tf.zeros_like(loss_background)

        bg_loss = loss_background * (1 - mask)
        fg_loss = loss_foreground * mask
        return tf.reduce_sum(
            tf.where(
                tf.broadcast_to(tf.less(mask, 0.1), bg_loss.shape), bg_loss, fg_loss
            ),
            -1,
        )
