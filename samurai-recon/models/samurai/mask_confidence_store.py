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
import tensorflow_addons as tfa
import nn_utils.math_utils as math_utils


class MaskConfidenceStore(tf.keras.layers.Layer):
    def __init__(self, num_samples: int, max_points: int = 34, **kwargs) -> None:
        super(MaskConfidenceStore, self).__init__()
        self.max_points = max_points
        self.confidences = tf.Variable(
            initial_value=tf.ones((num_samples, max_points * max_points)),
            name="confidence",
            trainable=True,
        )

    def get_confidences(self, idxs):
        idxs_lookup = tf.reshape(tf.convert_to_tensor(idxs, dtype=tf.int32), (-1, 1))
        return tf.gather_nd(self.confidences, idxs_lookup)

    def call(self, idxs, height: int, width: int):
        confidences = self.get_confidences(idxs)

        confidences_reshaped = tf.reshape(
            confidences, (-1, self.max_points, self.max_points, 1)
        )
        return math_utils.saturate(
            tf.image.resize(confidences_reshaped, [height, width], method="bicubic")
        )

    def get_regularization_loss(self, idxs):
        confs = self.get_confidences(idxs)
        return tf.reduce_sum(tf.square(confs - 1))

    def apply_confidence_to_mask(self, mask, confidence):
        return (confidence * mask) + ((1 - confidence) * (1 - mask))

    def apply_confidence_idx_to_masks(self, idxs, masks):
        confs = self.get_confidence_for_mask(idxs, masks)
        return self.apply_confidence_to_mask(masks, confs)

    def get_confidence_for_mask(self, idxs, masks):
        mask_shape = tf.shape(masks)
        return self.call(idxs, mask_shape[1], mask_shape[2])
