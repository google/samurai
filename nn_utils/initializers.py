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
from tensorflow.python.ops.init_ops_v2 import _compute_fans


class ScaledHyperInitializer(tf.keras.initializers.Initializer):
    def __init__(self, scale=1, seed=None):
        super().__init__()
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, _ = _compute_fans(shape)

        limit = tf.math.sqrt(6 / fan_in) * self.scale

        return tf.random.uniform(shape, -limit, limit, seed=self.seed)

    def get_config(self):
        base_config = super().get_config()
        config = {"scale": self.scale, "seed": self.seed}
        return dict(list(base_config.items()) + list(config.items()))


class HyperSirenFirstLayerInitializer(tf.keras.initializers.Initializer):
    def __init__(self, main_fan_in, scale=1.0, seed: int = None):
        super().__init__()
        self.main_fan_in = main_fan_in
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, _ = _compute_fans(shape)

        main_limit = tf.math.sqrt(3 / fan_in)

        siren_limit = self.scale / tf.math.maximum(1.0, fan_in)

        limit = main_limit * siren_limit
        return tf.random.uniform(shape, -limit, limit, seed=self.seed, dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        config = {"w0": self.w0, "c": self.c, "main_fan_in": self.main_fan_in}
        return dict(list(base_config.items()) + list(config.items()))


class HyperSirenInitializer(tf.keras.initializers.Initializer):
    def __init__(self, main_fan_in, w0: float = 30.0, c: float = 6.0, seed: int = None):
        super().__init__()
        self.main_fan_in = main_fan_in
        self.w0 = w0
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, _ = _compute_fans(shape)

        main_limit = tf.math.sqrt(3 / fan_in)
        siren_limit = (
            tf.math.sqrt(self.c / tf.math.maximum(1.0, self.main_fan_in)) / self.w0
        )

        limit = main_limit * siren_limit
        return tf.random.uniform(shape, -limit, limit, seed=self.seed, dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        config = {"w0": self.w0, "c": self.c, "main_fan_in": self.main_fan_in}
        return dict(list(base_config.items()) + list(config.items()))


class HyperInitializer(tf.keras.initializers.Initializer):
    def __init__(self, main_fan, seed=None):
        super().__init__()
        self.main_fan = main_fan
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, _ = _compute_fans(shape)

        limit = tf.math.sqrt(3 * 2 / (fan_in * self.main_fan))

        return tf.random.uniform(shape, -limit, limit, seed=self.seed)

    def get_config(self):
        base_config = super().get_config()
        config = {"main_fan": self.main_fan, "seed": self.seed}
        return dict(list(base_config.items()) + list(config.items()))


class SIRENFirstLayerInitializer(tf.keras.initializers.Initializer):
    def __init__(self, scale=1.0, seed=None):
        super().__init__()
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, _ = _compute_fans(shape)
        limit = self.scale / max(1.0, float(fan_in))
        return tf.random.uniform(shape, -limit, limit, seed=self.seed, dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        config = {"scale": self.scale, "seed": self.seed}
        return dict(list(base_config.items()) + list(config.items()))


class SIRENInitializer(tf.keras.initializers.Initializer):
    def __init__(self, w0: float = 30.0, c: float = 6.0, seed: int = None):
        super().__init__()
        self.w0 = w0
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, _ = _compute_fans(shape)
        limit = tf.math.sqrt(self.c / max(1.0, float(fan_in))) / self.w0
        return tf.random.uniform(shape, -limit, limit, seed=self.seed, dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        config = {"w0": self.w0, "c": self.c, "seed": self.seed}
        return dict(list(base_config.items()) + list(config.items()))
