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
import imageio
import tensorflow as tf
import tensorflow_addons as tfa

import nn_utils.math_utils as math_utils


class PreintegratedRenderer(tf.keras.layers.Layer):
    def __init__(
        self, brdf_integration_map_path: str, basecolor_metallic: bool = False, **kwargs
    ) -> None:
        super(PreintegratedRenderer, self).__init__(**kwargs)

        self.basecolor_metallic = basecolor_metallic

        # imageio.plugins.freeimage.download()

        brdf_map = imageio.imread(brdf_integration_map_path, format="HDR-FI")

        # Ensure B, H, W, C
        self.brdf_integration = tf.convert_to_tensor(brdf_map)[
            None, ...
        ]  # Read HDR data

    @tf.function(experimental_relax_shapes=True)
    def fresnel_schlick_roughness(self, ndotv, f0, roughness):
        with tf.name_scope("FresnelSchlickRoughness"):
            return f0 + (tf.maximum(1.0 - roughness, f0) - f0) * tf.pow(
                tf.maximum(1.0 - ndotv, 0.0), 5.0
            )

    @tf.function(experimental_relax_shapes=True)
    def calculate_reflection_direction(
        self,
        view_vector: tf.Tensor,
        normal: tf.Tensor,
        camera_pose: Optional[tf.Tensor] = None,
    ):
        """Calculates the relfection direction for a the specified view ray
        and normal. Can also use bend normals to approximate anisotropy and
        if a rotating object with a fixed illumination is used the reflected
        ray is also rotated with the inverse camera pose.

        Args:
            view_vector (tf.Tensor(float32), [batch, 3]): View vector pointing
                away from the surface.
            normal (tf.Tensor(float32), [batch, 3]): Normal vector of the surface.
            camera_pose (Optional[(tf.Tensor(float32), [3, 4])], optional):
                Defines the camera rotation matrix. If not None the rays are
                counterrotated. This is used if a capture is done with a rotating
                object. Defaults to None.

        Returns:
            reflected_ray (tf.Tensor(float32), [batch, 3]): the reflected ray in
                outgoing direction.
        """

        with tf.name_scope("ReflectionDirection"):
            normal = tf.where(normal == tf.zeros_like(normal), view_vector, normal)

            # Add the rotating object case
            if camera_pose is not None:
                with tf.name_scope("RotationCounterAct"):
                    rotation_matrix = tf.linalg.inv(camera_pose[..., :3, :3])

                    view_vector = view_vector[..., None, :] * rotation_matrix
                    view_vector = tf.reduce_sum(view_vector, -1)

            # View vector points away from the surface. For a reflection
            # The vector should be incoming (so flip) and the direction is
            # then outgoing
            reflection_vector = math_utils.reflect(-view_vector, normal)

            return view_vector, reflection_vector

    @tf.function(experimental_relax_shapes=True)
    def call(
        self,
        view_vector: tf.Tensor,
        normal: tf.Tensor,
        diffuse_irradiance: tf.Tensor,
        specular_irradiance: tf.Tensor,
        diffuse_or_basecolor: tf.Tensor,
        specular_or_metallic: tf.Tensor,
        roughness: tf.Tensor,
    ):
        """Performs the pre-integrated rendering.

        Args:
            view_vector (tf.Tensor(float32), [batch, 3]): View vector pointing
                away from the surface.
            normal (tf.Tensor(float32), [batch, 3]): Normal vector of the
                surface.
            diffuse_irradiance (tf.Tensor(float32), [batch, 3]): The diffuse
                preintegrated irradiance.
            specular_irradiance (tf.Tensor(float32), [batch, 3]): The specular
                preintegrated .
            diffuse_or_basecolor (tf.Tensor(float32), [batch, 3]): The diffuse material
                parameter.
            specular_or_metallic (tf.Tensor(float32), [batch, 3|1]): The specular material
                parameter.
            roughness (tf.Tensor(float32), [batch, 1]): The roughness material
                parameter.

        Returns:
            rendered_rgb (tf.Tensor(float32), [batch, 3]): The rendered result.
        """
        with tf.name_scope("CallPrepare"):
            normal = tf.where(normal == tf.zeros_like(normal), view_vector, normal)

            ndotv = math_utils.dot(normal, view_vector)

            if self.basecolor_metallic:
                lin_basecolor = math_utils.srgb_to_linear(diffuse_or_basecolor)

                non_metal = tf.ones_like(lin_basecolor) * 0.04

                interp_clamp = tf.clip_by_value(specular_or_metallic, 0, 1)

                lin_diffuse = lin_basecolor * (1.0 - interp_clamp)
                lin_specular = (
                    non_metal * (1.0 - interp_clamp) + lin_basecolor * interp_clamp
                )
            else:
                lin_diffuse = math_utils.srgb_to_linear(diffuse_or_basecolor)
                lin_specular = math_utils.srgb_to_linear(specular_or_metallic)

        F = self.fresnel_schlick_roughness(
            tf.maximum(ndotv, 0.0), lin_specular, roughness
        )
        kS = F
        kD = 1.0 - kS

        with tf.name_scope("Diffuse"):
            # Evaluate diffuse
            diffuse = diffuse_irradiance * lin_diffuse

        with tf.name_scope("Specular"):
            # Evaluate specular
            # Also swap from x,y indexing to ij
            envBrdfCoords = tf.concat(
                [
                    (1 - roughness) * (self.brdf_integration.shape[1] - 1),
                    tf.maximum(ndotv, 0.0) * (self.brdf_integration.shape[2] - 1),
                ],
                -1,
            )

            envBrdf = tfa.image.interpolate_bilinear(
                self.brdf_integration, envBrdfCoords[None, ...]
            )[
                0
            ]  # Remove fake batch dimension
            specular = specular_irradiance * (F * envBrdf[..., :1] + envBrdf[..., 1:2])

        # Joined ambient light
        ambient = kD * diffuse + specular
        return ambient
