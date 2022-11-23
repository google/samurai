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
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--coarse_samples",
        type=int,
        default=64,
        help="number of coarse samples per ray in a fixed grid",
    )
    parser.add_argument(
        "-lindisp",
        "--linear_disparity_sampling",
        action="store_true",
        help="Coarse sampling linearly in disparity rather than depth",
    )

    parser.add_argument(
        "--fine_samples",
        type=int,
        default=128,
        help="number of additional samples per ray based on the coarse samples",
    )
    parser.add_argument(
        "--fourier_frequency",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding",
    )
    parser.add_argument(
        "--direction_fourier_frequency",
        type=int,
        default=4,
        help="log2 of max freq for directional ray encoding",
    )
    parser.add_argument(
        "--random_encoding_offsets", dest="random_encoding_offsets", action="store_true"
    )
    parser.add_argument(
        "--no-random_encoding_offsets",
        dest="random_encoding_offsets",
        action="store_false",
    )

    parser.add_argument(
        "--fine_net_width", type=int, default=128, help="channels per layer"
    )
    parser.add_argument(
        "--fine_net_depth", type=int, default=8, help="layers in network"
    )
    parser.add_argument(
        "--coarse_net_width", type=int, default=128, help="channels per layer"
    )
    parser.add_argument(
        "--coarse_net_depth", type=int, default=6, help="layers in network"
    )

    parser.add_argument("--appearance_latent_dim", type=int, default=32)
    parser.add_argument("--diffuse_latent_dim", type=int, default=24)
    parser.add_argument("--fix_diffuse", action="store_true")

    parser.add_argument(
        "--camera_distribution",
        choices=["sphere", "hemisphere", "frontal"],
        default="sphere",
    )

    parser.add_argument("--use_fully_random_cameras", action="store_true")
    parser.add_argument("--random_cameras_per_view", type=int, default=8)
    parser.add_argument("--min_softmax_scaler", type=float, default=1)
    parser.add_argument("--max_softmax_scaler", type=float, default=30)
    parser.add_argument("--camera_weight_update_lr", type=float, default=0.5)
    parser.add_argument("--camera_weight_update_momentum", type=float, default=0.75)

    parser.add_argument(
        "--bounding_size",
        type=float,
        default=0.5,
        help="Creates a bounding sphere with 2*bounding_size radius",
    )
    parser.add_argument("--resolution_factor", type=int, default=8)

    # Coarse configs
    parser.add_argument(
        "--advanced_loss_done",
        type=int,
        default=60000,
        help=(
            "Exponentially decays losses. After this many steps the loss is reduced"
            "by 3 magnitudes"
        ),
    )

    # Gradient settings
    parser.add_argument("--network_gradient_norm_clipping", type=float, default=0.1)
    parser.add_argument("--camera_gradient_norm_clipping", type=float, default=-1)

    # Loss settings

    # Camera optimization
    parser.add_argument("--not_learn_r", action="store_true")
    parser.add_argument("--not_learn_t", action="store_true")
    parser.add_argument("--not_learn_f", action="store_true")

    parser.add_argument("--edge_align_step", type=int, default=200)
    parser.add_argument("--num_edge_align_steps", type=int, default=50)

    parser.add_argument("--pretrained_camera_poses_folder", type=str)

    parser.add_argument("--start_f_optimization", type=int, default=90000)
    parser.add_argument("--start_fourier_anneal", type=int, default=1000)
    parser.add_argument("--finish_fourier_anneal", type=int, default=40000)
    parser.add_argument("--slow_scheduler_decay", type=int, default=70000)
    parser.add_argument("--brdf_schedule_decay", type=int, default=150000)
    parser.add_argument("--lambda_smoothness", type=float, default=0.01)
    parser.add_argument("--smoothness_bound_dividier", type=int, default=200)

    parser.add_argument("--coarse_distortion_lambda", type=float, default=1e-3)
    parser.add_argument("--fine_distortion_lambda", type=float, default=0)
    parser.add_argument("--normal_direction_lambda", type=float, default=5e-3)
    parser.add_argument("--mlp_normal_direction_lambda", type=float, default=3e-4)

    parser.add_argument("--disable_posterior_scaling", action="store_true")
    parser.add_argument("--disable_mask_uncertainty", action="store_true")

    parser.add_argument("--lambda_brdf_decoder_smoothness", type=float, default=0.1)
    parser.add_argument("--lambda_brdf_decoder_sparsity", type=float, default=0.01)

    parser.add_argument("--camera_lr", type=float, default=1e-3)
    parser.add_argument("--camera_lr_decay", type=int, default=400)
    parser.add_argument("--camera_regularization", type=float, default=1.0)
    parser.add_argument("--aim_center_regularization", type=float, default=10.0)

    parser.add_argument(
        "--camera_rotation", choices=["lookat", "sixd"], default="lookat"
    )
    parser.add_argument("--learn_camera_offsets", action="store_true")

    # DECOMPOSITION SETTINGS
    # Decide if we are decomposing or not
    parser.add_argument("--basecolor_metallic", action="store_true")
    parser.add_argument("--skip_decomposition", action="store_true")

    parser.add_argument("--compose_on_white", action="store_true")

    # Illumination configs
    parser.add_argument(
        "--rotating_object",
        action="store_true",
        help=(
            "The object is rotating instead of the camera. The illumination then "
            "needs to stay static"
        ),
    )
    parser.add_argument(
        "--single_env",
        action="store_true",
        help="All input images are captured under a single environment",
    )

    parser.add_argument(
        "--brdf_preintegration_path",
        default="data/neural_pil/BRDFLut.hdr",
        help="Path to the preintegrated BRDF LUT.",
    )

    parser.add_argument(
        "--illumination_network_path",
        default="data/neural_pil/illumination-network",
        help="Path to the illumination network config and weights",
    )

    return parser
