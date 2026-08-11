#!/usr/bin/env python

# Copyright 2025 DIDEN Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("streaming_flow")
@dataclass
class StreamingFlowConfig(PreTrainedConfig):
    """Configuration for StreamingFlowPolicy (SFP-S, the stochastic variant).

    Streaming Flow Policy ("Streaming Flow Policy: Simplifying diffusion/flow-matching
    policies by treating action trajectories as flow trajectories",
    https://arxiv.org/abs/2505.21851) trains a velocity field over the *action
    trajectory itself*: the flow starts at the current action a(0) and integrates
    forward in trajectory-time, so actions can be streamed out during ODE integration.

    Observation encoding (vision backbone, crops, spatial softmax) and the
    FiLM-conditioned 1D UNet mirror lerobot's DiffusionPolicy so results are
    comparable; the UNet operates on a fixed length-2 sequence [a(t); z(t)]
    (action + stochastic latent), with fully-connected resampling instead of
    strided convolutions (fc_timesteps=2 in the reference implementation).

    SFP-specific args:
        sigma_0: std of the conditional flow around the demonstration at t=0.
        sigma_1: std of the conditional flow at t=1. Must satisfy 0 <= sigma_0 <= sigma_1.
        integration_method: fixed-step ODE integrator, one of ["euler", "midpoint", "rk4"].
        integration_steps_per_action: ODE substeps per action step at inference.
        a0_from_state_indices: indices into observation.state used to bootstrap the
            flow start point a(0) at the first inference of an episode (before any
            action has been executed). The gather happens in normalized space, which
            is a good approximation when action[i] commands the same physical
            quantity as state[a0_from_state_indices[i]] (e.g. joint position
            command vs measured joint position, at rest). After the first chunk,
            a(0) is the last executed action. If None, the first chunk starts from
            zeros (mid-range under MIN_MAX normalization).
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Avoid sampling chunks that are mostly copy-padding at episode ends.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone (identical to DiffusionPolicy).
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    # Velocity UNet (defaults follow the SFP reference implementation).
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    flow_step_embed_dim: int = 256
    use_film_scale_modulation: bool = True

    # Streaming flow.
    sigma_0: float = 0.1
    sigma_1: float = 0.1

    # Inference.
    integration_method: str = "rk4"
    integration_steps_per_action: int = 3
    a0_from_state_indices: tuple[int, ...] | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets (same values the SFP reference training loop uses).
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if not (0 <= self.sigma_0 <= self.sigma_1):
            raise ValueError(f"Need 0 <= sigma_0 <= sigma_1. Got {self.sigma_0=}, {self.sigma_1=}.")

        if self.integration_method not in ("euler", "midpoint", "rk4"):
            raise ValueError(f"Unsupported `integration_method` {self.integration_method}.")
        if self.integration_steps_per_action < 1:
            raise ValueError(f"`integration_steps_per_action` must be >= 1.")

        if self.horizon < 2:
            raise ValueError(f"`horizon` must be >= 2 to define a trajectory. Got {self.horizon}.")
        # The reference implementation makes the same restriction: with n_obs_steps=2 the
        # action grid starts one step in the past, so the flow start point a(0) is the
        # last *executed* action — the exact quantity available at inference time.
        if self.n_obs_steps != 2:
            raise ValueError(f"StreamingFlowPolicy requires n_obs_steps=2. Got {self.n_obs_steps}.")
        # The action grid spans indices 0..horizon-1 and index 0 is the flow start
        # point a(0) (the last executed action), so only horizon-1 future actions exist.
        if not (1 <= self.n_action_steps <= self.horizon - self.n_obs_steps + 1):
            raise ValueError(
                f"Need 1 <= n_action_steps <= horizon - n_obs_steps + 1. "
                f"Got {self.n_action_steps=}, {self.horizon=}, {self.n_obs_steps=}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.a0_from_state_indices is not None:
            action_dim = self.action_feature.shape[0]
            state_dim = self.robot_state_feature.shape[0]
            if len(self.a0_from_state_indices) != action_dim:
                raise ValueError(
                    f"`a0_from_state_indices` must have one index per action dim ({action_dim}). "
                    f"Got {len(self.a0_from_state_indices)}."
                )
            if any(not (0 <= i < state_dim) for i in self.a0_from_state_indices):
                raise ValueError(
                    f"`a0_from_state_indices` entries must be valid observation.state indices "
                    f"(state_dim={state_dim}). Got {self.a0_from_state_indices}."
                )

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        # Starts at 1 - n_obs_steps so that with n_obs_steps=2 the first action in the
        # chunk is the action taken at the *previous* step — i.e. the last executed
        # action — which is exactly the flow start point a(0) available at inference.
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
