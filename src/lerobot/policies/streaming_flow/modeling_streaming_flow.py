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
"""Streaming Flow Policy (stochastic variant, "SFP-S").

Paper: "Streaming Flow Policy: Simplifying diffusion/flow-matching policies by
treating action trajectories as flow trajectories" (https://arxiv.org/abs/2505.21851).
Reference implementation: https://github.com/siddancha/streaming-flow-policy
(`streaming_flow_policy/pusht/sfps.py`).

The demonstration action chunk xi (grid of `horizon` actions, uniformly spaced over
flow-time t in [0, 1]) defines a conditional flow:
    a(t) = xi(t) + eps0 + sigma_r * t * z0          eps0 ~ N(0, sigma_0), z0 ~ N(0, 1)
    z(t) = (1 - (1 - sigma_1) t) z0 + t xi(t)
with conditional velocity targets
    va(t) = xi'(t) + sigma_r * z0
    vz(t) = xi(t) + t xi'(t) - (1 - sigma_1) z0
where sigma_r = sqrt(sigma_1^2 - sigma_0^2). A FiLM-conditioned UNet over the fixed
length-2 sequence [a; z] regresses (va, vz). At inference the ODE is integrated
from a(0) = last executed action, z(0) ~ N(0, 1); intermediate integration states
at the action-grid times ARE the actions (that is the "streaming" property).

Observation encoding (per-camera ResNet + spatial softmax) is shared with
lerobot's DiffusionPolicy for comparability.
"""

import logging
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..diffusion.modeling_diffusion import (
    DiffusionConv1dBlock,
    DiffusionRgbEncoder,
    DiffusionSinusoidalPosEmb,
)
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_streaming_flow import StreamingFlowConfig

logger = logging.getLogger(__name__)


class StreamingFlowPolicy(PreTrainedPolicy):
    """Streaming Flow Policy (stochastic variant) with DiffusionPolicy-style obs encoding."""

    config_class = StreamingFlowConfig
    name = "streaming_flow"

    def __init__(self, config: StreamingFlowConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        # Last executed (normalized) action — the flow start point a(0) of the next chunk.
        self._last_executed_action: Tensor | None = None

        self.model = StreamingFlowModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        """Clear observation/action queues and the a(0) cache. Call on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        self._last_executed_action = None

    def _flow_start_point(self, batch: dict[str, Tensor]) -> Tensor:
        """a(0) for the next chunk: last executed action, else a bootstrap from state."""
        state = batch[OBS_STATE]  # (B, n_obs_steps, state_dim)
        batch_size = state.shape[0]
        action_dim = self.config.action_feature.shape[0]

        if (
            self._last_executed_action is not None
            and self._last_executed_action.shape[0] == batch_size
        ):
            return self._last_executed_action

        if self.config.a0_from_state_indices is not None:
            idx = torch.as_tensor(self.config.a0_from_state_indices, device=state.device)
            a0 = state[:, -1, :].index_select(-1, idx)
            logger.debug(
                "[streaming_flow] episode-start a(0) bootstrapped from state indices %s",
                self.config.a0_from_state_indices,
            )
            return a0

        logger.debug("[streaming_flow] episode-start a(0) fallback to zeros (no state indices configured)")
        return torch.zeros(batch_size, action_dim, dtype=state.dtype, device=state.device)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of `n_action_steps` actions given environment observations."""
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        else:
            batch = dict(batch)
            if self.config.image_features:
                for key in self.config.image_features:
                    if batch[key].ndim == 4:
                        batch[key] = batch[key].unsqueeze(1)
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        a0 = self._flow_start_point(batch)
        actions = self.model.generate_actions(batch, a0, noise=noise)
        # The last action of the chunk is executed right before the next chunk is
        # generated, making it the next chunk's flow start point.
        self._last_executed_action = actions[:, -1].detach().clone()
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations (queued, receding horizon)."""
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.model.compute_loss(batch)
        return loss, None


class StreamingFlowModel(nn.Module):
    def __init__(self, config: StreamingFlowConfig):
        super().__init__()
        self.config = config

        # Observation encoders — identical layout to DiffusionModel.
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.velocity_net = StreamingFlowVelocityUnet(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )

        sigma_r = (config.sigma_1**2 - config.sigma_0**2) ** 0.5
        self.register_buffer("sigma_r", torch.tensor(sigma_r, dtype=torch.float32))

        logger.debug(
            "[streaming_flow] model: horizon=%d n_action_steps=%d sigma_0=%.3f sigma_1=%.3f "
            "integration=%s x%d",
            config.horizon,
            config.n_action_steps,
            config.sigma_0,
            config.sigma_1,
            config.integration_method,
            config.integration_steps_per_action,
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Conditional flow-matching loss of SFP-S on a batch of action chunks."""
        assert set(batch).issuperset({OBS_STATE, ACTION})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        actions = batch[ACTION]  # (B, horizon, action_dim), normalized
        batch_size, horizon, action_dim = actions.shape
        assert horizon == self.config.horizon
        assert batch[OBS_STATE].shape[1] == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        device = actions.device
        sigma_0 = self.config.sigma_0
        sigma_1 = self.config.sigma_1
        sigma_r = self.sigma_r

        # Sample one flow time per batch element and linearly interpolate the
        # demonstration trajectory xi on the uniform action grid (first-order hold).
        t = torch.rand(batch_size, device=device)  # (B,)
        pos = t * (horizon - 1)
        idx = pos.floor().long().clamp(max=horizon - 2)  # (B,)
        frac = (pos - idx.to(pos.dtype)).unsqueeze(-1)  # (B, 1)
        batch_arange = torch.arange(batch_size, device=device)
        a_lo = actions[batch_arange, idx]  # (B, action_dim)
        a_hi = actions[batch_arange, idx + 1]  # (B, action_dim)
        xi = a_lo + frac * (a_hi - a_lo)  # xi(t), (B, action_dim)
        dxi = (a_hi - a_lo) * (horizon - 1)  # xi'(t), (B, action_dim)

        t_col = t.unsqueeze(-1)  # (B, 1)
        z0 = torch.randn_like(xi)
        a_t = xi + sigma_0 * torch.randn_like(xi) + sigma_r * t_col * z0
        z_t = (1 - (1 - sigma_1) * t_col) * z0 + t_col * xi
        va = dxi + sigma_r * z0
        vz = xi + t_col * dxi - (1 - sigma_1) * z0

        x = torch.stack((a_t, z_t), dim=1)  # (B, 2, action_dim)
        v_target = torch.stack((va, vz), dim=1)  # (B, 2, action_dim)

        v_pred = self.velocity_net(x, t, global_cond=global_cond)  # (B, 2, action_dim)

        loss = F.mse_loss(v_pred, v_target, reduction="none")

        # Down-weight samples whose interpolation segment lies in copy-padded actions
        # (episode ends). Off by default, mirroring DiffusionPolicy.
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_bounds = ~batch["action_is_pad"]  # (B, horizon)
            valid = (in_bounds[batch_arange, idx] & in_bounds[batch_arange, idx + 1]).float()
            mask = valid.view(batch_size, 1, 1)
            return (loss * mask).sum() / (mask.sum() * loss.shape[1] * loss.shape[2]).clamp_min(1)

        return loss.mean()

    def generate_actions(
        self, batch: dict[str, Tensor], a0: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        """
        Integrate the learned velocity field from a(0)=`a0`, z(0)~N(0,1) and return the
        integration states at the action-grid times — these are the actions.

        Expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        `a0`: (B, action_dim) normalized flow start point.
        Returns: (B, n_action_steps, action_dim).
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        z0 = noise if noise is not None else torch.randn_like(a0)
        x = torch.stack((a0, z0), dim=1)  # (B, 2, action_dim)

        horizon = self.config.horizon
        n_sub = self.config.integration_steps_per_action
        dt = (1.0 / (horizon - 1)) / n_sub

        def f(x_: Tensor, t_: float) -> Tensor:
            t_tensor = torch.full((batch_size,), t_, dtype=x_.dtype, device=x_.device)
            return self.velocity_net(x_, t_tensor, global_cond=global_cond)

        actions = []
        t = 0.0
        # Grid index 0 is a(0) itself; actions executed are grid indices 1..n_action_steps.
        for _ in range(self.config.n_action_steps):
            for _ in range(n_sub):
                x = _ode_step(f, x, t, dt, self.config.integration_method)
                t += dt
            actions.append(x[:, 0, :])

        return torch.stack(actions, dim=1)  # (B, n_action_steps, action_dim)


def _ode_step(f, x: Tensor, t: float, dt: float, method: str) -> Tensor:
    """One fixed-step ODE update of dx/dt = f(x, t)."""
    if method == "euler":
        return x + dt * f(x, t)
    if method == "midpoint":
        k1 = f(x, t)
        return x + dt * f(x + 0.5 * dt * k1, t + 0.5 * dt)
    if method == "rk4":
        k1 = f(x, t)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(x + dt * k3, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(f"Unsupported integration method {method}")


class _Linear1d(nn.Module):
    """Fully-connected stand-in for temporal down/up-sampling when the sequence is tiny.

    From the SFP reference implementation (`fc_timesteps`): the [a; z] sequence has
    length 2, which strided convolutions cannot resample, so each "resampling" stage
    is a Linear over the flattened (channels * time) features instead.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, c, t = x.shape
        return self.linear(x.reshape(b, c * t)).reshape(b, c, t)


class _FilmResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM scale+bias modulation."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int, n_groups: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)  # (B, 2*out_channels, 1)
        scale = cond_embed[:, : self.out_channels]
        bias = cond_embed[:, self.out_channels :]
        out = scale * out + bias
        out = self.conv2(out)
        return out + self.residual_conv(x)


class StreamingFlowVelocityUnet(nn.Module):
    """The SFP velocity network: ConditionalUnet1D with fc_timesteps=2.

    Structure follows the SFP reference implementation — a Diffusion-Policy-style
    FiLM UNet over the fixed length-2 sequence [a; z], with `_Linear1d` replacing
    the strided-conv temporal resampling (which is meaningless at T=2).
    """

    SEQ_LEN = 2  # [a; z]

    def __init__(self, config: StreamingFlowConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        input_dim = config.action_feature.shape[0]

        dsed = config.flow_step_embed_dim
        self.flow_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        all_dims = [input_dim, *config.down_dims]
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))
        block_kwargs = {"cond_dim": cond_dim, "kernel_size": config.kernel_size, "n_groups": config.n_groups}

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        _FilmResidualBlock1d(dim_in, dim_out, **block_kwargs),
                        _FilmResidualBlock1d(dim_out, dim_out, **block_kwargs),
                        _Linear1d(self.SEQ_LEN * dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                _FilmResidualBlock1d(mid_dim, mid_dim, **block_kwargs),
                _FilmResidualBlock1d(mid_dim, mid_dim, **block_kwargs),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        _FilmResidualBlock1d(dim_out * 2, dim_in, **block_kwargs),
                        _FilmResidualBlock1d(dim_in, dim_in, **block_kwargs),
                        _Linear1d(self.SEQ_LEN * dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        start_dim = config.down_dims[0]
        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(start_dim, start_dim, kernel_size=config.kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, x: Tensor, t: Tensor, global_cond: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, 2, action_dim) — the [a; z] pair.
            t: (B,) flow time in [0, 1].
            global_cond: (B, global_cond_dim).
        Returns:
            (B, 2, action_dim) predicted [va; vz].
        """
        x = einops.rearrange(x, "b t d -> b d t")

        global_feature = self.flow_step_encoder(t)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        skips: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")
