"""NavDP-inspired DDPM action decoder conditioned on frozen VLM features.

Actions are normalized continuous vectors in the order ``[fwd, down, yaw,
land]``. The data/inference adapters own physical-unit conversion and landing
thresholding; this module predicts epsilon and samples vectors in [-1, 1].
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


@dataclass(frozen=True)
class DiffusionActionHeadConfig:
    condition_dim: int
    action_dim: int = 4
    action_horizon: int = 1
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    dropout: float = 0.0
    state_dim: int = 0
    history_length: int = 0
    state_schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "condition_dim", "action_dim", "action_horizon", "hidden_dim",
            "num_layers", "num_heads", "num_train_timesteps", "num_inference_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.hidden_dim < 4 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be >= 4 and divisible by num_heads")
        if self.num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps cannot exceed num_train_timesteps")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("state_dim", "history_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
        if bool(self.state_dim) != bool(self.history_length):
            raise ValueError("state_dim and history_length must both be zero or both positive")
        if type(self.state_schema_version) is not int or self.state_schema_version != 1:
            raise ValueError("Unsupported state_schema_version; expected 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DiffusionActionHeadConfig":
        return cls(**values)


class SinusoidalTimestepEmbedding(nn.Module):
    """Sin/cos timestep features, also supporting odd embedding dimensions."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        half = dimension // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half).float() / (half - 1))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, timesteps: Tensor) -> Tensor:
        angles = timesteps.float()[:, None] * self.frequencies.float()[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dimension % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class DiffusionActionHead(nn.Module):
    """Trainable Transformer decoder replacing the vocabulary prediction head.

    ``condition_mask`` always uses True for real (unmasked) VLM tokens. The
    timestep is prepended as an always-valid memory token. Learned action
    positions and a causal target mask follow NavDP's action decoder; the VLM
    condition already contains positional information. Optional UAV state
    history is projected directly into this trainable head's memory, with
    learned temporal positions; it never passes through the frozen VLM.
    """

    def __init__(self, config: DiffusionActionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_projection = nn.Linear(config.condition_dim, config.hidden_dim)
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.action_positions = nn.Embedding(config.action_horizon, config.hidden_dim)
        self.time_embedding = SinusoidalTimestepEmbedding(config.hidden_dim)
        if config.state_dim:
            self.state_encoder = nn.Sequential(
                nn.Linear(config.state_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.state_positions = nn.Embedding(config.history_length, config.hidden_dim)
            self.state_type_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.noise_projection = nn.Linear(config.hidden_dim, config.action_dim)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.action_horizon, config.action_horizon, dtype=torch.bool).triu(1),
            persistent=False,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
            timestep_spacing="trailing",
        )

    @property
    def device(self) -> torch.device:
        return self.action_projection.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.action_projection.weight.dtype

    def _prepare_condition(
        self, condition: Tensor, condition_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        if condition.ndim != 3 or condition.shape[-1] != self.config.condition_dim:
            raise ValueError("condition must have shape [batch, tokens, condition_dim]")
        if condition.shape[0] == 0 or condition.shape[1] == 0:
            raise ValueError("condition requires a nonempty batch and token sequence")
        if condition_mask is None:
            valid = torch.ones(condition.shape[:2], dtype=torch.bool, device=self.device)
        else:
            if condition_mask.shape != condition.shape[:2]:
                raise ValueError("condition_mask must have shape [batch, tokens]")
            valid = condition_mask.to(device=self.device, dtype=torch.bool)
        # Zero masked tokens before projection, so even NaN padding is harmless.
        condition = condition.to(device=self.device, dtype=self.dtype)
        condition = condition.masked_fill(~valid[..., None], 0)
        return condition, valid

    def _prepare_state_history(
        self,
        state_history: Optional[Tensor],
        state_history_mask: Optional[Tensor],
        batch: int,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if not self.config.state_dim:
            if state_history is not None or state_history_mask is not None:
                raise ValueError("state_history inputs require a history-enabled head configuration")
            return None, None
        if state_history is None or state_history_mask is None:
            raise ValueError("history-enabled head requires state_history and state_history_mask")
        expected = (batch, self.config.history_length, self.config.state_dim)
        if not isinstance(state_history, Tensor) or tuple(state_history.shape) != expected:
            raise ValueError(f"state_history must have shape {expected}")
        if not isinstance(state_history_mask, Tensor) or tuple(state_history_mask.shape) != expected[:2]:
            raise ValueError(f"state_history_mask must have shape {expected[:2]}")
        if not torch.all((state_history_mask == 0) | (state_history_mask == 1)):
            raise ValueError("state_history_mask must contain only zero and one")
        valid = state_history_mask.to(device=self.device, dtype=torch.bool)
        state_history = state_history.to(device=self.device, dtype=self.dtype)
        # Mask before the MLP: a padding NaN must not poison activations or gradients.
        state_history = state_history.masked_fill(~valid[..., None], 0)
        if not torch.isfinite(state_history).all():
            raise ValueError("valid state_history values must be finite")
        return state_history, valid

    def _prepare_timesteps(self, timesteps: Union[int, Tensor], batch: int) -> Tensor:
        timesteps = torch.as_tensor(timesteps, device=self.device)
        if timesteps.is_floating_point() and not torch.equal(timesteps, timesteps.round()):
            raise ValueError("timesteps must contain integer indices")
        timesteps = timesteps.to(dtype=torch.long)
        if timesteps.ndim == 0 or timesteps.shape == (1,):
            timesteps = timesteps.reshape(1).expand(batch)
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must be a scalar or have shape [batch]")
        if torch.any((timesteps < 0) | (timesteps >= self.config.num_train_timesteps)):
            raise ValueError("timesteps are outside the training noise schedule")
        return timesteps

    def _check_actions(self, actions: Tensor, batch: int) -> None:
        expected = (batch, self.config.action_horizon, self.config.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}")

    def forward(
        self,
        noisy_actions: Tensor,
        timesteps: Union[int, Tensor],
        condition: Tensor,
        condition_mask: Optional[Tensor] = None,
        state_history: Optional[Tensor] = None,
        state_history_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict epsilon for normalized noisy actions; output uses float32."""
        condition, valid = self._prepare_condition(condition, condition_mask)
        batch = condition.shape[0]
        self._check_actions(noisy_actions, batch)
        timesteps = self._prepare_timesteps(timesteps, batch)
        time_tokens = self.time_embedding(timesteps).to(dtype=self.dtype)[:, None]
        memory = torch.cat((time_tokens, self.condition_projection(condition)), dim=1)
        memory_padding_mask = torch.cat(
            (torch.zeros(batch, 1, dtype=torch.bool, device=self.device), ~valid), dim=1
        )
        state_history, state_valid = self._prepare_state_history(state_history, state_history_mask, batch)
        if state_history is not None:
            state_positions = torch.arange(self.config.history_length, device=self.device)
            state_tokens = self.state_encoder(state_history)
            state_tokens = state_tokens + self.state_positions(state_positions)[None] + self.state_type_embedding
            memory = torch.cat((memory, state_tokens), dim=1)
            memory_padding_mask = torch.cat((memory_padding_mask, ~state_valid), dim=1)
        positions = torch.arange(self.config.action_horizon, device=self.device)
        targets = self.action_projection(noisy_actions.to(device=self.device, dtype=self.dtype))
        targets = targets + self.action_positions(positions)[None]
        decoded = self.decoder(
            tgt=targets,
            memory=memory,
            tgt_mask=self.causal_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.noise_projection(self.output_norm(decoded)).float()

    def compute_loss(
        self,
        condition: Tensor,
        actions: Tensor,
        condition_mask: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        state_history: Optional[Tensor] = None,
        state_history_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """DDPM epsilon MSE, averaged over valid scalar action coordinates.

        ``action_mask`` is [B, H] or [B, H, A], True for supervised values.
        Horizon padding must be trailing because action self-attention is causal.
        Supplying ``noise`` and ``timesteps`` enables deterministic diagnostics.
        """
        if condition.ndim != 3:
            raise ValueError("condition must have shape [batch, tokens, condition_dim]")
        batch = condition.shape[0]
        self._check_actions(actions, batch)
        actions = actions.to(device=self.device, dtype=torch.float32)
        if action_mask is None:
            valid = torch.ones_like(actions, dtype=torch.bool)
        else:
            if action_mask.shape == actions.shape[:2]:
                action_mask = action_mask[..., None].expand_as(actions)
            if action_mask.shape != actions.shape:
                raise ValueError("action_mask must have shape [batch, horizon] or [batch, horizon, action_dim]")
            valid = action_mask.to(device=self.device, dtype=torch.bool)
        if not valid.any():
            raise ValueError("action_mask must select at least one action coordinate")
        actions = actions.masked_fill(~valid, 0)
        if not torch.isfinite(actions).all():
            raise ValueError("valid actions must be finite")
        if (actions.abs() > 1.0).any():
            raise ValueError("valid actions must be normalized to [-1, 1]")
        if noise is None:
            noise = torch.randn_like(actions)
        else:
            self._check_actions(noise, batch)
            noise = noise.to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(noise).all():
                raise ValueError("noise must be finite")
        if timesteps is None:
            timesteps = torch.randint(self.config.num_train_timesteps, (batch,), device=self.device)
        timesteps = self._prepare_timesteps(timesteps, batch)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self(
            noisy_actions, timesteps, condition, condition_mask,
            state_history=state_history, state_history_mask=state_history_mask,
        )
        squared_error = (prediction.float() - noise).square()
        return squared_error.masked_select(valid).mean()

    @torch.no_grad()
    def sample(
        self,
        condition: Tensor,
        condition_mask: Optional[Tensor] = None,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        state_history: Optional[Tensor] = None,
        state_history_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate normalized actions with DDPM; pass a device-matched RNG."""
        condition, valid = self._prepare_condition(condition, condition_mask)
        state_history, state_history_mask = self._prepare_state_history(
            state_history, state_history_mask, condition.shape[0]
        )
        steps = self.config.num_inference_steps if num_inference_steps is None else num_inference_steps
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= self.config.num_train_timesteps:
            raise ValueError("num_inference_steps must be between 1 and num_train_timesteps")
        # A local scheduler avoids changing training or another sampling call's state.
        scheduler = DDPMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(steps, device=self.device)
        # Start at the final (nearly pure-noise) training timestep. Explicit
        # timesteps also make diffusers 0.30.3 use the actual next schedule entry
        # instead of floor(T / steps), which is wrong for non-divisible spacing.
        scheduler.set_timesteps(timesteps=scheduler.timesteps.tolist(), device=self.device)
        actions = torch.randn(
            (condition.shape[0], self.config.action_horizon, self.config.action_dim),
            device=self.device, dtype=torch.float32, generator=generator,
        )
        was_training = self.training
        self.eval()
        try:
            for timestep in scheduler.timesteps:
                predicted_noise = self(
                    actions, timestep, condition, valid,
                    state_history=state_history, state_history_mask=state_history_mask,
                )
                actions = scheduler.step(
                    model_output=predicted_noise,
                    timestep=timestep,
                    sample=actions,
                    generator=generator,
                ).prev_sample
        finally:
            self.train(was_training)
        return actions
