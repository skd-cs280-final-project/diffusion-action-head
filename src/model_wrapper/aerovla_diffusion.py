"""Frozen Prismatic/OpenVLA features followed by a trainable action diffusion head.

The causal LM's vocabulary projection is removed after loading its checkpoint.
Only the Llama decoder is called, so neither training nor inference allocates
vocabulary logits or relies on generated action tokens.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
from torch import nn

from .diffusion_action_head import (
    DiffusionActionHead,
    DiffusionActionHeadConfig,
)


DEFAULT_ACTION_SCHEMA = {
    "action_order": ["fwd", "down", "yaw", "land"],
    "minimum": [0.0, -5.0, -1.1, 0.0],
    "maximum": [5.0, 5.0, 1.1, 1.0],
    "normalized_range": [-1.0, 1.0],
    "land_threshold": 0.0,
    "land_label": "is_last_step OR is_penultimate",
}


def _reference(path: Optional[Union[str, Path]]) -> Optional[str]:
    if path is None:
        return None
    value = Path(path).expanduser()
    return str(value.resolve()) if value.exists() else str(path)


def _validate_schema(schema: Dict[str, Any]) -> None:
    if schema != DEFAULT_ACTION_SCHEMA:
        raise ValueError("Unsupported action_schema: expected AeroVLA [fwd, down, yaw, land] schema")


class AeroVLADiffusionModel(nn.Module):
    """Use all multimodal final hidden states as diffusion conditioning.

    ``backbone`` must be an unloaded/merged Prismatic model exposing
    ``vision_backbone``, ``projector`` and ``language_model.get_decoder()``.
    Frozen LoRA adapters are merged before this constructor is called.
    The head stays float32 by default even when the backbone uses bfloat16.
    """

    CONFIG_NAME = "diffusion_config.json"
    WEIGHTS_NAME = "diffusion_head.pt"
    FORMAT_VERSION = 1

    def __init__(
        self,
        backbone: nn.Module,
        head_config: Optional[DiffusionActionHeadConfig] = None,
        *,
        base_model_path: Optional[Union[str, Path]] = None,
        adapter_path: Optional[Union[str, Path]] = None,
        action_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        for name in ("vision_backbone", "projector", "language_model"):
            if not hasattr(backbone, name):
                raise TypeError(f"Prismatic backbone is missing {name}")
        embedding = backbone.get_input_embeddings()
        condition_dim = embedding.weight.shape[1]
        if head_config is None:
            head_config = DiffusionActionHeadConfig(condition_dim=condition_dim)
        if isinstance(head_config, dict):
            head_config = DiffusionActionHeadConfig.from_dict(head_config)
        if head_config.condition_dim != condition_dim:
            raise ValueError(f"Head condition_dim={head_config.condition_dim} != backbone hidden size={condition_dim}")
        if head_config.action_dim != 4:
            raise ValueError("AeroVLA requires four action channels: fwd, down, yaw, land")
        self.action_schema = copy.deepcopy(DEFAULT_ACTION_SCHEMA if action_schema is None else action_schema)
        _validate_schema(self.action_schema)
        self.base_model_path = _reference(base_model_path)
        self.adapter_path = _reference(adapter_path)
        self.head_config = head_config
        self.backbone = backbone
        if not callable(getattr(backbone.language_model, "get_decoder", None)):
            raise TypeError("language_model must expose get_decoder()")
        if hasattr(backbone.language_model, "lm_head"):
            del backbone.language_model.lm_head
        self._freeze_backbone()
        self.diffusion_head = DiffusionActionHead(head_config).to(device=embedding.weight.device)

    def _freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "AeroVLADiffusionModel":
        super().train(mode)
        self._freeze_backbone()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Optimizer input intentionally contains only diffusion-head parameters."""
        return (parameter for parameter in self.diffusion_head.parameters() if parameter.requires_grad)

    @property
    def device(self) -> torch.device:
        return self.backbone.get_input_embeddings().weight.device

    @classmethod
    def from_pretrained_backbone(
        cls,
        base_model_path: Union[str, Path],
        adapter_path: Optional[Union[str, Path]] = None,
        head_config: Optional[DiffusionActionHeadConfig] = None,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
        tokenizer_length: Optional[int] = None,
        action_schema: Optional[Dict[str, Any]] = None,
        **loading_kwargs: Any,
    ) -> "AeroVLADiffusionModel":
        from transformers import AutoModelForVision2Seq

        backbone = AutoModelForVision2Seq.from_pretrained(
            str(base_model_path),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **loading_kwargs,
        )
        if tokenizer_length is not None:
            backbone.resize_token_embeddings(tokenizer_length)
        if adapter_path is not None:
            from peft import PeftModel

            # Merge before extracting the decoder; this also restores the trained
            # projector saved by PEFT's modules_to_save mechanism.
            backbone = PeftModel.from_pretrained(backbone, str(adapter_path), is_trainable=False)
            backbone = backbone.merge_and_unload(safe_merge=True)
        if device is not None:
            backbone.to(device=device)
        return cls(
            backbone,
            head_config,
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            action_schema=action_schema,
        )

    @torch.no_grad()
    def encode_condition(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Insert image patches after BOS and preserve each text padding mask.

        As in the original Prismatic multimodal forward, input sequences start
        with BOS. Right padding is required and checked instead of silently
        inserting image patches after a left-padding token.
        """
        self._freeze_backbone()
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, nonempty text length]")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4 or pixel_values.shape[0] != input_ids.shape[0]:
            raise ValueError("pixel_values must have shape [batch, channels, height, width] matching input_ids")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        if attention_mask is None:
            pad_id = getattr(self.backbone, "pad_token_id", None)
            attention_mask = input_ids.ne(pad_id) if pad_id is not None else torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask must contain only zero and one")
            attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        if not attention_mask[:, 0].all() or ((~attention_mask[:, :-1]) & attention_mask[:, 1:]).any():
            raise ValueError("Expected nonempty, right-padded prompts starting with BOS; set tokenizer.padding_side='right'")

        vision_parameter = next(self.backbone.vision_backbone.parameters(), None)
        vision_dtype = vision_parameter.dtype if vision_parameter is not None else self.backbone.get_input_embeddings().weight.dtype
        pixel_values = pixel_values.to(device=self.device, dtype=vision_dtype)
        patch_features = self.backbone.vision_backbone(pixel_values)
        projected = self.backbone.projector(patch_features)
        text_embeddings = self.backbone.get_input_embeddings()(input_ids)
        projected = projected.to(dtype=text_embeddings.dtype)
        multimodal_embeddings = torch.cat((text_embeddings[:, :1], projected, text_embeddings[:, 1:]), dim=1)
        image_mask = torch.ones(projected.shape[:2], device=self.device, dtype=torch.bool)
        condition_mask = torch.cat((attention_mask[:, :1], image_mask, attention_mask[:, 1:]), dim=1)
        decoder = self.backbone.language_model.get_decoder()
        max_positions = getattr(decoder.config, "max_position_embeddings", None)
        if max_positions is not None and multimodal_embeddings.shape[1] > max_positions:
            raise ValueError(f"Text plus image sequence length {multimodal_embeddings.shape[1]} exceeds decoder limit {max_positions}")
        outputs = decoder(
            inputs_embeds=multimodal_embeddings,
            attention_mask=condition_mask,
            position_ids=(condition_mask.long().cumsum(-1) - 1).clamp_min(0),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        head_parameter = next(self.diffusion_head.parameters())
        condition = outputs.last_hidden_state.to(device=head_parameter.device, dtype=head_parameter.dtype)
        return condition, condition_mask.to(device=head_parameter.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition, condition_mask = self.encode_condition(input_ids, pixel_values, attention_mask)
        return self.diffusion_head.compute_loss(
            condition,
            actions,
            condition_mask=condition_mask,
            action_mask=action_mask,
            noise=noise,
            timesteps=timesteps,
        )

    @torch.no_grad()
    def predict_action(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        condition, condition_mask = self.encode_condition(input_ids, pixel_values, attention_mask)
        return self.diffusion_head.sample(
            condition,
            condition_mask=condition_mask,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    def save_pretrained(
        self,
        output_dir: Union[str, Path],
        *,
        action_schema: Optional[Dict[str, Any]] = None,
        training_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write only head weights, plus enough metadata to reload its backbone."""
        schema = copy.deepcopy(self.action_schema if action_schema is None else action_schema)
        _validate_schema(schema)
        config = {
            "format_version": self.FORMAT_VERSION,
            "architecture": type(self).__name__,
            "head_config": self.head_config.to_dict(),
            "base_model_path": self.base_model_path,
            "adapter_path": self.adapter_path,
            "tokenizer_length": self.backbone.get_input_embeddings().weight.shape[0],
            "action_schema": schema,
            "training_metadata": training_metadata or {},
        }
        # Validate JSON before writing weights to avoid partial output for an
        # unsupported metadata object such as a Path or Tensor.
        serialized = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_temp = output_dir / (self.WEIGHTS_NAME + ".tmp")
        config_temp = output_dir / (self.CONFIG_NAME + ".tmp")
        torch.save({name: tensor.detach().cpu() for name, tensor in self.diffusion_head.state_dict().items()}, weights_temp)
        config_temp.write_text(serialized, encoding="utf-8")
        os.replace(weights_temp, output_dir / self.WEIGHTS_NAME)
        os.replace(config_temp, output_dir / self.CONFIG_NAME)

    @classmethod
    def read_checkpoint_config(cls, checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
        config = json.loads((Path(checkpoint_dir) / cls.CONFIG_NAME).read_text(encoding="utf-8"))
        required = {"format_version", "architecture", "head_config", "base_model_path", "adapter_path", "tokenizer_length", "action_schema", "training_metadata"}
        if set(config) != required:
            raise ValueError(f"Invalid diffusion checkpoint metadata fields: {sorted(set(config) ^ required)}")
        if config["format_version"] != cls.FORMAT_VERSION or config["architecture"] != cls.__name__:
            raise ValueError("Unsupported diffusion checkpoint format or architecture")
        DiffusionActionHeadConfig.from_dict(config["head_config"])
        _validate_schema(config["action_schema"])
        if type(config["tokenizer_length"]) is not int or config["tokenizer_length"] <= 0:
            raise ValueError("Invalid checkpoint tokenizer_length")
        return config

    def load_head_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        config = self.read_checkpoint_config(checkpoint_dir)
        if config["head_config"] != self.head_config.to_dict():
            raise ValueError("Checkpoint head configuration does not match this model")
        if config["action_schema"] != self.action_schema:
            raise ValueError("Checkpoint action schema does not match this model")
        if config["tokenizer_length"] != self.backbone.get_input_embeddings().weight.shape[0]:
            raise ValueError("Checkpoint tokenizer size does not match this backbone")
        state = torch.load(Path(checkpoint_dir) / self.WEIGHTS_NAME, map_location="cpu", weights_only=True)
        self.diffusion_head.load_state_dict(state, strict=True)
        self._freeze_backbone()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        *,
        base_model_path: Optional[Union[str, Path]] = None,
        adapter_path: Optional[Union[str, Path]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
        tokenizer_length: Optional[int] = None,
        **loading_kwargs: Any,
    ) -> "AeroVLADiffusionModel":
        config = cls.read_checkpoint_config(checkpoint_dir)
        base_model_path = base_model_path or config["base_model_path"]
        if base_model_path is None:
            raise ValueError("Checkpoint has no base_model_path; provide it explicitly")
        if tokenizer_length is not None and tokenizer_length != config["tokenizer_length"]:
            raise ValueError("Requested tokenizer_length does not match the saved checkpoint")
        model = cls.from_pretrained_backbone(
            base_model_path,
            adapter_path=adapter_path if adapter_path is not None else config["adapter_path"],
            head_config=DiffusionActionHeadConfig.from_dict(config["head_config"]),
            torch_dtype=torch_dtype,
            device=device,
            tokenizer_length=config["tokenizer_length"],
            action_schema=config["action_schema"],
            **loading_kwargs,
        )
        model.load_head_checkpoint(checkpoint_dir)
        model.eval()
        return model
