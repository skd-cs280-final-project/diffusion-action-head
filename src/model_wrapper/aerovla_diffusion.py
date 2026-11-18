"""Prismatic/OpenVLA features followed by a trainable action diffusion head.

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
from torch.utils.checkpoint import checkpoint

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
    JOINT_WEIGHTS_NAME = "joint_adapter.pt"
    FORMAT_VERSION = 1
    JOINT_FORMAT_VERSION = 2

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
        self.joint_lora_config = None
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
        if self.joint_lora_config is None:
            self._freeze_backbone()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Return head parameters, plus the explicitly enabled joint adapters."""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _joint_target_modules(self):
        language_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        unused_vision_prefixes = []
        # Prismatic takes second-to-last block features with norm=False. Its
        # final blocks, pooling and classification layers cannot affect loss.
        for name in ("featurizer", "fused_featurizer"):
            featurizer = getattr(self.backbone.vision_backbone, name, None)
            if featurizer is not None and hasattr(featurizer, "blocks"):
                prefix = f"vision_backbone.{name}."
                unused_vision_prefixes.extend([
                    prefix + f"blocks.{len(featurizer.blocks) - 1}.",
                    prefix + "attn_pool.", prefix + "head.", prefix + "fc_norm.",
                    prefix + "norm.",
                ])
        targets = []
        for name, module in self.backbone.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            language = name.startswith("language_model.") and name.rsplit(".", 1)[-1] in language_targets
            vision = name.startswith("vision_backbone.") and not any(name.startswith(prefix) for prefix in unused_vision_prefixes)
            if language or vision:
                targets.append(name)
        if not any(name.startswith("language_model.") for name in targets):
            raise ValueError("Joint LoRA requires supported Llama projection layers")
        if not any(name.startswith("vision_backbone.") for name in targets):
            raise ValueError("Joint LoRA requires supported vision Linear layers")
        return sorted(targets)

    def enable_joint_lora(self, rank=16, alpha=32, dropout=0.0, gradient_checkpointing=True):
        """Jointly train visual/Llama LoRA, the full projector and the full head.

        The original AeroVLA adapter is already merged into the frozen base.
        Calling this again preserves an existing joint adapter; changing its
        architecture requires an explicitly different training initialization.
        """
        if type(rank) is not int or rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid joint LoRA rank, alpha or dropout")
        requested = {"rank": rank, "alpha": alpha, "dropout": dropout,
                     "gradient_checkpointing": bool(gradient_checkpointing)}
        if self.joint_lora_config is not None:
            if any(self.joint_lora_config[key] != requested[key] for key in ("rank", "alpha", "dropout")):
                raise ValueError("Existing joint LoRA configuration differs; refusing to reset trained adapters")
            self.joint_lora_config["gradient_checkpointing"] = bool(gradient_checkpointing)
            self._configure_joint_checkpointing()
            return self
        from peft import LoraConfig, inject_adapter_in_model

        targets = self._joint_target_modules()
        self.backbone.requires_grad_(False)
        self.backbone = inject_adapter_in_model(LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=targets, modules_to_save=["projector"], bias="none",
        ), self.backbone)
        # PEFT's low-level injector wraps modules_to_save but, unlike PeftModel,
        # does not reactivate their copy after its global adapter-only freeze.
        self.backbone.projector.modules_to_save["default"].requires_grad_(True)
        self.joint_lora_config = {**requested, "target_modules": targets}
        # AdamW moments follow parameter dtype. Keep every updated parameter in
        # fp32, while frozen base tensors remain in their original dtype.
        for parameter in self.backbone.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.diffusion_head.float().requires_grad_(True)
        self._configure_joint_checkpointing()
        self.train(self.training)
        return self

    def _configure_joint_checkpointing(self):
        decoder = self.backbone.language_model.get_decoder()
        if self.joint_lora_config["gradient_checkpointing"]:
            decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            decoder.gradient_checkpointing_disable()

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
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.training and self.joint_lora_config is not None):
            return self._encode_condition(input_ids, pixel_values, attention_mask)

    def _encode_condition(self, input_ids, pixel_values, attention_mask):
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
        if (self.joint_lora_config is not None and self.joint_lora_config["gradient_checkpointing"]
                and self.training and torch.is_grad_enabled()):
            # timm 0.9 get_intermediate_layers bypasses set_grad_checkpointing.
            # Non-reentrant checkpointing also works with frozen image inputs.
            patch_features = checkpoint(self.backbone.vision_backbone, pixel_values, use_reentrant=False)
        else:
            patch_features = self.backbone.vision_backbone(pixel_values)
        projector_dtype = torch.float32 if self.joint_lora_config is not None else next(self.backbone.projector.parameters()).dtype
        projected = self.backbone.projector(patch_features.to(dtype=projector_dtype))
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
        state_history: Optional[torch.Tensor] = None,
        state_history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition, condition_mask = self.encode_condition(input_ids, pixel_values, attention_mask)
        return self.diffusion_head.compute_loss(
            condition,
            actions,
            condition_mask=condition_mask,
            action_mask=action_mask,
            noise=noise,
            timesteps=timesteps,
            state_history=state_history,
            state_history_mask=state_history_mask,
        )

    @torch.no_grad()
    def predict_action(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        state_history: Optional[torch.Tensor] = None,
        state_history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition, condition_mask = self.encode_condition(input_ids, pixel_values, attention_mask)
        return self.diffusion_head.sample(
            condition,
            condition_mask=condition_mask,
            num_inference_steps=num_inference_steps,
            generator=generator,
            state_history=state_history,
            state_history_mask=state_history_mask,
        )

    def save_pretrained(
        self,
        output_dir: Union[str, Path],
        *,
        action_schema: Optional[Dict[str, Any]] = None,
        training_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write the head and, for joint training, its new LoRA/projector weights."""
        schema = copy.deepcopy(self.action_schema if action_schema is None else action_schema)
        _validate_schema(schema)
        config = {
            "format_version": self.JOINT_FORMAT_VERSION if self.joint_lora_config is not None else self.FORMAT_VERSION,
            "architecture": type(self).__name__,
            "head_config": self.head_config.to_dict(),
            "base_model_path": self.base_model_path,
            "adapter_path": self.adapter_path,
            "tokenizer_length": self.backbone.get_input_embeddings().weight.shape[0],
            "action_schema": schema,
            "training_metadata": training_metadata or {},
        }
        if self.joint_lora_config is not None:
            config["joint_lora_config"] = copy.deepcopy(self.joint_lora_config)
        # Validate JSON before writing weights to avoid partial output for an
        # unsupported metadata object such as a Path or Tensor.
        serialized = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_temp = output_dir / (self.WEIGHTS_NAME + ".tmp")
        config_temp = output_dir / (self.CONFIG_NAME + ".tmp")
        torch.save({name: tensor.detach().cpu() for name, tensor in self.diffusion_head.state_dict().items()}, weights_temp)
        if self.joint_lora_config is not None:
            from peft.utils import get_peft_model_state_dict

            adapter_state = get_peft_model_state_dict(self.backbone, save_embedding_layers=False)
            adapter_temp = output_dir / (self.JOINT_WEIGHTS_NAME + ".tmp")
            torch.save({name: tensor.detach().cpu() for name, tensor in adapter_state.items()}, adapter_temp)
            os.replace(adapter_temp, output_dir / self.JOINT_WEIGHTS_NAME)
        config_temp.write_text(serialized, encoding="utf-8")
        os.replace(weights_temp, output_dir / self.WEIGHTS_NAME)
        os.replace(config_temp, output_dir / self.CONFIG_NAME)

    @classmethod
    def read_checkpoint_config(cls, checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
        config = json.loads((Path(checkpoint_dir) / cls.CONFIG_NAME).read_text(encoding="utf-8"))
        required = {"format_version", "architecture", "head_config", "base_model_path", "adapter_path", "tokenizer_length", "action_schema", "training_metadata"}
        if config.get("format_version") == cls.JOINT_FORMAT_VERSION:
            required.add("joint_lora_config")
        if set(config) != required:
            raise ValueError(f"Invalid diffusion checkpoint metadata fields: {sorted(set(config) ^ required)}")
        if config["format_version"] not in (cls.FORMAT_VERSION, cls.JOINT_FORMAT_VERSION) or config["architecture"] != cls.__name__:
            raise ValueError("Unsupported diffusion checkpoint format or architecture")
        if config["format_version"] == cls.JOINT_FORMAT_VERSION:
            joint = config["joint_lora_config"]
            if not isinstance(joint, dict) or set(joint) != {"rank", "alpha", "dropout", "gradient_checkpointing", "target_modules"}:
                raise ValueError("Invalid joint_lora_config fields")
            if (type(joint["rank"]) is not int or joint["rank"] < 1 or joint["alpha"] <= 0
                    or not 0 <= joint["dropout"] < 1 or type(joint["gradient_checkpointing"]) is not bool
                    or not isinstance(joint["target_modules"], list) or not joint["target_modules"]
                    or any(not isinstance(name, str) for name in joint["target_modules"])):
                raise ValueError("Invalid joint_lora_config values")
        DiffusionActionHeadConfig.from_dict(config["head_config"])
        _validate_schema(config["action_schema"])
        if type(config["tokenizer_length"]) is not int or config["tokenizer_length"] <= 0:
            raise ValueError("Invalid checkpoint tokenizer_length")
        return config

    def load_head_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        config = self.read_checkpoint_config(checkpoint_dir)
        if DiffusionActionHeadConfig.from_dict(config["head_config"]) != self.head_config:
            raise ValueError("Checkpoint head configuration does not match this model")
        if config["action_schema"] != self.action_schema:
            raise ValueError("Checkpoint action schema does not match this model")
        if config["tokenizer_length"] != self.backbone.get_input_embeddings().weight.shape[0]:
            raise ValueError("Checkpoint tokenizer size does not match this backbone")
        state = torch.load(Path(checkpoint_dir) / self.WEIGHTS_NAME, map_location="cpu", weights_only=True)
        self.diffusion_head.load_state_dict(state, strict=True)
        if self.joint_lora_config is None:
            self._freeze_backbone()

    def _load_joint_checkpoint(self, checkpoint_dir, config):
        from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

        joint = config["joint_lora_config"]
        self.enable_joint_lora(**{key: joint[key] for key in ("rank", "alpha", "dropout", "gradient_checkpointing")})
        if self.joint_lora_config["target_modules"] != joint["target_modules"]:
            raise ValueError("Joint LoRA target modules do not match the saved backbone")
        state = torch.load(Path(checkpoint_dir) / self.JOINT_WEIGHTS_NAME, map_location="cpu", weights_only=True)
        expected = get_peft_model_state_dict(self.backbone, save_embedding_layers=False)
        if set(state) != set(expected) or any(state[name].shape != expected[name].shape for name in state if name in expected):
            raise ValueError("Joint adapter checkpoint has missing, extra or incompatible tensors")
        set_peft_model_state_dict(self.backbone, state)

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
        if config["format_version"] == cls.JOINT_FORMAT_VERSION:
            model._load_joint_checkpoint(checkpoint_dir, config)
        model.eval()
        return model
