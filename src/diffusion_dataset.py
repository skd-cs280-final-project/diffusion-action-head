"""Continuous AeroVLA targets and prompt-only batches for diffusion training.

The released AeroVLA JSON provides one action per image pair, so each target
has shape ``[1, 4]``: forward, down, yaw, landing. The first three coordinates
retain the original clipping ranges; landing retains the last/penultimate-step
rule. All four coordinates are scaled to [-1, 1], without quantization.
"""

from collections.abc import Mapping
from bisect import bisect_right
from functools import lru_cache
import json
from pathlib import Path
from typing import Optional

import torch

try:
    from .aerovla_dataset import ACTION_STATS, AeroVLADataset
    from .diffusion_split import select_trajectory_split
    from .uav_history import STATE_DIM, encode_state_history, observation_identity
except ImportError:  # Support ``python src/train_diffusion.py`` as well.
    from aerovla_dataset import ACTION_STATS, AeroVLADataset
    from diffusion_split import select_trajectory_split
    from uav_history import STATE_DIM, encode_state_history, observation_identity


ACTION_ORDER = ("fwd", "down", "yaw", "land")
ACTION_DIM = len(ACTION_ORDER)
ACTION_MIN = tuple(ACTION_STATS[axis]["min"] for axis in ("forward", "down", "yaw")) + (0.0,)
ACTION_MAX = tuple(ACTION_STATS[axis]["max"] for axis in ("forward", "down", "yaw")) + (1.0,)


@lru_cache(maxsize=4096)
def _read_history_log(path):
    """Read immutable observation logs with a bounded per-worker cache."""
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, Mapping):
        raise ValueError(f"History log must contain an observation object: {path}")
    return record


def _history_frame(sample):
    if not isinstance(sample, Mapping):
        raise ValueError("State history samples must be mappings.")
    trajectory, filename = sample.get("traj_rel_dir"), sample.get("img_name")
    if (
        not isinstance(trajectory, str) or not trajectory or trajectory.startswith("/")
        or "\\" in trajectory or any(part in ("", ".", "..") for part in trajectory.split("/"))
    ):
        raise ValueError("State history requires a canonical relative traj_rel_dir.")
    if not isinstance(filename, str) or "/" in filename or "\\" in filename or not Path(filename).stem.isdecimal():
        raise ValueError("State history requires img_name with a numeric frame stem.")
    return trajectory, int(Path(filename).stem)


def _action_tensor(actions) -> torch.Tensor:
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if actions.ndim == 0 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Actions must end in {ACTION_DIM} coordinates ordered as {ACTION_ORDER}.")
    if not torch.isfinite(actions).all():
        raise ValueError("Actions must contain only finite values.")
    return actions


def normalize_actions(actions) -> torch.Tensor:
    """Clip physical [..., 4] actions and map each coordinate to [-1, 1].

    Physical LAND values are 0 (continue) and 1 (land). Inputs are never
    modified, and the returned tensor is float32 on the input tensor's device.
    """
    actions = _action_tensor(actions)
    lower, upper = actions.new_tensor(ACTION_MIN), actions.new_tensor(ACTION_MAX)
    actions = torch.minimum(torch.maximum(actions, lower), upper)
    return 2.0 * (actions - lower) / (upper - lower) - 1.0


def denormalize_actions(actions) -> torch.Tensor:
    """Decode normalized [..., 4] actions, clipping to the physical ranges.

    LAND is returned as a continuous score in [0, 1]; inference can apply a
    threshold of 0.5. Forward/down are in the original distance units and yaw
    uses the original radians, exactly as in the existing AeroVLA decoder.
    """
    actions = _action_tensor(actions).clamp(-1.0, 1.0)
    lower, upper = actions.new_tensor(ACTION_MIN), actions.new_tensor(ACTION_MAX)
    return (actions + 1.0) * 0.5 * (upper - lower) + lower


def _remove_image_batch(pixel_values):
    if isinstance(pixel_values, Mapping):
        if not pixel_values:
            raise ValueError("Image processor returned an empty pixel_values mapping.")
        return {key: _remove_image_batch(value) for key, value in pixel_values.items()}
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
        raise ValueError("Image processor must return pixel_values of shape [1, C, H, W].")
    return pixel_values[0]


class AeroVLADiffusionDataset(AeroVLADataset):
    """Reuse AeroVLA's front-over-down image mosaic with continuous labels."""

    def __init__(
        self, data_root, clean_json_path, tokenizer, processor,
        action_horizon=1, split_json=None, split=None,
        history_length=0,
    ):
        if type(history_length) is not int or history_length < 0:
            raise ValueError("history_length must be a nonnegative integer.")
        if action_horizon != 1:
            raise ValueError(
                "The AeroVLA frame-label JSON supports action_horizon=1 only; "
                "explicit trajectory ordering and future-action labels are required for longer horizons."
            )
        super().__init__(data_root, clean_json_path, tokenizer, processor)
        if not isinstance(self.samples, list):
            raise ValueError("The training JSON must contain a list of AeroVLA frame-label records.")
        if split_json is None:
            if split is not None:
                raise ValueError("split requires split_json; refusing to silently use every sample.")
            self.split_info = {"split": None, "split_json": None, "num_samples": len(self.samples)}
        else:
            self.samples, self.split_info = select_trajectory_split(
                self.samples, split_json, split, data_json=clean_json_path,
            )
            print(
                f"[Dataset] Selected {self.split_info['num_samples']} {split} samples from "
                f"{self.split_info['num_trajectories']} trajectories."
            )
        self.action_horizon = action_horizon
        self.history_length = history_length
        self._history_frames = {}
        if history_length:
            # Build after trajectory splitting and before callers wrap this
            # dataset in a frame Subset; subsampling must not erase its past.
            for sample in self.samples:
                trajectory, frame = _history_frame(sample)
                frames = self._history_frames.setdefault(trajectory, set())
                if frame in frames:
                    raise ValueError(f"Duplicate history frame: {trajectory}/{frame:06d}.")
                frames.add(frame)
            self._history_frames = {key: tuple(sorted(frames)) for key, frames in self._history_frames.items()}

    def _get_state_history(self, sample):
        trajectory, frame = _history_frame(sample)
        frames = self._history_frames[trajectory]
        end = bisect_right(frames, frame)
        records, seen = [], set()
        for past_frame in reversed(frames[:end]):
            path = str(Path(self.data_root) / trajectory / "log" / f"{past_frame:06d}.json")
            # Only the matching observation log is consulted. Expert labels and
            # merged trajectories may contain future states and are never read.
            record = dict(_read_history_log(path))
            sensors = record.get("sensors", {})
            state = sensors.get("state", {}) if isinstance(sensors, Mapping) else {}
            for source in (record, state):
                if not isinstance(source, Mapping):
                    continue
                for field in ("frame", "frame_id", "frame_index"):
                    if field in source:
                        recorded = source[field]
                        if isinstance(recorded, bool) or str(recorded) != str(past_frame):
                            raise ValueError(f"History log {path} has mismatched {field}={recorded!r}; expected {past_frame}.")
            record.setdefault("frame_id", past_frame)
            identity = observation_identity(record)
            if identity not in seen:
                records.append(record)
                seen.add(identity)
            if len(records) == self.history_length:
                break
        records.reverse()
        return encode_state_history(records, self.history_length)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            instruction = sample["instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be a nonempty string")
            flags = (sample["is_last_step"], sample["is_penultimate"])
            if any(not isinstance(flag, (bool, int)) or flag not in (0, 1) for flag in flags):
                raise ValueError("is_last_step and is_penultimate must be booleans or 0/1")
            label = sample["label"]
            actions = normalize_actions([
                float(label["fwd"]), float(label["down"]), float(label["yaw"]), float(any(flags))
            ]).unsqueeze(0)
            image = self._get_images(sample["traj_rel_dir"], sample["img_name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid AeroVLA training sample at index {idx}: {exc}") from exc
        pixel_values = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
        feature = {
            "image": _remove_image_batch(pixel_values),
            "prompt_only": f"{instruction.strip()}\nAction: ",
            "actions": actions,
        }
        if self.history_length:
            feature["state_history"], feature["state_history_mask"] = self._get_state_history(sample)
        return feature


def _stack_images(images, dtype):
    first = images[0]
    if isinstance(first, Mapping):
        if any(not isinstance(image, Mapping) or set(image) != set(first) for image in images):
            raise ValueError("All examples must contain the same image processor keys.")
        return {key: _stack_images([image[key] for image in images], dtype) for key in first}
    if not all(isinstance(image, torch.Tensor) and image.ndim == 3 for image in images):
        raise ValueError("Each processed image must have shape [C, H, W].")
    result = torch.stack(images)
    return result.to(dtype=dtype) if dtype is not None else result


class DiffusionDataCollator:
    """Batch observations and targets without exposing action labels to the VLM.

    Long prompts fail explicitly instead of silently truncating the instruction
    or its ``Action:`` suffix. Image dtype defaults to the processor's dtype;
    the model controls precision when executing the frozen backbone.
    """

    def __init__(self, tokenizer, max_length=2048, pixel_dtype: Optional[torch.dtype] = None):
        if max_length < 1:
            raise ValueError("max_length must be positive.")
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer needs a pad_token before constructing the collator.")
        if getattr(tokenizer, "add_eos_token", False):
            raise ValueError("Disable tokenizer.add_eos_token for prompt-only diffusion inputs.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pixel_dtype = pixel_dtype

    def __call__(self, features):
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        batch = self.tokenizer(
            [feature["prompt_only"] for feature in features],
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Prompt contains {input_ids.shape[1]} tokens, exceeding max_length={self.max_length}; "
                "shorten the instruction or increase max_length within the backbone's context limit."
            )
        if not attention_mask.any(dim=1).all():
            raise ValueError("Every prompt must contain at least one non-padding token.")
        actions = torch.stack([_action_tensor(feature["actions"]) for feature in features])
        if actions.shape != (len(features), 1, ACTION_DIM):
            raise ValueError(f"Expected actions with shape [batch, 1, {ACTION_DIM}].")
        if (actions.abs() > 1.0).any():
            raise ValueError("Diffusion targets must already be normalized to [-1, 1].")
        result = {
            "pixel_values": _stack_images([feature["image"] for feature in features], self.pixel_dtype),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "actions": actions,
        }
        history_present = ["state_history" in feature or "state_history_mask" in feature for feature in features]
        if any(history_present):
            if not all("state_history" in feature and "state_history_mask" in feature for feature in features):
                raise ValueError("Every example must supply both state_history and state_history_mask.")
            histories = [feature["state_history"] for feature in features]
            masks = [feature["state_history_mask"] for feature in features]
            shape = histories[0].shape if isinstance(histories[0], torch.Tensor) else None
            if shape is None or len(shape) != 2 or shape[0] < 1 or shape[1] != STATE_DIM:
                raise ValueError(f"Each state_history must have shape [K, {STATE_DIM}] with K >= 1.")
            if not all(isinstance(history, torch.Tensor) and history.shape == shape for history in histories):
                raise ValueError("State history shapes must match within a batch.")
            if not all(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and mask.shape == shape[:1] for mask in masks):
                raise ValueError("Each state_history_mask must be bool with shape [K].")
            if not all(bool(mask[-1]) for mask in masks):
                raise ValueError("State histories must include a valid current observation in the last slot.")
            history = torch.stack(histories).to(dtype=torch.float32)
            if not torch.isfinite(history).all() or (history.abs() > 1).any():
                raise ValueError("State history values must be finite and normalized to [-1, 1].")
            result.update(state_history=history, state_history_mask=torch.stack(masks))
        return result
