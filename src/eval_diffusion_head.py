"""Offline validation of a frozen AeroVLA backbone and its action diffusion head."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__:
    from .diffusion_dataset import AeroVLADiffusionDataset, DiffusionDataCollator, denormalize_actions
    from .model_wrapper.aerovla_diffusion import AeroVLADiffusionModel
else:
    from diffusion_dataset import AeroVLADiffusionDataset, DiffusionDataCollator, denormalize_actions
    from model_wrapper.aerovla_diffusion import AeroVLADiffusionModel


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _valid_coordinates(actions, action_mask):
    if actions.ndim != 3 or actions.shape[-1] != 4:
        raise ValueError("Expected actions with shape [batch, horizon, 4]")
    if action_mask is None:
        return torch.ones_like(actions, dtype=torch.bool)
    if not torch.all((action_mask == 0) | (action_mask == 1)):
        raise ValueError("action_mask must contain only zero and one")
    if action_mask.shape == actions.shape[:2]:
        action_mask = action_mask[..., None].expand_as(actions)
    if action_mask.shape != actions.shape:
        raise ValueError("action_mask must have shape [batch, horizon] or [batch, horizon, 4]")
    return action_mask.to(device=actions.device, dtype=torch.bool)


def _loader_generators(dataloader):
    """Keep evaluation from advancing a caller-owned loader/sampler generator."""
    generators = {}
    for owner in (dataloader, getattr(dataloader, "sampler", None),
                  getattr(getattr(dataloader, "batch_sampler", None), "sampler", None)):
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator):
            generators[id(generator)] = (generator, generator.get_state())
    return list(generators.values())


@torch.no_grad()
def evaluate_diffusion(model, dataloader, *, device, seed=42,
                       num_inference_steps=None, sample_actions=True):
    """Evaluate epsilon MSE and optionally sampled actions with no training side effects.

    Each batch encodes only observations, once. Noise, timesteps and DDPM sampling
    have independent local generators. Repeated calls are reproducible for the
    same model, ordered batches, precision and device; changing the batch size or
    device need not produce identical stochastic samples. Global torch RNG and
    caller-owned loader generators are restored, including on evaluation errors.
    Metrics are weighted by valid scalar coordinates, not by batch means. Targets
    use the same clipped physical action ranges as training. There is deliberately
    no combined physical-unit error across distances and yaw radians.
    """
    device = torch.device(device)
    head = model.diffusion_head
    if num_inference_steps is None:
        num_inference_steps = head.config.num_inference_steps
    if (not isinstance(num_inference_steps, int) or isinstance(num_inference_steps, bool)
            or not 1 <= num_inference_steps <= head.config.num_train_timesteps):
        raise ValueError("num_inference_steps must be between 1 and num_train_timesteps")
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    time_generator = torch.Generator(device=device).manual_seed(seed + 1)
    sample_generator = torch.Generator(device=device).manual_seed(seed + 2)
    modes = [(module, module.training) for module in model.modules()]
    loader_generators = _loader_generators(dataloader)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    noise_sum, noise_count, samples, valid_samples = 0.0, 0, 0, 0
    continuous_sum, continuous_count = 0.0, 0
    physical_sum = torch.zeros(3, dtype=torch.float64)
    physical_squared = torch.zeros(3, dtype=torch.float64)
    physical_count = torch.zeros(3, dtype=torch.long)
    tp = tn = fp = fn = 0
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            for batch in dataloader:
                batch = _move_to_device(batch, device)
                actions = batch["actions"].to(dtype=torch.float32)
                valid = _valid_coordinates(actions, batch.get("action_mask"))
                samples += actions.shape[0]
                valid_samples += int(valid.flatten(1).any(dim=1).sum().item())
                count = int(valid.sum().item())
                if count == 0:
                    continue
                # No action targets are allowed in the backbone conditioning.
                observations = {key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values")
                                if key in batch}
                condition, condition_mask = model.encode_condition(**observations)
                history = {key: batch[key] for key in ("state_history", "state_history_mask")
                           if key in batch}
                noise = torch.randn(actions.shape, device=device, dtype=torch.float32,
                                    generator=noise_generator)
                timesteps = torch.randint(head.config.num_train_timesteps, (actions.shape[0],),
                                          device=device, generator=time_generator)
                loss = head.compute_loss(condition, actions, condition_mask=condition_mask,
                                         action_mask=valid, noise=noise, timesteps=timesteps, **history)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError("Evaluation noise loss must be a finite scalar")
                noise_sum += loss.double().item() * count
                noise_count += count
                if not sample_actions:
                    continue
                predictions = head.sample(condition, condition_mask=condition_mask,
                                          num_inference_steps=num_inference_steps,
                                          generator=sample_generator, **history)
                if predictions.shape != actions.shape:
                    raise ValueError("Sampled actions must have the same shape as targets")
                if not torch.isfinite(predictions.masked_select(valid)).all():
                    raise FloatingPointError("Sampled actions contain non-finite values")
                target = actions.masked_fill(~valid, 0).clamp(-1, 1)
                prediction = predictions.float().masked_fill(~valid, 0).clamp(-1, 1)
                continuous_valid = valid[..., :3]
                errors = (prediction[..., :3].double() - target[..., :3].double()).square()
                continuous_sum += errors.masked_select(continuous_valid).sum().item()
                continuous_count += int(continuous_valid.sum().item())
                physical_error = (denormalize_actions(prediction)[..., :3].double()
                                  - denormalize_actions(target)[..., :3].double())
                physical_error = physical_error.masked_fill(~continuous_valid, 0)
                physical_sum += physical_error.abs().sum(dim=(0, 1)).cpu()
                physical_squared += physical_error.square().sum(dim=(0, 1)).cpu()
                physical_count += continuous_valid.sum(dim=(0, 1)).cpu()
                land_valid = valid[..., 3]
                land_pred = prediction[..., 3] >= 0
                land_target = target[..., 3] >= 0
                tp += int((land_valid & land_pred & land_target).sum().item())
                tn += int((land_valid & ~land_pred & ~land_target).sum().item())
                fp += int((land_valid & land_pred & ~land_target).sum().item())
                fn += int((land_valid & ~land_pred & land_target).sum().item())
    finally:
        for generator, state in loader_generators:
            generator.set_state(state)
        # Restore mixed child modes too; model.train() alone would overwrite them.
        for module, training in modes:
            module.training = training
    if noise_count == 0:
        raise ValueError("Evaluation requires at least one valid action coordinate")
    result = {"num_samples": samples, "num_valid_samples": valid_samples,
              "num_valid_coordinates": noise_count, "noise_mse": noise_sum / noise_count,
              "seed": seed, "sample_actions": bool(sample_actions)}
    if sample_actions:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result.update({
            "num_inference_steps": num_inference_steps,
            "normalized_continuous_mse": continuous_sum / continuous_count if continuous_count else None,
            "num_valid_continuous_coordinates": continuous_count,
            "physical": {
                name: {"mae": physical_sum[index].item() / count if count else None,
                       "rmse": math.sqrt(physical_squared[index].item() / count) if count else None,
                       "count": count, "unit": "radians" if name == "yaw" else "original_distance_unit"}
                for index, name in enumerate(("fwd", "down", "yaw"))
                for count in [int(physical_count[index].item())]
            },
            "land": {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                     "accuracy": (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else None,
                     "support_positive": tp + fn, "support_negative": tn + fp,
                     "predicted_positive": tp + fp, "threshold_normalized": 0.0,
                     "zero_division": 0.0},
        })
    return result


def validate_checkpoint_split(checkpoint_config, split_info, data_json):
    """Reject known split mismatches and report legacy provenance as unknown."""
    if split_info.get("split") != "val":
        raise ValueError("Offline held-out evaluation requires the val split")
    metadata = checkpoint_config.get("training_metadata", {})
    train_split = metadata.get("train_split")
    if not train_split:
        return {"status": "unknown", "reason": "Checkpoint has no recorded training split; independence cannot be verified"}
    if train_split.get("split") != "train":
        raise ValueError("Checkpoint training metadata must identify the train split")
    checked = []
    for key in ("data_sha256", "data_json_sha256", "split_sha256", "trajectory_sha256"):
        saved, current = train_split.get(key), split_info.get(key)
        if saved is not None and current is not None:
            if saved != current:
                raise ValueError(f"Checkpoint training {key} does not match the evaluation data/split")
            checked.append(key)
    saved_data = train_split.get("data_json") or metadata.get("train_json")
    data_hash_verified = "data_sha256" in checked or "data_json_sha256" in checked
    if saved_data and not data_hash_verified:
        if Path(saved_data).expanduser().resolve() != Path(data_json).expanduser().resolve():
            raise ValueError("Checkpoint training data path differs and no matching content hash is available")
    if data_hash_verified and "split_sha256" in checked:
        return {"status": "verified_diffusion_head_holdout", "checked_hashes": checked,
                "scope": "Only diffusion-head train/val separation is verified; frozen adapter pretraining overlap is unknown"}
    return {"status": "unknown", "checked_hashes": checked,
            "reason": "Checkpoint lacks data/split content hashes needed to verify held-out independence"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-json", default="./data/aerovla_train_dataset.json")
    parser.add_argument("--split-json", default="./data/continuous_split_seed42.json")
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--data-root", default="./dataset_raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-dtype", "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, help="Deterministic subset for smoke checks; omit for all validation frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", "--inference-steps", type=int)
    parser.add_argument("--output", "--output-json", help="JSON report path; defaults to CHECKPOINT/eval_val.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process/GPU; launch with python, not torchrun")
    if args.batch_size < 1 or args.num_workers < 0 or args.max_length < 2:
        raise ValueError("Invalid batch-size, num-workers or max-length")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check GPU access or select --device cpu --backbone-dtype float32")
    if device.type == "cpu" and args.backbone_dtype != "float32":
        raise ValueError("Use --backbone-dtype float32 for CPU execution")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = AeroVLADiffusionModel.read_checkpoint_config(checkpoint)
    if __package__:
        from .uav_history import STATE_DIM, HISTORY_SCHEMA
    else:
        from uav_history import STATE_DIM, HISTORY_SCHEMA
    history_length = config["head_config"].get("history_length", 0)
    if history_length and config["head_config"].get("state_dim") != STATE_DIM:
        raise ValueError("Checkpoint UAV state dimension does not match the data schema")
    saved_schema = config.get("training_metadata", {}).get("history_schema")
    if history_length and saved_schema != HISTORY_SCHEMA:
        raise ValueError("Checkpoint history_schema is missing or differs from the data encoder")
    from transformers import AutoProcessor, AutoTokenizer
    if __package__:
        from .diffusion_split import deterministic_subset
    else:
        from diffusion_split import deterministic_subset
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    dataset = AeroVLADiffusionDataset(args.data_root, args.data_json, tokenizer, processor,
                                     split_json=args.split_json, split=args.split, history_length=history_length)
    provenance = validate_checkpoint_split(config, dataset.split_info, args.data_json)
    dataset = deterministic_subset(dataset, args.max_samples, args.seed)
    if len(dataset) == 0:
        raise ValueError("The validation dataset is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        collate_fn=DiffusionDataCollator(tokenizer, max_length=args.max_length),
                        generator=torch.Generator().manual_seed(args.seed))
    print(json.dumps({"evaluation_samples": len(dataset), "split_provenance": provenance}), flush=True)
    model = AeroVLADiffusionModel.from_pretrained(checkpoint, device=device,
                                                 torch_dtype=getattr(torch, args.backbone_dtype),
                                                 tokenizer_length=len(tokenizer))
    metrics = evaluate_diffusion(model, loader, device=device, seed=args.seed,
                                 num_inference_steps=args.num_inference_steps)
    report = {"evaluation_type": "offline_diffusion_action_validation", "checkpoint": str(checkpoint),
              "data_json": str(Path(args.data_json).expanduser().resolve()),
              "split_json": str(Path(args.split_json).expanduser().resolve()),
              "data_root": str(Path(args.data_root).expanduser().resolve()),
              "split": dataset.split_info, "split_provenance": provenance,
              "checkpoint_training_split": config["training_metadata"].get("train_split"),
              "metrics": metrics, "runtime": {"device": str(device), "backbone_dtype": args.backbone_dtype,
                                                 "batch_size": args.batch_size, "seed": args.seed},
              "target_semantics": "Clipped fwd [0,5], down [-5,5], yaw [-1.1,1.1]; LAND=is_last_step OR is_penultimate",
              "scope": "Offline action metrics do not measure closed-loop AirSim navigation success"}
    if history_length:
        report["history_schema"] = HISTORY_SCHEMA
        report["history_length"] = history_length
    output = Path(args.output) if args.output else checkpoint / "eval_val.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)
    print(f"Evaluation report: {output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    main()
