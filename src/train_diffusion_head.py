"""Train only the continuous action diffusion head on a frozen AeroVLA backbone."""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from .diffusion_dataset import AeroVLADiffusionDataset, DiffusionDataCollator
    from .diffusion_split import deterministic_subset
    from .eval_diffusion_head import evaluate_diffusion
    from .model_wrapper.aerovla_diffusion import AeroVLADiffusionModel
    from .model_wrapper.diffusion_action_head import DiffusionActionHeadConfig
else:
    from diffusion_dataset import AeroVLADiffusionDataset, DiffusionDataCollator
    from diffusion_split import deterministic_subset
    from eval_diffusion_head import evaluate_diffusion
    from model_wrapper.aerovla_diffusion import AeroVLADiffusionModel
    from model_wrapper.diffusion_action_head import DiffusionActionHeadConfig


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def batch_shapes(value):
    if isinstance(value, dict):
        return {key: batch_shapes(item) for key, item in value.items()}
    return list(value.shape)


def assert_only_head_trainable(model):
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not names or any(not name.startswith("diffusion_head.") for name in names):
        raise RuntimeError(f"Only diffusion_head parameters may train; got {names}")
    if model.backbone.training:
        raise RuntimeError("Frozen backbone must remain in eval mode")
    return [param for param in model.diffusion_head.parameters() if param.requires_grad]


def train_head(model, dataloader, *, device, epochs=5, learning_rate=1e-4,
               weight_decay=0.01, gradient_accumulation_steps=8, max_grad_norm=1.0,
               max_steps=None, warmup_ratio=0.03, log_callback=None, save_callback=None,
               save_steps=1000, validation_callback=None, eval_steps=0):
    """Single-process loop; accumulation is weighted by samples, including short tails.

    Callbacks see completed optimizer steps. No optimizer state is stored in head
    checkpoints: those are portable inference artifacts, not exact training resumes.
    """
    if epochs < 1 or gradient_accumulation_steps < 1 or len(dataloader) < 1:
        raise ValueError("epochs, accumulation and dataloader length must be positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if not 0 <= warmup_ratio < 1 or save_steps < 1:
        raise ValueError("warmup_ratio must be in [0,1); save_steps must be positive")
    if eval_steps < 0:
        raise ValueError("eval_steps must be nonnegative")
    model.train()
    parameters = assert_only_head_trainable(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * math.ceil(len(dataloader) / gradient_accumulation_steps)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    warmup_steps = min(int(total_steps * warmup_ratio), total_steps - 1)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(epochs):
        window_loss, window_samples = 0.0, 0
        for batch_index, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)
            batch_size = batch["actions"].shape[0]
            loss = model(**batch)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite/scalar loss at batch {batch_index}: {loss}")
            # Convert each micro-batch mean into a sum before dividing by the
            # true window size. This handles a smaller last micro-batch correctly.
            (loss * batch_size).backward()
            window_loss += loss.detach().item() * batch_size
            window_samples += batch_size
            boundary = ((batch_index + 1) % gradient_accumulation_steps == 0
                        or batch_index + 1 == len(dataloader))
            if not boundary:
                continue
            if any(param.grad is not None for param in model.backbone.parameters()):
                raise RuntimeError("A frozen backbone parameter received a gradient")
            for param in parameters:
                if param.grad is not None:
                    param.grad.div_(window_samples)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_grad_norm, error_if_nonfinite=True
            )
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            metrics = {"step": global_step, "epoch": epoch + 1,
                       "loss": window_loss / window_samples, "lr": current_lr,
                       "grad_norm": float(grad_norm), "samples": window_samples}
            if log_callback:
                log_callback(metrics)
            # Validate at completed epochs, at the final (possibly early) step,
            # and optionally at optimizer-step intervals; each step runs once.
            should_validate = (batch_index + 1 == len(dataloader)
                               or global_step == total_steps
                               or (eval_steps > 0 and global_step % eval_steps == 0))
            if validation_callback and should_validate:
                validation_callback(global_step, epoch + 1)
                model.train()
                assert_only_head_trainable(model)
            if save_callback and global_step % save_steps == 0:
                save_callback(global_step)
            window_loss, window_samples = 0.0, 0
            if global_step >= total_steps:
                return global_step
    return global_step


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="./openvla-7b")
    parser.add_argument("--adapter-path", default="./checkpoints/aero_vla",
                        help="Frozen AeroVLA LoRA; pass 'none' to use only the base model")
    parser.add_argument("--data-root", default="./dataset_raw")
    parser.add_argument("--train-json", default="./data/aerovla_train_dataset.json")
    parser.add_argument("--split-json", default="./data/continuous_split_seed42.json",
                        help="Required trajectory-disjoint train/val split; no full-data fallback")
    parser.add_argument("--output-dir", default="./checkpoints/aero_vla_diffusion")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-dtype", choices=["float32", "bfloat16", "float16"],
                        default="bfloat16", help="Diffusion head always trains in float32")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-val-samples", type=int,
                        help="Fixed random validation subset; omitted means all validation samples")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=0,
                        help="Also validate every N updates; 0 means each epoch and final step")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--check-data", action="store_true",
                        help="Validate split and preprocess a train/val batch, without loading the 7B model")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-train-timesteps", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This script uses one process/GPU; launch with python, not torchrun")
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.num_workers < 0 or args.max_length < 2:
        raise ValueError("Invalid batch-size, num-workers or max-length")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    if args.max_val_samples is not None and args.max_val_samples < 1:
        raise ValueError("max-val-samples must be positive")
    if args.epochs < 1 or args.gradient_accumulation_steps < 1 or args.save_steps < 1 or args.eval_steps < 0:
        raise ValueError("Invalid epoch, accumulation, saving or evaluation interval")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Invalid optimizer hyperparameters")
    train_json = Path(args.train_json)
    if not train_json.is_file():
        raise FileNotFoundError(f"Training manifest is missing: {train_json}. "
                                "Use the AeroVLA per-frame training JSON, not evaluation episodes.")
    split_json = Path(args.split_json)
    if not split_json.is_file():
        raise FileNotFoundError(f"Trajectory split is missing: {split_json}; train/val isolation is required")
    output_dir = Path(args.output_dir)
    if not args.check_data and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a fresh --output-dir; an existing run is at {output_dir}")
    device = torch.device(args.device)
    seed_everything(args.seed)
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = AeroVLADiffusionDataset(args.data_root, args.train_json, tokenizer, processor,
                                     split_json=args.split_json, split="train")
    val_dataset = AeroVLADiffusionDataset(args.data_root, args.train_json, tokenizer, processor,
                                         split_json=args.split_json, split="val")
    dataset = deterministic_subset(dataset, args.max_samples, args.seed)
    val_dataset = deterministic_subset(val_dataset, args.max_val_samples, args.eval_seed)
    collator = DiffusionDataCollator(tokenizer, max_length=args.max_length)
    print(json.dumps({"train_split": dataset.split_info, "val_split": val_dataset.split_info}), flush=True)
    if args.check_data:
        shapes = {}
        for name, selected in (("train", dataset), ("val", val_dataset)):
            batch = collator([selected[i] for i in range(min(args.batch_size, len(selected)))])
            shapes[name] = batch_shapes(batch)
        print(json.dumps({"data_check": "passed", "batch_shapes": shapes}), flush=True)
        return
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check GPU access or select --device cpu --backbone-dtype float32")
    if device.type == "cpu" and args.backbone_dtype != "float32":
        raise ValueError("Use --backbone-dtype float32 for CPU execution")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        collate_fn=collator,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            collate_fn=collator,
                            generator=torch.Generator().manual_seed(args.eval_seed))
    backbone_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    head_config = DiffusionActionHeadConfig(
        condition_dim=backbone_config.text_config.hidden_size, action_dim=4, action_horizon=1,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads,
        num_train_timesteps=args.num_train_timesteps, num_inference_steps=args.num_inference_steps,
        dropout=args.dropout,
    )
    adapter_path = None if args.adapter_path.lower() == "none" else args.adapter_path
    model = AeroVLADiffusionModel.from_pretrained_backbone(
        args.model_path, adapter_path=adapter_path, head_config=head_config,
        torch_dtype=getattr(torch, args.backbone_dtype), device=device,
        tokenizer_length=len(tokenizer),
    )
    model.train()
    parameters = assert_only_head_trainable(model)
    trainable = sum(param.numel() for param in parameters)
    frozen = sum(param.numel() for param in model.backbone.parameters())
    print(f"Frozen backbone: {frozen:,}; trainable diffusion head: {trainable:,}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**vars(args), "trainable_parameters": trainable, "frozen_parameters": frozen,
                  "head_config": head_config.to_dict(), "checkpoint_type": "head_only",
                  "train_json": str(train_json.resolve()), "split_json": str(split_json.resolve()),
                  "train_split": dataset.split_info, "val_split": val_dataset.split_info}
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    validation_state = {"best_noise_mse": float("inf"), "last_validation": None}

    def log(metrics):
        line = json.dumps(metrics)
        print(line, flush=True)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    def save(step, destination=None):
        destination = destination or output_dir / f"checkpoint-{step}"
        model.save_pretrained(destination, training_metadata={
            **run_config, "global_step": step, "last_validation": validation_state["last_validation"]
        })
        tokenizer.save_pretrained(destination)
        processor.save_pretrained(destination)
        shutil.copyfile(split_json, Path(destination) / "data_split.json")

    def validate(step, epoch):
        metrics = evaluate_diffusion(model, val_loader, device=device, seed=args.eval_seed,
                                     num_inference_steps=args.num_inference_steps)
        metrics = {**metrics, "step": step, "epoch": epoch, "split": "val"}
        validation_state["last_validation"] = metrics
        line = json.dumps(metrics, allow_nan=False)
        print("Validation: " + line, flush=True)
        with (output_dir / "validation_metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")
        if metrics["noise_mse"] < validation_state["best_noise_mse"]:
            validation_state["best_noise_mse"] = metrics["noise_mse"]
            save(step, output_dir / "best")
            (output_dir / "best_metrics.json").write_text(line + "\n")

    steps = train_head(
        model, loader, device=device, epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm, max_steps=args.max_steps, warmup_ratio=args.warmup_ratio,
        log_callback=log, save_callback=save, save_steps=args.save_steps,
        validation_callback=validate, eval_steps=args.eval_steps,
    )
    save(steps, output_dir)
    print(f"Completed {steps} optimizer steps. Head checkpoint: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
