"""Continue a trained diffusion policy with vision/LLM LoRA and a full action head.

Supports one GPU or torchrun DDP. Checkpoints are portable policy weights;
--checkpoint warm-starts a new schedule, not an exact optimizer/RNG resume.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

if __package__:
    from .diffusion_dataset import AeroVLADiffusionDataset, DiffusionDataCollator
    from .diffusion_split import deterministic_subset
    from .eval_diffusion_head import evaluate_diffusion, validate_checkpoint_split
    from .model_wrapper.aerovla_diffusion import AeroVLADiffusionModel
    from .train_diffusion_head import seed_everything, move_to_device, batch_shapes
    from .uav_history import HISTORY_SCHEMA, STATE_DIM
else:
    from diffusion_dataset import AeroVLADiffusionDataset, DiffusionDataCollator
    from diffusion_split import deterministic_subset
    from eval_diffusion_head import evaluate_diffusion, validate_checkpoint_split
    from model_wrapper.aerovla_diffusion import AeroVLADiffusionModel
    from train_diffusion_head import seed_everything, move_to_device, batch_shapes
    from uav_history import HISTORY_SCHEMA, STATE_DIM


def parameter_groups(model, *, lora_lr, projector_lr, head_lr, weight_decay):
    """Audit every trainable tensor and keep the four learning rates explicit."""
    groups = {name: [] for name in ("vision_lora", "llm_lora", "projector", "diffusion_head")}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("diffusion_head."):
            group = "diffusion_head"
        elif "projector" in name and name.startswith("backbone."):
            group = "projector"
        elif "lora_" in name and "vision_backbone" in name:
            group = "vision_lora"
        elif "lora_" in name and "language_model" in name:
            group = "llm_lora"
        else:
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"Trainable parameter must use float32: {name}")
        groups[group].append(parameter)
    if any(not values for values in groups.values()):
        raise RuntimeError(f"Joint training requires all four parameter groups: {list(groups)}")
    learning_rates = {"vision_lora": lora_lr, "llm_lora": lora_lr,
                      "projector": projector_lr, "diffusion_head": head_lr}
    if any(not math.isfinite(lr) or lr <= 0 for lr in learning_rates.values()):
        raise ValueError("All learning rates must be finite and positive")
    return [{"name": name, "params": values, "lr": learning_rates[name],
             "weight_decay": weight_decay} for name, values in groups.items()]


def train_joint(model, dataloader, *, device, epochs=1, lora_lr=2e-5,
                projector_lr=1e-5, head_lr=1e-5, weight_decay=0.01,
                gradient_accumulation_steps=8, max_grad_norm=1.0,
                max_steps=None, warmup_ratio=0.03, log_callback=None,
                validation_callback=None, eval_steps=1000,
                save_callback=None, save_steps=1000):
    """Sample-weighted accumulation; DDP averages equal-size rank windows.

    DistributedSampler pads at most world_size-1 training frames per epoch.
    Evaluation runs only on rank zero, on its unwrapped model, without padding.
    All callbacks are invoked on every rank so they may use barriers safely.
    """
    if epochs < 1 or gradient_accumulation_steps < 1 or len(dataloader) < 1:
        raise ValueError("epochs, accumulation and dataloader length must be positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if (not all(math.isfinite(value) for value in (weight_decay, max_grad_norm, warmup_ratio))
            or weight_decay < 0 or max_grad_norm <= 0 or not 0 <= warmup_ratio < 1):
        raise ValueError("Invalid optimizer hyperparameters")
    if save_steps < 1 or eval_steps < 0:
        raise ValueError("Invalid saving or evaluation interval")
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    groups = parameter_groups(raw, lora_lr=lora_lr, projector_lr=projector_lr,
                              head_lr=head_lr, weight_decay=weight_decay)
    parameters = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.AdamW(groups)
    total_steps = epochs * math.ceil(len(dataloader) / gradient_accumulation_steps)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    warmup_steps = min(int(total_steps * warmup_ratio), total_steps - 1)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    distributed = isinstance(model, DistributedDataParallel)
    world_size = dist.get_world_size() if distributed else 1
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    started = time.monotonic()
    model.train()
    for epoch in range(epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        window_loss, window_samples = 0.0, 0
        for batch_index, batch in enumerate(dataloader):
            boundary = ((batch_index + 1) % gradient_accumulation_steps == 0
                        or batch_index + 1 == len(dataloader))
            context = model.no_sync() if distributed and not boundary else nullcontext()
            with context:
                batch = move_to_device(batch, device)
                batch_size = batch["actions"].shape[0]
                loss = model(**batch)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite/scalar loss at batch {batch_index}")
                (loss * batch_size).backward()
            window_loss += loss.detach().item() * batch_size
            window_samples += batch_size
            if not boundary:
                continue
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(window_samples)
            group_norms = {}
            witnesses = {}
            for group in optimizer.param_groups:
                gradients = [p.grad for p in group["params"] if p.grad is not None]
                norm = torch.stack([g.detach().float().norm() for g in gradients]).norm() if gradients else torch.tensor(0.)
                group_norms[group["name"]] = float(norm)
                if global_step == 0:
                    if not torch.isfinite(norm) or norm <= 0:
                        raise RuntimeError(f"No finite nonzero first-step gradient for {group['name']}")
                    # A small representative slice confirms an actual optimizer update.
                    candidate = max((p for p in group["params"] if p.grad is not None),
                                    key=lambda p: float(p.grad.detach().abs().max()))
                    index = int(candidate.grad.detach().abs().reshape(-1).argmax())
                    witnesses[group["name"]] = (candidate, index, candidate.detach().reshape(-1)[index].clone())
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
            rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
            optimizer.step()
            updates = {name: bool(p.detach().reshape(-1)[index] != before)
                       for name, (p, index, before) in witnesses.items()}
            if updates and not all(updates.values()):
                raise RuntimeError(f"First optimizer step did not update every group: {updates}")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            statistics = torch.tensor([window_loss, window_samples], dtype=torch.float64, device=device)
            if distributed:
                dist.all_reduce(statistics)
            metrics = {"step": global_step, "epoch": epoch + 1,
                       "loss": float(statistics[0] / statistics[1]),
                       "samples": int(statistics[1]), "world_size": world_size,
                       "lr": rates, "grad_norm": float(grad_norm),
                       "group_grad_norm": group_norms,
                       "elapsed_seconds": time.monotonic() - started,
                       "total_steps": total_steps}
            if updates:
                metrics["first_step_parameter_updates"] = updates
            if torch.device(device).type == "cuda":
                metrics["peak_memory_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            if log_callback:
                log_callback(metrics)
            should_validate = (batch_index + 1 == len(dataloader) or global_step == total_steps
                               or (eval_steps > 0 and global_step % eval_steps == 0))
            if validation_callback and should_validate:
                validation_callback(global_step, epoch + 1)
                model.train()
            if save_callback and global_step % save_steps == 0:
                save_callback(global_step)
            window_loss, window_samples = 0.0, 0
            if global_step >= total_steps:
                return global_step
    return global_step


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="./checkpoints/best")
    parser.add_argument("--model-path", help="Override the source checkpoint's base model path")
    parser.add_argument("--adapter-path", help="Override the original, merged AeroVLA adapter path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="./dataset_raw")
    parser.add_argument("--train-json", default="./data/aerovla_train_dataset.json")
    parser.add_argument("--split-json", default="./data/continuous_split_seed42.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-val-samples", type=int, default=512,
                        help="Fixed validation subset; 0 evaluates the complete val split")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--projector-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.batch_size, args.eval_batch_size, args.epochs, args.gradient_accumulation_steps,
           args.save_steps, args.lora_rank, args.lora_alpha) < 1:
        raise ValueError("Batch sizes, epochs, accumulation, saving interval and LoRA dimensions must be positive")
    if args.num_workers < 0 or args.max_val_samples < 0 or args.eval_steps < 0 or args.max_length < 2:
        raise ValueError("Invalid data or validation parameters")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    if not 0 <= args.lora_dropout < 1 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Dropout and warmup ratio must be in [0,1)")
    if (not all(math.isfinite(value) for value in (args.lora_lr, args.projector_lr, args.head_lr,
                                                  args.max_grad_norm, args.weight_decay))
            or min(args.lora_lr, args.projector_lr, args.head_lr, args.max_grad_norm) <= 0 or args.weight_decay < 0):
        raise ValueError("Invalid optimizer hyperparameters")
    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check GPU access")
        torch.cuda.set_device(device)
    elif args.backbone_dtype != "float32" or world_size > 1:
        raise ValueError("CPU requires float32 and single-process execution")
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))

    def barrier():
        if world_size > 1:
            dist.barrier()

    output = Path(args.output_dir).resolve()
    if not args.check_data and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    seed_everything(args.seed)
    checkpoint = Path(args.checkpoint).resolve()
    config = AeroVLADiffusionModel.read_checkpoint_config(checkpoint)
    history_length = config["head_config"].get("history_length", 0)
    if history_length and (config["head_config"].get("state_dim") != STATE_DIM
                           or config["training_metadata"].get("history_schema") != HISTORY_SCHEMA):
        raise ValueError("Source checkpoint UAV history schema is incompatible")
    from transformers import AutoProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False
    train_data = AeroVLADiffusionDataset(args.data_root, args.train_json, tokenizer, processor,
                                       split_json=args.split_json, split="train", history_length=history_length)
    val_data = AeroVLADiffusionDataset(args.data_root, args.train_json, tokenizer, processor,
                                     split_json=args.split_json, split="val", history_length=history_length)
    provenance = validate_checkpoint_split(config, val_data.split_info, args.train_json)
    if provenance["status"] != "verified_diffusion_head_holdout":
        raise ValueError("Source checkpoint must record a verifiable training/validation split")
    train_data = deterministic_subset(train_data, args.max_samples, args.seed)
    val_data = deterministic_subset(val_data, args.max_val_samples or None, args.eval_seed)
    collator = DiffusionDataCollator(tokenizer, max_length=args.max_length)
    if args.check_data:
        if rank == 0:
            print(json.dumps({"data_check": "passed", "source_checkpoint": str(checkpoint),
                              "history_length": history_length, "split_provenance": provenance,
                              "train_split": train_data.split_info, "val_split": val_data.split_info,
                              "batch_shapes": {name: batch_shapes(collator([dataset[0]]))
                                               for name, dataset in (("train", train_data), ("val", val_data))}}), flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=args.seed) if world_size > 1 else None
    loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collator,
                        generator=torch.Generator().manual_seed(args.seed + rank))
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collator,
                            generator=torch.Generator().manual_seed(args.eval_seed)) if rank == 0 else None
    model = AeroVLADiffusionModel.from_pretrained(
        checkpoint, base_model_path=args.model_path, adapter_path=args.adapter_path,
        torch_dtype=getattr(torch, args.backbone_dtype), device=device, tokenizer_length=len(tokenizer))
    model.enable_joint_lora(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
                            gradient_checkpointing=not args.no_gradient_checkpointing)
    groups = parameter_groups(model, lora_lr=args.lora_lr, projector_lr=args.projector_lr,
                              head_lr=args.head_lr, weight_decay=args.weight_decay)
    group_counts = {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}
    run_config = {**vars(args), "checkpoint": str(checkpoint), "output_dir": str(output),
                  "train_json": str(Path(args.train_json).resolve()), "split_json": str(Path(args.split_json).resolve()),
                  "world_size": world_size, "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                  "history_length": history_length, "head_config": model.head_config.to_dict(),
                  "joint_lora_config": model.joint_lora_config, "checkpoint_type": "joint_lora",
                  "source_global_step": config["training_metadata"].get("global_step"),
                  "source_config_sha256": hashlib.sha256((checkpoint / model.CONFIG_NAME).read_bytes()).hexdigest(),
                  "trainable_parameter_groups": group_counts,
                  "trainable_parameters": sum(group_counts.values()),
                  "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
                  "train_split": train_data.split_info, "val_split": val_data.split_info,
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "resume_semantics": "weights_only_new_optimizer_and_schedule"}
    if history_length:
        run_config["history_schema"] = HISTORY_SCHEMA
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run_config.json", run_config)
        print(json.dumps(run_config), flush=True)
    barrier()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                      broadcast_buffers=False, find_unused_parameters=True,
                                      gradient_as_bucket_view=True) if world_size > 1 else model
    # Identical initialization, independent per-rank noise streams thereafter.
    seed_everything(args.seed + rank)
    state = {"best_noise_mse": float("inf"), "last_validation": None, "last_step": 0}

    def log(metrics):
        if rank != 0:
            return
        state["last_step"] = metrics["step"]
        metrics["updated_at"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(metrics), flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        write_json(output / "status.json", {"status": "running", **metrics})

    def save_local(step, destination):
        model.save_pretrained(destination, training_metadata={**run_config, "global_step": step,
                                                              "last_validation": state["last_validation"]})
        tokenizer.save_pretrained(destination)
        processor.save_pretrained(destination)
        shutil.copyfile(args.split_json, Path(destination) / "data_split.json")

    def save(step):
        barrier()
        if rank == 0:
            save_local(step, output / f"checkpoint-{step}")
        barrier()

    def validate(step, epoch):
        barrier()
        if rank == 0:
            metrics = evaluate_diffusion(model, val_loader, device=device, seed=args.eval_seed)
            metrics.update(step=step, epoch=epoch, split="val")
            state["last_validation"] = metrics
            print("Validation: " + json.dumps(metrics), flush=True)
            with (output / "validation_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if metrics["noise_mse"] < state["best_noise_mse"]:
                state["best_noise_mse"] = metrics["noise_mse"]
                save_local(step, output / "best")
                write_json(output / "best_metrics.json", metrics)
        barrier()

    try:
        validate(0, 0)
        steps = train_joint(wrapped, loader, device=device, epochs=args.epochs,
                            lora_lr=args.lora_lr, projector_lr=args.projector_lr, head_lr=args.head_lr,
                            weight_decay=args.weight_decay, gradient_accumulation_steps=args.gradient_accumulation_steps,
                            max_grad_norm=args.max_grad_norm, max_steps=args.max_steps, warmup_ratio=args.warmup_ratio,
                            log_callback=log, validation_callback=validate, eval_steps=args.eval_steps,
                            save_callback=save, save_steps=args.save_steps)
        barrier()
        if rank == 0:
            save_local(steps, output)
            write_json(output / "status.json", {"status": "completed", "step": steps,
                                                "last_validation": state["last_validation"],
                                                "updated_at": datetime.now(timezone.utc).isoformat()})
        barrier()
    except BaseException as error:
        if rank == 0:
            write_json(output / "status.json", {"status": "failed", "step": state["last_step"],
                                                "error": repr(error), "updated_at": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
