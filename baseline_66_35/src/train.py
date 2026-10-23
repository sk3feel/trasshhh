"""Explicit training/resume CLI. --check-resume restores state without any update."""

import argparse
import copy
import datetime
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .data import LengthBuckets, TokenDataset, train_loader
from .evaluate import validation_loss
from .generate import sha256_file
from .model import configure_runtime, create_model, load_resume

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def restore(model, optimizer, scheduler, scaler, checkpoint, fingerprint):
    if checkpoint["dataset_fingerprint"] != fingerprint:
        raise ValueError(
            "Dataset changed: explicitly reset sampler in a separate experiment"
        )
    if checkpoint["scheduler_policy"] != "constant":
        raise ValueError(
            "Only the selected baseline constant scheduler may be resumed here"
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    return (
        copy.deepcopy(checkpoint["state"]),
        np.asarray(checkpoint["seen_indices"]).copy(),
    )


def save(path, model, optimizer, scheduler, scaler, config, state, seen, fingerprint):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scheduler_policy": "constant",
        "scaler": scaler.state_dict(),
        "config": config,
        "state": copy.deepcopy(state),
        "seen_indices": seen,
        "dataset_fingerprint": fingerprint,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "tokenizer_sha256": sha256_file(
            Path(config["paths"]["bundle"]) / "tokenizer.json"
        ),
        "validation_sha256": sha256_file(
            Path(config["paths"]["validation"]) / "validation.parquet"
        ),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args):
    configure_runtime()
    config = json.loads(args.config.read_text())
    settings = config["training"]
    dataset = TokenDataset(config["paths"]["train"])
    expected_tokenizer = sha256_file(Path(config["paths"]["bundle"]) / "tokenizer.json")
    if (
        sha256_file(Path(config["paths"]["train"]) / "tokenizer_8192.json")
        != expected_tokenizer
    ):
        raise ValueError(
            "Training cache tokenizer differs from the configured frozen tokenizer"
        )
    fingerprint = sha256_file(Path(config["paths"]["train"]) / "manifest.json")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = create_model(config["model"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=settings["weight_decay"],
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = torch.amp.GradScaler("cuda")
    state = {
        "global_step": 0,
        "global_optimizer_updates": 0,
        "global_examples": 0,
        "global_source_tokens": 0,
        "global_target_tokens": 0,
        "global_skipped_updates": 0,
        "sampler_epoch": 0,
        "batch_in_epoch": 0,
    }
    seen = np.zeros(len(dataset), dtype=bool)
    if args.resume:
        checkpoint = load_resume(args.resume)
        expected_tokenizer = sha256_file(
            Path(config["paths"]["bundle"]) / "tokenizer.json"
        )
        expected_validation = sha256_file(
            Path(config["paths"]["validation"]) / "validation.parquet"
        )
        assert checkpoint["tokenizer_sha256"] == expected_tokenizer
        assert checkpoint["validation_sha256"] == expected_validation
        state, seen = restore(
            model, optimizer, scheduler, scaler, checkpoint, fingerprint
        )
        assert len(seen) == len(dataset)
        assert state["batch_in_epoch"] < len(
            LengthBuckets(dataset, settings["batch_size"], config["seed"])
        )
        assert [group["lr"] for group in optimizer.param_groups] == [settings["lr"]]
        if args.check_resume:
            assert scheduler.state_dict() == checkpoint["scheduler"]
            assert scaler.state_dict() == checkpoint["scaler"]
            for parameter, expected in zip(
                optimizer.state.values(), checkpoint["optimizer"]["state"].values()
            ):
                for name in ["exp_avg", "exp_avg_sq", "step"]:
                    assert torch.equal(parameter[name].cpu(), expected[name].cpu())
            report = {
                "optimizer_states_identical": len(optimizer.state),
                "scheduler_identical": True,
                "scaler_identical": True,
                "state": state,
                "lr": optimizer.param_groups[0]["lr"],
                "sampler_position_preserved": True,
                "rng_restored": True,
                "optimizer_updates_performed": 0,
                "baseline_file_modified": False,
            }
            print(json.dumps(report, indent=2))
            if args.audit_output:
                args.audit_output.write_text(json.dumps(report, indent=2))
            return
        del checkpoint
    elif args.check_resume:
        raise ValueError("--check-resume requires --resume")
    if args.output is None or args.seconds is None:
        raise ValueError("Training requires explicit --output and --seconds")
    baseline = Path(config["paths"]["bundle"]).resolve()
    if args.output.resolve() == baseline or baseline in args.output.resolve().parents:
        raise ValueError("The frozen baseline directory is never a training output")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    deadline = time.time() + args.seconds
    best_ce = float("inf")
    next_evaluation = time.time() + 600
    with (args.output / "metrics.jsonl").open("w", buffering=1) as log:
        while time.time() < deadline and not STOP:
            loader = train_loader(
                dataset, settings["batch_size"], config["seed"], state["sampler_epoch"]
            )
            for batch_index, batch in enumerate(loader):
                if batch_index < state["batch_in_epoch"]:
                    continue
                # Check wall clock directly, including after suspension/resume of the host.
                if time.time() >= deadline or STOP:
                    break
                indices = batch.pop("_indices")
                seen[indices] = True
                source_tokens = int(batch["attention_mask"].sum())
                target_tokens = int(batch["labels"].ne(-100).sum())
                batch = {
                    key: tensor.cuda(non_blocking=True) for key, tensor in batch.items()
                }
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch, use_cache=False).loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["gradient_clip"]
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < previous_scale
                if not skipped:
                    scheduler.step()
                state["global_step"] += 1
                state["global_optimizer_updates"] += int(not skipped)
                state["global_skipped_updates"] += int(skipped)
                state["global_examples"] += len(indices)
                state["global_source_tokens"] += source_tokens
                state["global_target_tokens"] += target_tokens
                state["batch_in_epoch"] = batch_index + 1
                if state["global_step"] % 100 == 0:
                    log.write(
                        json.dumps(
                            {
                                **state,
                                "loss": float(loss),
                                "lr": optimizer.param_groups[0]["lr"],
                            }
                        )
                        + "\n"
                    )
                if time.time() >= next_evaluation:
                    ce = validation_loss(model, config["paths"]["validation"])
                    save(
                        args.output / "last.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        state,
                        seen,
                        fingerprint,
                    )
                    if ce["validation_ce"] < best_ce:
                        best_ce = ce["validation_ce"]
                        save(
                            args.output / "best_ce.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            config,
                            state,
                            seen,
                            fingerprint,
                        )
                    log.write(json.dumps(ce) + "\n")
                    next_evaluation = time.time() + 600
            if state["batch_in_epoch"] >= len(loader):
                state["sampler_epoch"] += 1
                state["batch_in_epoch"] = 0
        save(
            args.output / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            state,
            seen,
            fingerprint,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-resume", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
