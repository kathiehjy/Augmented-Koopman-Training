import os
import csv
import json
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import Subset, DataLoader


def sync_cuda_if_needed(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def perf_counter_sync(device):
    sync_cuda_if_needed(device)
    return time.perf_counter()


def build_optimizer(
    parameters,
    optimizer_name="adam",
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    **kwargs,
):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )
    if optimizer_name == "adadelta":
        return torch.optim.Adadelta(
            parameters,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_optimizer_state_keys(optimizer_name, optimizer=None):
    optimizer_name = optimizer_name.lower()

    if optimizer_name in ("adam", "adamw"):
        return ("exp_avg", "exp_avg_sq")
    if optimizer_name == "adadelta":
        return ("square_avg", "acc_delta")
    if optimizer_name == "sgd":
        if optimizer is None:
            return ("momentum_buffer",)
        has_momentum = any(group.get("momentum", 0.0) > 0.0 for group in optimizer.param_groups)
        return ("momentum_buffer",) if has_momentum else ()

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_lifted_layout(optimizer_name, optimizer=None, include_optimizer_state=True):
    if not include_optimizer_state:
        return ("weight",)
    return ("weight", *get_optimizer_state_keys(optimizer_name, optimizer))


def get_lifted_num_blocks(optimizer_name, optimizer=None, include_optimizer_state=True):
    return len(get_lifted_layout(optimizer_name, optimizer, include_optimizer_state))


# ============================================================
# Model helpers
# ============================================================

def get_parametric_layers(model: nn.Module) -> List[nn.Module]:
    """Return supported trainable layers in forward order."""
    return [
        m for m in model.modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    ]


def get_linear_layers(model: nn.Module) -> List[nn.Module]:
    """Backward-compatible alias for supported trainable layers."""
    return get_parametric_layers(model)


def get_named_parametric_layers(model: nn.Module, layer_names=None) -> List[Tuple[str, nn.Module]]:
    """Return supported trainable layers with stable layerN names."""
    named_layers = [
        (f"layer{i}", module)
        for i, module in enumerate(get_parametric_layers(model))
    ]
    if layer_names is None:
        return named_layers

    selected = set(layer_names)
    unknown = selected - {name for name, _ in named_layers}
    if unknown:
        raise ValueError(f"Unknown Koopman record layer(s): {sorted(unknown)}")

    return [
        (name, module)
        for name, module in named_layers
        if name in selected
    ]


def concat_weight_bias(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Concatenate weight and bias into augmented matrix [w | b]."""
    weight_flat = weight.reshape(weight.shape[0], -1)
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    return torch.cat([weight_flat, bias.unsqueeze(1)], dim=-1)


# ============================================================
# Lifted state construction
# ============================================================

def _optimizer_state_tensor(optimizer, param, state_key):
    state = optimizer.state.get(param, {})
    value = state.get(state_key)
    if value is None:
        return torch.zeros_like(param.detach())
    return value.detach()


def build_lifted_node_state(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_name="adam",
    include_optimizer_state=True,
) -> torch.Tensor:
    """
    Build lifted node state [w | optimizer_state...] for one Linear/Conv2d layer.

    Adam/AdamW: [w | exp_avg | exp_avg_sq]
    SGD(momentum): [w | momentum_buffer]
    SGD(no momentum): [w]
    Adadelta: [w | square_avg | acc_delta]
    """
    weight = module.weight.detach()
    bias = module.bias.detach() if module.bias is not None else None
    blocks = [concat_weight_bias(weight, bias)]

    if not include_optimizer_state:
        return blocks[0]

    for state_key in get_optimizer_state_keys(optimizer_name, optimizer):
        state_w = _optimizer_state_tensor(optimizer, module.weight, state_key)
        state_b = (
            _optimizer_state_tensor(optimizer, module.bias, state_key)
            if module.bias is not None
            else None
        )
        blocks.append(concat_weight_bias(state_w, state_b))

    return torch.cat(blocks, dim=-1)


# ============================================================
# Koopman (Torch / GPU)
# ============================================================

def fit_koopman_from_traj(x_traj, rcond: float = 1e-6) -> torch.Tensor:
    """
    Fit Koopman operator U using least squares:
        U = F' pinv(F)
    x_traj: (T, dim)

    Robust Koopman fitting (numpy or torch input).
    """
    if isinstance(x_traj, np.ndarray):
        x_traj = torch.from_numpy(x_traj)

    device = x_traj.device
    x_traj = x_traj.to(device)

    if x_traj.shape[0] < 2:
        raise ValueError("Need at least 2 time points.")

    F = x_traj[:-1].T
    Fp = x_traj[1:].T

    U = Fp @ torch.linalg.pinv(F, rcond=rcond)
    return U


def fit_koopman_batch_from_traj(
    x_traj_batch,
    rcond: float = 1e-6,
    device=None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 256,
    method: str = "pinv",
    ridge: float = 1e-5,
) -> torch.Tensor:
    """
    Fit Koopman operators for a batch of trajectories.

    x_traj_batch: (batch, T, dim)
    returns: (batch, dim, dim)
    """
    if isinstance(x_traj_batch, np.ndarray):
        x_traj_batch = torch.from_numpy(x_traj_batch)

    if x_traj_batch.ndim != 3:
        raise ValueError("Expected x_traj_batch with shape (batch, T, dim).")
    if x_traj_batch.shape[1] < 2:
        raise ValueError("Need at least 2 time points.")

    device = torch.device(device) if device is not None else x_traj_batch.device
    ops = []

    for start in range(0, x_traj_batch.shape[0], batch_size):
        batch = x_traj_batch[start:start + batch_size].to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

        if method == "pinv":
            F = batch[:, :-1, :].transpose(1, 2) # (batch, dim, T-1)
            Fp = batch[:, 1:, :].transpose(1, 2)
            ops.append(Fp @ torch.linalg.pinv(F, rcond=rcond))
        elif method == "lstsq":
            # Solve min_U ||U F - F'|| via batched least-squares:
            # F.T @ U.T = F'.T, then transpose the solution back.
            F = batch[:, :-1, :] # (batch, T-1, dim)
            Fp = batch[:, 1:, :]
            solution = torch.linalg.lstsq(F, Fp, rcond=rcond).solution
            ops.append(solution.transpose(-1, -2).contiguous())
        elif method == "ridge":
            # Ridge least-squares solves:
            #   (F.T F + lambda I) A = F.T Fp
            # with U = A.T. Lambda is scaled by each trajectory's Gram magnitude.
            F = batch[:, :-1, :] # (batch, T-1, dim)
            Fp = batch[:, 1:, :]
            Ft = F.transpose(-1, -2)
            gram = Ft @ F
            rhs = Ft @ Fp

            dim = gram.shape[-1]
            eye = torch.eye(dim, device=device, dtype=dtype).expand_as(gram)
            scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
            gram_reg = gram + ridge * scale.view(-1, 1, 1) * eye

            solution = torch.linalg.solve(gram_reg, rhs)
            ops.append(solution.transpose(-1, -2).contiguous())
        else:
            raise ValueError(f"Unknown Koopman fit method: {method}")

    return torch.cat(ops, dim=0)

def rollout_operator(U: torch.Tensor, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Open-loop rollout:
        x_{t+1} = U x_t
    """
    traj = [x0]
    x = x0
    for _ in range(steps):
        x = U @ x
        traj.append(x)
    return torch.stack(traj)


def rollout_operator_batch(U: torch.Tensor, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Batched open-loop rollout.

    U: (batch, dim, dim)
    x0: (batch, dim)
    returns: (steps + 1, batch, dim)
    """
    traj = [x0]
    x = x0
    for _ in range(steps):
        x = torch.bmm(U, x.unsqueeze(-1)).squeeze(-1)
        traj.append(x)
    return torch.stack(traj, dim=0)


# ============================================================
# Lifted state utilities (NumPy ok)
# ============================================================

def split_lifted_array(x: np.ndarray, num_blocks: int = 3):
    """Split lifted state [w | optimizer_state...] into aligned blocks."""
    lifted_dim = x.shape[-1]
    if lifted_dim % num_blocks != 0:
        raise ValueError(f"Last dimension must be divisible by {num_blocks}.")
    block_dim = lifted_dim // num_blocks
    return tuple(
        x[..., i * block_dim:(i + 1) * block_dim]
        for i in range(num_blocks)
    )


def make_aligned_lifted_chunks(x: np.ndarray, block_size: int, num_blocks: int = 3):
    """Chunk lifted state while keeping all layout blocks aligned."""
    blocks = split_lifted_array(x, num_blocks=num_blocks)
    d = blocks[0].shape[-1]

    chunks = []
    for i in range(0, d, block_size):
        j = min(i + block_size, d)
        chunks.append(np.concatenate([block[..., i:j] for block in blocks], axis=-1))

    return chunks


def make_aligned_lifted_chunks_torch(x: torch.Tensor, block_size: int, num_blocks: int = 3):
    """Torch version of make_aligned_lifted_chunks."""
    lifted_dim = x.shape[-1]
    if lifted_dim % num_blocks != 0:
        raise ValueError(f"Last dimension must be divisible by {num_blocks}.")

    d = lifted_dim // num_blocks
    blocks = [
        x[..., i * d:(i + 1) * d]
        for i in range(num_blocks)
    ]

    chunks = []
    for i in range(0, d, block_size):
        j = min(i + block_size, d)
        chunks.append(torch.cat([block[..., i:j] for block in blocks], dim=-1))

    return chunks


def reassemble_chunk_rollouts(node_chunk_rollouts, num_blocks: int = 3):
    """Reassemble chunks into full lifted state."""
    block_parts = [[] for _ in range(num_blocks)]

    for chunk in node_chunk_rollouts:
        d = chunk.shape[1] // num_blocks
        for block_idx in range(num_blocks):
            block_parts[block_idx].append(chunk[:, block_idx * d:(block_idx + 1) * d])

    return np.concatenate([
        np.concatenate(parts, axis=1)
        for parts in block_parts
    ], axis=1)


def reassemble_chunk_rollouts_torch(chunk_rollouts, num_blocks: int = 3):
    """
    Reassemble batched chunk rollouts into full lifted states.

    Each chunk has shape (..., num_blocks * chunk_dim). The returned tensor has
    shape (..., num_blocks * full_dim).
    """
    block_parts = [[] for _ in range(num_blocks)]

    for chunk in chunk_rollouts:
        d = chunk.shape[-1] // num_blocks
        for block_idx in range(num_blocks):
            block_parts[block_idx].append(chunk[..., block_idx * d:(block_idx + 1) * d])

    return torch.cat([
        torch.cat(parts, dim=-1)
        for parts in block_parts
    ], dim=-1)


# ============================================================
# Prediction → model loading
# ============================================================

def extract_predicted_weights(
    predicted: Dict[str, np.ndarray],
    num_blocks: int = 3,
) -> Dict[str, np.ndarray]:
    """Extract weight part from [w | optimizer_state...]."""
    out = {}
    for k, arr in predicted.items():
        # Backward-compatible fallback for callers that do not have model layer
        # dimensions available. Prefer load_predicted_weights_into_model.
        d = arr.shape[-1] // num_blocks
        out[k] = arr[..., :d]
    return out


def load_predicted_weights_into_model(
    model: nn.Module,
    predicted: Dict[str, np.ndarray],
    step_idx: int
):
    """Load predicted weights into model, skipping layers not in predicted."""
    parametric_layers = get_parametric_layers(model)

    with torch.no_grad():
        for i, layer in enumerate(parametric_layers):
            name = f"layer{i}"
            if name not in predicted:
                continue
            block_dim = _module_block_dim(layer)
            w_node = predicted[name][step_idx, :, :block_dim]

            weight = torch.from_numpy(w_node[:, :-1]).to(layer.weight.device).type_as(layer.weight)
            bias = torch.from_numpy(w_node[:, -1]).to(layer.weight.device).type_as(layer.weight)

            layer.weight.copy_(weight.reshape_as(layer.weight))
            if layer.bias is not None:
                layer.bias.copy_(bias)

    return model


# ============================================================
# Restore optimizer state
# ============================================================
def _optimizer_step_tensor(model, optimizer, global_step):
    group = optimizer.param_groups[0]
    step_device = (
        next(model.parameters()).device
        if group.get("capturable", False) or group.get("fused", False)
        else torch.device("cpu")
    )
    step_dtype = torch.float64 if group.get("fused", False) else torch.float32
    return torch.tensor(float(global_step), dtype=step_dtype, device=step_device)


def _split_augmented_block(block):
    return block[:, :-1], block[:, -1]


def _module_block_dim(module):
    return module.weight[0].numel() + 1


def restore_optimizer_state_from_lifted_step(
    model,
    optimizer,
    predicted_trajectories,
    step_idx,
    global_step,
    optimizer_name="adam",
    clamp_v=True,
    v_min=1e-8,
    v_max=None,
):
    """
    Restore model weights + optimizer state from predicted lifted state.
    """
    parametric_layers = get_parametric_layers(model)
    optimizer_name = optimizer_name.lower()
    state_keys = get_optimizer_state_keys(optimizer_name, optimizer)
    needs_step = optimizer_name in ("adam", "adamw", "adadelta")
    step_tensor = _optimizer_step_tensor(model, optimizer, global_step) if needs_step else None
    nonnegative_state_keys = {"exp_avg_sq", "square_avg", "acc_delta"}

    with torch.no_grad():
        for layer_idx, module in enumerate(parametric_layers):
            layer_name = f"layer{layer_idx}"
            if layer_name not in predicted_trajectories:
                continue
            x_node = predicted_trajectories[layer_name][step_idx]  # numpy
            block_dim = _module_block_dim(module)
            param_device = module.weight.device
            param_dtype = module.weight.dtype

            expected_dim = block_dim * (1 + len(state_keys))
            if x_node.shape[1] < expected_dim:
                raise ValueError(
                    f"{layer_name} lifted dimension {x_node.shape[1]} is too small for "
                    f"optimizer layout {get_lifted_layout(optimizer_name, optimizer)}."
                )

            w_block = torch.from_numpy(x_node[:, :block_dim]).to(
                device=param_device,
                dtype=param_dtype,
            )
            pred_weight, pred_bias = _split_augmented_block(w_block)

            module.weight.copy_(pred_weight.reshape_as(module.weight))
            if module.bias is not None:
                module.bias.copy_(pred_bias)

            weight_state = {}
            bias_state = {}
            if needs_step:
                weight_state["step"] = step_tensor.clone()
                bias_state["step"] = step_tensor.clone()

            for state_idx, state_key in enumerate(state_keys, start=1):
                start = state_idx * block_dim
                end = (state_idx + 1) * block_dim
                state_block = torch.from_numpy(x_node[:, start:end]).to(
                    device=param_device,
                    dtype=param_dtype,
                )
                state_weight, state_bias = _split_augmented_block(state_block)

                if clamp_v and state_key in nonnegative_state_keys:
                    state_weight = torch.clamp(state_weight, min=v_min)
                    state_bias = torch.clamp(state_bias, min=v_min)
                if v_max is not None and state_key in nonnegative_state_keys:
                    state_weight = torch.clamp(state_weight, max=v_max)
                    state_bias = torch.clamp(state_bias, max=v_max)

                weight_state[state_key] = state_weight.clone()
                bias_state[state_key] = state_bias.clone()

            optimizer.state[module.weight] = weight_state
            if module.bias is not None:
                optimizer.state[module.bias] = bias_state


def restore_adam_state_from_lifted_step(*args, **kwargs):
    kwargs.setdefault("optimizer_name", "adam")
    return restore_optimizer_state_from_lifted_step(*args, **kwargs)

# def restore_adam_state_from_lifted_step(
#     model,
#     optimizer,
#     predicted_trajectories,
#     step_idx,
#     global_step,
#     clamp_v=True,
#     v_max=None,
# ):
#     linear_layers = get_linear_layers(model)

#     with torch.no_grad():
#         for i, layer in enumerate(linear_layers):
#             name = f"fc{i}"
#             x = predicted_trajectories[name][step_idx]

#             d = x.shape[1] // 3
#             w, m, v = x[:, :d], x[:, d:2*d], x[:, 2*d:]

#             device = layer.weight.device

#             w = torch.from_numpy(w).to(device)
#             m = torch.from_numpy(m).to(device)
#             v = torch.from_numpy(v).to(device)

#             if clamp_v:
#                 v = torch.clamp(v, min=1e-8)
#             if v_max:
#                 v = torch.clamp(v, max=v_max)

#             layer.weight.copy_(w[:, :-1])
#             layer.bias.copy_(w[:, -1])

#             optimizer.state[layer.weight] = {
#                 "step": torch.tensor(global_step, dtype=torch.int64, device=device),
#                 "exp_avg": m[:, :-1].clone(),
#                 "exp_avg_sq": v[:, :-1].clone(),
#             }

#             optimizer.state[layer.bias] = {
#                 "step": torch.tensor(global_step, dtype=torch.int64, device=device),
#                 "exp_avg": m[:, -1].clone(),
#                 "exp_avg_sq": v[:, -1].clone(),
#             }


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, loader, criterion, device):
    device = torch.device(device)
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

    return loss_sum / total, 100.0 * correct / total


# ============================================================
# Training (AMP-enabled for GPU)
# ============================================================

def continue_training_adam(
    # model,
    model_class,
    train_loader,
    test_loader,
    epochs=5, # Total epochs to continue training (not counting previous training)
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    optimizer_name="adam",
    optimizer_kwargs=None,
    device="cuda",
    predicted_trajectories=None,
    step_idx=None,
    global_step=0,
    restore_optimizer_state=False,
    output_dir="./outputs",
    t1=0,
    t2=None,
    record_stride=1,
    max_records_per_layer=None,
    checkpoint_every=None,
    lifted_record_dtype=torch.float32,
    metrics_history=None,
    runtime_start=None,
    phase_name="continue",
    metrics_log_every_steps=1,
    koopman_record_layers=None,
    base_state_dict=None,
    lifted_include_optimizer_state=True,
):
    device = torch.device(device)
    use_cuda_amp = device.type == "cuda"
    os.makedirs(output_dir, exist_ok=True)

    # model = model.to(device)
    model = model_class().to(device)
    if base_state_dict is not None:
        model.load_state_dict(base_state_dict)
    criterion = nn.CrossEntropyLoss()

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

    if restore_optimizer_state:
        restore_optimizer_state_from_lifted_step(
            model,
            optimizer,
            predicted_trajectories,
            step_idx,
            global_step,
            optimizer_name=optimizer_name,
        )
    else:
        load_predicted_weights_into_model(model, predicted_trajectories, step_idx)

    recorded_layers = get_named_parametric_layers(model, koopman_record_layers)
    lifted_records = {name: [] for name, _ in recorded_layers}
    last_snapshots = {}
    current_global_step = global_step
    last_lr = optimizer.param_groups[0]["lr"]
    runtime_origin = (
        perf_counter_sync(device)
        if runtime_start is None
        else runtime_start
    )

    for epoch in range(epochs):
        epoch_start_time = perf_counter_sync(device)
        epoch_record_time = 0.0
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for step_in_epoch, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_cuda_amp, dtype=torch.bfloat16):
                logits = model(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            last_lr = optimizer.param_groups[0]["lr"]

            if t2 is not None and current_global_step == (t2 - 1):
                record_start_time = perf_counter_sync(device)
                with torch.no_grad():
                    for layer_name, module in recorded_layers:
                        x_node = build_lifted_node_state(
                            module,
                            optimizer,
                            optimizer_name,
                            include_optimizer_state=lifted_include_optimizer_state,
                        )
                        last_snapshots[layer_name] = (
                            x_node.detach()
                            .cpu()
                            .to(lifted_record_dtype)
                            .numpy()
                        )
                epoch_record_time += perf_counter_sync(device) - record_start_time

            record_flag = False
            if current_global_step >= t1:
                if t2 is None or current_global_step < t2:
                    if ((current_global_step - t1) % record_stride) == 0:
                        record_flag = True

            if record_flag:
                record_start_time = perf_counter_sync(device)
                with torch.no_grad():
                    for layer_name, module in recorded_layers:
                        x_node = build_lifted_node_state(
                            module,
                            optimizer,
                            optimizer_name,
                            include_optimizer_state=lifted_include_optimizer_state,
                        )
                        x_np = (
                            x_node.detach()
                            .cpu()
                            .to(lifted_record_dtype)
                            .numpy()
                        )

                        lifted_records[layer_name].append(x_np)

                        if max_records_per_layer is not None:
                            if len(lifted_records[layer_name]) > max_records_per_layer:
                                lifted_records[layer_name].pop(0)
                epoch_record_time += perf_counter_sync(device) - record_start_time

            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            batch_correct = (preds == yb).sum().item()
            correct += batch_correct
            total += xb.size(0)
            current_global_step += 1

            if (
                metrics_history is not None
                and metrics_log_every_steps is not None
                and metrics_log_every_steps > 0
                and (current_global_step % metrics_log_every_steps) == 0
            ):
                metrics_history.append({
                    "phase": phase_name,
                    "epoch": epoch + 1,
                    "step_in_epoch": step_in_epoch,
                    "global_step": current_global_step,
                    "runtime_sec": perf_counter_sync(device) - runtime_origin,
                    "train_loss": loss.item(),
                    "train_acc": 100.0 * batch_correct / xb.size(0),
                })

        train_loss = loss_sum / total
        train_acc = 100.0 * correct / total
        # test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        epoch_total_time = perf_counter_sync(device) - epoch_start_time
        epoch_ex_record_time = max(0.0, epoch_total_time - epoch_record_time)

        print(
            f"[Epoch {epoch+1}] "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2f}% | "
            # f"test loss {test_loss:.4f} | test acc {test_acc:.2f}% | "
            f"time {epoch_total_time:.2f}s | "
            f"record {epoch_record_time:.2f}s | "
            f"excl record {epoch_ex_record_time:.2f}s"
        )

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_continue_epoch_{epoch + 1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": current_global_step,
            }, ckpt_path)

    lifted_dict_arr = {
        k: np.stack(v, axis=0) if len(v) > 0 else np.empty((0,))
        for k, v in lifted_records.items()
    }

    print("\nSaved continued lifted state array shapes:")
    for k, arr in lifted_dict_arr.items():
        print(f"{k}: {arr.shape}")

    # save_path = os.path.join(output_dir, "lifted_trajectory_adam_continue.npz")
    # np.savez_compressed(save_path, **lifted_dict_arr)
    # print(f"\nSaved continued trajectory to: {save_path}")

    return {
        "lifted": lifted_dict_arr,
        "last_snapshots": last_snapshots,
        "final_global_step": current_global_step,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "last_lr": last_lr,
        # "best_step": step_idx,
    }


def save_training_progress_data(history, output_dir, prefix="training_progress"):
    """
    Save training metrics for later plotting.

    Each row is one completed optimizer step, indexed by cumulative global
    batch-update step and elapsed runtime.
    """
    if not history:
        raise ValueError("Cannot save training progress: history is empty.")

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    json_path = os.path.join(output_dir, f"{prefix}.json")
    plot_path = os.path.join(output_dir, f"{prefix}.png")
    fieldnames = [
        "phase",
        "epoch",
        "step_in_epoch",
        "global_step",
        "runtime_sec",
        "train_loss",
        "train_acc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"csv": csv_path, "json": json_path, "plot": None}

    steps = np.array([row["global_step"] for row in history], dtype=float)
    runtime = np.array([row["runtime_sec"] for row in history], dtype=float)
    train_loss = np.array([row["train_loss"] for row in history], dtype=float)
    phases = [row["phase"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    axes[0].plot(steps, train_loss, linewidth=1.0)
    axes[0].set_xlabel("Batch update step")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Train loss vs steps")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(runtime, train_loss, linewidth=1.0)
    axes[1].set_xlabel("Runtime (seconds)")
    axes[1].set_ylabel("Train loss")
    axes[1].set_title("Train loss vs runtime")
    axes[1].grid(True, alpha=0.3)

    seen_phase_boundaries = set()
    for idx in range(1, len(history)):
        if phases[idx] == phases[idx - 1]:
            continue
        boundary_key = (phases[idx], steps[idx], runtime[idx])
        if boundary_key in seen_phase_boundaries:
            continue
        seen_phase_boundaries.add(boundary_key)
        axes[0].axvline(steps[idx], color="0.5", linestyle="--", linewidth=0.8, alpha=0.5)
        axes[1].axvline(runtime[idx], color="0.5", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return {"csv": csv_path, "json": json_path, "plot": plot_path}

def compute_steps_per_epoch(train_loader):
    return len(train_loader)

def epoch_to_step_range(t_epoch, steps_per_epoch):
    """
    Convert epoch range to global step range.
    """
    t = t_epoch * steps_per_epoch
    return t


def evaluate_koopman_predictions(
    model_class,
    predicted_trajectories,
    ds,
    criterion,
    device,
    step_stride=2,
    val_size=2000,
    batch_size=512,
    seed=1,
    num_workers=4,
    base_state_dict=None,
):
    """
    Evaluate Koopman-predicted weights only.

    Returns:
        results dict + best_step
    """
    device = torch.device(device)

    # -------------------------
    # Validation subset
    # -------------------------
    generator = torch.Generator()
    generator.manual_seed(seed)
    val_size = min(val_size, len(ds))
    indices = torch.randperm(len(ds), generator=generator)[:val_size]
    val_subset = Subset(ds, indices)

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    # -------------------------
    # Init tracking
    # -------------------------
    best_step = None
    best_loss = float("inf")

    results = {
        "steps": [],
        "koopman_loss": [],
        "koopman_acc": [],
    }

    print(f"{'Step':<8} | {'K-Loss':<12} {'K-Acc':<10}")
    print("-" * 40)
    print(f"Koopman validation samples: {val_size} | seed: {seed}")

    # -------------------------
    # Reuse model (GPU efficient)
    # -------------------------
    model = model_class().to(device)
    if base_state_dict is not None:
        model.load_state_dict(base_state_dict)

    # -------------------------
    # Loop over prediction steps
    # -------------------------
    first_layer_name = next(iter(predicted_trajectories))
    max_steps = predicted_trajectories[first_layer_name].shape[0]
    for step in range(0, max_steps, step_stride):
        if base_state_dict is not None:
            model.load_state_dict(base_state_dict)

        # Load predicted weights
        load_predicted_weights_into_model(
            model,
            predicted_trajectories,
            step
        )

        # Evaluate
        loss, acc = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        # Track best
        if loss < best_loss:
            best_loss = loss
            best_step = step

        # Store results
        results["steps"].append(step)
        results["koopman_loss"].append(loss)
        results["koopman_acc"].append(acc)

        print(f"{step:<8} | {loss:<12.4f} {acc:<10.2f}")

    # print(f"\nBest Koopman step: {best_step} (loss={best_loss:.4f})")

    return results, best_step
