import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.model import FCNet, MNIST_MLP
from . import utils


def resolve_koopman_record_layers(value):
    if value is None or value.strip().lower() == "default":
        return None, True
    value = value.strip()
    if value.lower() in {"none", "all", ""}:
        return None, False
    return [item.strip() for item in value.split(",") if item.strip()], False


def default_output_dir(base_dir):
    return os.path.join(base_dir, "logs")


def train_and_record_lifted(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    epochs,
    output_dir,
    use_amp=True,
    t1=0,
    t2=None,
    record_stride=1,
    max_records_per_layer=None,
    checkpoint_every=5,
    lifted_record_dtype=torch.float32,
    metrics_history=None,
    runtime_start=None,
    phase_name="warmup",
    metrics_log_every_steps=1,
    optimizer_name="adam",
    koopman_record_layers=None,
    lifted_include_optimizer_state=True,
):
    """Train model and record lifted layer states for Koopman fitting."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    use_cuda_amp = use_amp and device.type == "cuda"
    model = model.to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    recorded_layers = utils.get_named_parametric_layers(model, koopman_record_layers)
    lifted_records = {name: [] for name, _ in recorded_layers}
    last_snapshots = {}
    global_step = 0
    runtime_origin = (
        utils.perf_counter_sync(device)
        if runtime_start is None
        else runtime_start
    )

    for epoch in range(1, epochs + 1):
        epoch_start_time = utils.perf_counter_sync(device)
        epoch_record_time = 0.0
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step_in_epoch, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_cuda_amp, dtype=torch.bfloat16):
                logits = model(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            batch_correct = (preds == yb).sum().item()
            correct += batch_correct
            total += xb.size(0)

            if (
                metrics_history is not None
                and metrics_log_every_steps is not None
                and metrics_log_every_steps > 0
                and ((global_step + 1) % metrics_log_every_steps) == 0
            ):
                metrics_history.append({
                    "phase": phase_name,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "global_step": global_step + 1,
                    "runtime_sec": utils.perf_counter_sync(device) - runtime_origin,
                    "train_loss": loss.item(),
                    "train_acc": 100.0 * batch_correct / xb.size(0),
                })

            if t2 is not None and global_step == (t2 - 1):
                record_start_time = utils.perf_counter_sync(device)
                with torch.no_grad():
                    for layer_name, module in recorded_layers:
                        x_node = utils.build_lifted_node_state(
                            module,
                            optimizer,
                            optimizer_name,
                            include_optimizer_state=lifted_include_optimizer_state,
                        )
                        last_snapshots[layer_name] = (
                            x_node.detach().cpu().to(lifted_record_dtype).numpy()
                        )
                epoch_record_time += utils.perf_counter_sync(device) - record_start_time

            record_flag = False
            if global_step >= t1:
                if t2 is None or global_step < t2:
                    if ((global_step - t1) % record_stride) == 0:
                        record_flag = True

            if record_flag:
                record_start_time = utils.perf_counter_sync(device)
                with torch.no_grad():
                    for layer_name, module in recorded_layers:
                        x_node = utils.build_lifted_node_state(
                            module,
                            optimizer,
                            optimizer_name,
                            include_optimizer_state=lifted_include_optimizer_state,
                        )
                        lifted_records[layer_name].append(
                            x_node.detach().cpu().to(lifted_record_dtype).numpy()
                        )

                        if max_records_per_layer is not None:
                            if len(lifted_records[layer_name]) > max_records_per_layer:
                                lifted_records[layer_name].pop(0)
                epoch_record_time += utils.perf_counter_sync(device) - record_start_time

            global_step += 1

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        epoch_total_time = utils.perf_counter_sync(device) - epoch_start_time
        epoch_ex_record_time = max(0.0, epoch_total_time - epoch_record_time)

        print(
            f"Epoch {epoch:2d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:5.2f}% | "
            f"time {epoch_total_time:.2f}s | "
            f"record {epoch_record_time:.2f}s | "
            f"excl record {epoch_ex_record_time:.2f}s"
        )

        if checkpoint_every is not None and epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }, ckpt_path)

    lifted_dict_arr = {
        k: np.stack(v, axis=0) if len(v) > 0 else np.empty((0,))
        for k, v in lifted_records.items()
    }

    print("\nSaved lifted state array shapes:")
    for k, arr in lifted_dict_arr.items():
        print(f"{k}: {arr.shape}")

    return lifted_dict_arr, last_snapshots


def fit_koopman_operators_from_lifted(
    lifted_dict_arr,
    block_size=38,
    layer_block_sizes=None,
    pinv_rcond=1e-6,
    device=None,
    fit_dtype=torch.float32,
    fit_batch_size=256,
    fit_method="pinv",
    ridge=1e-5,
    lifted_num_blocks=3,
):
    """Fit node-wise or chunked Koopman operators from recorded lifted states."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    layer_block_sizes = layer_block_sizes or {"layer0": block_size}
    koopman_operators = {}

    for layer_name, history_stack in lifted_dict_arr.items():
        if history_stack.ndim != 3 or history_stack.shape[0] < 2:
            print(f"\nSkipping Koopman for {layer_name} | insufficient shape {history_stack.shape}")
            koopman_operators[layer_name] = []
            continue

        _, num_nodes, _ = history_stack.shape
        history_t = torch.as_tensor(history_stack)
        koopman_operators[layer_name] = [[] for _ in range(num_nodes)]

        if layer_name in layer_block_sizes:
            chunk_trajs = utils.make_aligned_lifted_chunks_torch(
                history_t,
                block_size=layer_block_sizes[layer_name],
                num_blocks=lifted_num_blocks,
            )

            for chunk_idx, chunk_traj in enumerate(chunk_trajs):
                node_chunk_trajs = chunk_traj.permute(1, 0, 2).contiguous()
                U_batch = utils.fit_koopman_batch_from_traj(
                    node_chunk_trajs,
                    rcond=pinv_rcond,
                    device=device,
                    dtype=fit_dtype,
                    batch_size=fit_batch_size,
                    method=fit_method,
                    ridge=ridge,
                )

                for j, U in enumerate(U_batch.unbind(dim=0)):
                    koopman_operators[layer_name][j].append(U.detach())

        else:
            node_trajs = history_t.permute(1, 0, 2).contiguous()
            U_batch = utils.fit_koopman_batch_from_traj(
                node_trajs,
                rcond=pinv_rcond,
                device=device,
                dtype=fit_dtype,
                batch_size=fit_batch_size,
                method=fit_method,
                ridge=ridge,
            )
            koopman_operators[layer_name] = [U.detach() for U in U_batch.unbind(dim=0)]

    return koopman_operators


def rollout_koopman_predictions(
    koopman_operators,
    last_snapshots,
    T_pred,
    block_size=38,
    layer_block_sizes=None,
    device=None,
    rollout_dtype=torch.float32,
    lifted_num_blocks=3,
):
    """Roll out Koopman-predicted lifted states from the last recorded snapshot."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    layer_block_sizes = layer_block_sizes or {"layer0": block_size}
    predicted_trajectories = {}

    for layer_name, layer_last in last_snapshots.items():
        num_nodes = layer_last.shape[0]

        if layer_name in layer_block_sizes:
            layer_last_t = torch.as_tensor(layer_last, device=device, dtype=rollout_dtype)
            x0_chunks = utils.make_aligned_lifted_chunks_torch(
                layer_last_t,
                block_size=layer_block_sizes[layer_name],
                num_blocks=lifted_num_blocks,
            )
            chunk_rollouts = []

            for chunk_idx, x0_chunk in enumerate(x0_chunks):
                U_batch = torch.stack([
                    koopman_operators[layer_name][j][chunk_idx]
                    for j in range(num_nodes)
                ]).to(device=device, dtype=rollout_dtype)

                chunk_rollouts.append(
                    utils.rollout_operator_batch(U_batch, x0_chunk, T_pred)
                )

            layer_pred_arr = (
                utils.reassemble_chunk_rollouts_torch(
                    chunk_rollouts,
                    num_blocks=lifted_num_blocks,
                )
                .detach()
                .cpu()
                .numpy()
            )
        else:
            x0 = torch.as_tensor(layer_last, device=device, dtype=rollout_dtype)
            U_batch = torch.stack([
                U if isinstance(U, torch.Tensor) else torch.from_numpy(U)
                for U in koopman_operators[layer_name]
            ]).to(device=device, dtype=rollout_dtype)

            layer_pred_arr = (
                utils.rollout_operator_batch(U_batch, x0, T_pred)
                .detach()
                .cpu()
                .numpy()
            )

        predicted_trajectories[layer_name] = layer_pred_arr

    return predicted_trajectories


# ============================================================
# Configuration
# ============================================================
batch_size = 64
dataset_name = "mnist"
model_name = "mnist_mlp"
num_classes = 10
max_epochs = 20
warmup_start_record_epoch = 7
warmup_epochs = 10
continue_record_warmup_epochs = 6
continue_epochs = 9
seed = 3
record_stride = 4
progress_log_every_steps = 1
lifted_include_optimizer_state = True
validation_size = 500
safe_guard_size = 10000
T_pred = 100
evaluation_stride = 10
pinv_rcond = 1e-5
koopman_dtype = torch.float32
koopman_fit_batch_size = 256
koopman_ridge = 1e-3
koopman_fit_method ="ridge" #pinv #lstsq
koopman_block_sizes = {
    "layer0": 19,
    "layer1": 26,
}

# AdamW parameters
# optimizer_name = "adamW"
# optimizer_betas = (0.9, 0.999)
# optimizer_eps = 1e-8
# optimizer_weight_decay = 0.01
# optimizer_kwargs = {}
# initial_lr = 1e-3
# continue_lr = initial_lr * 0.25

# Adam parameters 
# optimizer_name = "adam"
# optimizer_betas = (0.9, 0.999)
# optimizer_eps = 1e-8
# optimizer_weight_decay = 0.0
# optimizer_kwargs = {}
# initial_lr = 1e-3 
# continue_lr = initial_lr * 0.25


# Adadelta parameters (ignored if not using Adadelta)
optimizer_name = "adadelta"                  
optimizer_betas = (0.9, 0.999)  
optimizer_eps = 1e-6             
optimizer_weight_decay = 0.0
optimizer_kwargs = {
    "rho": 0.9,
}
initial_lr = 1
continue_lr = initial_lr * 0.25




BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description="Train with Koopman acceleration on MNIST.")
parser.add_argument("--dataset", default=dataset_name)
parser.add_argument("--model", default=model_name)
parser.add_argument("--num-classes", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=batch_size)
parser.add_argument("--max-epochs", type=int, default=max_epochs)
parser.add_argument("--warmup-start-record-epoch", type=int, default=warmup_start_record_epoch)
parser.add_argument("--warmup-epochs", type=int, default=warmup_epochs)
parser.add_argument("--continue-record-warmup-epochs", type=int, default=continue_record_warmup_epochs)
parser.add_argument("--continue-epochs", type=int, default=continue_epochs)
parser.add_argument("--record-stride", type=int, default=record_stride)
parser.add_argument("--validation-size", type=int, default=validation_size)
parser.add_argument("--safe-guard-size", type=int, default=safe_guard_size)
parser.add_argument("--t-pred", type=int, default=T_pred)
parser.add_argument("--evaluation-stride", type=int, default=evaluation_stride)
parser.add_argument("--progress-log-every-steps", type=int, default=progress_log_every_steps)
parser.add_argument("--optimizer", default=None, choices=["adam", "adamw", "adamW", "sgd", "adadelta"])
parser.add_argument("--initial-lr", type=float, default=None)
parser.add_argument("--continue-lr", type=float, default=None)
parser.add_argument("--weight-decay", type=float, default=None)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--output-dir", default=None)
parser.add_argument("--data-dir", default=None)
parser.add_argument(
    "--koopman-record-layers",
    default="default",
    help="Comma-separated stable layer names, e.g. layer0,layer1. Use 'default' for all MNIST parametric layers.",
)
args = parser.parse_args()

dataset_name = args.dataset.lower()
if dataset_name != "mnist":
    raise ValueError("This demo is MNIST-only. Use --dataset mnist.")

model_name = args.model.lower().replace("-", "_")
if model_name not in {"mnist_mlp", "fcnet"}:
    raise ValueError("This MNIST demo supports --model mnist_mlp or --model fcnet.")

if args.num_classes is not None:
    num_classes = args.num_classes
batch_size = args.batch_size
max_epochs = args.max_epochs
warmup_start_record_epoch = args.warmup_start_record_epoch
warmup_epochs = args.warmup_epochs
continue_record_warmup_epochs = args.continue_record_warmup_epochs
continue_epochs = args.continue_epochs
record_stride = args.record_stride
validation_size = args.validation_size
safe_guard_size = args.safe_guard_size
T_pred = args.t_pred
evaluation_stride = args.evaluation_stride
progress_log_every_steps = args.progress_log_every_steps
koopman_record_layers, _ = resolve_koopman_record_layers(args.koopman_record_layers)

if args.optimizer is not None:
    optimizer_name = args.optimizer
    optimizer_kwargs = {}
if args.momentum is not None:
    optimizer_kwargs = {"momentum": args.momentum, "nesterov": False}
if args.initial_lr is not None:
    initial_lr = args.initial_lr
if args.continue_lr is not None:
    continue_lr = args.continue_lr
elif args.initial_lr is not None:
    continue_lr = initial_lr * 0.1
if args.weight_decay is not None:
    optimizer_weight_decay = args.weight_decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME = f"{dataset_name}_{model_name}_{optimizer_name}"
OUTPUT_DIR = args.output_dir or default_output_dir(BASE_DIR)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

data_dir = args.data_dir or os.path.join(BASE_DIR, "data")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
validation_base_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

if model_name == "mnist_mlp":
    model_class = lambda: MNIST_MLP(num_classes=num_classes)
else:
    model_class = FCNet

validation_generator = torch.Generator()
validation_generator.manual_seed(seed)
validation_eval_size = min(validation_size, len(train_ds))
validation_indices = torch.randperm(
    len(train_ds),
    generator=validation_generator,
)[:validation_eval_size]
validation_ds = Subset(validation_base_ds, validation_indices)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
    persistent_workers=True,
)
eval_batch_size = 1000
test_loader = DataLoader(
    test_ds,
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
    persistent_workers=True,
)
validation_loader = DataLoader(
    validation_ds,
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
    persistent_workers=True,
)

model = model_class().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = utils.build_optimizer(
    model.parameters(),
    optimizer_name=optimizer_name,
    lr=initial_lr,
    betas=optimizer_betas,
    eps=optimizer_eps,
    weight_decay=optimizer_weight_decay,
    **optimizer_kwargs,
)
lifted_num_blocks = utils.get_lifted_num_blocks(
    optimizer_name,
    optimizer,
    include_optimizer_state=lifted_include_optimizer_state,
)
lifted_layout = utils.get_lifted_layout(
    optimizer_name,
    optimizer,
    include_optimizer_state=lifted_include_optimizer_state,
)
print(f"Lifted optimizer layout: {lifted_layout}")

step_per_epoch = utils.compute_steps_per_epoch(train_loader)
warmup_run_epochs = min(warmup_epochs, max_epochs)
warmup_record_start_epoch = min(warmup_start_record_epoch, warmup_run_epochs)
warmup_t1_step = utils.epoch_to_step_range(warmup_record_start_epoch, step_per_epoch)
warmup_t2_step = utils.epoch_to_step_range(warmup_run_epochs, step_per_epoch)

training_history = []
training_start_time = utils.perf_counter_sync(device)

lifted_dict_arr, last_snapshots = train_and_record_lifted(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device=device,
    epochs=warmup_run_epochs,
    output_dir="./outputs/warmup",
    t1=warmup_t1_step,
    t2=warmup_t2_step,
    record_stride=record_stride,
    lifted_record_dtype=koopman_dtype,
    metrics_history=training_history,
    runtime_start=training_start_time,
    phase_name="warmup",
    metrics_log_every_steps=progress_log_every_steps,
    optimizer_name=optimizer_name,
    koopman_record_layers=koopman_record_layers,
    lifted_include_optimizer_state=lifted_include_optimizer_state,
)

trained_epochs = warmup_run_epochs
current_global_step = warmup_t2_step
koopman_iteration = 1
last_backprop_lr = optimizer.param_groups[0]["lr"]
current_model_state_dict = {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
}

while trained_epochs < max_epochs:
    segment_epochs = min(continue_epochs, max_epochs - trained_epochs)
    segment_t1_step = current_global_step
    segment_t2_step = current_global_step + segment_epochs * step_per_epoch
    segment_record_warmup_steps = min(
        continue_record_warmup_epochs * step_per_epoch,
        segment_epochs * step_per_epoch,
    )
    segment_record_t1_step = segment_t1_step + segment_record_warmup_steps

    koopman_accel_start_time = utils.perf_counter_sync(device)
    print(
        f"Koopman acceleration {koopman_iteration} | "
        f"last optimizer lr before acceleration {last_backprop_lr:g}"
    )

    koopman_operators = fit_koopman_operators_from_lifted(
        lifted_dict_arr,
        layer_block_sizes=koopman_block_sizes,
        pinv_rcond=pinv_rcond,
        device=device,
        fit_dtype=koopman_dtype,
        fit_batch_size=koopman_fit_batch_size,
        fit_method=koopman_fit_method,
        ridge=koopman_ridge,
        lifted_num_blocks=lifted_num_blocks,
    )

    predicted_trajectories = rollout_koopman_predictions(
        koopman_operators,
        last_snapshots,
        T_pred=T_pred,
        layer_block_sizes=koopman_block_sizes,
        device=device,
        rollout_dtype=koopman_dtype,
        lifted_num_blocks=lifted_num_blocks,
    )

    results, best_step = utils.evaluate_koopman_predictions(
        model_class=model_class,
        predicted_trajectories=predicted_trajectories,
        ds=validation_base_ds,
        criterion=criterion,
        step_stride=evaluation_stride,
        device=device,
        val_size=validation_size,
        batch_size=eval_batch_size,
        seed=seed + koopman_iteration,
        base_state_dict=current_model_state_dict,
    )

    comparison_model = model_class().to(device)
    comparison_model.load_state_dict(current_model_state_dict)
    utils.load_predicted_weights_into_model(
        comparison_model,
        predicted_trajectories,
        step_idx=best_step,
    )
    best_koopman_val_loss, best_koopman_val_acc = utils.evaluate_model(
        comparison_model,
        validation_loader,
        criterion,
        device,
    )
    print(
        f"Validation at best Koopman step | "
        f"val loss {best_koopman_val_loss:.4f}, val acc {best_koopman_val_acc:.2f}%"
    )

    koopman_accel_time = utils.perf_counter_sync(device) - koopman_accel_start_time
    print(
        f"Koopman acceleration {koopman_iteration} | "
        f"fit + rollout + best-step eval time {koopman_accel_time:.2f}s | "
        f"best step {best_step}"
    )
    training_history.append({
        "phase": f"koopman_selected_{koopman_iteration}",
        "epoch": trained_epochs,
        "step_in_epoch": 0,
        "global_step": segment_t1_step,
        "runtime_sec": utils.perf_counter_sync(device) - training_start_time,
        "train_loss": best_koopman_val_loss,
        "train_acc": best_koopman_val_acc,
    })
    # For baseline setup
    # best_step = T_pred 
    continued_records = utils.continue_training_adam(
        model_class=model_class,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=segment_epochs,
        lr=continue_lr,
        betas=optimizer_betas,
        eps=optimizer_eps,
        weight_decay=optimizer_weight_decay,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
        predicted_trajectories=predicted_trajectories,
        step_idx=best_step,
        global_step=segment_t1_step,
        restore_optimizer_state=False,
        output_dir=f"./outputs/iter_{koopman_iteration:03d}",
        t1=segment_record_t1_step,
        t2=segment_t2_step,
        record_stride=record_stride,
        lifted_record_dtype=koopman_dtype,
        metrics_history=training_history,
        runtime_start=training_start_time,
        phase_name=f"koopman_iter_{koopman_iteration}",
        metrics_log_every_steps=progress_log_every_steps,
        koopman_record_layers=koopman_record_layers,
        base_state_dict=current_model_state_dict,
        lifted_include_optimizer_state=lifted_include_optimizer_state,
    )

    lifted_dict_arr = continued_records["lifted"]
    last_snapshots = continued_records["last_snapshots"]
    current_global_step = continued_records["final_global_step"]
    current_model_state_dict = continued_records["model_state_dict"]
    last_backprop_lr = continued_records["last_lr"]
    trained_epochs += segment_epochs
    koopman_iteration += 1

print(f"\nFinished Koopman-accelerated training at {trained_epochs} epochs.")
progress_paths = utils.save_training_progress_data(training_history, OUTPUT_DIR)
print(f"Saved training progress CSV to: {progress_paths['csv']}")
print(f"Saved training progress JSON to: {progress_paths['json']}")
if progress_paths["plot"] is not None:
    print(f"Saved training progress plot to: {progress_paths['plot']}")
else:
    print("Skipped training progress plot because matplotlib is not installed.")
