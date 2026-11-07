# Accelerating Neural Network Training with Augmented Koopman Dynamics

This repository contains a demo implementation for accelerating neural network training with augmented Koopman dynamics on the MNIST dataset.


## Contents

```text
src/
  train.py      Main MNIST training and Koopman acceleration script
  model.py      MNIST model definitions
  utils.py      Optimizer, lifted-state, Koopman, evaluation, and logging utilities

scripts/
  visualization.py      Plot training loss against runtime and optimization step
  visualization_acc.py  Plot accuracy against runtime and optimization step
```

## Requirements

Install the Python dependencies in a fresh environment:

```bash
pip install torch torchvision numpy matplotlib
```

## Quick Start

From the repository root:

```bash
python -m src.train
```

By default, this command trains the MNIST demo model and writes progress files to:

```text
logs/training_progress.csv
logs/training_progress.json
logs/training_progress.png
```

The CSV and JSON files contain step-level training metrics. The PNG file plots training loss against update steps and runtime.

## Useful Options

Supported models:

```text
mnist_mlp
fcnet
```

Supported optimizer:

```text
adam
adamw
adadelta
```

The baseline setting used for comparison can be configured in `src/train.py`. For example:

```text
record_stride = 1
lifted_include_optimizer_state = False
koopman_fit_method ='pinv'
best_step = T_pred 
```
