# Score Based Generative Modeling (SBGM) via SDEs

This repository implements **score–based diffusion models** using the
**stochastic differential equation (SDE) formulation** described in
Song et al. ([2021]【615325094571085†screenshot】). It supports both 2D image data (e.g. MNIST) and 1D time
series with a flexible, well–tested codebase that runs on **CPU**,
**CUDA** and Apple Silicon **MPS** devices. The goal is to provide a
complete end–to–end pipeline covering SDE families, score networks,
training, sampling, visualisation and packaging.

## Overview

In score–based generative models we define a continuous–time forward
process that gradually perturbs data with increasing noise according
to an SDE

\[\mathrm{d}x = f(x,t)\,\mathrm{d}t + g(t)\,\mathrm{d}W_t,\]

where \(f\) is the drift, \(g\) the diffusion coefficient and \(W_t\)
a Wiener process. By training a neural network \(s_\theta(x,t)\) to
approximate the *score* \(\nabla_x \log p_t(x)\) of the perturbed
distribution, one can simulate a reverse–time SDE to generate novel
data. Three SDE families are implemented:

* **VE–SDE (variance exploding)**: \(f=0\), \(g(t)=\sigma(t)\sqrt{2\log\tfrac{\sigma_\text{max}}{\sigma_\text{min}}}\) with
  \(\sigma(t)=\sigma_\text{min}(\sigma_\text{max}/\sigma_\text{min})^t\). Noise variance grows exponentially.
* **VP–SDE (variance preserving)**: \(f=-\tfrac12 \beta(t)x\), \(g(t)=\sqrt{\beta(t)}\) with
  \(\beta(t)\) a linear or cosine schedule. The variance of \(x(t)\)
  remains roughly constant.
* **subVP–SDE**: identical drift to VP but diffusion
  \(g(t)=\sqrt{\beta(t)\tfrac{1-\alpha_{\bar t}(t)}{\alpha_{\bar t}(t)}}\) which better matches the discrete
  diffusion process used in DDPMs.

The reverse–time dynamics are simulated using one of three
algorithms:

| Method                | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| **Euler–Maruyama (EM)** | Predictor only; simplest discretisation of the reverse SDE.    |
| **Predictor–Corrector (PC)** | Adds Langevin corrector steps at each time step to refine samples |
| **Probability flow ODE** | Solves a deterministic ODE equivalent to the reverse SDE; uses a fixed–step RK4 solver. |

For a deeper mathematical treatment see Song et al. (2021) and
Nichol & Dhariwal (2021)【615325094571085†screenshot】.

## Installation

The project uses a standard Python package layout. Clone the
repository and install the dependencies in a virtual environment:

```bash
git clone <this repository> sbgm
cd sbgm
pip install -e .
```

The minimal dependencies are listed in `requirements.txt` (PyTorch,
torchvision, numpy, matplotlib, tqdm, pyyaml). Optional extras
include `tensorboard` for logging and `torchdiffeq` for adaptive ODE
solvers.

## Quick Start

Example commands for training and sampling are provided in `scripts/`.
On CPU these run within a few minutes using the provided configs:

```bash
# VE‑SDE on MNIST
python -m sbgm.cli.main --config configs/mnist_ve.yaml --train

# VP‑SDE on MNIST
python -m sbgm.cli.main --config configs/mnist_vp.yaml --train

# VE‑SDE on synthetic 1D data
python -m sbgm.cli.main --config configs/ts1d_ve.yaml --train
```

The training script logs progress with `tqdm` and writes optional
TensorBoard files if `tensorboard` is installed and `tensorboard: true`
is set in the config. Upon completion it produces a set of samples
saved as PNGs in `outputs/<date_run>/`. Loss curves are also plotted.

To sample from a saved model without retraining use the `--sample`
flag and specify the output directory in the config:

```bash
python -m sbgm.cli.main --config configs/mnist_ve.yaml --sample
```

## Configuration

All experiment parameters reside in YAML files under `configs/`.
Nested dictionaries group settings for the dataset, model, SDE,
training and sampler. Command line flags override YAML entries using
dotted keys; for example to change the learning rate on the fly:

```bash
python -m sbgm.cli.main --config configs/mnist_ve.yaml --train \
    --training.lr 1e-4 --training.epochs 2 --model.base_channels 32
```

The configuration system uses dataclasses internally and supports
arbitrary nesting. See the example configs for common options.

## Package Structure

```
sbgm/
  config.py           # YAML parsing and CLI overrides
  utils/              # Device, EMA, logging, plotting, schedulers, seeding
  data/               # MNIST loader, CSV loader, synthetic generator
  models/             # Building blocks, UNet1D and UNet2D definitions
  sde/                # Base SDE and VE/VP/subVP implementations
  training/           # Losses and training loops for images and time series
  sampling/           # EM, predictor–corrector and ODE samplers
  cli/                # Command line entry point
configs/              # Ready‑to‑use YAML configs
scripts/              # Bash scripts wrapping common runs
tests/                # Unit tests covering models, SDEs and samplers
examples/             # Tiny CSV for time series
```

## Citations

Please cite the following works if you use this code:

* **Score-Based Generative Modeling through SDEs** – Yang Song et al., 2021【615325094571085†screenshot】.
* **Improved Denoising Diffusion Probabilistic Models** – Alex Nichol & Prafulla Dhariwal, 2021【615325094571085†screenshot】.

These references introduce the continuous–time formulation of
diffusion models and the cosine noise schedule.
