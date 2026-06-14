# PINN-Beam-Benchmarks

> **Reproducibility artifact for the paper:**
> *"A Topical Review of Machine Learning and Physics-Informed Neural Networks for Beam Analysis and Control."*

This repository contains the complete suite of Physics-Informed Neural Network (PINN) benchmark codes discussed in the paper. We provide these to ensure full reproducibility of the quantitative comparisons, architecture sweeps, and advanced case studies, and to serve as a baseline for future research.

## Benchmark Suite

### 1. Classical Beam Theories
- **`ebb_pinn.py`**: Euler–Bernoulli free-vibration benchmark (`w_xxxx + w_tt = 0`). Reproduces the cost comparisons (FEM vs PINN), architecture sweeps, and optimizer scaling sweeps (Adam vs. L-BFGS) shown in **Tables 14–16**.
- **`timoshenko_pinn.py`**: Timoshenko beam formulation. Models two coupled fields `(w, φ)` and demonstrates the handling of shear deformation and rotary inertia.

### 2. Advanced Case Studies
- **`heterogeneous_pinn.py`**: Functionally graded beam benchmark. Demonstrates how spatially varying parameters $E(x)I(x)$ and $m(x)$ are handled pointwise in the residual without remeshing.
- **`nonlinear_pinn.py`**: Geometrically nonlinear (von Kármán) beam. Implements the integro-differential stretching-bending coupling and amplitude curriculum learning to reproduce the hardening backbone curve.
- **`elasticity_2d_pinn.py`**: Full 2D plane-stress linear-elasticity solver. Demonstrates the modeling error of 1D beam reductions as a function of aspect ratio $L/h$.

## Environment Setup

The code is lightweight and relies primarily on PyTorch.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Benchmarks

All scripts support command-line arguments to reproduce the paper's tables. For example, to reproduce the Euler-Bernoulli architecture sweep:
```bash
python ebb_pinn.py --depth 2 --width 64
python ebb_pinn.py --depth 4 --width 128
python ebb_pinn.py --depth 4 --width 200      # reference configuration
python ebb_pinn.py --depth 6 --width 200
```
To test the impact of Adam vs. L-BFGS, omit the L-BFGS stage:
```bash
python ebb_pinn.py --lbfgs_iters 0
```

## Notes for Researchers
- **Hardware Variation**: Absolute wall-clock training and inference times will vary depending on your hardware (CPU vs. GPU). The *ratios* reported in the paper (e.g., PINN vs. FEM inference time) are the primary reproducible metrics.
- **Random Seeds**: PINN training can be sensitive to initialization. Use the `--seed` flag to average results across multiple runs as done in the paper.

## Citation
If you use these benchmark codes or find the review paper helpful, please consider citing:
```bibtex
@article{imran2026topical,
  title={A Topical Review of Machine Learning and Physics-Informed Neural Networks for Beam Analysis and Control},
  author={Imran and Khan, Suliman and Meligy, Rowida and Nouioua, Mourad},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026}
}
```
