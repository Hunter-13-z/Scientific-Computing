# Scientific Computing

This repository contains assignment for Scientific Computing

## Files
```
Scientific-Computing/
├── Assignment2/
│   │
│   ├── outputs/                        # Saved simulation outputs from Sections 2.2 and 2.3 used in the report
│   │   │
│   │   ├── f0p025_k0p051/              # Gray–Scott results for (f,k) = (0.025, 0.051), chaotic
│   │   ├── f0p025_k0p052/              # Gray–Scott results for (f,k) = (0.025, 0.052), oscillating
│   │   ├── f0p030_k0p060/              # Gray–Scott results for (f,k) = (0.030, 0.060), dissipative
│   │   ├── f0p035_k0p058/              # Gray–Scott results for (f,k) = (0.035, 0.058), steady theta pattern
│   │   ├── f0p035_k0p060/              # Gray–Scott results for (f,k) = (0.035, 0.060), steady kappa pattern
│   │   ├── f0p050_k0p060/              # Gray–Scott results for (f,k) = (0.050, 0.060), steady iota pattern
│   │
│   │   ├── cluster_ps_0.05.npy         # DLA cluster array for ps = 0.05
│   │   ├── cluster_ps_0.05_steps_1000.png  # Visualization of cluster for ps = 0.05
│   │   ├── cluster_ps_0.2.npy          # DLA cluster array for ps = 0.2
│   │   ├── cluster_ps_0.2_steps_958.png    # Visualization of cluster for ps = 0.2
│   │   ├── cluster_ps_0.5.npy          # DLA cluster array for ps = 0.5
│   │   ├── cluster_ps_0.5_steps_816.png    # Visualization of cluster for ps = 0.5
│   │   ├── cluster_ps_1.0.npy          # DLA cluster array for ps = 1
│   │   ├── cluster_ps_1.0_steps_856.png    # Visualization of cluster for ps = 1
│   │
│   ├── 2.1_DLA.ipynb                   # Numerical solution for DLA growth (Task 2.1)
│   ├── 2.2_monte_carlo_DLA.ipynb       # Monte Carlo simulation of Diffusion Limited Aggregation (Task 2.2)
│   ├── 2.3_Gray_Scott_model.ipynb      # Gray–Scott reaction–diffusion model and pattern visualization (Task 2.3)
│   ├── requirements.txt                # Dependencies list for Python installation
├── .gitignore                          # Files ignored by Git
├── README.md                           # Project description and usage information
```



## Environment

- Python 3.13
- Jupyter Notebook / JupyterLab

Required libraries may include:

- numpy
- numba
- matplotlib
