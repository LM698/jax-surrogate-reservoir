# Differentiable Reservoir Operation Modeling with JAX

This repository contains the code and notebooks for a Master of Engineering project focused on developing a differentiable, data-driven framework for reservoir operation modeling. The project uses JAX-based simulation and gradient-based optimization to learn reservoir release policies from inflow and storage data, with an extension to hydropower-related objectives.

## Project Overview

Reservoir operation models are essential for representing human water management within hydrologic systems. Traditional rule-based approaches are useful, but they often lack flexibility and can be difficult to calibrate for different reservoirs or operating conditions.

This project develops a differentiable surrogate reservoir operation model that integrates physical constraints with data-driven optimization. The model uses outputs from `mosartwmpy` as reference trajectories and learns release policies based on storage, inflow, and seasonal variation. The goal is to reproduce storage dynamics while keeping the learned operating policy interpretable.

The project includes two main components:

1. **Storage matching model**  
   A JAX-based reservoir model is calibrated to match reference storage trajectories.

2. **Hydropower extension**  
   A simplified hydropower formulation is added to evaluate trade-offs between storage accuracy and hydropower performance.

## Main Results

The framework is first evaluated on Moore Dam. The storage-only model achieves strong agreement with the reference storage trajectory, with a root mean square error (RMSE) of 1.98 million cubic meters (MCM) and a Nash-Sutcliffe efficiency (NSE) of 0.995.

The learned parameters show stable and physically interpretable seasonal patterns, suggesting that the model captures meaningful operating behavior rather than simply fitting the data numerically.

The model is then extended to include hydropower generation as an additional objective. Results show a clear trade-off between storage accuracy and hydropower performance. Improved seasonal generation patterns can be achieved, but this comes at the cost of increased storage error. Gains in annual energy production remain limited. Similar trade-off behavior is observed across multiple reservoirs, including Comerford, Bellows Falls, and Wilder.

## Repository Structure

```text
MEng_Project_LM698/
├── README.md
├── environment.yml
├── .gitignore
└── LM_26/
    ├── storage_model.py              # Differentiable reservoir storage simulation model
    ├── storage_loss.py               # Storage-matching loss function and performance metrics
    ├── storage_optimizer.py          # Optimization routines for storage-only experiments
    ├── hydro_model.py                # Reservoir model with hydropower extension
    ├── hydro_loss.py                 # Multi-objective loss including hydropower terms
    ├── hydro_optimizer.py            # Optimization routines for hydropower experiments
    ├── util.py                       # Data loading and time-index helper functions
    ├── input_data_Moore/             # Processed Moore Dam input data
    ├── experiment_inputs/            # Processed input data for other dams
    └── notebooks/                    # Experimental notebooks
```

## Model Description

The reservoir model follows a daily mass-balance structure:

```text
Storage(t+1) = Storage(t) + Inflow(t) - Release(t)
```

The release policy is parameterized using seasonal and state-dependent terms, including:

- baseline seasonal release
- target storage
- storage feedback
- inflow sensitivity

Physical constraints are included to prevent storage from exceeding reservoir capacity and to avoid releasing more water than is available. The model is implemented in JAX so that the simulation can be optimized using automatic differentiation.

## Hydropower Extension

The hydropower extension estimates power generation using a simplified physical relationship between turbine flow, hydraulic head, and efficiency:

```text
Power = density × gravity × hydraulic head × turbine flow × efficiency
```

This extension allows the model to evaluate multi-objective trade-offs between storage matching and hydropower-related performance.

## Environment Setup

This project was developed using Python 3.11 with Miniforge/conda.

To create the environment:

```bash
conda env create -f environment.yml
conda activate lm26_jax
```

If using VS Code notebooks, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name lm26_jax --display-name "Python (lm26_jax)"
```

Then select `Python (lm26_jax)` as the interpreter or notebook kernel in VS Code.

## Data

The model uses processed reservoir inflow and storage time series extracted from `mosartwmpy` tutorial outputs. Large raw NetCDF files are not included in this repository.

Processed input files should be placed under:

```text
LM_26/input_data_Moore/
LM_26/experiment_inputs/
```


## Running the Notebooks

The notebooks are designed to be run from the project root directory. In VS Code, it is recommended to open the full project folder rather than only the `LM_26` subfolder.

Example:

```bash
cd "/path/to/MEng_Project_LM698"
code .
```

Each notebook searches for the `LM_26` folder and adds the project root to the Python path so that local modules can be imported correctly.

## Notes

The notebooks are organized as experimental workflows rather than a fully packaged software library. The main purpose of this repository is to document the model formulation, optimization experiments, and project workflow for reproducibility.

## Author

Liying Ma  
Master of Engineering Project  
Department of Biological and Environmental Engineering
Cornell University
