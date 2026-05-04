"""
LM_26 — MOSART-WBM JAX Surrogate Dam Model

Storage matching:
    model, loss, utils, optimize

Storage + hydropower:
    hydro_model, hydro_loss, hydro_optimize
"""

# Storage Matching Model
from storage_model import step_biweekly_target_inflow, run_scan, init_params
from storage_loss import loss_fn, grad_loss, smooth_penalty, compute_metrics
from util import load_dam_data, to_biweek
from storage_optimizer import run_optimization, run_multistart

# Storage + Hydropower Matching Model
from hydro_model import (
    step_biweekly_target_inflow as hydro_step,
    run_scan as hydro_run_scan,
    init_params as hydro_init_params,
)
from hydro_loss import (
    loss_fn as hydro_loss_fn,
    grad_loss as hydro_grad_loss,
    evaluate_params,
    compute_hydro_targets,
)
from hydro_optimizer import (
    run_optimization as hydro_run_optimization,
    run_multistart as hydro_run_multistart,
)