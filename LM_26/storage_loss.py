"""
Loss functions for storage-matching optimization.

Total loss = RMSE(S_sim, S_obs) + lambda * smoothness_regularization

Smoothness regularization penalizes abrupt changes between adjacent
bi-weekly parameter values, encouraging physically plausible seasonal curves.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad

from storage_model import run_scan


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------

def smooth_penalty(x):
    """
    Mean squared difference between adjacent bi-weekly parameter values.
    Encourages smooth seasonal parameter curves.

    - Parameters -
    x: jnp.ndarray, shape (26,)
        Raw bi-weekly parameter vector.

   - Returns -
    float
        Smoothness penalty (lower = smoother).
    """
    return jnp.mean((x[1:] - x[:-1]) ** 2)


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------

@jit
def loss_fn(params, inflow_data, storage_obs_data, biweek_data, Smax,
            lam_smooth=0.005, kq_scale=2.0):
    """
    Total loss: RMSE + weighted smoothness regularization.

    Uses a one-step-ahead formulation [inflow(t) predicts storage(t+1)]:
        - Input: inflow_data[:-1] (days 0 to T-2)
        - Target: storage_obs_data[1:] (days 1 to T-1)

    Regularization weights:
        Q_base, S_target : 1.0 strongly smoothed — seasonal baselines)
        kS: 0.5 (moderately smoothed)
        kQ: 0.2 (loosely smoothed — can respond faster)

    - Parameters -
    params: jnp.ndarray, shape (104,)
        Model parameters.
    inflow_data: jnp.ndarray, shape (T,)
        Daily inflow (MCM/day).
    storage_obs_data: jnp.ndarray, shape (T,)
        Observed storage from mosartwmpy (MCM).
    biweek_data: jnp.ndarray, shape (T,), dtype int
        Bi-week index for each day (1..26).
    Smax: float
        Maximum reservoir capacity (MCM).
    lam_smooth: float
        Regularization strength (default 0.005).
    kq_scale: float
        Inflow sensitivity bound.

    - Returns -
    float
        Total loss value.
    """
    S0_local = storage_obs_data[0]

    # One-step-ahead simulation
    S_sim, _ = run_scan(
        inflow_data[:-1], S0_local, biweek_data[:-1], params, Smax, kq_scale
    )

    # RMSE against observed storage
    rmse = jnp.sqrt(jnp.mean((S_sim - storage_obs_data[1:]) ** 2))

    # Smoothness regularization on raw parameter vectors
    qbase_raw = params[0:26]
    starget_raw = params[26:52]
    ks_raw = params[52:78]
    kq_raw = params[78:104]

    reg = (
          1.0 * smooth_penalty(qbase_raw)
        + 1.0 * smooth_penalty(starget_raw)
        + 0.5 * smooth_penalty(ks_raw)
        + 0.2 * smooth_penalty(kq_raw)
    )

    return rmse + lam_smooth * reg


# ---------------------------------------------------------------------------
# Gradient (pre-compiled)
# ---------------------------------------------------------------------------

grad_loss = jit(grad(loss_fn))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(S_sim, S_obs):
    """
    Compute standard hydrology evaluation metrics.

    - Parameters -
    S_sim: array-like, shape (T,)
        Simulated storage.
    S_obs: array-like, shape (T,)
        Observed storage.

    - Returns -
    dict with keys: rmse, nse, bias_pct
        rmse: Root Mean Square Error (MCM)
        nse: Nash-Sutcliffe Efficiency [-inf, 1], 1 = perfect
        bias_pct: Mean bias as percentage of observed mean
    """
    S_sim = jnp.array(S_sim)
    S_obs = jnp.array(S_obs)

    rmse = float(jnp.sqrt(jnp.mean((S_sim - S_obs) ** 2)))

    ss_res = jnp.sum((S_sim - S_obs) ** 2)
    ss_tot = jnp.sum((S_obs - jnp.mean(S_obs)) ** 2)
    nse = float(1.0 - ss_res / ss_tot)

    bias_pct = float(jnp.mean(S_sim - S_obs) / jnp.mean(S_obs) * 100.0)

    return {"rmse": rmse, "nse": nse, "bias_pct": bias_pct}