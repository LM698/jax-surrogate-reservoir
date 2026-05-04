"""
Multi-objective loss for joint storage + hydropower matching.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad

from hydro_model import run_scan


def smooth_penalty(x):
    """Mean squared difference between adjacent bi-weekly parameter values."""
    return jnp.mean((x[1:] - x[:-1]) ** 2)


def compute_hydro_targets(P_sim, month_id_jax, n_months=12):
    """
    Aggregate daily power (MW) into monthly energy fractions and annual total.

    - Parameters -
    P_sim : jnp.ndarray, shape (T,)
        Daily average power (MW).
    month_id_jax : jnp.ndarray, shape (T,), dtype int
        0-based month index for each day (0=Jan .. 11=Dec).
    n_months : int

    - Returns -
    monthly_gen : jnp.ndarray, shape (12,)
        Monthly energy proxy (sum of daily mean MW values).
    sim_frac_12 : jnp.ndarray, shape (12,)
        Fraction of annual energy in each month.
    annual_sim_mwh : float
        Annual energy (MWh), assuming 24 hours per day.
    """
    monthly_gen = jnp.zeros(n_months).at[month_id_jax].add(P_sim)
    sim_frac_12 = monthly_gen / (jnp.sum(monthly_gen) + 1e-12)
    annual_sim_mwh = jnp.sum(P_sim) * 24.0
    return monthly_gen, sim_frac_12, annual_sim_mwh


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

@jit
def loss_fn(params,
            inflow_data, storage_obs_data, biweek_data, Smax,
            month_id_jax, obs_frac_12_jax, annual_obs_mwh, sigma_S,
            H_m, P_cap, eta=0.8,
            lam_storage=1.0,
            lam_hydro_shape=1.0, lam_hydro_annual=0.5,
            lam_smooth=0.005, kq_scale=2.0):
    """
    Multi-objective loss: storage matching + hydropower shape + annual energy.

    Total loss:
        L = lam_storage * storage_loss_norm
          + lam_hydro_shape * hydro_shape_loss
          + lam_hydro_annual * hydro_annual_loss
          + lam_smooth * smoothness_reg

    Notes:
    - storage_loss_norm is normalized by sigma_S
    - hydro_shape_loss is a relative MSE on monthly fractions
    - hydro_annual_loss is a relative squared error on annual energy
    """
    S_sim, _, P_sim = run_scan(
        inflow_data[:-1], storage_obs_data[0], biweek_data[:-1],
        params, Smax, H_m, P_cap, eta, kq_scale
    )

    # Storage: normalized MSE
    mse_storage = jnp.mean(((S_sim - storage_obs_data[1:]) / (sigma_S + 1e-12)) ** 2)

    # Hydropower shape: relative MSE on monthly fractions
    _, sim_frac_12, annual_sim_mwh = compute_hydro_targets(P_sim, month_id_jax)
    mse_shape = jnp.mean(((sim_frac_12 - obs_frac_12_jax) / (obs_frac_12_jax + 1e-12)) ** 2)

    # Hydropower annual: relative squared error
    mse_annual = ((annual_sim_mwh - annual_obs_mwh) / (annual_obs_mwh + 1e-12)) ** 2

    # Smoothness on Q_base and S_target only
    reg = smooth_penalty(params[0:26]) + smooth_penalty(params[26:52])

    return (
        lam_storage * mse_storage
        + lam_hydro_shape * mse_shape
        + lam_hydro_annual * mse_annual
        + lam_smooth * reg
    )


grad_loss = jit(grad(loss_fn))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_params(params,
                    inflow_data, storage_obs_data, biweek_data, Smax,
                    month_id_jax, obs_frac_12_jax, annual_obs_mwh, sigma_S,
                    H_m, P_cap, eta=0.8,
                    lam_storage=1.0,
                    lam_hydro_shape=1.0, lam_hydro_annual=0.5,
                    lam_smooth=0.005, kq_scale=2.0):
    """
    Compute full diagnostics for a given parameter set.

    - Returns -
    dict with keys including:
        storage_rmse, storage_loss_norm,
        hydro_shape_loss, hydro_annual_loss,
        annual_sim_mwh, annual_sim_gwh,
        bias_mwh, bias_percent,
        smooth_penalty,
        weighted_storage_term, weighted_shape_term, weighted_annual_term,
        total_loss,
        sim_frac_12, S_sim, Q_out, P_sim
    """
    S_sim, Q_out, P_sim = run_scan(
        inflow_data[:-1], storage_obs_data[0], biweek_data[:-1],
        params, Smax, H_m, P_cap, eta, kq_scale
    )

    storage_rmse = float(jnp.sqrt(jnp.mean((S_sim - storage_obs_data[1:]) ** 2)))
    storage_loss_norm = float(jnp.mean(((S_sim - storage_obs_data[1:]) / (sigma_S + 1e-12)) ** 2))

    _, sim_frac_12, annual_sim_mwh = compute_hydro_targets(P_sim, month_id_jax)
    annual_sim_mwh = float(annual_sim_mwh)

    hydro_shape_loss = float(jnp.mean(((sim_frac_12 - obs_frac_12_jax) / (obs_frac_12_jax + 1e-12)) ** 2))
    hydro_annual_loss = float(((annual_sim_mwh - annual_obs_mwh) / (annual_obs_mwh + 1e-12)) ** 2)

    reg = float(smooth_penalty(params[0:26]) + smooth_penalty(params[26:52]))

    weighted_storage_term = lam_storage * storage_loss_norm
    weighted_shape_term = lam_hydro_shape * hydro_shape_loss
    weighted_annual_term = lam_hydro_annual * hydro_annual_loss
    weighted_smooth_term = lam_smooth * reg

    total_loss = (
        weighted_storage_term
        + weighted_shape_term
        + weighted_annual_term
        + weighted_smooth_term
    )

    bias_mwh = annual_sim_mwh - annual_obs_mwh
    bias_percent = (bias_mwh / annual_obs_mwh) * 100.0

    return {
        "storage_rmse": storage_rmse,
        "storage_loss_norm": storage_loss_norm,
        "hydro_shape_loss": hydro_shape_loss,
        "hydro_annual_loss": hydro_annual_loss,
        "annual_sim_mwh": annual_sim_mwh,
        "annual_sim_gwh": annual_sim_mwh / 1000.0,
        "bias_mwh": bias_mwh,
        "bias_percent": bias_percent,
        "smooth_penalty": reg,
        "weighted_storage_term": weighted_storage_term,
        "weighted_shape_term": weighted_shape_term,
        "weighted_annual_term": weighted_annual_term,
        "weighted_smooth_term": weighted_smooth_term,
        "lam_storage": lam_storage,
        "lam_hydro_shape": lam_hydro_shape,
        "lam_hydro_annual": lam_hydro_annual,
        "total_loss": total_loss,
        "sim_frac_12": np.array(sim_frac_12),
        "S_sim": np.array(S_sim),
        "Q_out": np.array(Q_out),
        "P_sim": np.array(P_sim),
    }
