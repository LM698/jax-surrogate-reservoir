"""
Multi-start Adam optimization for joint storage + hydropower matching.
"""

import numpy as np
import jax.numpy as jnp
import jax.random as random
import optax
from jax import jit

from hydro_loss import loss_fn, grad_loss, evaluate_params
from hydro_model import init_params


# ---------------------------------------------------------------------------
# Single optimization run
# ---------------------------------------------------------------------------

def run_optimization(params_init,
                     inflow_data, storage_obs_data, biweek_data, Smax,
                     month_id_jax, obs_frac_12_jax, annual_obs_mwh, sigma_S,
                     H_m, P_cap, eta=0.8,
                     lam_storage=1.0,
                     lam_hydro_shape=1.0, lam_hydro_annual=0.5,
                     learning_rate=1e-2, num_iterations=20000,
                     lam_smooth=0.005, kq_scale=2.0, verbose=False):
    """Run Adam optimization from a single initialization."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params_init)
    params = params_init

    @jit
    def update(p, state):
        grads = grad_loss(
            p,
            inflow_data, storage_obs_data, biweek_data, Smax,
            month_id_jax, obs_frac_12_jax, annual_obs_mwh, sigma_S,
            H_m, P_cap, eta,
            lam_storage, lam_hydro_shape, lam_hydro_annual,
            lam_smooth, kq_scale
        )
        updates, next_state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), next_state

    loss_hist = []
    for i in range(num_iterations):
        params, opt_state = update(params, opt_state)

        if (i % 200 == 0) or (i == num_iterations - 1):
            curr = loss_fn(
                params,
                inflow_data, storage_obs_data, biweek_data, Smax,
                month_id_jax, obs_frac_12_jax, annual_obs_mwh, sigma_S,
                H_m, P_cap, eta,
                lam_storage, lam_hydro_shape, lam_hydro_annual,
                lam_smooth, kq_scale
            )
            loss_hist.append(float(curr))
            if verbose:
                print(f"iter {i:5d}: loss = {float(curr):.6f}")
            if jnp.isnan(curr):
                print("NaN encountered — stopping early.")
                break

    metrics = evaluate_params(
        params,
        inflow_data, storage_obs_data, biweek_data, Smax,
        month_id_jax, obs_frac_12_jax, annual_obs_mwh, sigma_S,
        H_m, P_cap, eta,
        lam_storage, lam_hydro_shape, lam_hydro_annual,
        lam_smooth, kq_scale
    )

    return params, jnp.array(loss_hist), metrics


# ---------------------------------------------------------------------------
# Multi-start runner
# ---------------------------------------------------------------------------

def run_multistart(inflow_data, storage_obs_data, biweek_data, Smax,
                   month_id_jax, obs_frac_12_jax, annual_obs_mwh, sigma_S,
                   H_m, P_cap, eta=0.8,
                   lam_storage=1.0,
                   lam_hydro_shape=1.0, lam_hydro_annual=0.5,
                   num_runs=10, learning_rate=1e-2, num_iterations=20000,
                   lam_smooth=0.005, kq_scale=2.0, verbose=False):
    """
    Run optimization from multiple random seeds and return the best result.

    Best is selected by lowest total_loss so the storage and hydropower
    objectives are both included in the selection criterion.
    """
    results = []

    for seed in range(num_runs):
        print(f"\n--- Seed {seed + 1}/{num_runs} (seed={seed}) ---")
        key = random.PRNGKey(seed)

        params_init = init_params(key, storage_obs_data, inflow_data, biweek_data, Smax)

        params_final, loss_hist, metrics = run_optimization(
            params_init,
            inflow_data, storage_obs_data, biweek_data, Smax,
            month_id_jax, obs_frac_12_jax, annual_obs_mwh, sigma_S,
            H_m, P_cap, eta,
            lam_storage,
            lam_hydro_shape, lam_hydro_annual,
            learning_rate, num_iterations,
            lam_smooth, kq_scale, verbose
        )

        print(
            f"Total Loss: {metrics['total_loss']:.4f} | "
            f"Storage RMSE: {metrics['storage_rmse']:.3f} MCM | "
            f"Annual Bias: {metrics['bias_percent']:.2f}% | "
            f"Shape Loss: {metrics['hydro_shape_loss']:.4f}"
        )

        results.append({
            "seed": seed,
            "params_final": params_final,
            "loss_history": loss_hist,
            "metrics": metrics,
        })

    all_total_loss = np.array([r["metrics"]["total_loss"] for r in results])
    best_idx = int(np.argmin(all_total_loss))

    summary = {
        "min_total_loss": float(all_total_loss.min()),
        "mean_total_loss": float(all_total_loss.mean()),
        "max_total_loss": float(all_total_loss.max()),
        "best_seed": results[best_idx]["seed"],
        "best_metrics": results[best_idx]["metrics"],
    }

    print(f"\n=== Multi-start Summary ===")
    print(f"Min Total Loss: {summary['min_total_loss']:.4f}")
    print(f"Mean Total Loss: {summary['mean_total_loss']:.4f}")
    print(f"Best Seed: {summary['best_seed']}")
    print(f"Storage RMSE: {summary['best_metrics']['storage_rmse']:.3f} MCM")
    print(f"Annual Sim: {summary['best_metrics']['annual_sim_gwh']:.3f} GWh")
    print(f"Annual Bias: {summary['best_metrics']['bias_percent']:.2f}%")

    return results[best_idx]["params_final"], results, summary


# ---------------------------------------------------------------------------
# Helpers for multi-case trade-off experiments
# ---------------------------------------------------------------------------

def collect_case_summary(case_name, summary):
    """Convert one case summary into a flat dict for comparison tables."""
    m = summary["best_metrics"]
    return {
        "case": case_name,
        "lam_storage": m["lam_storage"],
        "lam_hydro_shape": m["lam_hydro_shape"],
        "lam_hydro_annual": m["lam_hydro_annual"],
        "total_loss": m["total_loss"],
        "storage_rmse": m["storage_rmse"],
        "storage_loss_norm": m["storage_loss_norm"],
        "hydro_shape_loss": m["hydro_shape_loss"],
        "hydro_annual_loss": m["hydro_annual_loss"],
        "annual_sim_gwh": m["annual_sim_gwh"],
        "bias_percent": m["bias_percent"],
        "weighted_storage_term": m["weighted_storage_term"],
        "weighted_shape_term": m["weighted_shape_term"],
        "weighted_annual_term": m["weighted_annual_term"],
    }


def summarize_cases(case_summaries):
    """
    Build a compact comparison table from a dict like
    {case_name: summary_from_run_multistart}.
    """
    rows = [collect_case_summary(name, summary)
            for name, summary in case_summaries.items()]
    return rows