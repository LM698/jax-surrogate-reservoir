"""
Multi-start Adam optimization for storage-matching parameter estimation.

Best run is selected by the minimum final loss
(instead of minimum final RMSE).

This version includes early stopping.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax import jit

from storage_loss import loss_fn, grad_loss, compute_metrics
from storage_model import run_scan, init_params


# ---------------------------------------------------------------------------
# Single optimization run
# ---------------------------------------------------------------------------

def run_optimization(params_init, inflow_data, storage_obs_data, biweek_data, Smax,
                     learning_rate=1e-2, num_iterations=20000,
                     lam_smooth=0.02, verbose=False,
                     early_stopping=True, patience=1000, tol=1e-6,
                     log_every=200):
    """
    Run Adam optimization from a single initialization.

    Uses gradient clipping (global norm = 1.0) to prevent instability,
    followed by Adam updates.

    Early stopping:
        Stop if the loss does not improve by at least 'tol'
        for 'patience' consecutive iterations.

    - Parameters -
    params_init: jnp.ndarray, shape (104,)
        Starting parameters.
    inflow_data: jnp.ndarray, shape (T,)
    storage_obs_data: jnp.ndarray, shape (T,)
    biweek_data: jnp.ndarray, shape (T,), dtype int
    Smax: float
    learning_rate: float
    num_iterations: int
    lam_smooth: float
        Regularization strength passed to loss_fn.
    verbose: bool
        If True, print loss every 'log_every' iterations.
    early_stopping: bool
        Whether to enable early stopping.
    patience: int
        Number of consecutive iterations without sufficient improvement
        before stopping.
    tol: float
        Minimum loss improvement required to reset patience.
    log_every: int
        Interval for recording/printing loss.

    - Returns -
    best_params: jnp.ndarray, shape (104,)
        Best parameters found during optimization.
    loss_hist: jnp.ndarray
        Loss values recorded every `log_every` iterations
        (and at the final iteration if needed).
    final_rmse: float
        RMSE of the full simulation with the best parameters.
    final_loss: float
        Final objective value from loss_fn using the best parameters.
    best_iter: int
        Iteration index where the best loss was achieved.
    n_iters_run: int
        Number of iterations actually executed.
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params_init)
    params = params_init

    @jit
    def update(params, opt_state):
        grads = grad_loss(
            params,
            inflow_data,
            storage_obs_data,
            biweek_data,
            Smax,
            lam_smooth
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    loss_hist = []

    # Track best solution
    best_loss = np.inf
    best_params = params
    best_iter = -1
    wait = 0
    n_iters_run = 0

    for i in range(num_iterations):
        params, opt_state = update(params, opt_state)
        n_iters_run = i + 1

        curr = loss_fn(
            params,
            inflow_data,
            storage_obs_data,
            biweek_data,
            Smax,
            lam_smooth
        )
        curr_float = float(curr)

        # Update best if sufficiently improved
        if curr_float < best_loss - tol:
            best_loss = curr_float
            best_params = params
            best_iter = i
            wait = 0
        else:
            wait += 1

        # Logging
        if (i % log_every == 0) or (i == num_iterations - 1):
            loss_hist.append(curr_float)
            if verbose:
                print(f"iter {i:5d}: loss = {curr_float:.6f}")

        # Stop on NaN
        if jnp.isnan(curr):
            print("NaN encountered — stopping early.")
            break

        # Early stopping
        if early_stopping and (wait >= patience):
            print(f"Early stopping at iteration {i} "
                  f"(best loss = {best_loss:.6f} at iter {best_iter})")
            break

    # Make sure final/best loss is included in history if last logged point missed it
    if len(loss_hist) == 0 or loss_hist[-1] != best_loss:
        loss_hist.append(best_loss)

    # Final evaluation uses the BEST parameters found, not just the last ones
    S_sim_full, _ = run_scan(
        inflow_data,
        storage_obs_data[0],
        biweek_data,
        best_params,
        Smax
    )
    final_rmse = float(jnp.sqrt(jnp.mean((S_sim_full - storage_obs_data) ** 2)))
    final_loss = float(
        loss_fn(best_params, inflow_data, storage_obs_data, biweek_data, Smax, lam_smooth)
    )

    return best_params, jnp.array(loss_hist), final_rmse, final_loss, best_iter, n_iters_run


# ---------------------------------------------------------------------------
# Multi-start runner
# ---------------------------------------------------------------------------

def run_multistart(inflow_data, storage_obs_data, biweek_data, Smax,
                   num_runs=20, learning_rate=1e-2, num_iterations=20000,
                   lam_smooth=0.02, verbose=False,
                   early_stopping=True, patience=1000, tol=1e-6,
                   log_every=200):
    """
    Run optimization from multiple random initializations and return the best.

    Multi-start is important because Adam can converge to different local
    minima depending on initialization. Running multiple seeds and selecting
    the lowest-loss solution improves robustness.

    - Parameters -
    inflow_data: jnp.ndarray, shape (T,)
    storage_obs_data: jnp.ndarray, shape (T,)
    biweek_data: jnp.ndarray, shape (T,), dtype int
    Smax: float
    num_runs: int
        Number of random seeds to try.
    learning_rate: float
    num_iterations: int
    lam_smooth: float
    verbose: bool
    early_stopping: bool
    patience: int
    tol: float
    log_every: int

    - Returns -
    best_params: jnp.ndarray, shape (104,)
        Parameters from the best (lowest final loss) run.
    results: list of dict
        Full results for all runs.
    summary: dict
        Summary statistics for final loss and final RMSE.
    """
    results = []

    for seed in range(num_runs):
        print(f"\n===== Run {seed + 1}/{num_runs} (seed={seed}) =====")
        key = random.PRNGKey(seed)

        params_init = init_params(
            key,
            storage_obs_data,
            inflow_data,
            biweek_data,
            Smax
        )

        params_final, loss_hist, final_rmse, final_loss, best_iter, n_iters_run = run_optimization(
            params_init,
            inflow_data,
            storage_obs_data,
            biweek_data,
            Smax,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            lam_smooth=lam_smooth,
            verbose=verbose,
            early_stopping=early_stopping,
            patience=patience,
            tol=tol,
            log_every=log_every
        )

        print(f"Final loss: {final_loss:.6f}")
        print(f"Final RMSE: {final_rmse:.4f} MCM")
        print(f"Best iteration: {best_iter}")
        print(f"Iterations run: {n_iters_run}")

        results.append({
            "seed": seed,
            "final_rmse": final_rmse,
            "final_loss": final_loss,
            "best_iter": best_iter,
            "n_iters_run": n_iters_run,
            "params_final": params_final,
            "loss_history": loss_hist
        })

    # Select best run by final loss
    all_loss = np.array([r["final_loss"] for r in results])
    best_idx = int(np.argmin(all_loss))

    # Keep RMSE summary too
    all_rmse = np.array([r["final_rmse"] for r in results])
    all_best_iter = np.array([r["best_iter"] for r in results])
    all_n_iters = np.array([r["n_iters_run"] for r in results])

    summary = {
        "min_loss": float(all_loss.min()),
        "max_loss": float(all_loss.max()),
        "mean_loss": float(all_loss.mean()),
        "min_rmse": float(all_rmse.min()),
        "max_rmse": float(all_rmse.max()),
        "mean_rmse": float(all_rmse.mean()),
        "min_best_iter": int(all_best_iter.min()),
        "max_best_iter": int(all_best_iter.max()),
        "mean_best_iter": float(all_best_iter.mean()),
        "min_iters_run": int(all_n_iters.min()),
        "max_iters_run": int(all_n_iters.max()),
        "mean_iters_run": float(all_n_iters.mean()),
        "best_seed": results[best_idx]["seed"]
    }

    print("\n=== Multi-start Summary ===")
    print(f"Min loss: {summary['min_loss']:.6f}")
    print(f"Max loss: {summary['max_loss']:.6f}")
    print(f"Mean loss: {summary['mean_loss']:.6f}")
    print(f"Min RMSE: {summary['min_rmse']:.4f} MCM")
    print(f"Max RMSE: {summary['max_rmse']:.4f} MCM")
    print(f"Mean RMSE: {summary['mean_rmse']:.4f} MCM")
    print(f"Mean best iteration: {summary['mean_best_iter']:.1f}")
    print(f"Mean iterations run: {summary['mean_iters_run']:.1f}")
    print(f"Best seed: {summary['best_seed']}")

    return results[best_idx]["params_final"], results, summary