"""
Core storage model functions for biweekly storage matching experiments.

This file contains:
- parameter transforms
- parameter initialization
- one-step reservoir update
- full simulation over time
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
from functools import partial


# ---------------------------------------------------------------------------
# Single time-step
# ---------------------------------------------------------------------------

@jit
def step_biweekly_target_inflow(params, carry, inputs, Smax, kq_scale=2.0):
    """
    Simulate one daily time step of reservoir storage.

    Policy (linear feedback control):
        Q_target = Q_base + kS * (S_prev - S_target) + kQ * Q_in

    - Parameters -
    params: jnp.ndarray, shape (104,)
        Flat parameter vector: [qbase_raw(26), starget_raw(26), ks_raw(26), kq_raw(26)]
        Raw values are transformed to physical space via activation functions, and the range of them is:
            Q_base: [0, inf)
            S_target: [0, Smax]
            kS: [0, inf)
            kQ: [-kq_scale, +kq_scale]

    carry: tuple
        (S_prev,): reservoir storage at the previous time step (MCM)

    inputs: tuple
        (Q_in_t, biweek_t): inflow (MCM/day) and bi-week index (1..26)

    Smax: float
        Maximum reservoir capacity (MCM).

    kq_scale: float
        Bounds the inflow sensitivity: kQ in [-kq_scale, +kq_scale].

    - Returns -
    S_next: float
        Storage at the next time step (MCM).
    (S_next, Q_rel): tuple
        Outputs collected by lax.scan.
    """
    qbase_raw = params[0:26]
    starget_raw = params[26:52]
    ks_raw = params[52:78]
    kq_raw = params[78:104]

    S_prev, = (carry,)
    Q_in_t, biweek_t = inputs
    dt = 1.0

    idx = biweek_t - 1  # convert 1-based to 0-based index

    # Transform raw parameters to physical space
    Q_base = jax.nn.softplus(qbase_raw[idx])        # baseline seasonal release
    S_target = jax.nn.sigmoid(starget_raw[idx]) * Smax  # seasonal target storage
    kS = jax.nn.softplus(ks_raw[idx])            # storage feedback gain
    kQ = jnp.tanh(kq_raw[idx]) * kq_scale       # inflow sensitivity

    # Linear feedback control law
    Q_target = Q_base + kS * (S_prev - S_target) + kQ * Q_in_t

    # Physical constraints
    # Spill: if storage exceeds capacity, release the excess
    excess = S_prev - Smax
    Q_spill = jnp.maximum(0.0, excess / dt)
    Q_rel_base = jnp.where(excess > 0.0, Q_in_t + Q_spill, Q_target)

    # Mass balance: cannot release more water than available
    max_available = (S_prev / dt) + Q_in_t
    Q_rel = jnp.minimum(Q_rel_base, max_available)

    # Non-negativity: release cannot be negative
    Q_rel = jnp.maximum(Q_rel, 0.0)

    # Storage update (water balance)
    S_next = S_prev + (Q_in_t - Q_rel) * dt
    S_next = jnp.clip(S_next, 0.0, Smax)

    return S_next, (S_next, Q_rel)


# ---------------------------------------------------------------------------
# Full time-series scan
# ---------------------------------------------------------------------------

@jit
def run_scan(inflow_data, initial_storage, biweek_data, params, Smax, kq_scale=2.0):
    """
    Run the reservoir model over a full time series using lax.scan.

    - Parameters -
    inflow_data: jnp.ndarray, shape (T,)
        Daily inflow sequence (MCM/day).
    initial_storage: float
        Starting reservoir storage S0 (MCM).
    biweek_data: jnp.ndarray, shape (T,), dtype int
        Bi-week index for each day (1..26).
    params: jnp.ndarray, shape (104,)
        Model parameters (see step_biweekly_target_inflow).
    Smax: float
        Maximum reservoir capacity (MCM).
    kq_scale: float
        Inflow sensitivity bound.

    - Returns -
    S_sim: jnp.ndarray, shape (T,)
        Simulated daily storage (MCM).
    Q_out: jnp.ndarray, shape (T,)
        Simulated daily release (MCM/day).
    """
    inputs = (inflow_data, biweek_data)
    fn = partial(step_biweekly_target_inflow, params, Smax=Smax, kq_scale=kq_scale)
    _, (S_sim, Q_out) = lax.scan(fn, initial_storage, inputs)
    return S_sim, Q_out


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

def _inv_softplus(y):
    """
    Inverse of softplus: maps y > 0 back to raw space.
    """
    return jnp.log(jnp.expm1(y))


def init_params(key, storage_obs, inflow, biweek, Smax, kq_init=0.0):
    """
    Physically-informed parameter initialization.

    - Q_base: initialized near 80% of mean inflow (stable baseline release)
    - S_target: initialized from the per-biweek median observed storage
                 (provides a data-informed seasonal rule curve)
    - kS: initialized small (softplus(-2) ~ 0.13) to avoid aggressive
                 storage correction early in optimization
    - kQ: initialized near kq_init (default 0 = neutral inflow response)

    Small Gaussian noise is added to all parameters to break symmetry
    across seeds in multi-start optimization.

    - Parameters -
    key: jax.random.PRNGKey
    storage_obs: jnp.ndarray, shape (T,)
    inflow: jnp.ndarray, shape (T,)
    biweek: jnp.ndarray, shape (T,), dtype int
    Smax: float
    kq_init: float
        Initial inflow sensitivity (in physical space, before arctanh).
        Default 0 = no inflow feed-forward at initialization.

    - Returns -
    params : jnp.ndarray, shape (104,)
    """
    key_q, key_s, key_ks, key_kq = random.split(key, 4)

    # Q_base: 80% of mean inflow, with small noise
    q_mean = jnp.mean(inflow)
    qbase_raw = _inv_softplus(0.8 * q_mean) + 0.05 * random.normal(key_q, (26,))

    # S_target: per-biweek median of observed storage (as storage goal)
    bw = np.array(biweek).astype(int)
    s_np = np.array(storage_obs).astype(float)
    med = np.zeros(26, dtype=float)
    for i in range(1, 27):
        vals = s_np[bw == i]
        med[i-1] = np.median(vals) if vals.size > 0 else np.median(s_np)

    frac = np.clip(med / float(Smax), 1e-4, 1 - 1e-4)
    starget_raw = jnp.array(np.log(frac / (1 - frac)))
    starget_raw = starget_raw + 0.05 * random.normal(key_s, (26,))

    # kS: small initial feedback gain
    ks_raw = -2.0 + 0.1 * random.normal(key_ks, (26,))

    # kQ: initialized at kq_init via arctanh (inverse of tanh)
    kq_raw = jnp.arctanh(jnp.clip(jnp.ones((26,)) * kq_init, -0.95, 0.95))
    kq_raw = kq_raw + 0.05 * random.normal(key_kq, (26,))

    return jnp.concatenate([qbase_raw, starget_raw, ks_raw, kq_raw])