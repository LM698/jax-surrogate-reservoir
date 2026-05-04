"""
Reservoir simulation model extended with hydropower generation.

Extends the bi-weekly target-tracking policy by computing 
power output (MW) from simulated release, using a fixed design head.

Power equation:
    P = eta * rho * g * H * min(Q_rel, Q_turb_max)  [MW]

where Q_turb_max is derived from installed capacity P_cap.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
from functools import partial


# ---------------------------------------------------------------------------
# Single time-step (storage + hydropower)
# ---------------------------------------------------------------------------

@jit
def step_biweekly_target_inflow(params, carry, inputs, Smax,
                                 H_m, P_cap, eta=0.8, kq_scale=2.0):
    """
    Simulate one daily time step: storage update + hydropower generation.

    Policy (same as storage model):
        Q_target = Q_base + kS * (S_prev - S_target) + kQ * Q_in

    Hydropower:
        Q_turb_max = P_cap / (rho * g * H * eta)   [m³/s]
        Q_actual = min(Q_rel_m3s, Q_turb_max)
        P = (rho * g * H * Q_actual * eta) / 1e6  [MW]

    - Parameters -
    params: jnp.ndarray, shape (104,)
        [qbase_raw(26), starget_raw(26), ks_raw(26), kq_raw(26)]
    carry: tuple
        (S_prev,) — storage at previous time step (MCM)
    inputs: tuple
        (Q_in_t, biweek_t) — inflow (MCM/day) and bi-week index (1..26)
    Smax: float
        Maximum reservoir capacity (MCM).
    H_m: float
        Design head (metres). Fixed per dam.
    P_cap: float
        Installed turbine capacity (MW). Used to derive max turbine flow.
    eta: float
        Turbine efficiency (default 0.8).
    kq_scale: float
        Inflow sensitivity bound.

    - Returns -
    S_next : float
    (S_next, Q_rel, power_mw) : tuple collected by lax.scan
    """
    qbase_raw   = params[0:26]
    starget_raw = params[26:52]
    ks_raw      = params[52:78]
    kq_raw      = params[78:104]

    S_prev, = (carry,)
    Q_in_t, biweek_t = inputs
    dt = 1.0

    idx = biweek_t - 1

    # Transform raw → physical
    Q_base   = jax.nn.softplus(qbase_raw[idx])
    S_target = jax.nn.sigmoid(starget_raw[idx]) * Smax
    kS       = jax.nn.softplus(ks_raw[idx])
    kQ       = jnp.tanh(kq_raw[idx]) * kq_scale

    # Release decision
    Q_target = Q_base + kS * (S_prev - S_target) + kQ * Q_in_t

    # Spill to prevent exceeding capacity
    projected = S_prev + (Q_in_t - Q_target) * dt
    Q_spill = jnp.maximum(0.0, (projected - Smax) / dt)
    Q_rel_base = Q_target + Q_spill

    # Mass balance constraint
    max_available = S_prev / dt + Q_in_t
    Q_rel = jnp.clip(Q_rel_base, 0.0, max_available)

    # Storage update
    S_next = jnp.clip(S_prev + (Q_in_t - Q_rel) * dt, 0.0, Smax)

    # Hydropower (fixed head)
    rho, g = 1000.0, 9.81
    Q_turb_max_m3s = (P_cap * 1e6) / (rho * g * H_m * eta + 1e-12)
    Q_rel_m3s = Q_rel * 1e6 / 86400.0           # MCM/day -> m^2/s
    Q_actual_m3s = jnp.minimum(Q_rel_m3s, Q_turb_max_m3s)
    power_mw = (rho * g * H_m * Q_actual_m3s * eta) / 1e6

    return S_next, (S_next, Q_rel, power_mw)


# ---------------------------------------------------------------------------
# Full time-series scan
# ---------------------------------------------------------------------------

@jit
def run_scan(inflow_data, initial_storage, biweek_data, params,
             Smax, H_m, P_cap, eta=0.8, kq_scale=2.0):
    """
    Run the hydro model over a full time series using lax.scan.

    - Parameters -
    inflow_data: jnp.ndarray, shape (T,)
    initial_storage: float
    biweek_data: jnp.ndarray, shape (T,), dtype int
    params: jnp.ndarray, shape (104,)
    Smax: float
    H_m: float — design head (m)
    P_cap: float — installed capacity (MW)
    eta: float — turbine efficiency
    kq_scale: float

    - Returns -
    S_sim: jnp.ndarray, shape (T,) — simulated storage (MCM)
    Q_out: jnp.ndarray, shape (T,) — simulated release (MCM/day)
    P_sim: jnp.ndarray, shape (T,) — simulated power (MW, daily average)
    """
    inputs = (inflow_data, biweek_data)
    fn = partial(step_biweekly_target_inflow, params,
                 Smax=Smax, H_m=H_m, P_cap=P_cap,
                 eta=eta, kq_scale=kq_scale)
    _, (S_sim, Q_out, P_sim) = lax.scan(fn, initial_storage, inputs)
    return S_sim, Q_out, P_sim


# ---------------------------------------------------------------------------
# Parameter initialization (identical to storage model)
# ---------------------------------------------------------------------------

def _inv_softplus(y):
    return jnp.log(jnp.expm1(y))


def init_params(key, storage_obs, inflow, biweek, Smax, kq_init=0.0):
    """
    Physically-informed parameter initialization (same strategy as Phase 1).

    - Parameters -
    key: jax.random.PRNGKey
    storage_obs: jnp.ndarray, shape (T,)
    inflow: jnp.ndarray, shape (T,)
    biweek: jnp.ndarray, shape (T,), dtype int
    Smax: float
    kq_init: float

    - Returns -
    params: jnp.ndarray, shape (104,)
    """
    key_q, key_s, key_ks, key_kq = random.split(key, 4)

    q_mean = jnp.mean(inflow)
    qbase_raw = _inv_softplus(0.8 * q_mean) + 0.05 * random.normal(key_q, (26,))

    bw = np.array(biweek).astype(int)
    s_np = np.array(storage_obs).astype(float)
    med = np.zeros(26, dtype=float)
    for i in range(1, 27):
        vals = s_np[bw == i]
        med[i-1] = np.median(vals) if vals.size > 0 else np.median(s_np)

    frac = np.clip(med / float(Smax), 1e-4, 1 - 1e-4)
    starget_raw = jnp.array(np.log(frac / (1 - frac)))
    starget_raw = starget_raw + 0.05 * random.normal(key_s, (26,))

    ks_raw = -2.0 + 0.1 * random.normal(key_ks, (26,))

    kq_raw = jnp.arctanh(jnp.clip(jnp.ones((26,)) * kq_init, -0.95, 0.95))
    kq_raw = kq_raw + 0.05 * random.normal(key_kq, (26,))

    return jnp.concatenate([qbase_raw, starget_raw, ks_raw, kq_raw])