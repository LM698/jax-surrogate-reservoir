"""
Data loading and preprocessing utilities.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def to_biweek(dts: pd.Series) -> np.ndarray:
    """
    Map a datetime Series to bi-week indices (1..26).

    A bi-week is a 14-day period. Day-of-year 1-14 → biweek 1,
    day 15-28 → biweek 2, ..., day 351-365 → biweek 26.

    - Parameters -
    dts : pd.Series of datetime64
        Time column from the loaded CSV.

    - Returns -
    np.ndarray, dtype int, shape (T,)
        Bi-week index for each day, clipped to [1, 26].
    """
    day_of_year = dts.dt.dayofyear.values
    bw = ((day_of_year - 1) // 14) + 1
    bw = np.minimum(bw, 26)
    return bw.astype(int)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dam_data(file_path: str):
    """
    Load a dam CSV file and return processed arrays.

    Expected CSV columns: time, storage_MCM, inflow_MCM

    - Parameters -
    file_path: str
        Path to the dam CSV file.

    - Returns -
    data: pd.DataFrame
        Raw dataframe with parsed datetime column.
    storage_obs: jnp.ndarray, shape (T,)
        Observed storage from MOSARTWBM (MCM).
    inflow: jnp.ndarray, shape (T,)
        Daily inflow (MCM/day).
    biweek_of_year: jnp.ndarray, shape (T,), dtype int
        Bi-week index for each day (1..26).
    Smax: float
        Effective maximum storage (max of observed series, MCM).
        Note: this is a data-driven proxy for reservoir capacity.
              It may underestimate true capacity if the reservoir
              never reached full in the observed period.
    """
    data = pd.read_csv(file_path)
    data["time"] = pd.to_datetime(data["time"])

    storage_obs = jnp.array(data["storage_MCM"].values)
    inflow = jnp.array(data["inflow_MCM"].values)
    biweek_of_year = jnp.array(to_biweek(data["time"]))
    Smax = float(jnp.max(storage_obs))

    print(f"Loaded {len(storage_obs)} days from: {file_path}")
    print(f"S0 = {float(storage_obs[0]):.2f} MCM")
    print(f"Smax = {Smax:.2f} MCM  (proxy from observed max)")
    print(f"Mean inflow = {float(jnp.mean(inflow)):.2f} MCM/day")

    return data, storage_obs, inflow, biweek_of_year, Smax