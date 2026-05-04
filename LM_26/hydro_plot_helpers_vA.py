"""
Plot helpers for hydropower extension experiments (Version A).

These helpers are intentionally lightweight so they can be dropped into the
existing notebook workflow without restructuring the project.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_tradeoff(case_rows,
                  x_key="storage_rmse",
                  y_key="bias_percent",
                  title="Trade-off: Storage vs Hydropower",
                  annotate=True):
    """
    Scatter plot for comparing weighted cases.

    Parameters
    ----------
    case_rows : list[dict]
        Output from summarize_cases(...)
    x_key : str
        Metric on x-axis. Usually "storage_rmse".
    y_key : str
        Metric on y-axis. Usually "bias_percent" or "hydro_shape_loss".
    """
    plt.figure(figsize=(6, 5))
    for row in case_rows:
        x = row[x_key]
        y = row[y_key]
        plt.scatter(x, y, s=70)
        if annotate:
            plt.annotate(row["case"], (x, y), xytext=(5, 5), textcoords="offset points")

    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel(y_key.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gca()


def plot_monthly_fraction_cases(obs_frac_12, case_to_sim_frac,
                                title="Monthly Hydropower Distribution"):
    """
    Compare observed monthly generation fractions with multiple simulated cases.

    Parameters
    ----------
    obs_frac_12 : array-like, shape (12,)
    case_to_sim_frac : dict[str, array-like]
        Example: {"Base": sim_frac_base, "Balanced": sim_frac_balanced}
    """
    months = np.arange(1, 13)
    plt.figure(figsize=(8, 4.5))
    plt.plot(months, obs_frac_12, marker="o", linewidth=2, label="Observed")

    for case_name, sim_frac in case_to_sim_frac.items():
        plt.plot(months, np.asarray(sim_frac), marker="o", linewidth=1.8, label=case_name)

    plt.xlabel("Month")
    plt.ylabel("Fraction of Annual Generation")
    plt.title(title)
    plt.xticks(months)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gca()


def plot_annual_generation_bars(case_rows, annual_obs_gwh,
                                title="Annual Hydropower Comparison"):
    """Bar plot comparing observed annual generation with case results."""
    labels = ["Observed"] + [row["case"] for row in case_rows]
    values = [annual_obs_gwh] + [row["annual_sim_gwh"] for row in case_rows]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Annual Generation (GWh)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return plt.gca()


def print_case_table(case_rows):
    """Pretty-print a compact case comparison table in the notebook."""
    headers = [
        "case", "lam_storage", "lam_hydro_shape", "lam_hydro_annual",
        "storage_rmse", "hydro_shape_loss", "hydro_annual_loss", "bias_percent"
    ]
    line = " | ".join(f"{h:>18s}" for h in headers)
    print(line)
    print("-" * len(line))
    for row in case_rows:
        print(" | ".join([
            f"{str(row['case']):>18s}",
            f"{row['lam_storage']:18.3f}",
            f"{row['lam_hydro_shape']:18.3f}",
            f"{row['lam_hydro_annual']:18.3f}",
            f"{row['storage_rmse']:18.3f}",
            f"{row['hydro_shape_loss']:18.4f}",
            f"{row['hydro_annual_loss']:18.4f}",
            f"{row['bias_percent']:18.2f}",
        ]))
