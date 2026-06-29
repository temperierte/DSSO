"""
Plot pure-SFFA parameter budget vs test RMSE from the new reconstruction logs.

This version preserves the plotting style, title/axis labels, legend text, point
annotations, and printed table columns of plot_sffa_budget.py, but reads only two
logs:

- SFFA points are read from result-reconstruction-budget.txt.
- The Subgraph numerical LMMSE horizontal reference is read from
  result-reconstruction.txt at the same p.
- The three best polynomial-LIM horizontal references are also read from
  result-reconstruction.txt at the same p.
- No zero-fill / no-processing baseline is drawn.
- No asymptotic full-graph baseline is drawn.
- No Kron-SFFA points are drawn; the budget curve is pure SFFA only.
- No standard deviations are plotted.

Usage:
    python plot_sffa_budget_new_logs.py
    python plot_sffa_budget_new_logs.py --budget-log result-reconstruction-budget.txt \
        --reconstruction-log result-reconstruction.txt
    python plot_sffa_budget_new_logs.py --x budget
    python plot_sffa_budget_new_logs.py --x params
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

DEFAULT_BUDGET_LOG_PATH = Path("result-reconstruction-budget.txt")
DEFAULT_RECONSTRUCTION_LOG_PATH = Path("result-reconstruction.txt")


# -----------------------------------------------------------------------------
# Small parsing helpers
# -----------------------------------------------------------------------------


def resolve_existing_path(path: Path) -> Path:
    """Resolve path, with a /mnt/data fallback for notebook/ChatGPT sandboxes."""
    path = Path(path)
    if path.exists():
        return path

    fallback = Path("/mnt/data") / path.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Cannot find log file: {path} or {fallback}")


def read_lines(path: Path) -> list[str]:
    return resolve_existing_path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def nearly_equal(a: float, b: float, tol: float = 5e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def compact_label(k: int, r: int) -> str:
    return f"({k},{r})"


def parse_rmse_row(line: str) -> Optional[tuple[str, float, float, float]]:
    """Parse one printed RMSE table row: method | rmse±std | mse."""
    m = re.match(
        r"^(.+?)\s+\|\s+([0-9.eE+-]+)±([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s*$",
        line.strip(),
    )
    if not m:
        return None

    method, rmse, rmse_std, mse = m.groups()
    return method.strip(), float(rmse), float(rmse_std), float(mse)


def parse_summary_row(line: str) -> Optional[tuple[float, str, float, float, float]]:
    """Parse one row from the optional final p-sweep summary table.

    Format:
        p | method_key | 0.1234±0.0056 | 1.234e-02

    This is a fallback only. The primary parser uses the detailed p-sweep blocks.
    """
    m = re.match(
        r"^\s*([0-9.]+)\s+\|\s+(.+?)\s+\|\s+([0-9.eE+-]+)±([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s*$",
        line.rstrip(),
    )
    if not m:
        return None
    p_val, method_key, rmse, rmse_std, mse = m.groups()
    return float(p_val), method_key.strip(), float(rmse), float(rmse_std), float(mse)


# -----------------------------------------------------------------------------
# Data extraction from the two current stdout logs
# -----------------------------------------------------------------------------


def parse_budget_log(log_path: Path, p_target: float = 0.5) -> pd.DataFrame:
    """Extract pure-SFFA budget sweep points from result-reconstruction-budget.txt."""
    lines = read_lines(log_path)

    header_re = re.compile(
        r"Budget analysis:\s*p=([0-9.]+),\s*k=(\d+),\s*r=(\d+),\s*budget=(\d+)"
    )
    learned_re = re.compile(
        r"^pure_sffa\s+\|\s+params=\s*([0-9]+)\s+\|\s+"
        r"train missing MSE=([0-9.eE+-]+)\s+\|\s+\|\|F\|\|_F=([0-9.eE+-]+)"
    )
    method_re = re.compile(r"Pure distance-based SFFA\s*\(k=(\d+),\s*r=(\d+)\)")

    rows: list[dict[str, object]] = []
    current: dict[str, object] = {}

    for line in lines:
        m = header_re.search(line)
        if m:
            p_val, k, r, budget = m.groups()
            current = {
                "p": float(p_val),
                "k": int(k),
                "r": int(r),
                "budget": int(budget),  # experiment budget k + ... + k^r, excluding identity
                "params": None,         # actual printed pure_sffa params, usually budget + 1
                "train_missing_mse": None,
                "fro_norm": None,
            }
            continue

        if not current:
            continue

        m = learned_re.match(line.strip())
        if m:
            params, train_mse, fro_norm = m.groups()
            current["params"] = int(params)
            current["train_missing_mse"] = float(train_mse)
            current["fro_norm"] = float(fro_norm)
            continue

        parsed = parse_rmse_row(line)
        if parsed is None:
            continue

        method, rmse, _rmse_std, mse = parsed
        m = method_re.search(method)
        if not m:
            continue

        k_from_method, r_from_method = int(m.group(1)), int(m.group(2))
        p_val = float(current["p"])
        if not nearly_equal(p_val, p_target):
            continue
        if k_from_method != int(current["k"]) or r_from_method != int(current["r"]):
            raise ValueError(
                "Budget header and SFFA method row disagree: "
                f"header={(current['k'], current['r'])}, method={(k_from_method, r_from_method)}"
            )

        params = current["params"]
        if params is None:
            # Fallback only; normally we should use the printed learned-operator params.
            params = int(current["budget"]) + 1

        rows.append(
            {
                "p": p_val,
                "k": int(current["k"]),
                "r": int(current["r"]),
                "budget": int(current["budget"]),
                "params": int(params),
                "train_missing_mse": current["train_missing_mse"],
                "fro_norm": current["fro_norm"],
                "method": method,
                "rmse": float(rmse),
                "mse": float(mse),
                "label": compact_label(int(current["k"]), int(current["r"])),
            }
        )

    if not rows:
        raise ValueError(f"No pure-SFFA budget rows found for p={p_target} in {log_path}.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["budget", "params", "rmse", "k", "r"]).reset_index(drop=True)
    return df


def parse_num_lmmse_baseline(log_path: Path, p_target: float = 0.5) -> float:
    """Extract Subgraph numerical LMMSE at the selected p from result-reconstruction.txt."""
    lines = read_lines(log_path)
    p_re = re.compile(r"=====\s*p sweep\s+\d+/\d+:\s*p=([0-9.]+),")

    current_p: Optional[float] = None
    for line in lines:
        m = p_re.search(line)
        if m:
            current_p = float(m.group(1))
            continue

        if current_p is None or not nearly_equal(current_p, p_target):
            continue

        parsed = parse_rmse_row(line)
        if parsed is None:
            continue

        method, rmse, _rmse_std, _mse = parsed
        if method == "Subgraph numerical LMMSE":
            return float(rmse)

    # Fallback: final p-sweep summary table may use a compact method key.
    for line in lines:
        parsed_summary = parse_summary_row(line)
        if parsed_summary is None:
            continue
        p_val, method_key, rmse, _rmse_std, _mse = parsed_summary
        if nearly_equal(p_val, p_target) and method_key == "numerical_lmmse":
            return float(rmse)

    raise ValueError(f"Cannot find Subgraph numerical LMMSE for p={p_target} in {log_path}.")


def parse_best_lim_baselines(log_path: Path, p_target: float = 0.5) -> dict[str, tuple[float, str]]:
    """Extract the best LIM RMSE for induced/Kron/union polynomial families.

    The primary parser reads detailed rows inside the matching p-sweep block.
    The summary parser is only a fallback and keeps the same family keys as the
    original plotting script.
    """
    lines = read_lines(log_path)
    p_re = re.compile(r"=====\s*p sweep\s+\d+/\d+:\s*p=([0-9.]+),")

    family_patterns: dict[str, re.Pattern[str]] = {
        "Best induced polynomial LIM": re.compile(r"^Induced Laplacian polynomial LIM \(deg≤(\d+)\)$"),
        "Best Kron polynomial LIM": re.compile(r"^Kron Laplacian polynomial LIM \(deg≤(\d+)\)$"),
        "Best union polynomial LIM": re.compile(r"^1\.\.\d+-union Laplacian polynomial LIM \(deg≤(\d+)\)$"),
    }

    current_p: Optional[float] = None
    best: dict[str, tuple[float, str, int]] = {}

    for line in lines:
        m = p_re.search(line)
        if m:
            current_p = float(m.group(1))
            continue

        if current_p is None or not nearly_equal(current_p, p_target):
            continue

        parsed = parse_rmse_row(line)
        if parsed is None:
            continue

        method, rmse, _rmse_std, _mse = parsed
        for family, pattern in family_patterns.items():
            m_family = pattern.match(method)
            if not m_family:
                continue
            degree = int(m_family.group(1))
            old = best.get(family)
            if old is None or rmse < old[0]:
                best[family] = (float(rmse), method, degree)

    # Fallback for logs where only the final compact p-sweep summary is available.
    summary_key_patterns: dict[str, re.Pattern[str]] = {
        "Best induced polynomial LIM": re.compile(r"^induced_poly_lim_r(\d+)$"),
        "Best Kron polynomial LIM": re.compile(r"^kron_poly_lim_r(\d+)$"),
        "Best union polynomial LIM": re.compile(r"^union_poly_lim_r(\d+)$"),
    }
    for line in lines:
        parsed_summary = parse_summary_row(line)
        if parsed_summary is None:
            continue
        p_val, method_key, rmse, _rmse_std, _mse = parsed_summary
        if not nearly_equal(p_val, p_target):
            continue
        for family, pattern in summary_key_patterns.items():
            m_family = pattern.match(method_key)
            if not m_family:
                continue
            degree = int(m_family.group(1))
            old = best.get(family)
            if old is None or rmse < old[0]:
                best[family] = (float(rmse), method_key, degree)

    missing = [family for family in family_patterns if family not in best]
    if missing:
        raise ValueError(f"Cannot find LIM baselines {missing} for p={p_target} in {log_path}.")

    # Drop the degree from the internal tuple; keep it in the method string for printing/legend.
    return {family: (rmse, method) for family, (rmse, method, _degree) in best.items()}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def sffa_r_color_map(r_values: list[int]) -> dict[int, str]:
    """Assign one stable color to each SFFA word length r."""
    palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:cyan",
        "tab:olive",
    ]
    return {int(r): palette[i % len(palette)] for i, r in enumerate(sorted(set(int(v) for v in r_values)))}


def annotate_points(ax, plot_df: pd.DataFrame, x_col: str, r_to_color: dict[int, str]) -> None:
    """Annotate each SFFA point by (k,r), using the same color as its r group."""
    for _, row in plot_df.iterrows():
        r_val = int(row["r"])
        ax.annotate(
            row["label"],
            xy=(row[x_col], row["rmse"]),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=12,
            color=r_to_color.get(r_val, "black"),
            ha="left",
            va="top",
        )


def plot_budget_rmse(
    plot_df: pd.DataFrame,
    num_lmmse_rmse: float,
    lim_baselines: dict[str, tuple[float, str]],
    x_col: str = "budget",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8.5))

    plot_df = plot_df.sort_values([x_col, "rmse", "k", "r"]).reset_index(drop=True)
    r_to_color = sffa_r_color_map([int(v) for v in plot_df["r"].tolist()])

    # Pure SFFA points, grouped and colored by word length r.
    for r_val in sorted(r_to_color):
        group = plot_df[plot_df["r"] == r_val].sort_values([x_col, "k"])
        ax.scatter(
            group[x_col],
            group["rmse"],
            s=58,
            marker="o",
            color=r_to_color[r_val],
            edgecolors="white",
            linewidths=0.8,
            label=fr"SFFA $r={r_val}$",
            zorder=3,
        )

    # Horizontal references: distinct colors and thicker lines.
    reference_colors = {
        "Numerical LMMSE": "black",
        "Induced LIM": "tab:brown",
        "Kron LIM": "tab:pink",
        "Union LIM": "tab:gray",
    }

    ax.axhline(
        num_lmmse_rmse,
        linestyle="-",
        linewidth=1.8,
        color=reference_colors["Numerical LMMSE"],
        label="Numerical LMMSE",
        zorder=1,
    )

    lim_legend_name = {
        "Best induced polynomial LIM": r"Best $\mathbb{R}_{\leq r}[\mathbf{L}_\mathrm{ind}]$",
        "Best Kron polynomial LIM": r"Best $\mathbb{R}_{\leq r}[\mathbf{L}_\mathrm{Kron}]$",
        "Best union polynomial LIM": r"Best $\mathbb{R}_{\leq r}[\mathbf{L}_U^{\cup_{12}}]$",
    }

    lim_color_key = {
        "Best induced polynomial LIM": "Induced LIM",
        "Best Kron polynomial LIM": "Kron LIM",
        "Best union polynomial LIM": "Union LIM",
    }

    for family, (rmse, method) in lim_baselines.items():
        ax.axhline(
            rmse,
            linestyle="--",
            linewidth=1.8,
            color=reference_colors[lim_color_key[family]],
            label=lim_legend_name.get(family, family),
            zorder=1,
        )


    annotate_points(ax, plot_df, x_col=x_col, r_to_color=r_to_color)

    ax.set_xlabel("Number of learned SFFA coefficients")
    ax.set_ylabel("Test RMSE on missing nodes")
    ax.set_title("SFFA budget analysis")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-log",
        "--log",
        dest="budget_log",
        type=Path,
        default=DEFAULT_BUDGET_LOG_PATH,
        help="Path to result-reconstruction-budget.txt. --log is kept as a compatibility alias.",
    )
    parser.add_argument(
        "--reconstruction-log",
        "--baseline-log",
        dest="reconstruction_log",
        type=Path,
        default=DEFAULT_RECONSTRUCTION_LOG_PATH,
        help="Path to result-reconstruction.txt. --baseline-log is kept as a compatibility alias.",
    )
    parser.add_argument("--p", type=float, default=0.5, help="Which p value to extract.")
    parser.add_argument(
        "--x",
        choices=["budget", "params"],
        default="budget",
        help="x-axis: budget excludes identity; params uses printed pure_sffa params and includes identity.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the figure.")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show(); useful for batch runs.")
    args = parser.parse_args()

    plot_df = parse_budget_log(args.budget_log, p_target=args.p)
    num_lmmse_rmse = parse_num_lmmse_baseline(args.reconstruction_log, p_target=args.p)
    lim_baselines = parse_best_lim_baselines(args.reconstruction_log, p_target=args.p)

    print(f"Loaded budget log : {resolve_existing_path(args.budget_log)}")
    print(f"Loaded reconstruction log: {resolve_existing_path(args.reconstruction_log)}")
    print(f"Using p={args.p:g}; {len(plot_df)} pure-SFFA rows")
    print(f"Subgraph numerical LMMSE: {num_lmmse_rmse:.4f}")
    for family, (rmse, method) in lim_baselines.items():
        print(f"{family}: {rmse:.4f}  [{method}]")

    print(
        plot_df[["budget", "params", "k", "r", "rmse", "mse", "label", "method"]]
        .to_string(index=False)
    )

    plot_budget_rmse(
        plot_df=plot_df,
        num_lmmse_rmse=num_lmmse_rmse,
        lim_baselines=lim_baselines,
        x_col=args.x,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()


# Example:
# python plot_sffa_budget.py --budget-log result-reconstruction-budget.txt --reconstruction-log result-reconstruction.txt --x budget --p 0.5