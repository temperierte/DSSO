from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import Tensor


CURRENT_DIR = Path(__file__).resolve().parent
RUN_ROOT_DIR = CURRENT_DIR
PROJECT_ROOT_DIR = CURRENT_DIR.parent

DEVICE = "cpu"
DTYPE = torch.float64

DEFAULT_PLOT_PATH = Path("outputs") / "subgraph_reconstruction_rmse.png"
DEFAULT_GRAPH_PLOT_PATH = Path("outputs") / "graph_subgraph_reconstruction_layout.png"


DEFAULT_METHODS = "zero_baseline,asymptotic_full_graph,subgraph_asymptotic_lmmse,numerical_lmmse,induced_poly,kron_poly,union_poly,distance_laplacian_poly,random_group_algebra,pure_sffa"
LIM_METHOD_BASES = ("induced_poly_lim", "kron_poly_lim", "union_poly_lim")


METHOD_SPECS = {
    "zero_baseline": {
        "enabled": True,
        "label": "No processing (zero fill)",
        "group": "baseline",
    },
    # Optional compatibility baseline.  It is not in DEFAULT_METHODS, because the
    # requested full-graph baseline for the regular run is asymptotic_full_graph.
    "full_graph": {
        "enabled": False,
        "label": "Full-graph numerical LMMSE",
        "group": "baseline",
    },
    "asymptotic_full_graph": {
        "enabled": True,
        "label": "Asymptotic full-graph numerical LMMSE",
        "group": "baseline",
    },
    "subgraph_asymptotic_lmmse": {
        "enabled": True,
        "label": "Asymptotic subgraph numerical LMMSE",
        "group": "empirical",
    },
    "numerical_lmmse": {
        "enabled": True,
        "label": "Subgraph numerical LMMSE",
        "group": "empirical",
    },
    "induced_poly": {
        "enabled": True,
        "label": "Induced Laplacian polynomial",
        "group": "learned",
    },
    "kron_poly": {
        "enabled": True,
        "label": "Kron Laplacian polynomial",
        "group": "learned",
    },
    "union_poly": {
        "enabled": True,
        "label": "Union distance-Laplacian polynomial",
        "group": "learned",
    },
    "distance_laplacian_poly": {
        "enabled": True,
        "label": "Distance-k Laplacian polynomial",
        "group": "learned",
    },
    "random_group_algebra": {
        "enabled": True,
        "label": "Random-group algebra",
        "group": "learned",
    },
    "pure_sffa": {
        "enabled": True,
        "label": "Pure distance-based SFFA",
        "group": "learned",
    },
    # Selectors only.  They do not appear as result rows.  If explicitly named
    # in --methods, each expands into *_lim_r2, ..., *_lim_r{rmax}.
    "induced_poly_lim": {
        "enabled": False,
        "label": "Induced polynomial degree sweep selector",
        "group": "lim_selector",
    },
    "kron_poly_lim": {
        "enabled": False,
        "label": "Kron polynomial degree sweep selector",
        "group": "lim_selector",
    },
    "union_poly_lim": {
        "enabled": False,
        "label": "Union polynomial degree sweep selector",
        "group": "lim_selector",
    },
}


def get_enabled_methods() -> List[str]:
    return [
        name
        for name, spec in METHOD_SPECS.items()
        if bool(spec.get("enabled", False)) and spec.get("group") != "lim_selector"
    ]


def get_enabled_learned_methods() -> List[str]:
    return [
        name
        for name, spec in METHOD_SPECS.items()
        if bool(spec.get("enabled", False)) and spec.get("group") == "learned"
    ]


def split_method_tokens(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").replace(";", ",").split(",") if item.strip()]


def is_lim_variant_name(name: str) -> bool:
    return any(name.startswith(f"{base}_r") for base in LIM_METHOD_BASES)


def lim_variant_degree(name: str) -> int | None:
    for base in LIM_METHOD_BASES:
        prefix = f"{base}_r"
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix.isdigit():
                return int(suffix)
    return None


def lim_variant_base(name: str) -> str | None:
    for base in LIM_METHOD_BASES:
        if name.startswith(f"{base}_r"):
            return base
    return None


def iter_enabled_lim_variants(base: str) -> List[Tuple[str, int]]:
    prefix = f"{base}_r"
    out: List[Tuple[str, int]] = []
    for name, spec in METHOD_SPECS.items():
        if not bool(spec.get("enabled", False)) or not name.startswith(prefix):
            continue
        degree = lim_variant_degree(name)
        if degree is not None:
            out.append((name, degree))
    out.sort(key=lambda pair: pair[1])
    return out


def normalize_method_name(token: str) -> str:
    name = str(token or "").strip().replace("-", "_")
    aliases = {
        "default": "default",
        "defaults": "default",
        "regular": "default",
        "regular_methods": "default",
        "all_regular": "default",
        "zero": "zero_baseline",
        "zero_fill": "zero_baseline",
        "no_processing": "zero_baseline",
        "asymptotic": "asymptotic_full_graph",
        "asymptotic_lmmse": "asymptotic_full_graph",
        "asymptotic_full_graph_lmmse": "asymptotic_full_graph",
        "asymptotic_full_graph_num_lmmse": "asymptotic_full_graph",
        "asymptotic_full_graph_numerical_lmmse": "asymptotic_full_graph",
        "subgraph_asymptotic": "subgraph_asymptotic_lmmse",
        "asymptotic_subgraph": "subgraph_asymptotic_lmmse",
        "subgraph_asymptotic_lmmse": "subgraph_asymptotic_lmmse",
        "asymptotic_subgraph_lmmse": "subgraph_asymptotic_lmmse",
        "subgraph_asymptotic_numerical_lmmse": "subgraph_asymptotic_lmmse",
        "asymptotic_subgraph_numerical_lmmse": "subgraph_asymptotic_lmmse",
        "full_graph_num_lmmse": "full_graph",
        "full_graph_lmmse": "full_graph",
        "num_lmmse": "numerical_lmmse",
        "subgraph_lmmse": "numerical_lmmse",
        "subgraph_numerical_lmmse": "numerical_lmmse",
        "plugin_oracle_noisy": "numerical_lmmse",
        "induced_laplacian_polynomial": "induced_poly",
        "sub_lp": "induced_poly",
        "kron_laplacian_polynomial": "kron_poly",
        "kron_lp": "kron_poly",
        "union123_laplacian_polynomial": "union_poly",
        "union_laplacian_polynomial": "union_poly",
        "union_lp": "union_poly",
        "distance_poly": "distance_laplacian_poly",
        "distance_laplacian_polynomial": "distance_laplacian_poly",
        "distance_based_laplacian_polynomial": "distance_laplacian_poly",
        "dk_lap_poly": "distance_laplacian_poly",
        "dk_lp": "distance_laplacian_poly",
        "rg_alg": "random_group_algebra",
        "random_group": "random_group_algebra",
        "sffa": "pure_sffa",
        "sffa_3_3_d123": "pure_sffa",
    }
    return aliases.get(name, name)


def register_lim_variant(base: str, degree: int) -> str:
    if base not in LIM_METHOD_BASES:
        raise ValueError(f"Unknown lim selector: {base}")
    degree = int(degree)
    key = f"{base}_r{degree}"
    base_label = {
        "induced_poly_lim": "Induced Laplacian polynomial LIM",
        "kron_poly_lim": "Kron Laplacian polynomial LIM",
        "union_poly_lim": "Union distance-Laplacian polynomial LIM",
    }[base]
    METHOD_SPECS[key] = {"enabled": True, "label": f"{base_label} (deg≤{degree})", "group": "learned"}
    return key


def configure_enabled_methods(method_text: str, rmax: int) -> List[str]:
    """Enable methods and expand *_poly_lim selectors into r=2..rmax variants."""
    for name in list(METHOD_SPECS.keys()):
        if is_lim_variant_name(name):
            del METHOD_SPECS[name]
    for spec in METHOD_SPECS.values():
        spec["enabled"] = False

    raw_methods = str(method_text or "").strip() or DEFAULT_METHODS
    requested = split_method_tokens(raw_methods)
    expanded: List[str] = []
    unknown: List[str] = []

    def add_default_methods() -> None:
        expanded.extend(split_method_tokens(DEFAULT_METHODS))

    for raw in requested:
        method = normalize_method_name(raw)
        if method in {"default", "all"}:
            add_default_methods()
            continue
        if method in LIM_METHOD_BASES:
            if int(rmax) < 2:
                raise ValueError("--rmax must be >= 2 when any *_poly_lim method is enabled.")
            for degree in range(2, int(rmax) + 1):
                expanded.append(register_lim_variant(method, degree))
            continue
        if method in METHOD_SPECS and METHOD_SPECS[method].get("group") != "lim_selector":
            expanded.append(method)
            continue
        unknown.append(raw)

    if unknown:
        known = sorted(
            name
            for name, spec in METHOD_SPECS.items()
            if spec.get("group") != "lim_selector" and not is_lim_variant_name(name)
        ) + list(LIM_METHOD_BASES)
        raise ValueError(f"Unknown methods: {unknown}. Known methods/selectors: {known}")

    seen = set()
    enabled_order: List[str] = []
    for method in expanded:
        if method in seen:
            continue
        METHOD_SPECS[method]["enabled"] = True
        enabled_order.append(method)
        seen.add(method)
    if not enabled_order:
        raise ValueError("At least one method must be enabled.")
    return enabled_order


@dataclass
class ExperimentConfig:
    seed: int = 1
    device: str = "cuda"
    data_dir: str = "processed_molene"
    adjacency_file: str = "A.npy"
    signal_file: str = "X_clean.npy"
    adjacency_key: str = ""
    signal_key: str = ""
    split_mode: str = "ratio"  # ratio: split all samples by ratio; number: first train_size then next test_size
    train_ratio: float = 0.7
    test_ratio: float | None = None
    train_size: int | None = None
    test_size: int | None = None
    p_fixed: float = 0.25  # p: subgraph node ratio |V0|/|V|
    p_values: str = ""  # Optional CSV/range sweep for p_fixed, e.g. "0.3,0.4,0.5" or "0.3:0.7:0.1"
    p_missing: float = 0.1  # p1: hidden node ratio |M0|/|V| in the original graph
    methods: str = DEFAULT_METHODS
    rmax: int = 5  # independent maximum degree for *_poly_lim sweeps, emitted for r=2..rmax when enabled
    ridge: float = 1e-8
    # Spectral preprocessing of raw T x N graph signals.
    # Default reproduces the original behavior: keep L_sym eigenvalues lambda <= 1.
    spectral_preprocess: str = "cutoff"
    bandlimit_cutoff: float = 1.0
    bandlimit_k: int = 0
    # Normalization is disabled in this bandlimited variant.
    # These fields are kept only for CLI/backward compatibility and are ignored.
    normalize_by_all: bool = False
    normalize_by_train: bool = False
    use_normalized_laplacian: bool = False
    sffa_distance_k: int = 3  # k: controls distance-k primitives, union size, and random-group count
    auto_budget_k: bool = False  # choose max k with SFFA basis budget <= N0 for each p
    sffa_word_len: int = 3  # r: maximum polynomial degree / algebra word length for regular learned methods
    random_group_ratio: float = 0.9
    sffa_grid: str = ""  # Deprecated compatibility field; ignored by the revised runner.
    max_sffa_basis_size: int = 2000
    save_outputs: bool = True
    epochs: int = 10
    lr: float = 1.0
    optimizer: str = "lbfgs"
    lbfgs_max_iter: int = 20
    num_runs: int = 1
    plot_path: str = str(DEFAULT_PLOT_PATH)
    graph_plot_path: str = str(DEFAULT_GRAPH_PLOT_PATH)


def parse_sffa_grid_spec(spec: str, default_k: int, default_r: int) -> List[Tuple[int, int]]:
    """Parse a grid specification such as "2,2;3,2;4,3" into (k,r) pairs."""
    spec = (spec or "").strip()
    if not spec:
        return [(int(default_k), int(default_r))]

    out: List[Tuple[int, int]] = []
    for raw_item in spec.replace(" ", "").split(";"):
        if not raw_item:
            continue
        if "," not in raw_item:
            raise ValueError(
                "Each --sffa_grid item must have the form k,r; "
                f"got {raw_item!r}. Example: --sffa_grid '2,2;3,2;4,3'."
            )
        k_str, r_str = raw_item.split(",", 1)
        k = int(k_str)
        r = int(r_str)
        if k < 1:
            raise ValueError("Every k in --sffa_grid must be >= 1.")
        if r < 0:
            raise ValueError("Every r in --sffa_grid must be >= 0.")
        out.append((k, r))

    if not out:
        raise ValueError("--sffa_grid did not contain any valid k,r pair.")

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[Tuple[int, int]] = []
    for pair in out:
        if pair not in seen:
            deduped.append(pair)
            seen.add(pair)
    return deduped


def parse_float_range_or_csv(text: str) -> List[float]:
    """Parse CSV like '0.3,0.4' or inclusive range '0.3:0.7:0.1'."""
    text = str(text or "").strip()
    if not text:
        return []
    if ":" in text and "," not in text:
        parts = [float(p.strip()) for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError("range form must be start:stop:step, e.g. 0.3:0.7:0.1")
        start, stop, step = parts
        if step == 0:
            raise ValueError("range step cannot be zero")
        vals: List[float] = []
        x = start
        if step > 0:
            while x <= stop + 1e-12:
                vals.append(round(float(x), 10))
                x += step
        else:
            while x >= stop - 1e-12:
                vals.append(round(float(x), 10))
                x += step
        return vals
    return [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]


def p_values_from_cfg(cfg: ExperimentConfig) -> List[float]:
    vals = parse_float_range_or_csv(cfg.p_values) if str(cfg.p_values).strip() else [float(cfg.p_fixed)]
    if not vals:
        vals = [float(cfg.p_fixed)]
    out: List[float] = []
    seen = set()
    for p_val in vals:
        p_float = float(p_val)
        if p_float <= 0 or p_float > 1:
            raise ValueError("Every p value must lie in (0, 1].")
        key = round(p_float, 12)
        if key not in seen:
            out.append(p_float)
            seen.add(key)
    return out


def is_sffa_grid_mode(cfg: ExperimentConfig) -> bool:
    return bool((cfg.sffa_grid or "").strip())


def update_dynamic_method_labels(cfg: ExperimentConfig | None = None) -> None:
    """Update user-facing labels for the current fair-comparison k/r setting."""
    if cfg is None:
        return
    k = int(cfg.sffa_distance_k)
    r = int(cfg.sffa_word_len)
    if "zero_baseline" in METHOD_SPECS:
        METHOD_SPECS["zero_baseline"]["label"] = "No processing (zero fill)"
    if "full_graph" in METHOD_SPECS:
        METHOD_SPECS["full_graph"]["label"] = "Full-graph numerical LMMSE"
    if "asymptotic_full_graph" in METHOD_SPECS:
        METHOD_SPECS["asymptotic_full_graph"]["label"] = "Asymptotic full-graph numerical LMMSE"
    if "subgraph_asymptotic_lmmse" in METHOD_SPECS:
        METHOD_SPECS["subgraph_asymptotic_lmmse"]["label"] = "Asymptotic subgraph numerical LMMSE"
    if "numerical_lmmse" in METHOD_SPECS:
        METHOD_SPECS["numerical_lmmse"]["label"] = "Subgraph numerical LMMSE"
    if "induced_poly" in METHOD_SPECS:
        METHOD_SPECS["induced_poly"]["label"] = f"Induced Laplacian polynomial (deg≤{r})"
    if "kron_poly" in METHOD_SPECS:
        METHOD_SPECS["kron_poly"]["label"] = f"Kron Laplacian polynomial (deg≤{r})"
    if "union_poly" in METHOD_SPECS:
        METHOD_SPECS["union_poly"]["label"] = f"1..{k}-union Laplacian polynomial (deg≤{r})"
    if "distance_laplacian_poly" in METHOD_SPECS:
        METHOD_SPECS["distance_laplacian_poly"]["label"] = f"Distance-1..{k} Laplacian polynomial (deg≤{r})"
    if "random_group_algebra" in METHOD_SPECS:
        METHOD_SPECS["random_group_algebra"]["label"] = f"Random-group algebra (q={k}, r={r})"
    if "pure_sffa" in METHOD_SPECS:
        METHOD_SPECS["pure_sffa"]["label"] = f"Pure distance-based SFFA (k={k}, r={r})"

    for name in list(METHOD_SPECS.keys()):
        degree = lim_variant_degree(name)
        if degree is None:
            continue
        base = lim_variant_base(name)
        if base == "induced_poly_lim":
            label = f"Induced Laplacian polynomial LIM (deg≤{degree})"
        elif base == "kron_poly_lim":
            label = f"Kron Laplacian polynomial LIM (deg≤{degree})"
        elif base == "union_poly_lim":
            label = f"1..{k}-union Laplacian polynomial LIM (deg≤{degree})"
        else:
            label = name
        METHOD_SPECS[name]["label"] = label


def register_dynamic_method(name: str, label: str, group: str = "learned") -> None:
    if name not in METHOD_SPECS:
        METHOD_SPECS[name] = {"enabled": True, "label": label, "group": group}
    else:
        METHOD_SPECS[name]["enabled"] = True
        METHOD_SPECS[name]["label"] = label
        METHOD_SPECS[name]["group"] = group


def max_sffa_k_under_budget(n0: int, r: int) -> int:
    """Largest primitive count k with SFFA basis count 1+k+...+k^r <= n0."""
    n0 = int(n0)
    r = int(r)
    if n0 < 1:
        raise ValueError("n0 must be positive for budgeted k selection.")
    if r <= 0:
        return 1
    best = 1
    k = 1
    while True:
        size = sffa_basis_size(k, r, include_identity=True)
        if size <= n0:
            best = k
            k += 1
            if k > max(1, n0):
                break
        else:
            break
    return int(best)


def unregister_grid_base_methods() -> None:
    """Deprecated no-op retained for old imports; grid mode is removed in this revision."""
    return None


@dataclass
class GraphSignalData:
    A_raw: np.ndarray
    A_graph: np.ndarray
    X_train: Tensor
    X_test: Tensor
    X_asymptotic: Tensor
    mean: np.ndarray
    std: np.ndarray
    train_size: int
    test_size: int
    asymptotic_size: int
    node_count: int


@dataclass
class LearnedOperator:
    name: str
    operator: Tensor
    train_mse_missing: float
    num_parameters: int


@dataclass
class ReconstructionResult:
    rmse_dict: Dict[str, float]
    mse_dict: Dict[str, float]
    rmse_std_dict: Dict[str, float] | None = None


@dataclass
class RunSummary:
    run_index: int
    seed: int
    idx0: List[int]
    obs_local: List[int]
    miss_local: List[int]
    obs_global: List[int]
    miss_global: List[int]
    learned_train_mse_missing: Dict[str, float]
    result: ReconstructionResult


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(requested: str) -> str:
    requested = requested.lower().strip()
    if requested not in {"cuda", "cpu"}:
        raise ValueError("device must be either 'cuda' or 'cpu'.")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warning] --device cuda was requested, but CUDA is not available. Falling back to CPU.")
        return "cpu"
    return requested


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = RUN_ROOT_DIR / p
    return p


def resolve_output_path(save_path: str | Path) -> Path:
    path = resolve_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_timestamped_paths(plot_path: str | Path, graph_plot_path: str | Path, timestamp: str) -> Dict[str, Path]:
    plot_base = resolve_output_path(plot_path)
    graph_base = resolve_output_path(graph_plot_path)
    return {
        "plot": plot_base.with_name(f"{plot_base.stem}_{timestamp}{plot_base.suffix}"),
        "graph_plot": graph_base.with_name(f"{graph_base.stem}_{timestamp}{graph_base.suffix}"),
        "json": plot_base.with_name(f"{plot_base.stem}_{timestamp}.json"),
    }


def root_relative_path(path: str | Path) -> str:
    p = Path(path)
    try:
        p_abs = p.resolve()
    except FileNotFoundError:
        p_abs = p.absolute()
    try:
        return str(p_abs.relative_to(PROJECT_ROOT_DIR))
    except ValueError:
        return p_abs.name if p_abs.is_absolute() else str(p)


def submatrix(M: Tensor, row_idx: Tensor, col_idx: Tensor) -> Tensor:
    return M.index_select(0, row_idx).index_select(1, col_idx)


def choose_random_subgraph_indices(N: int, p: float, generator: torch.Generator) -> Tensor:
    if p <= 0 or p > 1:
        raise ValueError("p_fixed must be in (0, 1].")
    n_sub = max(1, min(N, int(round(p * N))))
    idx = torch.randperm(N, generator=generator)[:n_sub]
    idx, _ = torch.sort(idx)
    return idx


def choose_missing_observed_local_indices(
    N: int,
    idx0: Tensor,
    p_missing: float,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor]:
    """Split V0 into observed/missing nodes when p_missing is |M0|/|V|.

    Earlier versions interpreted p_missing as |M0|/|V0|.  In this revision it is
    the hidden-node ratio in the original graph.  Therefore the requested hidden
    count is round(p_missing * N), while the sampled subgraph size is |V0|.  We
    the caller asserts p_missing <= p_fixed.  We also require at least one
    observed and one missing node after integer rounding.
    """
    n0 = int(idx0.numel())
    if n0 < 2:
        raise ValueError("Subgraph must contain at least 2 nodes to split observed and missing sets.")
    if p_missing <= 0 or p_missing >= 1:
        raise ValueError("p_missing/p1 must be in (0, 1).")

    m0 = int(round(float(p_missing) * int(N)))
    m0 = max(1, m0)
    if m0 >= n0:
        raise ValueError(
            f"Resolved hidden node count m0={m0} leaves no observed nodes in V0 (n0={n0}). "
            "Use a smaller --p_missing/--p1 or a larger --p/--p_fixed."
        )

    perm = torch.randperm(n0, generator=generator)
    miss_local = perm[:m0]
    obs_local = perm[m0:]
    miss_local, _ = torch.sort(miss_local)
    obs_local, _ = torch.sort(obs_local)
    return obs_local, miss_local


# -----------------------------------------------------------------------------
# Data interface
# -----------------------------------------------------------------------------


def load_numpy_array(path: Path, preferred_keys: Sequence[str], array_role: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {array_role} file: {root_relative_path(path)}")

    if path.suffix.lower() == ".npy":
        return np.load(path)

    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        keys = list(archive.keys())
        for key in preferred_keys:
            if key and key in archive:
                return archive[key]
        if len(keys) == 1:
            return archive[keys[0]]
        raise ValueError(
            f"{array_role} npz archive {root_relative_path(path)} contains multiple arrays {keys}; "
            f"set the corresponding key argument explicitly."
        )

    raise ValueError(f"Unsupported {array_role} file extension for {root_relative_path(path)}. Use .npy or .npz.")


def prepare_unweighted_adjacency(A_raw: np.ndarray) -> np.ndarray:
    A = np.asarray(A_raw, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency matrix must be N x N, got shape {A.shape}.")
    if np.isnan(A).any():
        raise ValueError("Adjacency matrix contains NaN.")

    A_bin = (A != 0).astype(np.float64)
    np.fill_diagonal(A_bin, 0.0)
    A_bin = np.logical_or(A_bin > 0, A_bin.T > 0).astype(np.float64)
    np.fill_diagonal(A_bin, 0.0)

    degrees = A_bin.sum(axis=1)
    if np.any(degrees <= 0):
        isolated = np.where(degrees <= 0)[0].tolist()
        print(f"[warning] adjacency contains isolated nodes: {isolated}")

    edge_count = int(np.count_nonzero(np.triu(A_bin, k=1)))
    total_possible = A_bin.shape[0] * (A_bin.shape[0] - 1) // 2
    print(f"[graph] unweighted undirected edges={edge_count}/{total_possible}")
    return A_bin



def symmetric_normalized_laplacian_np(A_np: np.ndarray) -> np.ndarray:
    """Build L_sym = I - D^{-1/2} A D^{-1/2} from an undirected adjacency."""
    A = np.asarray(A_np, dtype=np.float64)
    degree = A.sum(axis=1)
    if np.any(degree <= 0):
        isolated = np.where(degree <= 0)[0].tolist()
        raise ValueError(
            "Cannot run symmetric-normalized-Laplacian GFT with isolated nodes; "
            f"isolated nodes={isolated}."
        )
    inv_sqrt = 1.0 / np.sqrt(degree)
    L_sym = np.eye(A.shape[0], dtype=np.float64) - (inv_sqrt[:, None] * A * inv_sqrt[None, :])
    return 0.5 * (L_sym + L_sym.T)


def preprocess_signals_by_normalized_laplacian_spectrum(
    X_raw: np.ndarray,
    A_graph: np.ndarray,
    mode: str = "cutoff",
    cutoff: float = 1.0,
    k: int = 0,
    eig_tol: float = 1e-10,
    energy_tol: float = 1e-12,
) -> np.ndarray:
    """Optionally project T x N signals onto selected eigenspaces of L_sym.

    mode:
      - "none": no spectral preprocessing; return X_raw unchanged.
      - "cutoff": keep eigenspaces with eigenvalue <= cutoff.
      - "topk": keep the first k eigenvectors with the smallest eigenvalues.

    X_raw is T x N. For row-wise graph signals, the GFT coefficients are X_raw @ U,
    where L_sym = U diag(lambda) U^T.
    """
    mode = (mode or "cutoff").lower().strip()
    if mode not in {"none", "cutoff", "topk"}:
        raise ValueError("spectral_preprocess must be one of {'none', 'cutoff', 'topk'}.")

    if mode == "none":
        X_out = np.asarray(X_raw, dtype=np.float64)
        print("Spectral preprocessing")
        print("----------------------")
        print("mode                    : none")
        print("operation               : no spectral truncation; all node-space energy is retained")
        print()
        return X_out

    L_sym = symmetric_normalized_laplacian_np(A_graph)
    eigvals, U = np.linalg.eigh(L_sym)
    n_eigs = eigvals.size

    if mode == "cutoff":
        keep = eigvals <= (cutoff + eig_tol)
        selection_desc = f"lambda <= {cutoff:g}"
    else:  # mode == "topk"
        if k <= 0:
            raise ValueError("bandlimit_k must be positive when spectral_preprocess='topk'.")
        if k > n_eigs:
            raise ValueError(f"bandlimit_k={k} exceeds the number of nodes/eigenvalues {n_eigs}.")
        keep = np.zeros(n_eigs, dtype=bool)
        keep[: int(k)] = True
        selection_desc = f"first k={int(k)} eigenvectors with smallest eigenvalues"

    X_hat = X_raw @ U
    spectral_energy = np.sum(X_hat * X_hat, axis=0)
    total_energy = float(np.sum(spectral_energy))
    kept_energy = float(np.sum(spectral_energy[keep]))
    removed_energy = float(np.sum(spectral_energy[~keep]))
    retained_nonzero = int(np.count_nonzero(spectral_energy[keep] > energy_tol))
    removed_nonzero = int(np.count_nonzero(spectral_energy[~keep] > energy_tol))

    X_hat_band = X_hat.copy()
    X_hat_band[:, ~keep] = 0.0
    X_band = X_hat_band @ U.T
    X_band = np.asarray(X_band, dtype=np.float64)

    kept_count = int(np.count_nonzero(keep))
    kept_lambda_max = float(eigvals[keep][-1]) if kept_count > 0 else float("nan")

    print("Spectral preprocessing")
    print("----------------------")
    print("GFT operator             : symmetric normalized Laplacian")
    print(f"mode                     : {mode}")
    print(f"selection                : {selection_desc}")
    print(f"eigenvalues kept/total   : {kept_count} / {n_eigs}")
    print(f"largest kept eigenvalue  : {kept_lambda_max:.6e}")
    print(f"nonzero kept components  : {retained_nonzero}")
    print(f"nonzero removed comps    : {removed_nonzero}")
    if total_energy > 0:
        print(f"kept spectral energy     : {kept_energy:.6e} ({kept_energy / total_energy:.6%})")
        print(f"removed spectral energy  : {removed_energy:.6e} ({removed_energy / total_energy:.6%})")
    else:
        print("kept spectral energy     : 0.000000e+00 (total signal energy is zero)")
        print("removed spectral energy  : 0.000000e+00 (total signal energy is zero)")
    print(f"lambda min/max           : {eigvals[0]:.6e} / {eigvals[-1]:.6e}")
    print()

    return X_band


# Backward-compatible alias for older downstream imports.
def bandlimit_signals_by_normalized_laplacian(
    X_raw: np.ndarray,
    A_graph: np.ndarray,
    cutoff: float = 1.0,
    eig_tol: float = 1e-10,
    energy_tol: float = 1e-12,
) -> np.ndarray:
    return preprocess_signals_by_normalized_laplacian_spectrum(
        X_raw=X_raw,
        A_graph=A_graph,
        mode="cutoff",
        cutoff=cutoff,
        eig_tol=eig_tol,
        energy_tol=energy_tol,
    )

def load_graph_signal_data(cfg: ExperimentConfig) -> GraphSignalData:
    data_dir = resolve_path(cfg.data_dir)
    A_path = data_dir / cfg.adjacency_file
    X_path = data_dir / cfg.signal_file

    A_raw = load_numpy_array(A_path, preferred_keys=[cfg.adjacency_key, "A", "adjacency", "adj"], array_role="adjacency")
    X_raw = load_numpy_array(X_path, preferred_keys=[cfg.signal_key, "X", "X_clean", "data", "signals"], array_role="signal")

    A_raw = np.asarray(A_raw, dtype=np.float64)
    X_raw = np.asarray(X_raw, dtype=np.float64)

    if X_raw.ndim != 2:
        raise ValueError(f"Expected signal array to be T x N, got shape {X_raw.shape}.")
    if np.isnan(X_raw).any():
        raise ValueError("Signal array contains NaN.")

    time_count, node_count = X_raw.shape
    A_graph = prepare_unweighted_adjacency(A_raw)
    if A_graph.shape != (node_count, node_count):
        raise ValueError(f"Adjacency shape {A_graph.shape} is incompatible with signal shape {X_raw.shape}.")

    split_mode = str(cfg.split_mode or "ratio").lower().strip()
    if split_mode not in {"ratio", "number"}:
        raise ValueError("split_mode must be either 'ratio' or 'number'.")

    # Absolute counts imply number mode for backward compatibility.  In number
    # mode, the code uses the first train_size rows and then the immediately
    # following test_size rows.  In ratio mode, all available rows are split into
    # train/test blocks; train_ratio and test_ratio are normalized to sum to one.
    number_mode = split_mode == "number" or cfg.train_size is not None or cfg.test_size is not None

    if number_mode:
        if cfg.train_size is not None:
            if cfg.train_size <= 0:
                raise ValueError("train_size must be positive when provided.")
            train_size = int(cfg.train_size)
            train_split_desc = f"absolute train_size={train_size}"
        else:
            if cfg.train_ratio <= 0 or cfg.train_ratio >= 1:
                raise ValueError("train_ratio must be in (0, 1) when train_size is not provided.")
            train_size = int(round(float(cfg.train_ratio) * time_count))
            train_split_desc = f"number mode fallback train_ratio={cfg.train_ratio:g} -> train_size={train_size}"

        remaining = time_count - train_size
        if cfg.test_size is not None:
            if cfg.test_size <= 0:
                raise ValueError("test_size must be positive when provided.")
            test_size = int(cfg.test_size)
            test_split_desc = f"absolute test_size={test_size}"
        elif cfg.test_ratio is not None:
            if cfg.test_ratio <= 0:
                raise ValueError("test_ratio must be positive when provided.")
            test_size = int(round(float(cfg.test_ratio) * time_count))
            test_split_desc = f"number mode fallback test_ratio={cfg.test_ratio:g} -> test_size={test_size}"
        else:
            test_size = remaining
            test_split_desc = f"all remaining after train -> test_size={test_size}"

        if train_size <= 0 or train_size >= time_count:
            raise ValueError(f"train_size must be in [1, {time_count - 1}], got {train_size}.")
        if test_size <= 0:
            raise ValueError("Resolved test_size must be positive.")
        if test_size > remaining:
            raise ValueError(
                f"Requested number split uses too many time points: "
                f"train_size={train_size}, test_size={test_size}, total={time_count}."
            )
        split_mode_desc = "number"
    else:
        train_ratio = float(cfg.train_ratio)
        test_ratio = float(1.0 - train_ratio) if cfg.test_ratio is None else float(cfg.test_ratio)
        if train_ratio <= 0 or test_ratio <= 0:
            raise ValueError("train_ratio and test_ratio must be positive in ratio mode.")
        ratio_sum = train_ratio + test_ratio
        train_fraction = train_ratio / ratio_sum
        train_size = int(round(train_fraction * time_count))
        train_size = max(1, min(time_count - 1, train_size))
        test_size = time_count - train_size
        train_split_desc = f"ratio train={train_ratio:g}, test={test_ratio:g}, normalized train_fraction={train_fraction:g}"
        test_split_desc = f"all remaining after normalized ratio split -> test_size={test_size}"
        split_mode_desc = "ratio"

    used_size = train_size + test_size
    dropped_size = time_count - used_size

    # This variant intentionally performs no z-score normalization.
    # Optional spectral preprocessing is performed before any train/test split.
    # The asymptotic LMMSE baseline therefore sees the same preprocessed signal
    # distribution as the train/test experiment.
    if cfg.normalize_by_all or cfg.normalize_by_train:
        print("[warning] z-score normalization flags are ignored in this bandlimited variant.")

    X_all = preprocess_signals_by_normalized_laplacian_spectrum(
        X_raw=X_raw,
        A_graph=A_graph,
        mode=cfg.spectral_preprocess,
        cutoff=cfg.bandlimit_cutoff,
        k=cfg.bandlimit_k,
    )
    X_used_raw = X_all[:used_size]
    X_train_raw = X_used_raw[:train_size]
    X_test_raw = X_used_raw[train_size:used_size]

    mean = np.zeros((1, node_count), dtype=np.float64)
    std = np.ones((1, node_count), dtype=np.float64)
    X_train = X_train_raw
    X_test = X_test_raw
    if cfg.spectral_preprocess == "none":
        normalization_mode = "none; no spectral preprocessing"
    elif cfg.spectral_preprocess == "cutoff":
        normalization_mode = f"none; preprocessed by L_sym GFT cutoff lambda<={cfg.bandlimit_cutoff:g}"
    elif cfg.spectral_preprocess == "topk":
        normalization_mode = f"none; preprocessed by L_sym GFT top-k k={cfg.bandlimit_k}"
    else:
        raise ValueError("Unexpected spectral_preprocess mode.")

    print("Graph signal data loaded")
    print("------------------------")
    print(f"data dir               : {root_relative_path(data_dir)}")
    print(f"adjacency file         : {root_relative_path(A_path)}")
    print(f"signal file            : {root_relative_path(X_path)}")
    print(f"A graph shape          : {A_graph.shape}  (N x N, unweighted)")
    print(f"signal shape           : {X_raw.shape}  (T x N)")
    print(f"train/test split       : {X_train.shape} / {X_test.shape}")
    print(f"split mode             : {split_mode_desc}")
    print(f"train split mode       : {train_split_desc}")
    print(f"test split mode        : {test_split_desc}")
    print(f"used/dropped time pts  : {used_size} / {dropped_size}")
    print(f"asymptotic oracle pts  : {X_all.shape[0]}")
    print(f"normalization mode     : {normalization_mode}")
    print(f"X_train mean/std       : {X_train.mean():.6f} / {X_train.std():.6f}")
    print(f"X_test mean/std        : {X_test.mean():.6f} / {X_test.std():.6f}")
    print(f"X_asym mean/std        : {X_all.mean():.6f} / {X_all.std():.6f}")
    print()

    return GraphSignalData(
        A_raw=A_raw,
        A_graph=A_graph,
        X_train=torch.tensor(X_train, dtype=DTYPE, device=DEVICE),
        X_test=torch.tensor(X_test, dtype=DTYPE, device=DEVICE),
        X_asymptotic=torch.tensor(X_all, dtype=DTYPE, device=DEVICE),
        mean=mean,
        std=std,
        train_size=train_size,
        test_size=test_size,
        asymptotic_size=int(X_all.shape[0]),
        node_count=node_count,
    )


# -----------------------------------------------------------------------------
# Graph/operator construction
# -----------------------------------------------------------------------------


def build_graph_laplacian_from_adjacency(A_np: np.ndarray, use_normalized_laplacian: bool) -> Tuple[Tensor, Tensor, nx.Graph]:
    A_np = np.asarray(A_np, dtype=np.float64)
    A = torch.tensor(A_np, dtype=DTYPE, device=DEVICE)
    L = laplacian_from_adjacency(
        A,
        use_normalized_laplacian=use_normalized_laplacian,
        zero_isolated=False,
        fail_on_isolated=bool(use_normalized_laplacian),
        context="full graph",
    )

    G_nx = nx.from_numpy_array((A_np > 0).astype(np.float64))
    return L, A, G_nx


def laplacian_from_adjacency(
    A: Tensor,
    use_normalized_laplacian: bool = False,
    zero_isolated: bool = True,
    fail_on_isolated: bool = False,
    context: str = "graph",
) -> Tensor:
    """Build either a combinatorial or symmetric-normalized Laplacian.

    For subgraph-derived primitives we default to zero_isolated=True.  This keeps
    isolated rows/columns as zero in normalized distance-k and random-group
    primitives, preserving the old padded-subgraph semantics as closely as
    possible.  For the full graph we keep the previous stricter behavior by
    setting fail_on_isolated=True when --use_normalized_laplacian is enabled.
    """
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency for {context} must be square, got shape {tuple(A.shape)}.")

    A = A.clone()
    A.fill_diagonal_(0.0)
    A = torch.maximum(A, A.T)
    degree = A.sum(dim=1)

    if not use_normalized_laplacian:
        return torch.diag(degree) - A

    isolated_mask = degree <= 0
    if fail_on_isolated and torch.any(isolated_mask):
        isolated = torch.where(isolated_mask)[0].detach().cpu().tolist()
        raise ValueError(f"Normalized Laplacian cannot be built for {context} with isolated nodes: {isolated}")

    inv_sqrt = torch.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = torch.rsqrt(degree[nonzero])
    normalized_adj = inv_sqrt[:, None] * A * inv_sqrt[None, :]

    diag = torch.ones_like(degree)
    if zero_isolated:
        diag = torch.where(nonzero, diag, torch.zeros_like(diag))
    L = torch.diag(diag) - normalized_adj
    return 0.5 * (L + L.T)


def combinatorial_laplacian_from_adjacency(A: Tensor) -> Tensor:
    return laplacian_from_adjacency(A, use_normalized_laplacian=False)


def induced_subgraph_laplacian(A: Tensor, idx0: Tensor, use_normalized_laplacian: bool = False) -> Tensor:
    A_sub = submatrix(A, idx0, idx0)
    return laplacian_from_adjacency(
        A_sub,
        use_normalized_laplacian=use_normalized_laplacian,
        zero_isolated=True,
        context="induced subgraph primitive",
    )


def truncated_laplacian(L: Tensor, idx0: Tensor) -> Tensor:
    return submatrix(L, idx0, idx0)


def kron_laplacian(L: Tensor, idx0: Tensor, ridge: float) -> Tensor:
    N = L.shape[0]
    mask = torch.ones(N, dtype=torch.bool, device=L.device)
    mask[idx0] = False
    idxc = torch.arange(N, device=L.device)[mask]
    L00 = submatrix(L, idx0, idx0)
    if idxc.numel() == 0:
        return L00
    L0c = submatrix(L, idx0, idxc)
    Lc0 = submatrix(L, idxc, idx0)
    Lcc = submatrix(L, idxc, idxc)
    eye_c = torch.eye(Lcc.shape[0], dtype=L.dtype, device=L.device)
    # Ridge-stabilized Schur complement. When --use_normalized_laplacian is set,
    # this is the Schur complement of the full symmetric-normalized Laplacian.
    sol = torch.linalg.solve(Lcc + ridge * eye_c, Lc0)
    K = L00 - L0c @ sol
    return 0.5 * (K + K.T)


def distance_laplacians_from_full_graph(
    G_nx: nx.Graph,
    idx0: Tensor,
    max_k: int,
    device: torch.device,
    dtype: torch.dtype,
    use_normalized_laplacian: bool = False,
) -> Dict[str, Tensor]:
    """Build exact distance-k Laplacians on V0, where distances are measured in the full graph.

    The algebraic bases are still constructed in the full V0 space. This function only
    generalizes the previous hard-coded k in {1,2,3} to k in {1,...,max_k}.
    """
    if max_k < 1:
        raise ValueError("sffa_distance_k must be >= 1.")

    nodes = [int(v) for v in idx0.detach().cpu().tolist()]
    n0 = len(nodes)
    node_to_local = {v: i for i, v in enumerate(nodes)}
    node_set = set(nodes)

    # Use CPU numpy for incremental filling; it is faster than many small torch writes.
    A_by_k_np = {k: np.zeros((n0, n0), dtype=np.float64) for k in range(1, max_k + 1)}

    for u in nodes:
        lengths = nx.single_source_shortest_path_length(G_nx, u, cutoff=max_k)
        i = node_to_local[u]
        for v, dist in lengths.items():
            if v in node_set and v != u and 1 <= int(dist) <= max_k:
                j = node_to_local[v]
                A_by_k_np[int(dist)][i, j] = 1.0

    out: Dict[str, Tensor] = {}
    for k in range(1, max_k + 1):
        A_k_np = np.maximum(A_by_k_np[k], A_by_k_np[k].T)
        A_k = torch.tensor(A_k_np, dtype=dtype, device=device)
        out[f"distance_{k}_laplacian"] = laplacian_from_adjacency(
            A_k,
            use_normalized_laplacian=use_normalized_laplacian,
            zero_isolated=True,
            context=f"distance-{k} subgraph primitive",
        )
    return out


def union_laplacian_from_distance_ops(
    distance_ops: Dict[str, Tensor],
    max_k: int,
    use_normalized_laplacian: bool = False,
) -> Tensor:
    # Reconstruct union adjacency from Laplacian off-diagonal signs.
    A_union = None
    for k in range(1, max_k + 1):
        key = f"distance_{k}_laplacian"
        Lk = distance_ops[key]
        Ak = (Lk < 0).to(dtype=Lk.dtype, device=Lk.device)
        Ak = Ak - torch.diag(torch.diag(Ak))
        A_union = Ak if A_union is None else torch.maximum(A_union, Ak)
    assert A_union is not None
    return laplacian_from_adjacency(
        A_union,
        use_normalized_laplacian=use_normalized_laplacian,
        zero_isolated=True,
        context="union distance subgraph primitive",
    )


def random_group_laplacians_from_full_graph(
    G_nx: nx.Graph,
    idx0: Tensor,
    num_groups: int,
    group_ratio: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    use_normalized_laplacian: bool = False,
) -> List[Tensor]:
    """Build random induced sub-subgraph Laplacians embedded in V0 coordinates."""
    if int(num_groups) < 1:
        raise ValueError("num_groups/k must be >= 1.")
    if not (0.0 < float(group_ratio) <= 1.0):
        raise ValueError("random_group_ratio must lie in (0, 1].")

    nodes = [int(v) for v in idx0.detach().cpu().tolist()]
    n0 = len(nodes)
    node_to_local = {v: i for i, v in enumerate(nodes)}
    group_size = max(1, min(n0, int(round(float(group_ratio) * n0))))
    out: List[Tensor] = []

    for _ in range(int(num_groups)):
        perm = torch.randperm(n0, generator=generator)[:group_size].tolist()
        chosen_nodes = [nodes[int(i)] for i in perm]
        A_group_np = np.zeros((n0, n0), dtype=np.float64)
        for u, v in G_nx.subgraph(chosen_nodes).edges():
            i = node_to_local[int(u)]
            j = node_to_local[int(v)]
            if i == j:
                continue
            A_group_np[i, j] = 1.0
            A_group_np[j, i] = 1.0
        A_group = torch.tensor(A_group_np, dtype=dtype, device=device)
        out.append(
            laplacian_from_adjacency(
                A_group,
                use_normalized_laplacian=use_normalized_laplacian,
                zero_isolated=True,
                context="random-group subgraph primitive",
            )
        )
    return out


def build_filter_primitives(
    L: Tensor,
    A: Tensor,
    G_nx: nx.Graph,
    idx0: Tensor,
    ridge: float,
    max_distance_k: int,
    num_random_groups: int,
    random_group_ratio: float,
    generator: torch.Generator,
    use_normalized_laplacian: bool = False,
) -> Dict[str, Tensor]:
    L_induced = induced_subgraph_laplacian(A, idx0, use_normalized_laplacian=use_normalized_laplacian)
    L_kron = kron_laplacian(L, idx0, ridge=ridge)
    distance_ops = distance_laplacians_from_full_graph(
        G_nx,
        idx0,
        max_k=max_distance_k,
        device=L.device,
        dtype=L.dtype,
        use_normalized_laplacian=use_normalized_laplacian,
    )
    L_union = union_laplacian_from_distance_ops(
        distance_ops,
        max_k=max_distance_k,
        use_normalized_laplacian=use_normalized_laplacian,
    )
    random_group_ops = random_group_laplacians_from_full_graph(
        G_nx=G_nx,
        idx0=idx0,
        num_groups=num_random_groups,
        group_ratio=random_group_ratio,
        generator=generator,
        device=L.device,
        dtype=L.dtype,
        use_normalized_laplacian=use_normalized_laplacian,
    )

    ops: Dict[str, Tensor] = {
        "induced_laplacian": L_induced,
        "kron_laplacian": L_kron,
        "union_laplacian": L_union,
    }
    ops.update(distance_ops)
    for i, Lg in enumerate(random_group_ops, start=1):
        ops[f"random_group_{i}_laplacian"] = Lg
    return ops


# -----------------------------------------------------------------------------
# Reconstruction targets and numerical LMMSE
# -----------------------------------------------------------------------------


def make_subgraph_observed_input(X_sub: Tensor, obs_local: Tensor) -> Tensor:
    X_obs = torch.zeros_like(X_sub)
    X_obs[:, obs_local] = X_sub[:, obs_local]
    return X_obs


def ridge_regression_map(Y: Tensor, X: Tensor, ridge: float) -> Tensor:
    """Return F minimizing ||Y - X F^T||_F^2 + n*ridge*||F||_F^2.

    X is samples x input_dim; Y is samples x output_dim.
    Returned F has shape output_dim x input_dim so prediction is X @ F.T.
    """
    n = X.shape[0]
    input_dim = X.shape[1]
    gram = X.T @ X
    eye = torch.eye(input_dim, dtype=X.dtype, device=X.device)
    return Y.T @ X @ torch.linalg.solve(gram + n * ridge * eye, eye)


def subgraph_numerical_lmmse_operator(X_train_sub: Tensor, obs_local: Tensor, miss_local: Tensor, ridge: float) -> Tensor:
    """Ridge LMMSE for the task-relevant block M0 <- O0.

    This is equivalent to the zero-padded full-N0 formulation on the missing output block,
    but avoids solving an N0-dimensional ridge system with many structurally zero columns.
    The returned matrix is embedded back into N0 x N0 for downstream compatibility.
    """
    X_obs = X_train_sub[:, obs_local]
    Y_miss = X_train_sub[:, miss_local]
    H_mo = ridge_regression_map(Y=Y_miss, X=X_obs, ridge=ridge)

    n0 = X_train_sub.shape[1]
    H_full = torch.zeros((n0, n0), dtype=X_train_sub.dtype, device=X_train_sub.device)
    H_full[miss_local[:, None], obs_local[None, :]] = H_mo
    return H_full


def full_graph_numerical_lmmse_predict(
    X_train_full: Tensor,
    X_test_full: Tensor,
    miss_global: Tensor,
    ridge: float,
) -> Tensor:
    """Unreachable full-graph baseline.

    It predicts M0 using all test-time visible full-graph nodes V minus M0.
    Returned tensor has shape test_samples x |M0|.
    """
    N = X_train_full.shape[1]
    visible_mask = torch.ones(N, dtype=torch.bool, device=X_train_full.device)
    visible_mask[miss_global] = False
    visible_global = torch.arange(N, device=X_train_full.device)[visible_mask]

    X_train_visible = X_train_full[:, visible_global]
    Y_train_missing = X_train_full[:, miss_global]
    F_miss_from_visible = ridge_regression_map(Y=Y_train_missing, X=X_train_visible, ridge=ridge)
    return X_test_full[:, visible_global] @ F_miss_from_visible.T


def mse_rmse_on_missing(pred_missing: Tensor, target_missing: Tensor) -> Tuple[float, float]:
    err = pred_missing - target_missing
    mse = torch.mean(err * err).item()
    return float(mse), float(math.sqrt(max(mse, 0.0)))


# -----------------------------------------------------------------------------
# Learned filter-bank models
# -----------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_operator(S: Tensor) -> Tensor:
    norm = torch.linalg.norm(S, ord="fro")
    if torch.isfinite(norm) and norm.item() > 0:
        return S / norm
    return S


def polynomial_basis(S: Tensor, degree: int = 3) -> List[Tensor]:
    n = S.shape[0]
    I = torch.eye(n, dtype=S.dtype, device=S.device)
    bases = [I]
    current = I
    S_norm = normalize_operator(S)
    for _ in range(1, degree + 1):
        current = current @ S_norm
        bases.append(current)
    return bases


def multi_laplacian_polynomial_basis(primitives: Sequence[Tensor], degree: int = 3) -> List[Tensor]:
    """Return I plus powers 1..degree of each primitive, sharing one identity term."""
    if not primitives:
        raise ValueError("primitives must be nonempty.")
    if degree < 0:
        raise ValueError("degree/r must be non-negative.")
    n = primitives[0].shape[0]
    bases = [torch.eye(n, dtype=primitives[0].dtype, device=primitives[0].device)]
    for S in primitives:
        S_norm = normalize_operator(S)
        current = torch.eye(n, dtype=S.dtype, device=S.device)
        for _ in range(1, int(degree) + 1):
            current = current @ S_norm
            bases.append(current)
    return bases


def sffa_basis_size(num_primitives: int, max_word_len: int, include_identity: bool = True) -> int:
    """Return the number of SFFA word bases without constructing them."""
    if num_primitives <= 0:
        raise ValueError("num_primitives must be positive.")
    if max_word_len < 0:
        raise ValueError("max_word_len must be non-negative.")

    if num_primitives == 1:
        total = max_word_len + 1 if include_identity else max_word_len
    else:
        if include_identity:
            total = (num_primitives ** (max_word_len + 1) - 1) // (num_primitives - 1)
        else:
            total = (num_primitives ** (max_word_len + 1) - num_primitives) // (num_primitives - 1)
    return int(total)


def guard_sffa_basis_size(
    method_name: str,
    num_primitives: int,
    max_word_len: int,
    max_sffa_basis_size: int,
    include_identity: bool = True,
) -> int:
    basis_size = sffa_basis_size(num_primitives, max_word_len, include_identity=include_identity)
    if max_sffa_basis_size > 0 and basis_size > max_sffa_basis_size:
        raise RuntimeError(
            f"[SFFA guard] {method_name} would construct {basis_size} bases "
            f"with q={num_primitives}, r={max_word_len}, include_identity={include_identity}. "
            f"This exceeds --max_sffa_basis_size={max_sffa_basis_size}. "
            "Abort before materializing full V0 x V0 word matrices."
        )
    print(
        f"[SFFA guard] {method_name}: q={num_primitives}, r={max_word_len}, "
        f"basis_size={basis_size}, threshold={max_sffa_basis_size}"
    )
    return basis_size


def sffa_word_basis(primitives: Sequence[Tensor], max_word_len: int = 3, include_identity: bool = True) -> List[Tensor]:
    """Enumerate noncommutative word bases up to a configurable word length.

    For q primitives and length r, the number of bases is
        1 + q + q^2 + ... + q^r
    when include_identity=True. Products are always formed in the full V0 x V0
    space; reconstruction-specific M0 <- O0 slicing is applied only after the
    full word matrices have been constructed.
    """
    if not primitives:
        raise ValueError("At least one primitive is required for SFFA.")
    if max_word_len < 0:
        raise ValueError("sffa_word_len must be non-negative.")

    prims = [normalize_operator(P) for P in primitives]
    n = prims[0].shape[0]
    bases: List[Tensor] = []
    if include_identity:
        bases.append(torch.eye(n, dtype=prims[0].dtype, device=prims[0].device))

    current_words: List[Tensor] = [torch.eye(n, dtype=prims[0].dtype, device=prims[0].device)]
    for _length in range(1, max_word_len + 1):
        next_words: List[Tensor] = []
        for B_left in current_words:
            for Pj in prims:
                next_words.append(B_left @ Pj)
        bases.extend(next_words)
        current_words = next_words
    return bases


class LinearFilterBank(torch.nn.Module):
    def __init__(self, bases: Sequence[Tensor], init_scale: float = 1e-2):
        super().__init__()
        if not bases:
            raise ValueError("bases must be nonempty.")
        B = torch.stack([b.detach().clone() for b in bases], dim=0)
        self.register_buffer("bases", B)
        self.theta = torch.nn.Parameter(init_scale * torch.randn(B.shape[0], dtype=B.dtype, device=B.device))

    def get_filter_matrix(self) -> Tensor:
        return torch.einsum("m,mij->ij", self.theta, self.bases)

    def forward(self, X: Tensor) -> Tensor:
        H = self.get_filter_matrix()
        return X @ H.T


class RectangularLinearFilterBank(torch.nn.Module):
    """Reconstruction-specific filter bank using only the M0 <- O0 block.

    The full algebraic/polynomial bases are constructed before this class is called.
    This class then slices each basis to its task-relevant block, which preserves the
    intended algebra while avoiding useless N0 x N0 forward passes during training.
    """
    def __init__(
        self,
        bases_full: Sequence[Tensor],
        obs_local: Tensor,
        miss_local: Tensor,
        init_scale: float = 1e-2,
    ):
        super().__init__()
        if not bases_full:
            raise ValueError("bases_full must be nonempty.")
        blocks = [submatrix(B, miss_local, obs_local).detach().clone() for B in bases_full]
        B = torch.stack(blocks, dim=0)  # [num_basis, |M0|, |O0|]
        self.register_buffer("bases", B)
        self.theta = torch.nn.Parameter(init_scale * torch.randn(B.shape[0], dtype=B.dtype, device=B.device))

    def get_block_matrix(self) -> Tensor:
        return torch.einsum("m,mij->ij", self.theta, self.bases)

    def forward(self, X_obs_values: Tensor) -> Tensor:
        H_mo = self.get_block_matrix()
        return X_obs_values @ H_mo.T

    def get_filter_matrix(self, n0: int, obs_local: Tensor, miss_local: Tensor) -> Tensor:
        H_full = torch.zeros((n0, n0), dtype=self.bases.dtype, device=self.bases.device)
        H_full[miss_local[:, None], obs_local[None, :]] = self.get_block_matrix()
        return H_full


def train_operator_model(
    name: str,
    model: torch.nn.Module,
    X_train_obs: Tensor,
    X_train_sub: Tensor,
    miss_local: Tensor,
    epochs: int,
    lr: float,
    ridge: float,
) -> LearnedOperator:
    # Kept for backward compatibility; the reconstruction code below uses the
    # faster block-only training routine.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target_missing = X_train_sub[:, miss_local]

    for _ in range(epochs):
        pred = model(X_train_obs)
        pred_missing = pred[:, miss_local]
        fit_loss = torch.mean((pred_missing - target_missing) ** 2)
        H = model.get_filter_matrix()
        reg_loss = ridge * torch.mean(H * H)
        loss = fit_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    with torch.no_grad():
        H = model.get_filter_matrix().detach().clone()
        pred_missing = model(X_train_obs)[:, miss_local]
        train_mse_missing = torch.mean((pred_missing - target_missing) ** 2).item()

    return LearnedOperator(
        name=name,
        operator=H,
        train_mse_missing=float(train_mse_missing),
        num_parameters=count_parameters(model),
    )


def train_rectangular_operator_model(
    name: str,
    model: RectangularLinearFilterBank,
    X_train_sub: Tensor,
    obs_local: Tensor,
    miss_local: Tensor,
    epochs: int,
    lr: float,
    ridge: float,
    optimizer_name: str = "lbfgs",
    lbfgs_max_iter: int = 20,
) -> LearnedOperator:
    """Train only the reconstruction-relevant M0 <- O0 block.

    This gives the same missing-node predictions as training a masked full matrix
    built from the same full-space bases, while avoiding N0 x N0 forward products.
    """
    X_obs_values = X_train_sub[:, obs_local]
    target_missing = X_train_sub[:, miss_local]

    def compute_loss() -> Tensor:
        pred_missing = model(X_obs_values)
        fit_loss = torch.mean((pred_missing - target_missing) ** 2)
        H_mo = model.get_block_matrix()
        reg_loss = ridge * torch.mean(H_mo * H_mo)
        return fit_loss + reg_loss

    optimizer_name = optimizer_name.lower().strip()

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=lbfgs_max_iter,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            return loss

        for _ in range(epochs):
            optimizer.step(closure)

    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            loss = compute_loss()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    else:
        raise ValueError("optimizer_name must be either 'adam' or 'lbfgs'.")

    with torch.no_grad():
        pred_missing = model(X_obs_values)
        train_mse_missing = torch.mean((pred_missing - target_missing) ** 2).item()
        H_full = model.get_filter_matrix(X_train_sub.shape[1], obs_local, miss_local).detach().clone()

    return LearnedOperator(
        name=name,
        operator=H_full,
        train_mse_missing=float(train_mse_missing),
        num_parameters=count_parameters(model),
    )


def build_learned_operators(
    primitives: Dict[str, Tensor],
    X_train_sub: Tensor,
    obs_local: Tensor,
    miss_local: Tensor,
    k: int,
    r: int,
    rmax: int,
    max_sffa_basis_size: int,
    epochs: int,
    lr: float,
    ridge: float,
    optimizer_name: str,
    lbfgs_max_iter: int,
) -> Dict[str, LearnedOperator]:
    learned: Dict[str, LearnedOperator] = {}
    k = int(k)
    r = int(r)
    rmax = int(rmax)
    if k < 1:
        raise ValueError("k/sffa_distance_k must be >= 1.")
    if r < 0:
        raise ValueError("r/sffa_word_len must be >= 0.")

    distance_prims = [primitives[f"distance_{kk}_laplacian"] for kk in range(1, k + 1)]
    random_group_prims = [primitives[f"random_group_{kk}_laplacian"] for kk in range(1, k + 1)]

    def train_from_full_bases(name: str, bases: Sequence[Tensor]) -> None:
        model = RectangularLinearFilterBank(
            bases_full=bases,
            obs_local=obs_local,
            miss_local=miss_local,
        )
        learned[name] = train_rectangular_operator_model(
            name=name,
            model=model,
            X_train_sub=X_train_sub,
            obs_local=obs_local,
            miss_local=miss_local,
            epochs=epochs,
            lr=lr,
            ridge=ridge,
            optimizer_name=optimizer_name,
            lbfgs_max_iter=lbfgs_max_iter,
        )

    if METHOD_SPECS["induced_poly"]["enabled"]:
        train_from_full_bases(
            "induced_poly",
            polynomial_basis(primitives["induced_laplacian"], degree=r),
        )

    for lim_name, lim_r in iter_enabled_lim_variants("induced_poly_lim"):
        train_from_full_bases(
            lim_name,
            polynomial_basis(primitives["induced_laplacian"], degree=lim_r),
        )

    if METHOD_SPECS["kron_poly"]["enabled"]:
        train_from_full_bases(
            "kron_poly",
            polynomial_basis(primitives["kron_laplacian"], degree=r),
        )

    for lim_name, lim_r in iter_enabled_lim_variants("kron_poly_lim"):
        train_from_full_bases(
            lim_name,
            polynomial_basis(primitives["kron_laplacian"], degree=lim_r),
        )

    L_union_for_poly: Tensor | None = None
    if METHOD_SPECS["union_poly"]["enabled"] or iter_enabled_lim_variants("union_poly_lim"):
        # Reuse the primitive constructed in build_filter_primitives so that
        # --use_normalized_laplacian applies consistently to union_poly and
        # union_poly_lim as well.
        L_union_for_poly = primitives["union_laplacian"]

    if METHOD_SPECS["union_poly"]["enabled"]:
        assert L_union_for_poly is not None
        train_from_full_bases(
            "union_poly",
            polynomial_basis(L_union_for_poly, degree=r),
        )

    for lim_name, lim_r in iter_enabled_lim_variants("union_poly_lim"):
        assert L_union_for_poly is not None
        train_from_full_bases(
            lim_name,
            polynomial_basis(L_union_for_poly, degree=lim_r),
        )

    if METHOD_SPECS["distance_laplacian_poly"]["enabled"]:
        train_from_full_bases(
            "distance_laplacian_poly",
            multi_laplacian_polynomial_basis(distance_prims, degree=r),
        )

    if METHOD_SPECS["random_group_algebra"]["enabled"]:
        guard_sffa_basis_size(
            method_name="random_group_algebra",
            num_primitives=len(random_group_prims),
            max_word_len=r,
            max_sffa_basis_size=max_sffa_basis_size,
            include_identity=True,
        )
        train_from_full_bases(
            "random_group_algebra",
            sffa_word_basis(random_group_prims, max_word_len=r, include_identity=True),
        )

    if METHOD_SPECS["pure_sffa"]["enabled"]:
        guard_sffa_basis_size(
            method_name="pure_sffa",
            num_primitives=len(distance_prims),
            max_word_len=r,
            max_sffa_basis_size=max_sffa_basis_size,
            include_identity=True,
        )
        train_from_full_bases(
            "pure_sffa",
            sffa_word_basis(distance_prims, max_word_len=r, include_identity=True),
        )

    return learned


# -----------------------------------------------------------------------------
# Evaluation, plotting, records
# -----------------------------------------------------------------------------


def evaluate_reconstruction_methods(
    cfg: ExperimentConfig,
    dataset: GraphSignalData,
    learned_ops: Dict[str, LearnedOperator],
    subgraph_lmmse: Tensor,
    subgraph_asymptotic_lmmse: Tensor,
    idx0: Tensor,
    obs_local: Tensor,
    miss_local: Tensor,
    miss_global: Tensor,
) -> ReconstructionResult:
    X_test_sub = dataset.X_test[:, idx0]
    X_test_obs = make_subgraph_observed_input(X_test_sub, obs_local)
    target_missing = X_test_sub[:, miss_local]

    mse_dict: Dict[str, float] = {}
    rmse_dict: Dict[str, float] = {}

    if METHOD_SPECS["zero_baseline"]["enabled"]:
        pred_zero_missing = torch.zeros_like(target_missing)
        mse, rmse = mse_rmse_on_missing(pred_zero_missing, target_missing)
        mse_dict["zero_baseline"] = mse
        rmse_dict["zero_baseline"] = rmse

    if METHOD_SPECS.get("full_graph", {}).get("enabled", False):
        pred_full_missing = full_graph_numerical_lmmse_predict(
            X_train_full=dataset.X_train,
            X_test_full=dataset.X_test,
            miss_global=miss_global,
            ridge=cfg.ridge,
        )
        mse, rmse = mse_rmse_on_missing(pred_full_missing, target_missing)
        mse_dict["full_graph"] = mse
        rmse_dict["full_graph"] = rmse

    if METHOD_SPECS["asymptotic_full_graph"]["enabled"]:
        pred_asym_missing = full_graph_numerical_lmmse_predict(
            X_train_full=dataset.X_asymptotic,
            X_test_full=dataset.X_test,
            miss_global=miss_global,
            ridge=cfg.ridge,
        )
        mse, rmse = mse_rmse_on_missing(pred_asym_missing, target_missing)
        mse_dict["asymptotic_full_graph"] = mse
        rmse_dict["asymptotic_full_graph"] = rmse

    if METHOD_SPECS["subgraph_asymptotic_lmmse"]["enabled"]:
        pred_sub_asym = X_test_obs @ subgraph_asymptotic_lmmse.T
        mse, rmse = mse_rmse_on_missing(pred_sub_asym[:, miss_local], target_missing)
        mse_dict["subgraph_asymptotic_lmmse"] = mse
        rmse_dict["subgraph_asymptotic_lmmse"] = rmse

    if METHOD_SPECS["numerical_lmmse"]["enabled"]:
        pred_sub = X_test_obs @ subgraph_lmmse.T
        mse, rmse = mse_rmse_on_missing(pred_sub[:, miss_local], target_missing)
        mse_dict["numerical_lmmse"] = mse
        rmse_dict["numerical_lmmse"] = rmse

    for name, obj in learned_ops.items():
        if not METHOD_SPECS[name]["enabled"]:
            continue
        pred = X_test_obs @ obj.operator.T
        mse, rmse = mse_rmse_on_missing(pred[:, miss_local], target_missing)
        mse_dict[name] = mse
        rmse_dict[name] = rmse

    return ReconstructionResult(rmse_dict=rmse_dict, mse_dict=mse_dict)


def average_results(results: List[ReconstructionResult]) -> ReconstructionResult:
    if not results:
        raise ValueError("results must be nonempty.")
    method_order = get_enabled_methods()
    mse_dict: Dict[str, float] = {}
    rmse_dict: Dict[str, float] = {}
    rmse_std_dict: Dict[str, float] = {}
    for method in method_order:
        vals_mse = [r.mse_dict[method] for r in results if method in r.mse_dict]
        vals_rmse = [r.rmse_dict[method] for r in results if method in r.rmse_dict]
        if not vals_mse:
            continue
        mse_dict[method] = float(np.mean(vals_mse))
        rmse_dict[method] = float(np.mean(vals_rmse))
        rmse_std_dict[method] = float(np.std(vals_rmse, ddof=1)) if len(vals_rmse) > 1 else 0.0
    return ReconstructionResult(mse_dict=mse_dict, rmse_dict=rmse_dict, rmse_std_dict=rmse_std_dict)


def format_mean_std(mean: float, std: float | None, width: int = 24) -> str:
    if std is None:
        return f"{mean:.4f}".rjust(width)
    return f"{mean:.4f}±{std:.4f}".rjust(width)


def print_result_table(result: ReconstructionResult) -> None:
    col_width = 26
    print("=== Averaged test RMSE on missing nodes ===")
    print("method".ljust(36) + " | " + "RMSE".rjust(col_width) + " | " + "MSE".rjust(col_width))
    print("-" * (36 + 3 + col_width + 3 + col_width))
    for name in get_enabled_methods():
        if name not in result.rmse_dict:
            continue
        label = METHOD_SPECS[name]["label"]
        std = result.rmse_std_dict.get(name) if result.rmse_std_dict is not None else None
        print(label.ljust(36) + " | " + format_mean_std(result.rmse_dict[name], std, col_width) + " | " + f"{result.mse_dict[name]:.6e}".rjust(col_width))


def plot_results(result: ReconstructionResult, save_path: str | Path) -> None:
    path = resolve_output_path(save_path)
    names = [name for name in get_enabled_methods() if name in result.rmse_dict]
    labels = [METHOD_SPECS[name]["label"] for name in names]
    values = [result.rmse_dict[name] for name in names]
    x = np.arange(len(names))

    plt.figure(figsize=(max(8, 1.3 * len(names)), 5))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("RMSE on missing nodes")
    plt.title("SFL signal reconstruction")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_graph_subgraph(A_np: np.ndarray, idx0: Tensor, obs_local: Tensor, miss_local: Tensor, save_path: str | Path) -> None:
    path = resolve_output_path(save_path)
    G = nx.from_numpy_array((A_np > 0).astype(np.float64))
    pos = nx.spring_layout(G, seed=0)

    idx0_cpu = idx0.detach().cpu().numpy()
    obs_global = idx0[obs_local].detach().cpu().numpy()
    miss_global = idx0[miss_local].detach().cpu().numpy()
    sub_set = set(int(v) for v in idx0_cpu.tolist())
    obs_set = set(int(v) for v in obs_global.tolist())
    miss_set = set(int(v) for v in miss_global.tolist())
    other_nodes = [v for v in G.nodes if v not in sub_set]

    plt.figure(figsize=(7, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=18, alpha=0.25)
    nx.draw_networkx_nodes(G, pos, nodelist=list(obs_set), node_size=35, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=list(miss_set), node_size=45, alpha=0.9, node_shape="s")
    plt.title("Subgraph reconstruction split: circles=observed, squares=missing")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def print_config(cfg: ExperimentConfig) -> None:
    print("Experiment configuration")
    print("------------------------")
    for key, value in asdict(cfg).items():
        print(f"{key:24s}: {value}")
    print()


def print_split_info(idx0: Tensor, obs_local: Tensor, miss_local: Tensor) -> None:
    obs_global = idx0[obs_local]
    miss_global = idx0[miss_local]
    print("Subgraph reconstruction split")
    print("-----------------------------")
    print(f"N0                       : {idx0.numel()}")
    print(f"observed nodes in V0     : {obs_local.numel()}")
    print(f"missing nodes in V0      : {miss_local.numel()}")
    print(f"idx0 first 20            : {idx0.detach().cpu().tolist()[:20]}")
    print(f"obs_global first 20      : {obs_global.detach().cpu().tolist()[:20]}")
    print(f"miss_global first 20     : {miss_global.detach().cpu().tolist()[:20]}")
    print()


def print_learned_operator_info(learned_ops: Dict[str, LearnedOperator]) -> None:
    print("Learned SFL operators")
    print("---------------------")
    for name, obj in learned_ops.items():
        print(
            f"{name:32s} | params={obj.num_parameters:4d} | "
            f"train missing MSE={obj.train_mse_missing:.6e} | ||F||_F={torch.linalg.norm(obj.operator, ord='fro').item():.6e}"
        )
    print()


def save_experiment_record(out: Dict[str, object], output_json_path: str | Path, timestamp: str) -> None:
    json_path = resolve_output_path(output_json_path)
    cfg: ExperimentConfig = out["cfg"]  # type: ignore[assignment]
    dataset: GraphSignalData = out["dataset"]  # type: ignore[assignment]
    avg: ReconstructionResult = out["result"]  # type: ignore[assignment]

    config_payload = asdict(cfg)
    for key in ["data_dir", "plot_path", "graph_plot_path"]:
        config_payload[key] = root_relative_path(resolve_path(config_payload[key]))

    payload = {
        "timestamp": timestamp,
        "config": config_payload,
        "enabled_methods": get_enabled_methods(),
        "method_specs": METHOD_SPECS,
        "run_seeds": list(out["run_seeds"]),
        "first_run_idx0": out["idx0"].detach().cpu().tolist(),
        "first_run_obs_local": out["obs_local"].detach().cpu().tolist(),
        "first_run_miss_local": out["miss_local"].detach().cpu().tolist(),
        "first_run_obs_global": out["idx0"][out["obs_local"]].detach().cpu().tolist(),
        "first_run_miss_global": out["idx0"][out["miss_local"]].detach().cpu().tolist(),
        "data_summary": {
            "node_count": dataset.node_count,
            "train_size": dataset.train_size,
            "test_size": dataset.test_size,
            "asymptotic_size": dataset.asymptotic_size,
            "raw_edge_count": int(np.count_nonzero(np.triu(dataset.A_raw, k=1))),
            "graph_edge_count": int(np.count_nonzero(np.triu(dataset.A_graph, k=1))),
        },
        "averaged_result": {
            "rmse_dict": avg.rmse_dict,
            "rmse_std_dict": avg.rmse_std_dict,
            "mse_dict": avg.mse_dict,
        },
        "run_summaries": [asdict(s) for s in out["run_summaries"]],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def run_single_experiment(
    cfg: ExperimentConfig,
    dataset: GraphSignalData,
    L: Tensor,
    A: Tensor,
    G_nx: nx.Graph,
    run_index: int,
    run_seed: int,
) -> Dict[str, object]:
    set_seed(run_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(run_seed)

    idx0_cpu = choose_random_subgraph_indices(dataset.node_count, cfg.p_fixed, generator=generator)
    obs_local_cpu, miss_local_cpu = choose_missing_observed_local_indices(
        dataset.node_count,
        idx0_cpu,
        cfg.p_missing,
        generator=generator,
    )

    idx0 = idx0_cpu.to(device=L.device)
    obs_local = obs_local_cpu.to(device=L.device)
    miss_local = miss_local_cpu.to(device=L.device)
    obs_global = idx0[obs_local]
    miss_global = idx0[miss_local]

    X_train_sub = dataset.X_train[:, idx0]
    X_asymptotic_sub = dataset.X_asymptotic[:, idx0]

    k = int(cfg.sffa_distance_k)
    r = int(cfg.sffa_word_len)

    primitives = build_filter_primitives(
        L=L,
        A=A,
        G_nx=G_nx,
        idx0=idx0,
        ridge=cfg.ridge,
        max_distance_k=k,
        num_random_groups=k,
        random_group_ratio=cfg.random_group_ratio,
        generator=generator,
        use_normalized_laplacian=cfg.use_normalized_laplacian,
    )
    sub_lmmse = subgraph_numerical_lmmse_operator(X_train_sub, obs_local, miss_local, ridge=cfg.ridge)
    sub_asym_lmmse = subgraph_numerical_lmmse_operator(X_asymptotic_sub, obs_local, miss_local, ridge=cfg.ridge)
    learned_ops = build_learned_operators(
        primitives=primitives,
        X_train_sub=X_train_sub,
        obs_local=obs_local,
        miss_local=miss_local,
        k=k,
        r=r,
        rmax=cfg.rmax,
        max_sffa_basis_size=cfg.max_sffa_basis_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        ridge=cfg.ridge,
        optimizer_name=cfg.optimizer,
        lbfgs_max_iter=cfg.lbfgs_max_iter,
    )

    result = evaluate_reconstruction_methods(
        cfg=cfg,
        dataset=dataset,
        learned_ops=learned_ops,
        subgraph_lmmse=sub_lmmse,
        subgraph_asymptotic_lmmse=sub_asym_lmmse,
        idx0=idx0,
        obs_local=obs_local,
        miss_local=miss_local,
        miss_global=miss_global,
    )

    summary = RunSummary(
        run_index=run_index,
        seed=run_seed,
        idx0=idx0.detach().cpu().tolist(),
        obs_local=obs_local.detach().cpu().tolist(),
        miss_local=miss_local.detach().cpu().tolist(),
        obs_global=obs_global.detach().cpu().tolist(),
        miss_global=miss_global.detach().cpu().tolist(),
        learned_train_mse_missing={name: obj.train_mse_missing for name, obj in learned_ops.items()},
        result=result,
    )

    return {
        "idx0": idx0,
        "obs_local": obs_local,
        "miss_local": miss_local,
        "obs_global": obs_global,
        "miss_global": miss_global,
        "primitives": primitives,
        "subgraph_lmmse": sub_lmmse,
        "subgraph_asymptotic_lmmse": sub_asym_lmmse,
        "learned_ops": learned_ops,
        "result": result,
        "summary": summary,
    }


def run_experiment(cfg: ExperimentConfig) -> Dict[str, object]:
    if cfg.sffa_word_len < 0:
        raise ValueError("r/sffa_word_len must be >= 0.")
    assert float(cfg.p_missing) <= float(cfg.p_fixed) + 1e-12, (
        f"p_missing/p1={cfg.p_missing:g} must be <= p_fixed/p={cfg.p_fixed:g}."
    )

    dataset = load_graph_signal_data(cfg)

    if cfg.auto_budget_k:
        n0_budget = max(1, min(dataset.node_count, int(round(float(cfg.p_fixed) * dataset.node_count))))
        budgeted_k = max_sffa_k_under_budget(n0_budget, int(cfg.sffa_word_len))
        print(
            f"[auto_budget_k] p={cfg.p_fixed:g}, N={dataset.node_count}, N0={n0_budget}, "
            f"r={cfg.sffa_word_len}, selected k={budgeted_k}, "
            f"SFFA params={sffa_basis_size(budgeted_k, int(cfg.sffa_word_len), include_identity=True)}"
        )
        cfg.sffa_distance_k = int(budgeted_k)

    if cfg.sffa_distance_k < 1:
        raise ValueError("k/sffa_distance_k must be >= 1.")

    configure_enabled_methods(cfg.methods, cfg.rmax)
    update_dynamic_method_labels(cfg)
    L, A, G_nx = build_graph_laplacian_from_adjacency(dataset.A_graph, cfg.use_normalized_laplacian)
    lap_mode = "symmetric normalized" if cfg.use_normalized_laplacian else "combinatorial"
    print(f"SFL Laplacian primitive mode: {lap_mode}")
    print()

    run_outputs: List[Dict[str, object]] = []
    run_results: List[ReconstructionResult] = []
    run_summaries: List[RunSummary] = []

    for run_index in range(cfg.num_runs):
        run_seed = cfg.seed + run_index
        print(f"[p={cfg.p_fixed:g} | run {run_index + 1}/{cfg.num_runs}] seed={run_seed}")
        out = run_single_experiment(cfg, dataset, L, A, G_nx, run_index, run_seed)
        run_outputs.append(out)
        run_results.append(out["result"])  # type: ignore[arg-type]
        run_summaries.append(out["summary"])  # type: ignore[arg-type]

    avg_result = average_results(run_results)
    ref_out = run_outputs[0]

    return {
        "cfg": cfg,
        "dataset": dataset,
        "L": L,
        "A": A,
        "G_nx": G_nx,
        "idx0": ref_out["idx0"],
        "obs_local": ref_out["obs_local"],
        "miss_local": ref_out["miss_local"],
        "obs_global": ref_out["obs_global"],
        "miss_global": ref_out["miss_global"],
        "primitives": ref_out["primitives"],
        "subgraph_lmmse": ref_out["subgraph_lmmse"],
        "subgraph_asymptotic_lmmse": ref_out["subgraph_asymptotic_lmmse"],
        "learned_ops": ref_out["learned_ops"],
        "run_seeds": [cfg.seed + i for i in range(cfg.num_runs)],
        "enabled_methods": get_enabled_methods(),
        "method_labels": {name: METHOD_SPECS[name]["label"] for name in get_enabled_methods()},
        "result": avg_result,
        "run_summaries": run_summaries,
    }


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generic T x N graph-signal SFL reconstruction. "
            "The task samples a random subgraph V0 and reconstructs M0 from O0, "
            "where p=|V0|/|V| and p1=|M0|/|V|."
        )
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_dir", type=str, default="processed_molene")
    parser.add_argument("--adjacency_file", type=str, default="A.npy")
    parser.add_argument("--signal_file", type=str, default="X_clean.npy")
    parser.add_argument("--adjacency_key", type=str, default="")
    parser.add_argument("--signal_key", type=str, default="")

    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["ratio", "number"],
        default="ratio",
        help=(
            "ratio: split all available samples into train/test according to train_ratio/test_ratio; "
            "number: take the first train_size samples, then the next test_size samples."
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=None,
        help=(
            "Optional test ratio in ratio mode. If omitted, test_ratio=1-train_ratio. "
            "If supplied, train_ratio and test_ratio are normalized to split all samples."
        ),
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Absolute number of training time points in number mode.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="Absolute number of test time points in number mode, immediately after training rows.",
    )

    parser.add_argument(
        "--p_fixed",
        "--p",
        type=str,
        default="0.25",
        help=(
            "Subgraph node ratio p=|V0|/|V|. Accepts one value, CSV, or range start:stop:step; "
            "e.g. --p 0.3,0.4,0.5 or --p 0.3:0.7:0.1."
        ),
    )
    parser.add_argument(
        "--p_values",
        type=str,
        default="",
        help="Optional explicit p sweep. Overrides --p_fixed/--p when nonempty.",
    )
    parser.add_argument(
        "--p_missing",
        "--p1",
        type=float,
        default=0.1,
        help="Hidden-node ratio p1=|M0|/|V| in the original graph. Must satisfy p1 <= p.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=DEFAULT_METHODS,
        help=(
            "Comma-separated methods to run. Default runs regular methods only. "
            "Optional limit sweeps are induced_poly_lim,kron_poly_lim,union_poly_lim; "
            "they expand into *_r2..*_rmax and are not included by default."
        ),
    )
    parser.add_argument(
        "--rmax",
        type=int,
        default=5,
        help="Maximum degree for induced_poly_lim/kron_poly_lim/union_poly_lim. These run degrees 2..rmax.",
    )
    parser.add_argument("--ridge", type=float, default=1e-8, help="Ridge used in numerical LMMSE and learned-filter Frobenius penalty.")
    parser.add_argument(
        "--spectral_preprocess",
        type=str,
        choices=["none", "cutoff", "topk"],
        default="cutoff",
        help=(
            "Spectral preprocessing for raw signals before train/test split. "
            "'none' keeps all node-space signals unchanged; "
            "'cutoff' keeps normalized-Laplacian eigenvectors with lambda <= --bandlimit_cutoff; "
            "'topk' keeps the first --bandlimit_k smallest-eigenvalue eigenvectors."
        ),
    )
    parser.add_argument(
        "--bandlimit_cutoff",
        type=float,
        default=1.0,
        help="Eigenvalue cutoff used when --spectral_preprocess cutoff. Default 1.0 reproduces the original version.",
    )
    parser.add_argument(
        "--bandlimit_k",
        type=int,
        default=0,
        help="Number of smallest normalized-Laplacian eigenvectors kept when --spectral_preprocess topk.",
    )
    parser.add_argument(
        "--normalize_by_all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignored in this bandlimited variant; z-score normalization is disabled.",
    )
    parser.add_argument(
        "--normalize_by_train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignored in this bandlimited variant; z-score normalization is disabled.",
    )
    parser.add_argument("--use_normalized_laplacian", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sffa_distance_k",
        "--k",
        dest="sffa_distance_k",
        type=int,
        default=3,
        help=(
            "k. Controls the number of distance-k Laplacians, the union_poly union size, "
            "the random_group_algebra random sub-subgraph count, and pure_sffa primitives."
        ),
    )
    parser.add_argument(
        "--auto_budget_k",
        "--auto-budget-k",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For each p, set k to the largest value whose pure-SFFA basis count "
            "1+k+...+k^r does not exceed N0=round(p*N)."
        ),
    )
    parser.add_argument(
        "--sffa_word_len",
        "--r",
        dest="sffa_word_len",
        type=int,
        default=3,
        help="r. Maximum polynomial degree for regular polynomial methods and maximum word length for algebra/SFFA methods.",
    )
    parser.add_argument(
        "--random_group_ratio",
        type=float,
        default=0.9,
        help="Each random sub-subgraph in random_group_algebra samples this ratio of V0 nodes.",
    )
    parser.add_argument(
        "--max_sffa_basis_size",
        type=int,
        default=2000,
        help=(
            "Abort before constructing algebra/SFFA word bases if the geometric-series basis count exceeds this threshold. "
            "Set <=0 to disable this guard."
        ),
    )
    parser.add_argument(
        "--save_outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save RMSE plots, graph plots, and JSON/CSV records. Use --no-save_outputs for notebook sweeps.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "lbfgs"],
        default="lbfgs",
        help="Optimizer for learned SFL filter coefficients. Use lbfgs for faster convergence on low-dimensional filter banks.",
    )
    parser.add_argument(
        "--lbfgs_max_iter",
        type=int,
        default=20,
        help="Internal LBFGS iterations per outer epoch.",
    )
    parser.add_argument("--num_runs", "--runs", dest="num_runs", type=int, default=1)
    parser.add_argument("--plot_path", type=str, default=str(DEFAULT_PLOT_PATH))
    parser.add_argument("--graph_plot_path", type=str, default=str(DEFAULT_GRAPH_PLOT_PATH))
    args = parser.parse_args()

    p_values = parse_float_range_or_csv(args.p_values) if str(args.p_values).strip() else parse_float_range_or_csv(args.p_fixed)
    if not p_values:
        raise ValueError("At least one p value must be supplied.")
    p_values_csv = ",".join(f"{float(p):.12g}" for p in p_values)

    kwargs = vars(args).copy()
    kwargs["p_fixed"] = float(p_values[0])
    kwargs["p_values"] = p_values_csv
    cfg = ExperimentConfig(**kwargs)
    if cfg.num_runs <= 0:
        raise ValueError("num_runs/runs must be positive.")
    return cfg


def clone_config_for_p(cfg: ExperimentConfig, p_value: float) -> ExperimentConfig:
    payload = asdict(cfg)
    payload["p_fixed"] = float(p_value)
    return ExperimentConfig(**payload)


def p_token(p_value: float) -> str:
    return (f"p{float(p_value):.6g}".replace(".", "p").replace("-", "m"))


def sweep_summary_rows(outputs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for out in outputs:
        cfg: ExperimentConfig = out["cfg"]  # type: ignore[assignment]
        dataset: GraphSignalData = out["dataset"]  # type: ignore[assignment]
        result: ReconstructionResult = out["result"]  # type: ignore[assignment]
        method_order = list(out.get("enabled_methods", get_enabled_methods()))
        method_labels = dict(out.get("method_labels", {}))
        for method in method_order:
            if method not in result.rmse_dict:
                continue
            rows.append(
                {
                    "p": float(cfg.p_fixed),
                    "p1_hidden_original_ratio": float(cfg.p_missing),
                    "n0_first_run": int(out["idx0"].numel()),  # type: ignore[union-attr]
                    "m0_first_run": int(out["miss_local"].numel()),  # type: ignore[union-attr]
                    "runs": int(cfg.num_runs),
                    "seeds": ";".join(str(s) for s in out["run_seeds"]),
                    "method": method,
                    "label": method_labels.get(method, METHOD_SPECS.get(method, {"label": method})["label"]),
                    "rmse_mean": result.rmse_dict[method],
                    "rmse_std": result.rmse_std_dict.get(method, 0.0) if result.rmse_std_dict is not None else 0.0,
                    "mse_mean": result.mse_dict[method],
                    "train_size": int(dataset.train_size),
                    "test_size": int(dataset.test_size),
                    "k": int(cfg.sffa_distance_k),
                    "r": int(lim_variant_degree(method) or cfg.sffa_word_len),
                    "rmax": int(cfg.rmax),
                    "optimizer": cfg.optimizer,
                }
            )
    return rows


def print_sweep_summary(outputs: Sequence[Dict[str, object]]) -> None:
    rows = sweep_summary_rows(outputs)
    if not rows:
        return
    print("=== p-sweep averaged RMSE summary ===")
    print("p".rjust(8) + " | " + "method".ljust(30) + " | " + "RMSE mean±std".rjust(24) + " | " + "MSE mean".rjust(14))
    print("-" * (8 + 3 + 30 + 3 + 24 + 3 + 14))
    for row in rows:
        rmse = f"{row['rmse_mean']:.4f}±{row['rmse_std']:.4f}"
        print(
            f"{row['p']:8.4g} | "
            + str(row["method"]).ljust(30)
            + " | "
            + rmse.rjust(24)
            + " | "
            + f"{row['mse_mean']:.6e}".rjust(14)
        )
    print()


def save_sweep_summary_files(outputs: Sequence[Dict[str, object]], plot_path: str | Path, timestamp: str) -> Tuple[Path, Path]:
    rows = sweep_summary_rows(outputs)
    plot_base = resolve_output_path(plot_path)
    csv_path = plot_base.with_name(f"{plot_base.stem}_{timestamp}_summary.csv")
    json_path = plot_base.with_name(f"{plot_base.stem}_{timestamp}_summary.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "timestamp": timestamp,
        "summary_rows": rows,
        "p_values": [float(out["cfg"].p_fixed) for out in outputs],  # type: ignore[union-attr]
        "run_seeds_by_p": {
            str(float(out["cfg"].p_fixed)): list(out["run_seeds"]) for out in outputs  # type: ignore[union-attr]
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return csv_path, json_path


if __name__ == "__main__":
    cfg = parse_args()
    DEVICE = resolve_torch_device(cfg.device)
    cfg.device = DEVICE
    set_seed(cfg.seed)

    p_values = p_values_from_cfg(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_outputs: List[Dict[str, object]] = []

    for p_idx, p_value in enumerate(p_values, start=1):
        cfg_p = clone_config_for_p(cfg, p_value)
        print(f"\n===== p sweep {p_idx}/{len(p_values)}: p={p_value:g}, p1={cfg_p.p_missing:g}, runs={cfg_p.num_runs} =====")
        out = run_experiment(cfg_p)
        all_outputs.append(out)

        print_config(cfg_p)
        print_split_info(out["idx0"], out["obs_local"], out["miss_local"])
        print_learned_operator_info(out["learned_ops"])
        print_result_table(out["result"])

        if cfg_p.save_outputs:
            stamped = f"{timestamp}_{p_token(p_value)}"
            output_paths = build_timestamped_paths(cfg_p.plot_path, cfg_p.graph_plot_path, stamped)
            dataset: GraphSignalData = out["dataset"]  # type: ignore[assignment]
            plot_graph_subgraph(dataset.A_graph, out["idx0"], out["obs_local"], out["miss_local"], output_paths["graph_plot"])
            plot_results(out["result"], output_paths["plot"])
            save_experiment_record(out, output_paths["json"], stamped)

            print(f"Saved graph plot to: {root_relative_path(output_paths['graph_plot'])}")
            print(f"Saved RMSE plot to: {root_relative_path(output_paths['plot'])}")
            print(f"Saved experiment record to: {root_relative_path(output_paths['json'])}")
        else:
            print("Skipped saving plots and JSON because --no-save_outputs was set.")

    if len(all_outputs) > 1:
        print_sweep_summary(all_outputs)
    if cfg.save_outputs:
        csv_path, json_path = save_sweep_summary_files(all_outputs, cfg.plot_path, timestamp)
        print(f"Saved p-sweep summary CSV to: {root_relative_path(csv_path)}")
        print(f"Saved p-sweep summary JSON to: {root_relative_path(json_path)}")
