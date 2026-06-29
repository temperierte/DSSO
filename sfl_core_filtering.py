"""Core utilities for filtering-style Subgraph Filter Learning experiments.

This module is a minimally intrusive rewrite of the simplified ``sfl_core.py``.
It keeps the dense-matrix graph primitive/basis pipeline, but makes the three
experiment components configurable:

    truth f:        linear Laplacian polynomial or nonlinear Volterra-Hadamard
    noise:          Gaussian, uniform, Rademacher, or skewed bounded sub-Gaussian noise
    train/test loss: MSE, Huber, log-cosh, or smooth pinball, tied between fitting and evaluation

Conventions
-----------
Signals are row-major batches: X has shape [T, N].  A node-space operator H is
applied as

    Y_hat = X @ H.T.

All Laplacian primitives returned here are dense tensors.  This remains suitable
for METR-LA-scale filtering experiments and keeps LBFGS fitting simple.
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from torch import Tensor

try:
    from torch_geometric.data import Data
except ModuleNotFoundError:
    class Data:  # minimal fallback for environments without torch_geometric
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)



NodeIds = Union[Sequence[int], Tensor]


# -----------------------------------------------------------------------------
# Method names shared with the filtering runner
# -----------------------------------------------------------------------------

TRAINABLE_METHODS = frozenset(
    {
        "sub_lp",      # induced-subgraph Laplacian polynomial
        "dk_lap_poly", # separate distance-k Laplacian polynomials
        "kron_lp",     # Kron-reduced Laplacian polynomial
        "union_lp",    # polynomial of union_{1..k_union} distance graph
        "rg_alg",      # random-group free algebra
        "sffa",        # pure distance-based Laplacian SFFA/free algebra
    }
)

CLOSED_FORM_METHODS = frozenset({"num_lmmse"})

DEFAULT_METHODS = "sub_lp,kron_lp,union_lp,dk_lap_poly,rg_alg,sffa,num_lmmse"

METHOD_LABELS: Dict[str, str] = {
    "sub_lp": "Induced-subgraph Laplacian polynomial",
    "dk_lap_poly": "Distance-k Laplacian polynomial",
    "kron_lp": "Kron Laplacian polynomial",
    "union_lp": "Union distance-Laplacian polynomial",
    "rg_alg": "Random-group algebra",
    "sffa": "Distance-based Laplacian SFFA",
    "num_lmmse": "Numerical LMMSE (MSE-trained baseline)",
}


# -----------------------------------------------------------------------------
# Reproducibility and small helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch RNG states."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_seed_sequence(seed: int, runs: int = 1) -> List[int]:
    """Return the consecutive seeds used by the runner for repeated runs."""
    runs = int(runs)
    if runs <= 0:
        raise ValueError("runs must be positive.")
    start = int(seed)
    return list(range(start, start + runs))


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve 'auto', 'cuda', or 'cpu' to a torch.device."""
    device = str(device).lower().strip()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of {'auto', 'cpu', 'cuda'}.")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def make_torch_generator(seed: Optional[int], device: Union[str, torch.device]) -> Optional[torch.Generator]:
    """Create a PyTorch generator on the requested device when a seed is given."""
    if seed is None:
        return None
    dev = torch.device(device)
    try:
        g = torch.Generator(device=dev)
    except TypeError:
        g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def as_long_tensor(idx: NodeIds, device: Optional[torch.device] = None) -> Tensor:
    if isinstance(idx, Tensor):
        return idx.to(device=device, dtype=torch.long)
    return torch.tensor([int(v) for v in idx], dtype=torch.long, device=device)


def node_ids_list(idx: NodeIds) -> List[int]:
    if isinstance(idx, Tensor):
        return [int(v) for v in idx.detach().cpu().tolist()]
    return [int(v) for v in idx]


def submatrix(M: Tensor, row_idx: NodeIds, col_idx: Optional[NodeIds] = None) -> Tensor:
    """Return M[row_idx, col_idx], preserving the supplied node order."""
    r = as_long_tensor(row_idx, device=M.device)
    c = r if col_idx is None else as_long_tensor(col_idx, device=M.device)
    return M.index_select(0, r).index_select(1, c)


def symmetrize(M: Tensor) -> Tensor:
    return 0.5 * (M + M.T)


def normalize_fro(S: Tensor, eps: float = 1e-12) -> Tensor:
    """Frobenius-normalize an operator, used before optional polynomial powers."""
    norm = torch.linalg.norm(S, ord="fro")
    if torch.isfinite(norm) and float(norm.item()) > eps:
        return S / norm
    return S


# -----------------------------------------------------------------------------
# Data loading: METR-LA only
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrafficData:
    graph: Data
    graph_nx: nx.Graph
    x_full: Tensor
    num_nodes: int
    adjacency: Tensor
    raw_indices: Tensor
    raw_total_timestamps: int
    missing_timestamps: int


def default_dataset_dir() -> Path:
    """Assume this file lives in <project_root>/revisedfilter/."""
    return Path(__file__).resolve().parents[1] / "dataset"


def load_metr_la_traffic(
    num_timesteps: Optional[int] = None,
    dataset_dir: Optional[Union[str, Path]] = None,
    adj_filename: str = "adj_METR-LA.pkl",
    h5_filename: str = "METR-LA.h5",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> TrafficData:
    """Load the METR-LA graph and complete single-time-step signals.

    Missingness is detected only through non-finite values such as NaN/Inf.
    Zero traffic speeds are valid signal values and are not filtered out.

    If ``num_timesteps`` is positive, the cleaned sequence is truncated to that
    many rows.  Passing ``None`` or a non-positive value keeps all complete rows.
    The new filtering runner can omit the old total-sample flag and control the
    effective sample count through train/test counts or train ratio instead.
    """
    root = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    adj_path = root / adj_filename
    h5_path = root / h5_filename
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing adjacency file: {adj_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing METR-LA h5 file: {h5_path}")

    with open(str(adj_path), "rb") as f:
        _sensor_ids, _sensor_id_to_ind, adj = pickle.load(f, encoding="latin1")
    if isinstance(adj, list):
        adj = adj[0]

    A = torch.as_tensor(adj, dtype=dtype)
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"METR-LA adjacency must be square; got {tuple(A.shape)}.")
    A = A.clone()
    A.fill_diagonal_(0.0)
    A = torch.maximum(A, A.T)
    A = (A > 0).to(dtype)
    n = int(A.shape[0])

    edge_index = torch.nonzero(A, as_tuple=False).T.contiguous()
    edge_weight = torch.ones(edge_index.shape[1], dtype=dtype)
    graph = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=n)

    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(n))
    graph_nx.add_edges_from(edge_index.T.detach().cpu().numpy().tolist())

    with h5py.File(str(h5_path), "r") as f:
        raw_values = f["df"]["block0_values"][:]
        # Validate the HDF5 layout and preserve parity with previous loaders.
        _ = pd.to_datetime(f["df"]["axis1"][:])

    if raw_values.ndim != 2 or raw_values.shape[1] != n:
        raise ValueError(f"Expected METR-LA values with shape [T,{n}], got {raw_values.shape}.")

    finite_mask = np.isfinite(raw_values).all(axis=1)
    raw_indices_np = np.flatnonzero(finite_mask).astype(np.int64)
    clean_values = raw_values[finite_mask]

    if num_timesteps is not None and int(num_timesteps) > 0:
        keep = min(int(num_timesteps), int(clean_values.shape[0]))
        clean_values = clean_values[:keep]
        raw_indices_np = raw_indices_np[:keep]

    if clean_values.shape[0] == 0:
        raise ValueError("No complete METR-LA samples remained after NaN/Inf filtering.")

    x_full = torch.as_tensor(clean_values, dtype=dtype, device=device)
    raw_indices = torch.as_tensor(raw_indices_np, dtype=torch.long, device=device)
    A = A.to(device=device)
    graph.edge_index = graph.edge_index.to(device=device)
    graph.edge_attr = graph.edge_attr.to(device=device)

    return TrafficData(
        graph=graph,
        graph_nx=graph_nx,
        x_full=x_full,
        num_nodes=n,
        adjacency=A,
        raw_indices=raw_indices,
        raw_total_timestamps=int(raw_values.shape[0]),
        missing_timestamps=int((~finite_mask).sum()),
    )


# -----------------------------------------------------------------------------
# Dense Laplacian and graph primitive construction
# -----------------------------------------------------------------------------


def dense_adjacency_from_edge_index(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    make_undirected_or: bool = True,
) -> Tensor:
    if device is None:
        device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=dtype, device=device)
    else:
        edge_weight = edge_weight.to(device=device, dtype=dtype)
    A = torch.zeros((int(num_nodes), int(num_nodes)), dtype=dtype, device=device)
    row = edge_index[0].to(device=device, dtype=torch.long)
    col = edge_index[1].to(device=device, dtype=torch.long)
    A[row, col] = edge_weight
    if make_undirected_or:
        A = torch.maximum(A, A.T)
        A = (A > 0).to(dtype)
    A.fill_diagonal_(0.0)
    return A


def symmetric_normalized_laplacian_from_adjacency(
    A: Tensor,
    zero_isolated: bool = False,
) -> Tensor:
    """Return L_sym = I - D^{-1/2} A D^{-1/2}."""
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    A = A.clone()
    A.fill_diagonal_(0.0)
    A = torch.maximum(A, A.T)
    n = int(A.shape[0])
    deg = A.sum(dim=1)
    inv_sqrt = torch.zeros_like(deg)
    nonzero = deg > 0
    inv_sqrt[nonzero] = deg[nonzero].rsqrt()
    off = inv_sqrt[:, None] * A * inv_sqrt[None, :]
    diag = torch.ones(n, dtype=A.dtype, device=A.device)
    if zero_isolated:
        diag = torch.where(nonzero, diag, torch.zeros_like(diag))
    L = torch.diag(diag) - off
    return symmetrize(L)


def symmetric_normalized_laplacian_from_edges(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    zero_isolated: bool = False,
) -> Tensor:
    A = dense_adjacency_from_edge_index(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=int(num_nodes),
        dtype=dtype,
        device=device,
        make_undirected_or=True,
    )
    return symmetric_normalized_laplacian_from_adjacency(A, zero_isolated=zero_isolated)


def full_laplacian_from_graph(
    graph: Data,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    zero_isolated: bool = False,
) -> Tensor:
    return symmetric_normalized_laplacian_from_edges(
        graph.edge_index,
        num_nodes=int(graph.num_nodes),
        edge_weight=getattr(graph, "edge_attr", None),
        dtype=dtype,
        device=device,
        zero_isolated=zero_isolated,
    )


def induced_adjacency(A_full: Tensor, v0_idx: NodeIds) -> Tensor:
    return submatrix(A_full, v0_idx, v0_idx)


def induced_laplacian(
    A_full: Tensor,
    v0_idx: NodeIds,
    zero_isolated: bool = False,
) -> Tensor:
    return symmetric_normalized_laplacian_from_adjacency(
        induced_adjacency(A_full, v0_idx),
        zero_isolated=zero_isolated,
    )


def _edges_to_laplacian(
    edges_local: Sequence[Tuple[int, int]],
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
    zero_isolated: bool = False,
) -> Tensor:
    A = torch.zeros((int(num_nodes), int(num_nodes)), dtype=dtype, device=device)
    if len(edges_local) > 0:
        ei = torch.tensor(edges_local, dtype=torch.long, device=device).T.contiguous()
        A[ei[0], ei[1]] = 1.0
        A[ei[1], ei[0]] = 1.0
    return symmetric_normalized_laplacian_from_adjacency(A, zero_isolated=zero_isolated)


def distance_k_laplacians(
    graph_nx: nx.Graph,
    v0_idx: NodeIds,
    k_max: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    zero_isolated: bool = False,
) -> List[Tensor]:
    """Build distance-k subgraph symmetric normalized Laplacians for k=1..k_max."""
    if int(k_max) < 1:
        raise ValueError("k_max must be >= 1.")
    if device is None:
        device = torch.device("cpu")

    v0 = node_ids_list(v0_idx)
    n0 = len(v0)
    local = {node: i for i, node in enumerate(v0)}
    v0_set = set(v0)
    edges_by_k: Dict[int, set[Tuple[int, int]]] = {k: set() for k in range(1, int(k_max) + 1)}

    for u in v0:
        lengths = nx.single_source_shortest_path_length(graph_nx, u, cutoff=int(k_max))
        i = local[u]
        for v, d in lengths.items():
            d = int(d)
            if d < 1 or d > int(k_max) or v not in v0_set:
                continue
            j = local[int(v)]
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges_by_k[d].add((a, b))

    return [
        _edges_to_laplacian(
            sorted(edges_by_k[k]),
            num_nodes=n0,
            dtype=dtype,
            device=device,
            zero_isolated=zero_isolated,
        )
        for k in range(1, int(k_max) + 1)
    ]


def union_laplacian_from_distance_laplacians(
    distance_laps: Sequence[Tensor],
    upto_k: Optional[int] = None,
    zero_isolated: bool = False,
) -> Tensor:
    """Build a union graph Laplacian from distance-k Laplacians."""
    if len(distance_laps) == 0:
        raise ValueError("distance_laps must be non-empty.")
    k = len(distance_laps) if upto_k is None else int(upto_k)
    if k < 1 or k > len(distance_laps):
        raise ValueError("upto_k must lie in [1, len(distance_laps)].")

    n = int(distance_laps[0].shape[0])
    A = torch.zeros((n, n), dtype=distance_laps[0].dtype, device=distance_laps[0].device)
    for L in distance_laps[:k]:
        Ak = (L < 0).to(dtype=A.dtype)
        Ak.fill_diagonal_(0.0)
        A = torch.maximum(A, Ak)
    return symmetric_normalized_laplacian_from_adjacency(A, zero_isolated=zero_isolated)


def union_distance_laplacian(
    graph_nx: nx.Graph,
    v0_idx: NodeIds,
    ks: Sequence[int] = (1, 2, 3),
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    zero_isolated: bool = False,
) -> Tensor:
    if len(ks) == 0:
        raise ValueError("ks must be non-empty.")
    if device is None:
        device = torch.device("cpu")

    v0 = node_ids_list(v0_idx)
    n0 = len(v0)
    local = {node: i for i, node in enumerate(v0)}
    v0_set = set(v0)
    k_set = {int(k) for k in ks}
    k_max = max(k_set)
    edges_local: set[Tuple[int, int]] = set()

    for u in v0:
        lengths = nx.single_source_shortest_path_length(graph_nx, u, cutoff=k_max)
        i = local[u]
        for v, d in lengths.items():
            if int(d) not in k_set or v not in v0_set:
                continue
            j = local[int(v)]
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges_local.add((a, b))

    return _edges_to_laplacian(
        sorted(edges_local),
        num_nodes=n0,
        dtype=dtype,
        device=device,
        zero_isolated=zero_isolated,
    )


def random_group_laplacians(
    graph_nx: nx.Graph,
    v0_idx: NodeIds,
    num_groups: int = 3,
    p0: float = 0.9,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    zero_isolated: bool = False,
) -> List[Tensor]:
    """Build random induced-subgraph Laplacians embedded in V0 coordinates."""
    if int(num_groups) <= 0:
        raise ValueError("num_groups must be positive.")
    if not (0.0 < float(p0) <= 1.0):
        raise ValueError("p0 must lie in (0, 1].")
    if device is None:
        device = torch.device("cpu")

    v0 = node_ids_list(v0_idx)
    n0 = len(v0)
    v0_set = set(v0)
    local = {node: i for i, node in enumerate(v0)}
    out: List[Tensor] = []
    group_size = max(1, int(n0 * float(p0)))

    for _ in range(int(num_groups)):
        perm = torch.randperm(n0)[:group_size].tolist()
        chosen = {v0[int(i)] for i in perm}
        edges_local: set[Tuple[int, int]] = set()
        for u, v in graph_nx.subgraph(list(chosen)).edges():
            if u not in v0_set or v not in v0_set:
                continue
            a, b = local[int(u)], local[int(v)]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges_local.add((a, b))
        out.append(
            _edges_to_laplacian(
                sorted(edges_local),
                num_nodes=n0,
                dtype=dtype,
                device=device,
                zero_isolated=zero_isolated,
            )
        )
    return out


def kron_reduced_laplacian(
    L_full: Tensor,
    v0_idx: NodeIds,
    ridge: float = 0.0,
    use_pinv: bool = False,
) -> Tensor:
    """Kron-reduce L_full onto V0 via a ridge-stabilized Schur complement."""
    if L_full.dim() != 2 or L_full.shape[0] != L_full.shape[1]:
        raise ValueError("L_full must be square.")
    n_full = int(L_full.shape[0])
    keep = as_long_tensor(v0_idx, device=L_full.device)
    if keep.numel() == 0:
        raise ValueError("v0_idx must be non-empty.")
    if torch.unique(keep).numel() != keep.numel():
        raise ValueError("v0_idx contains duplicate nodes.")

    mask = torch.ones(n_full, dtype=torch.bool, device=L_full.device)
    mask[keep] = False
    comp = torch.arange(n_full, dtype=torch.long, device=L_full.device)[mask]

    L00 = submatrix(L_full, keep, keep)
    if comp.numel() == 0:
        return symmetrize(L00)
    L0c = submatrix(L_full, keep, comp)
    Lc0 = submatrix(L_full, comp, keep)
    Lcc = submatrix(L_full, comp, comp)
    if float(ridge) != 0.0:
        Lcc = Lcc + float(ridge) * torch.eye(Lcc.shape[0], dtype=L_full.dtype, device=L_full.device)
    if use_pinv:
        correction = L0c @ torch.linalg.pinv(Lcc) @ Lc0
    else:
        correction = L0c @ torch.linalg.solve(Lcc, Lc0)
    return symmetrize(L00 - correction)


# -----------------------------------------------------------------------------
# Truth filters and noisy labels
# -----------------------------------------------------------------------------


DEFAULT_TRUTH_COEFFS: Tuple[float, ...] = (-1.0, 9.0, -12.0, 4.0)
DEFAULT_VOLTERRA_QUADRATIC_COEFFS: Tuple[float, ...] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class TruthResult:
    y_full: Tensor
    F_full: Optional[Tensor]
    summary: str
    truth_type: str


@dataclass(frozen=True)
class TruthFilterSpec:
    """Container for the switchable filtering truth model.

    ``truth_type`` accepts:
      - ``linear_poly`` / ``poly3``: y = x F^T, F=sum_l coeffs[l] L^l.
      - ``volterra``: y = lambda * ((x_c) * tanh(x_c / scale)) B^T.
      - ``hadamard``: y = lambda * (((x B1^T) * tanh((x B2^T) / scale)) B3^T),
        where B1=L, B2=L^2, and B3 is controlled by ``quadratic_coeffs``.

    The nonlinear options intentionally have no linear x A^T channel.
    """

    truth_type: str = "linear_poly"
    coeffs: Tuple[float, ...] = DEFAULT_TRUTH_COEFFS
    quadratic_coeffs: Tuple[float, ...] = DEFAULT_VOLTERRA_QUADRATIC_COEFFS
    volterra_lambda: float = 1.0
    nonlinear_scale: float = 0.0
    center_nonlinear_input: bool = True
    normalize_base: bool = False


def normalized_truth_type(truth_type: str) -> str:
    t = str(truth_type).lower().strip().replace("-", "_")
    aliases = {
        "linear": "linear_poly",
        "poly": "linear_poly",
        "poly3": "linear_poly",
        "lap_poly": "linear_poly",
        "laplacian_poly": "linear_poly",
        "polynomial": "linear_poly",
        "volterra2": "volterra",
        "volterra_hadamard": "volterra",
        "hadamard_volterra": "volterra",
        "hadamard_product": "hadamard",
        "lap_hadamard": "hadamard",
    }
    t = aliases.get(t, t)
    if t not in {"linear_poly", "volterra", "hadamard"}:
        raise ValueError("truth_type must be one of {'linear_poly', 'poly3', 'volterra', 'hadamard'}.")
    return t


def polynomial_filter_from_coeffs(S: Tensor, coeffs: Sequence[float], normalize_base: bool = False) -> Tensor:
    """Return F = sum_l coeffs[l] S^l."""
    if S.dim() != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be square.")
    base = normalize_fro(S) if normalize_base else S
    n = int(base.shape[0])
    I = torch.eye(n, dtype=base.dtype, device=base.device)
    F = torch.zeros_like(base)
    power = I
    for a in coeffs:
        F = F + float(a) * power
        power = power @ base
    return F.contiguous()


def build_linear_poly_truth(
    L_full: Tensor,
    x_full: Tensor,
    coeffs: Sequence[float] = DEFAULT_TRUTH_COEFFS,
    normalize_base: bool = False,
) -> TruthResult:
    """Build the old third-order Laplacian-polynomial truth y = xF^T."""
    F = polynomial_filter_from_coeffs(L_full, coeffs=coeffs, normalize_base=normalize_base)
    X = x_full.to(device=F.device, dtype=F.dtype)
    Y = X @ F.T
    summary = (
        "[TruthFilter] type=linear_poly | formula=y=xF^T | "
        f"coeffs={tuple(float(c) for c in coeffs)} | normalize_base={bool(normalize_base)} | "
        "F_full=complete_linear_operator"
    )
    return TruthResult(y_full=Y.contiguous(), F_full=F.contiguous(), summary=summary, truth_type="linear_poly")


def build_volterra_hadamard_truth(
    L_full: Tensor,
    x_full: Tensor,
    quadratic_coeffs: Sequence[float] = DEFAULT_VOLTERRA_QUADRATIC_COEFFS,
    volterra_lambda: float = 1.0,
    nonlinear_scale: float = 0.0,
    center_nonlinear_input: bool = True,
    normalize_base: bool = False,
) -> TruthResult:
    """Build the nonlinear Volterra-Hadamard truth used in filtering runs.

    The retained channel is

        y = lambda * ((x_c) * tanh(x_c / scale)) B^T,

    where B is a quadratic polynomial of the full symmetric normalized
    Laplacian.  The linear xA^T channel from the reference implementation is
    intentionally omitted.  Since the map is nonlinear, ``F_full`` is ``None``.
    """
    B = polynomial_filter_from_coeffs(L_full, coeffs=quadratic_coeffs, normalize_base=normalize_base)
    X = x_full.to(device=B.device, dtype=B.dtype)

    x_nl = X
    if bool(center_nonlinear_input):
        x_nl = x_nl - x_nl.mean(dim=-1, keepdim=True)

    if float(nonlinear_scale) > 0.0:
        scale = torch.as_tensor(float(nonlinear_scale), device=X.device, dtype=X.dtype)
        scale_desc = f"{float(nonlinear_scale):.6g}(fixed)"
    else:
        scale = torch.sqrt(torch.mean(x_nl.detach() ** 2, dim=-1, keepdim=True)).clamp_min(1e-6)
        scale_desc = "auto_per_sample_rms"

    phi = torch.tanh(x_nl / scale)
    hadamard_signal = x_nl * phi
    Y = float(volterra_lambda) * (hadamard_signal @ B.T)

    with torch.no_grad():
        y_rms = torch.sqrt(torch.mean(Y.detach() ** 2)).clamp_min(1e-12)
        signal_rms = torch.sqrt(torch.mean(hadamard_signal.detach() ** 2)).clamp_min(1e-12)

    summary = (
        "[TruthFilter] type=volterra | "
        "formula=y=lambda*((x_c*tanh(x_c/scale))B^T) | "
        f"quadratic_coeffs={tuple(float(c) for c in quadratic_coeffs)} | "
        f"volterra_lambda={float(volterra_lambda):.6g} | "
        f"nonlinear_scale={scale_desc} | "
        f"center_nonlinear_input={bool(center_nonlinear_input)} | "
        f"normalize_base={bool(normalize_base)} | "
        f"rms_hadamard_signal={float(signal_rms):.6g} | "
        f"rms_y={float(y_rms):.6g} | "
        "F_full=None_nonlinear_truth"
    )
    return TruthResult(y_full=Y.contiguous(), F_full=None, summary=summary, truth_type="volterra")




def build_hadamard_truth(
    L_full: Tensor,
    x_full: Tensor,
    quadratic_coeffs: Sequence[float] = DEFAULT_VOLTERRA_QUADRATIC_COEFFS,
    volterra_lambda: float = 1.0,
    nonlinear_scale: float = 0.0,
    normalize_base: bool = False,
) -> TruthResult:
    """Build the explicit Hadamard nonlinear truth.

    The retained channel is

        Y = lambda * [((X B1^T) * tanh((X B2^T) / scale)) B3^T],

    where B1=L, B2=L^2, and B3 is a quadratic polynomial of the full
    symmetric normalized Laplacian.  The command-line controls are intentionally
    shared with the Volterra option: ``quadratic_coeffs`` controls B3,
    ``volterra_lambda`` controls lambda, and ``nonlinear_scale`` controls scale.
    Since the map is nonlinear, ``F_full`` is ``None``.
    """
    base = normalize_fro(L_full) if bool(normalize_base) else L_full
    B1 = base.contiguous()
    B2 = (base @ base).contiguous()
    B3 = polynomial_filter_from_coeffs(L_full, coeffs=quadratic_coeffs, normalize_base=normalize_base)
    X = x_full.to(device=B1.device, dtype=B1.dtype)

    z1 = X @ B1.T
    z2 = X @ B2.T

    if float(nonlinear_scale) > 0.0:
        scale = torch.as_tensor(float(nonlinear_scale), device=X.device, dtype=X.dtype)
        scale_desc = f"{float(nonlinear_scale):.6g}(fixed)"
    else:
        scale = torch.sqrt(torch.mean(z2.detach() ** 2, dim=-1, keepdim=True)).clamp_min(1e-6)
        scale_desc = "auto_per_sample_rms_of_XB2T"

    hadamard_signal = z1 * torch.tanh(z2 / scale)
    Y = float(volterra_lambda) * (hadamard_signal @ B3.T)

    with torch.no_grad():
        z1_rms = torch.sqrt(torch.mean(z1.detach() ** 2)).clamp_min(1e-12)
        z2_rms = torch.sqrt(torch.mean(z2.detach() ** 2)).clamp_min(1e-12)
        signal_rms = torch.sqrt(torch.mean(hadamard_signal.detach() ** 2)).clamp_min(1e-12)
        y_rms = torch.sqrt(torch.mean(Y.detach() ** 2)).clamp_min(1e-12)

    summary = (
        "[TruthFilter] type=hadamard | "
        "formula=y=lambda*(((XB1^T)*tanh((XB2^T)/scale))B3^T), B1=L, B2=L^2 | "
        f"B3_quadratic_coeffs={tuple(float(c) for c in quadratic_coeffs)} | "
        f"volterra_lambda={float(volterra_lambda):.6g} | "
        f"nonlinear_scale={scale_desc} | "
        f"normalize_base={bool(normalize_base)} | "
        f"rms_XB1T={float(z1_rms):.6g} | "
        f"rms_XB2T={float(z2_rms):.6g} | "
        f"rms_hadamard_signal={float(signal_rms):.6g} | "
        f"rms_y={float(y_rms):.6g} | "
        "F_full=None_nonlinear_truth"
    )
    return TruthResult(y_full=Y.contiguous(), F_full=None, summary=summary, truth_type="hadamard")

def build_truth_labels(
    x_full: Tensor,
    L_full: Tensor,
    truth_type: str = "linear_poly",
    coeffs: Sequence[float] = DEFAULT_TRUTH_COEFFS,
    quadratic_coeffs: Sequence[float] = DEFAULT_VOLTERRA_QUADRATIC_COEFFS,
    volterra_lambda: float = 1.0,
    nonlinear_scale: float = 0.0,
    center_nonlinear_input: bool = True,
    normalize_base: bool = False,
) -> TruthResult:
    """Build noiseless full-graph labels from a selected truth model."""
    t = normalized_truth_type(truth_type)
    if t == "linear_poly":
        return build_linear_poly_truth(
            L_full=L_full,
            x_full=x_full,
            coeffs=coeffs,
            normalize_base=normalize_base,
        )
    if t == "volterra":
        return build_volterra_hadamard_truth(
            L_full=L_full,
            x_full=x_full,
            quadratic_coeffs=quadratic_coeffs,
            volterra_lambda=volterra_lambda,
            nonlinear_scale=nonlinear_scale,
            center_nonlinear_input=center_nonlinear_input,
            normalize_base=normalize_base,
        )
    return build_hadamard_truth(
        L_full=L_full,
        x_full=x_full,
        quadratic_coeffs=quadratic_coeffs,
        volterra_lambda=volterra_lambda,
        nonlinear_scale=nonlinear_scale,
        normalize_base=normalize_base,
    )


def build_truth_labels_from_spec(L_full: Tensor, x_full: Tensor, spec: TruthFilterSpec) -> TruthResult:
    return build_truth_labels(
        x_full=x_full,
        L_full=L_full,
        truth_type=spec.truth_type,
        coeffs=spec.coeffs,
        quadratic_coeffs=spec.quadratic_coeffs,
        volterra_lambda=spec.volterra_lambda,
        nonlinear_scale=spec.nonlinear_scale,
        center_nonlinear_input=spec.center_nonlinear_input,
        normalize_base=spec.normalize_base,
    )


def normalized_noise_type(noise_type: str) -> str:
    n = str(noise_type).lower().strip().replace("-", "_")
    aliases = {
        "normal": "gaussian",
        "gauss": "gaussian",
        "wgn": "gaussian",
        "rad": "rademacher",
        "bernoulli_pm1": "rademacher",
        "uniform_symmetric": "uniform",
        "skew": "skew_bernoulli",
        "skewed": "skew_bernoulli",
        "skew_bernoulli": "skew_bernoulli",
        "skewed_bernoulli": "skew_bernoulli",
        "asymmetric_bernoulli": "skew_bernoulli",
    }
    n = aliases.get(n, n)
    if n not in {"gaussian", "uniform", "rademacher", "skew_bernoulli"}:
        raise ValueError("noise_type must be one of {'gaussian', 'uniform', 'rademacher', 'skew_bernoulli'}.")
    return n


def sample_noise_like(
    ref: Tensor,
    noise_type: str = "gaussian",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Return mean-zero, unit-variance sub-Gaussian noise with shape like ref.

    ``skew_bernoulli`` is a bounded two-point distribution with p=0.8:
    it returns +sqrt((1-p)/p) with probability p and -sqrt(p/(1-p))
    otherwise.  Hence E[Z]=0 and Var[Z]=1, but the law is not centrally
    symmetric.  Boundedness keeps it sub-Gaussian, and ``sigma`` still controls
    the final amplitude through Y + sigma * Z.
    """
    n = normalized_noise_type(noise_type)
    if n == "gaussian":
        return torch.randn(ref.shape, dtype=ref.dtype, device=ref.device, generator=generator)
    if n == "uniform":
        u = torch.rand(ref.shape, dtype=ref.dtype, device=ref.device, generator=generator)
        return math.sqrt(3.0) * (2.0 * u - 1.0)
    if n == "rademacher":
        z = torch.randint(0, 2, ref.shape, device=ref.device, generator=generator)
        return (2.0 * z.to(dtype=ref.dtype) - 1.0)
    # Skewed bounded Bernoulli: mean 0, variance 1, nonzero skewness.
    p = 0.8
    hi = math.sqrt((1.0 - p) / p)
    lo = -math.sqrt(p / (1.0 - p))
    u = torch.rand(ref.shape, dtype=ref.dtype, device=ref.device, generator=generator)
    return torch.where(
        u < p,
        torch.full(ref.shape, hi, dtype=ref.dtype, device=ref.device),
        torch.full(ref.shape, lo, dtype=ref.dtype, device=ref.device),
    )


def make_noisy_labels(
    x_full: Tensor,
    L_full: Tensor,
    truth_type: str = "linear_poly",
    coeffs: Sequence[float] = DEFAULT_TRUTH_COEFFS,
    quadratic_coeffs: Sequence[float] = DEFAULT_VOLTERRA_QUADRATIC_COEFFS,
    volterra_lambda: float = 1.0,
    nonlinear_scale: float = 0.0,
    center_nonlinear_input: bool = True,
    normalize_base: bool = False,
    sigma: float = 5.0,
    noise_type: str = "gaussian",
    noise_seed: Optional[int] = None,
    add_noise: bool = True,
) -> TruthResult:
    """Build noisy labels for the selected truth and noise law."""
    truth = build_truth_labels(
        x_full=x_full,
        L_full=L_full,
        truth_type=truth_type,
        coeffs=coeffs,
        quadratic_coeffs=quadratic_coeffs,
        volterra_lambda=volterra_lambda,
        nonlinear_scale=nonlinear_scale,
        center_nonlinear_input=center_nonlinear_input,
        normalize_base=normalize_base,
    )
    Y = truth.y_full
    noise_desc = "none"
    if bool(add_noise) and float(sigma) != 0.0:
        g = make_torch_generator(noise_seed, Y.device)
        ntype = normalized_noise_type(noise_type)
        noise = sample_noise_like(Y, noise_type=ntype, generator=g)
        Y = Y + float(sigma) * noise
        noise_desc = f"{ntype},sigma={float(sigma):.6g}"

    return TruthResult(
        y_full=Y.contiguous(),
        F_full=None if truth.F_full is None else truth.F_full.contiguous(),
        summary=f"{truth.summary} | noise={noise_desc}",
        truth_type=truth.truth_type,
    )


def make_linear_noisy_labels(
    x_full: Tensor,
    L_full: Tensor,
    coeffs: Sequence[float] = DEFAULT_TRUTH_COEFFS,
    sigma: float = 5.0,
    noise_seed: Optional[int] = None,
    normalize_base: bool = False,
    noise_type: str = "gaussian",
) -> Tuple[Tensor, Tensor]:
    """Backward-compatible wrapper for the old linear-polynomial interface."""
    result = make_noisy_labels(
        x_full=x_full,
        L_full=L_full,
        truth_type="linear_poly",
        coeffs=coeffs,
        sigma=sigma,
        noise_seed=noise_seed,
        normalize_base=normalize_base,
        noise_type=noise_type,
        add_noise=True,
    )
    if result.F_full is None:
        raise RuntimeError("linear truth unexpectedly returned F_full=None.")
    return result.F_full, result.y_full


# -----------------------------------------------------------------------------
# Basis construction
# -----------------------------------------------------------------------------


def polynomial_bases(S: Tensor, degree: int = 3, normalize_base: bool = False) -> List[Tensor]:
    """Return [I, S, ..., S^degree], optionally after Frobenius normalization."""
    if int(degree) < 0:
        raise ValueError("degree must be non-negative.")
    if S.dim() != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be square.")
    base = normalize_fro(S) if normalize_base else S
    n = int(base.shape[0])
    I = torch.eye(n, dtype=base.dtype, device=base.device)
    out = [I]
    cur = I
    for _ in range(int(degree)):
        cur = cur @ base
        out.append(cur)
    return out


def multi_polynomial_bases(
    primitives: Sequence[Tensor],
    degree: int = 3,
    path_order: bool = False,
    normalize_base: bool = False,
) -> List[Tensor]:
    """Polynomial bases for several primitives with per-primitive coefficients."""
    if len(primitives) == 0:
        raise ValueError("primitives must be non-empty.")
    out: List[Tensor] = []
    for i, S in enumerate(primitives):
        hop_base = i + 1
        max_deg = int(degree)
        if path_order:
            max_deg = int(degree) // hop_base
        out.extend(polynomial_bases(S, degree=max_deg, normalize_base=normalize_base))
    return out


def sffa_basis_size(num_primitives: int, word_len: int, include_identity: bool = True) -> int:
    if int(num_primitives) <= 0:
        raise ValueError("num_primitives must be positive.")
    if int(word_len) < 0:
        raise ValueError("word_len must be non-negative.")
    q = int(num_primitives)
    r = int(word_len)
    if q == 1:
        return int(r + 1 if include_identity else r)
    if include_identity:
        return int((q ** (r + 1) - 1) // (q - 1))
    return int((q ** (r + 1) - q) // (q - 1))


def sffa_word_bases(
    primitives: Sequence[Tensor],
    word_len: int = 3,
    include_identity: bool = True,
    max_basis_size: int = 2000,
    normalize_primitives: bool = False,
) -> List[Tensor]:
    """Enumerate noncommutative word bases up to ``word_len``."""
    if len(primitives) == 0:
        raise ValueError("primitives must be non-empty.")
    q = len(primitives)
    size = sffa_basis_size(q, int(word_len), include_identity=include_identity)
    if int(max_basis_size) > 0 and size > int(max_basis_size):
        raise RuntimeError(
            f"SFFA basis would contain {size} matrices with q={q}, r={word_len}; "
            f"max_basis_size={max_basis_size}."
        )

    prim = [normalize_fro(P) if normalize_primitives else P for P in primitives]
    n = int(prim[0].shape[0])
    for P in prim:
        if P.shape != prim[0].shape:
            raise ValueError("All SFFA primitives must have the same shape.")

    I = torch.eye(n, dtype=prim[0].dtype, device=prim[0].device)
    bases: List[Tensor] = []
    if include_identity:
        bases.append(I)

    current_words = [I]
    for _length in range(1, int(word_len) + 1):
        next_words: List[Tensor] = []
        for B in current_words:
            for P in prim:
                next_words.append(B @ P)
        bases.extend(next_words)
        current_words = next_words
    return bases


def distance_sffa_primitives(distance_laps: Sequence[Tensor], k: int) -> List[Tensor]:
    """Primitives for pure distance-based Laplacian SFFA."""
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if len(distance_laps) < k:
        raise ValueError(f"Need at least {k} distance Laplacians; got {len(distance_laps)}.")
    return list(distance_laps[:k])


def standard_sffa_primitives(
    induced_L: Tensor,
    distance_laps: Sequence[Tensor],
    k: int,
) -> List[Tensor]:
    """Backward-compatible alias for pure distance-based SFFA primitives."""
    del induced_L
    return distance_sffa_primitives(distance_laps, k=k)


# -----------------------------------------------------------------------------
# Losses, fitting, and evaluation
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BasisFitResult:
    theta: Tensor
    operator: Tensor
    train_loss: float
    num_parameters: int
    loss_type: str

    @property
    def train_mse(self) -> float:
        """Backward-compatible alias; equals train_loss for non-MSE runs."""
        return self.train_loss


def stack_bases(bases: Sequence[Tensor]) -> Tensor:
    if len(bases) == 0:
        raise ValueError("bases must be non-empty.")
    B0 = bases[0]
    for B in bases:
        if B.shape != B0.shape:
            raise ValueError("All bases must have the same shape.")
    return torch.stack([B.detach().clone().to(device=B0.device, dtype=B0.dtype) for B in bases], dim=0)


def combine_bases(theta: Tensor, bases_tensor: Tensor) -> Tensor:
    return torch.einsum("m,mij->ij", theta.to(device=bases_tensor.device, dtype=bases_tensor.dtype), bases_tensor)


def normalized_loss_type(loss_type: str) -> str:
    loss_type = str(loss_type).lower().strip().replace("-", "_")
    aliases = {
        "l2": "mse",
        "square": "mse",
        "squared": "mse",
        "huber_loss": "huber",
        "log_cosh": "logcosh",
        "logcosh_loss": "logcosh",
        "pinball": "smooth_pinball",
        "pinball_loss": "smooth_pinball",
        "smooth_pinball_loss": "smooth_pinball",
        "quantile": "smooth_pinball",
        "quantile_loss": "smooth_pinball",
    }
    loss_type = aliases.get(loss_type, loss_type)
    if loss_type not in {"mse", "huber", "logcosh", "smooth_pinball"}:
        raise ValueError("loss_type must be one of {'mse', 'huber', 'logcosh', 'smooth_pinball'}.")
    return loss_type


def loss_tensor(
    y_pred: Tensor,
    y_true: Tensor,
    loss_type: str = "mse",
    *,
    huber_delta: float = 1.0,
    tau_pinball: float = 0.8,
) -> Tensor:
    loss_type = normalized_loss_type(loss_type)
    if loss_type == "mse":
        return torch.mean((y_pred - y_true) ** 2)
    delta = float(huber_delta)
    if delta <= 0.0:
        raise ValueError("huber_delta must be positive.")
    if loss_type == "huber":
        return Fnn.huber_loss(y_pred, y_true, reduction="mean", delta=delta)
    if loss_type == "smooth_pinball":
        tau = float(tau_pinball)
        if not (0.0 < tau < 1.0):
            raise ValueError("tau_pinball must lie in (0, 1).")
        # Smooth pinball / quantile loss using softplus-smoothed one-sided
        # linear pieces.  With residual r=pred-true, its asymptotic slopes are
        # tau on r>0 and tau-1 on r<0.  huber_delta is reused as the smoothing
        # temperature to keep the CLI minimally invasive.
        z = (y_pred - y_true) / delta
        return torch.mean(delta * (tau * Fnn.softplus(z) + (1.0 - tau) * Fnn.softplus(-z)))
    # Smooth convex robust loss: delta^2 * log(cosh((pred-true)/delta)).
    # The stable identity log(cosh z)=z+softplus(-2z)-log(2) avoids overflow.
    z = (y_pred - y_true) / delta
    return torch.mean((delta ** 2) * (z + Fnn.softplus(-2.0 * z) - math.log(2.0)))


def loss_value(
    y_pred: Tensor,
    y_true: Tensor,
    loss_type: str = "mse",
    *,
    huber_delta: float = 1.0,
    tau_pinball: float = 0.8,
) -> float:
    return float(
        loss_tensor(
            y_pred,
            y_true,
            loss_type=loss_type,
            huber_delta=huber_delta,
            tau_pinball=tau_pinball,
        ).item()
    )


def mse_loss(y_pred: Tensor, y_true: Tensor) -> float:
    return loss_value(y_pred, y_true, loss_type="mse")


def huber_loss(y_pred: Tensor, y_true: Tensor, huber_delta: float = 1.0) -> float:
    return loss_value(y_pred, y_true, loss_type="huber", huber_delta=huber_delta)


def logcosh_loss(y_pred: Tensor, y_true: Tensor, huber_delta: float = 1.0) -> float:
    return loss_value(y_pred, y_true, loss_type="logcosh", huber_delta=huber_delta)


def smooth_pinball_loss(
    y_pred: Tensor,
    y_true: Tensor,
    huber_delta: float = 1.0,
    tau_pinball: float = 0.8,
) -> float:
    return loss_value(
        y_pred,
        y_true,
        loss_type="smooth_pinball",
        huber_delta=huber_delta,
        tau_pinball=tau_pinball,
    )


def fit_basis_operator(
    x_train: Tensor,
    y_train: Tensor,
    bases: Sequence[Tensor],
    epochs: int = 20,
    lr: float = 0.5,
    ridge: float = 0.0,
    optimizer_name: str = "lbfgs",
    lbfgs_max_iter: int = 50,
    init_scale: float = 0.0,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    tau_pinball: float = 0.8,
) -> BasisFitResult:
    """Fit H=sum_j theta_j B_j by minimizing MSE, Huber, log-cosh, or smooth pinball loss.

    LBFGS is the default optimizer because the basis problem is low-dimensional;
    log-cosh and smooth pinball are smooth and convex in the residual, which makes
    them friendly non-quadratic options for LBFGS.  Adam remains available through
    ``optimizer_name='adam'``.
    """
    if x_train.dim() != 2 or y_train.dim() != 2:
        raise ValueError("x_train and y_train must be [T, n].")
    if x_train.shape != y_train.shape:
        raise ValueError("x_train and y_train must have the same shape.")
    loss_type = normalized_loss_type(loss_type)
    B = stack_bases(bases).to(device=x_train.device, dtype=x_train.dtype)
    m = int(B.shape[0])
    if float(init_scale) == 0.0:
        theta = torch.zeros(m, dtype=x_train.dtype, device=x_train.device, requires_grad=True)
    else:
        theta = (float(init_scale) * torch.randn(m, dtype=x_train.dtype, device=x_train.device)).requires_grad_()

    def objective() -> Tensor:
        H = combine_bases(theta, B)
        pred = x_train @ H.T
        fit = loss_tensor(
            pred,
            y_train,
            loss_type=loss_type,
            huber_delta=huber_delta,
            tau_pinball=tau_pinball,
        )
        if float(ridge) != 0.0:
            reg = float(ridge) * torch.mean(H * H)
            return fit + reg
        return fit

    opt_name = str(optimizer_name).lower().strip()
    if opt_name == "adam":
        opt = torch.optim.Adam([theta], lr=float(lr))
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = objective()
            loss.backward()
            opt.step()
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS(
            [theta],
            lr=float(lr),
            max_iter=int(lbfgs_max_iter),
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        for _ in range(int(epochs)):
            opt.step(closure)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'lbfgs'.")

    with torch.no_grad():
        H = combine_bases(theta, B).detach().clone()
        train_loss = loss_value(
            x_train @ H.T,
            y_train,
            loss_type=loss_type,
            huber_delta=huber_delta,
            tau_pinball=tau_pinball,
        )
    return BasisFitResult(
        theta=theta.detach().clone(),
        operator=H,
        train_loss=train_loss,
        num_parameters=m,
        loss_type=loss_type,
    )


def fit_lmmse_operator(x_train: Tensor, y_train: Tensor, ridge: float = 1e-9) -> Tensor:
    """Closed-form ridge LMMSE for y_hat = x @ H.T.

    This is always the MSE-trained numerical LMMSE baseline.  Under Huber
    evaluation it is intentionally not re-optimized for the Huber objective.
    """
    if x_train.dim() != 2 or y_train.dim() != 2:
        raise ValueError("x_train and y_train must be [T, n].")
    if x_train.shape != y_train.shape:
        raise ValueError("x_train and y_train must have the same shape.")
    X = x_train
    Y = y_train
    n = int(X.shape[1])
    eye = torch.eye(n, dtype=X.dtype, device=X.device)
    gram = X.T @ X
    rhs = X.T @ Y
    W = torch.linalg.solve(gram + float(ridge) * X.shape[0] * eye, rhs)
    return W.T.contiguous()


def apply_operator(x: Tensor, H: Tensor) -> Tensor:
    if x.dim() != 2:
        raise ValueError("x must have shape [T, n].")
    return x @ H.to(device=x.device, dtype=x.dtype).T


def eval_operator_loss(
    x_test: Tensor,
    y_test: Tensor,
    H: Tensor,
    loss_type: str = "mse",
    *,
    huber_delta: float = 1.0,
    tau_pinball: float = 0.8,
) -> float:
    with torch.no_grad():
        pred = apply_operator(x_test, H)
        return loss_value(
            pred,
            y_test,
            loss_type=loss_type,
            huber_delta=huber_delta,
            tau_pinball=tau_pinball,
        )


def eval_operator_mse(x_test: Tensor, y_test: Tensor, H: Tensor) -> float:
    return eval_operator_loss(x_test, y_test, H, loss_type="mse")


def eval_operator_huber(x_test: Tensor, y_test: Tensor, H: Tensor, huber_delta: float = 1.0) -> float:
    return eval_operator_loss(x_test, y_test, H, loss_type="huber", huber_delta=huber_delta)


def eval_operator_logcosh(x_test: Tensor, y_test: Tensor, H: Tensor, huber_delta: float = 1.0) -> float:
    return eval_operator_loss(x_test, y_test, H, loss_type="logcosh", huber_delta=huber_delta)


def eval_operator_smooth_pinball(
    x_test: Tensor,
    y_test: Tensor,
    H: Tensor,
    huber_delta: float = 1.0,
    tau_pinball: float = 0.8,
) -> float:
    return eval_operator_loss(
        x_test,
        y_test,
        H,
        loss_type="smooth_pinball",
        huber_delta=huber_delta,
        tau_pinball=tau_pinball,
    )


def train_eval_basis_loss(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    bases: Sequence[Tensor],
    epochs: int = 20,
    lr: float = 0.5,
    ridge: float = 0.0,
    optimizer_name: str = "lbfgs",
    lbfgs_max_iter: int = 50,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    tau_pinball: float = 0.8,
) -> Tuple[float, BasisFitResult]:
    fit = fit_basis_operator(
        x_train=x_train,
        y_train=y_train,
        bases=bases,
        epochs=epochs,
        lr=lr,
        ridge=ridge,
        optimizer_name=optimizer_name,
        lbfgs_max_iter=lbfgs_max_iter,
        loss_type=loss_type,
        huber_delta=huber_delta,
        tau_pinball=tau_pinball,
    )
    test_loss = eval_operator_loss(
        x_test,
        y_test,
        fit.operator,
        loss_type=loss_type,
        huber_delta=huber_delta,
        tau_pinball=tau_pinball,
    )
    return test_loss, fit


def train_eval_basis_mse(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    bases: Sequence[Tensor],
    epochs: int = 20,
    lr: float = 0.5,
    ridge: float = 0.0,
    optimizer_name: str = "lbfgs",
    lbfgs_max_iter: int = 50,
) -> Tuple[float, BasisFitResult]:
    return train_eval_basis_loss(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        bases=bases,
        epochs=epochs,
        lr=lr,
        ridge=ridge,
        optimizer_name=optimizer_name,
        lbfgs_max_iter=lbfgs_max_iter,
        loss_type="mse",
    )


# -----------------------------------------------------------------------------
# Experiment slicing helpers and lazy primitive cache
# -----------------------------------------------------------------------------


def choose_v0(
    num_nodes: int,
    p_visible: float,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tensor:
    if not (0.0 < float(p_visible) <= 1.0):
        raise ValueError("p_visible must lie in (0, 1].")
    n0 = max(1, int(float(p_visible) * int(num_nodes)))

    if seed is None:
        return torch.randperm(int(num_nodes), device=device)[:n0].to(dtype=torch.long)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randperm(int(num_nodes), generator=g)[:n0].to(dtype=torch.long)
    if device is not None:
        idx = idx.to(device=device)
    return idx


def split_train_test_number(
    x: Tensor,
    y: Tensor,
    train_steps: int = 50,
    test_steps: int = 50,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be [T, n].")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    train_steps = int(train_steps)
    test_steps = int(test_steps)
    if train_steps <= 0 or test_steps <= 0:
        raise ValueError("train_steps and test_steps must be positive.")
    if train_steps + test_steps > int(x.shape[0]):
        raise ValueError("train_steps + test_steps exceeds available samples.")
    return (
        x[:train_steps],
        y[:train_steps],
        x[train_steps : train_steps + test_steps],
        y[train_steps : train_steps + test_steps],
    )


def split_train_test_ratio(
    x: Tensor,
    y: Tensor,
    train_ratio: float = 0.7,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be [T, n].")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if not (0.0 < float(train_ratio) < 1.0):
        raise ValueError("train_ratio must lie in (0, 1).")
    T = int(x.shape[0])
    split = int(T * float(train_ratio))
    if split <= 0 or split >= T:
        raise ValueError(f"train_ratio={train_ratio} produced an invalid split for T={T}.")
    return x[:split], y[:split], x[split:], y[split:]


def split_train_test(
    x: Tensor,
    y: Tensor,
    train_steps: int = 50,
    test_steps: int = 50,
    *,
    split_mode: str = "number",
    train_ratio: float = 0.7,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split filtering samples by counts or by sequential train ratio."""
    mode = str(split_mode).lower().strip()
    if mode in {"number", "count", "counts"}:
        return split_train_test_number(x, y, train_steps=train_steps, test_steps=test_steps)
    if mode in {"ratio", "fraction"}:
        return split_train_test_ratio(x, y, train_ratio=train_ratio)
    raise ValueError("split_mode must be 'number' or 'ratio'.")


def split_train_test_with_meta(
    x: Tensor,
    y: Tensor,
    train_steps: int = 50,
    test_steps: int = 50,
    *,
    split_mode: str = "number",
    train_ratio: float = 0.7,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Union[int, float, str]]]:
    """Split filtering samples and return metadata for CSV/JSON records."""
    T = int(x.shape[0])
    mode = str(split_mode).lower().strip()
    if mode in {"number", "count", "counts"}:
        out = split_train_test_number(x, y, train_steps=train_steps, test_steps=test_steps)
        meta: Dict[str, Union[int, float, str]] = {
            "split_mode": "number",
            "clean_T": T,
            "train_samples": int(train_steps),
            "test_samples": int(test_steps),
            "split_index": int(train_steps),
            "train_ratio": float(train_steps) / float(T),
        }
        return (*out, meta)
    if mode in {"ratio", "fraction"}:
        split = int(T * float(train_ratio))
        out = split_train_test_ratio(x, y, train_ratio=train_ratio)
        meta = {
            "split_mode": "ratio",
            "clean_T": T,
            "train_samples": int(split),
            "test_samples": int(T - split),
            "split_index": int(split),
            "train_ratio": float(train_ratio),
        }
        return (*out, meta)
    raise ValueError("split_mode must be 'number' or 'ratio'.")


@dataclass
class PrimitiveCache:
    """Method-aware lazy matrix cache for one sampled V0."""

    A_full: Tensor
    L_full: Tensor
    graph_nx: nx.Graph
    v0_idx: Tensor
    dtype: torch.dtype = torch.float32
    device: Union[str, torch.device] = "cpu"
    zero_isolated: bool = False
    kron_ridge: float = 0.0
    kron_use_pinv: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.v0_idx = self.v0_idx.to(device=self.device, dtype=torch.long)
        self.A_full = self.A_full.to(device=self.device, dtype=self.dtype)
        self.L_full = self.L_full.to(device=self.device, dtype=self.dtype)
        self._cache: Dict[str, object] = {}

    @property
    def n0(self) -> int:
        return int(self.v0_idx.numel())

    def induced(self) -> Tensor:
        key = "induced"
        if key not in self._cache:
            self._cache[key] = induced_laplacian(self.A_full, self.v0_idx, zero_isolated=self.zero_isolated)
        return self._cache[key]  # type: ignore[return-value]

    def kron(self) -> Tensor:
        key = "kron"
        if key not in self._cache:
            self._cache[key] = kron_reduced_laplacian(
                self.L_full,
                self.v0_idx,
                ridge=float(self.kron_ridge),
                use_pinv=bool(self.kron_use_pinv),
            )
        return self._cache[key]  # type: ignore[return-value]

    def distance_laps(self, k_max: int) -> List[Tensor]:
        k_max = int(k_max)
        if k_max < 1:
            raise ValueError("k_max must be >= 1.")
        key = f"distance_{k_max}"
        if key not in self._cache:
            best_key = None
            best_k = 0
            for existing in self._cache:
                if existing.startswith("distance_"):
                    try:
                        kk = int(existing.split("_", 1)[1])
                    except ValueError:
                        continue
                    if kk >= k_max and (best_key is None or kk < best_k):
                        best_key, best_k = existing, kk
            if best_key is not None:
                self._cache[key] = list(self._cache[best_key])[:k_max]  # type: ignore[arg-type]
            else:
                self._cache[key] = distance_k_laplacians(
                    self.graph_nx,
                    self.v0_idx,
                    k_max=k_max,
                    dtype=self.dtype,
                    device=self.device,
                    zero_isolated=self.zero_isolated,
                )
        return self._cache[key]  # type: ignore[return-value]

    def union(self, k_max: int) -> Tensor:
        k_max = int(k_max)
        if k_max < 1:
            raise ValueError("k_max must be >= 1.")
        key = f"union_{k_max}"
        if key not in self._cache:
            self._cache[key] = union_distance_laplacian(
                self.graph_nx,
                self.v0_idx,
                ks=list(range(1, k_max + 1)),
                dtype=self.dtype,
                device=self.device,
                zero_isolated=self.zero_isolated,
            )
        return self._cache[key]  # type: ignore[return-value]

    def random_groups(self, num_groups: int = 3, p0: float = 0.9) -> List[Tensor]:
        key = f"random_groups_{int(num_groups)}_{float(p0):.8f}"
        if key not in self._cache:
            self._cache[key] = random_group_laplacians(
                self.graph_nx,
                self.v0_idx,
                num_groups=int(num_groups),
                p0=float(p0),
                dtype=self.dtype,
                device=self.device,
                zero_isolated=self.zero_isolated,
            )
        return self._cache[key]  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Method basis factory
# -----------------------------------------------------------------------------


def build_method_bases(
    method: str,
    cache: PrimitiveCache,
    r_poly: int = 3,
    k_union: int = 3,
    k_sffa: int = 3,
    r_sffa: int = 3,
    rg_p0: float = 0.9,
    max_sffa_basis_size: int = 2000,
    *,
    # Backward-compatible names accepted but no longer preferred.
    poly_degree: Optional[int] = None,
    sffa_k: Optional[int] = None,
    sffa_r: Optional[int] = None,
    rg_num: Optional[int] = None,
) -> Optional[List[Tensor]]:
    """Return trainable bases for a named filtering method.

    New parameter semantics:
      - ``r_poly`` controls every polynomial degree: induced, Kron, distance-k,
        and union polynomials.
      - ``k_union`` controls how many distance graphs are unioned in ``union_lp``.
      - ``k_sffa`` and ``r_sffa`` control ``dk_lap_poly``, ``sffa``, and ``rg_alg``.
        Thus ``dk_lap_poly`` and ``sffa`` use the same distance scale and order
        for a direct polynomial-vs-free-algebra comparison.  For ``rg_alg``,
        ``k_sffa`` is the number of random groups.

    Removed methods intentionally return ``None``: ``rg_lp``, ``kron12``,
    ``kron_sffa``, and all direct/truncated operators.
    """
    method = str(method).strip()
    if poly_degree is not None:
        r_poly = int(poly_degree)
    if sffa_k is not None:
        k_sffa = int(sffa_k)
    if sffa_r is not None:
        r_sffa = int(sffa_r)
    # ``rg_num`` is intentionally ignored: random-group algebra now uses k_sffa.
    _ = rg_num

    r_poly = int(r_poly)
    k_union = int(k_union)
    k_sffa = int(k_sffa)
    r_sffa = int(r_sffa)

    if method in {"plain_trunc", "truncated", "induced_direct", "kron_direct", "union_direct", "rg_lp", "kron12", "kron_sffa"}:
        return None

    if method in {"sub_lp", "induced_poly"}:
        return polynomial_bases(cache.induced(), degree=r_poly)

    if method in {"dk_lap_poly", "dk_lp", "distance_poly"}:
        return multi_polynomial_bases(
            cache.distance_laps(k_sffa),
            degree=r_sffa,
            path_order=False,
        )

    if method in {"kron_lp", "kron_poly"}:
        return polynomial_bases(cache.kron(), degree=r_poly)

    if method in {"union_lp", "union_poly"}:
        return polynomial_bases(cache.union(k_union), degree=r_poly)

    if method in {"rg_alg", "random_group_alg", "random_group_algebra"}:
        return sffa_word_bases(
            cache.random_groups(k_sffa, rg_p0),
            word_len=r_sffa,
            include_identity=True,
            max_basis_size=max_sffa_basis_size,
        )

    if method in {"sffa", "distance_sffa", "dk_alg"}:
        prim = distance_sffa_primitives(cache.distance_laps(k_sffa), k=k_sffa)
        return sffa_word_bases(
            prim,
            word_len=r_sffa,
            include_identity=True,
            max_basis_size=max_sffa_basis_size,
        )

    return None


__all__ = [
    "TrafficData",
    "PrimitiveCache",
    "TruthFilterSpec",
    "TruthResult",
    "BasisFitResult",
    "TRAINABLE_METHODS",
    "CLOSED_FORM_METHODS",
    "DEFAULT_METHODS",
    "METHOD_LABELS",
    "DEFAULT_TRUTH_COEFFS",
    "DEFAULT_VOLTERRA_QUADRATIC_COEFFS",
    "set_seed",
    "run_seed_sequence",
    "resolve_device",
    "make_torch_generator",
    "load_metr_la_traffic",
    "full_laplacian_from_graph",
    "symmetric_normalized_laplacian_from_adjacency",
    "induced_laplacian",
    "distance_k_laplacians",
    "union_laplacian_from_distance_laplacians",
    "union_distance_laplacian",
    "random_group_laplacians",
    "kron_reduced_laplacian",
    "polynomial_filter_from_coeffs",
    "build_linear_poly_truth",
    "build_volterra_hadamard_truth",
    "build_truth_labels",
    "build_truth_labels_from_spec",
    "make_noisy_labels",
    "make_linear_noisy_labels",
    "normalized_noise_type",
    "sample_noise_like",
    "polynomial_bases",
    "multi_polynomial_bases",
    "sffa_basis_size",
    "sffa_word_bases",
    "distance_sffa_primitives",
    "standard_sffa_primitives",
    "normalized_loss_type",
    "loss_tensor",
    "loss_value",
    "mse_loss",
    "huber_loss",
    "logcosh_loss",
    "smooth_pinball_loss",
    "fit_basis_operator",
    "fit_lmmse_operator",
    "apply_operator",
    "eval_operator_loss",
    "eval_operator_mse",
    "eval_operator_huber",
    "eval_operator_logcosh",
    "eval_operator_smooth_pinball",
    "train_eval_basis_loss",
    "train_eval_basis_mse",
    "choose_v0",
    "split_train_test_number",
    "split_train_test_ratio",
    "split_train_test",
    "split_train_test_with_meta",
    "build_method_bases",
]
