"""Core utilities for horizon Subgraph Filter Learning experiments.

This module keeps the graph primitive, filter-bank, loss, and one-step fitting
utilities from ``sfl_core_msemae.py`` and adds horizon/window prediction routines
for traffic forecasting.  The horizon task uses input windows X with shape
[S, P, N] and targets Y with shape [S, H, N].

For an SFL basis {B_j}, the horizon model is

    Y_hat[:, h, :] = sum_p X[:, p, :] @ H_{p,h}.T,
    H_{p,h} = sum_j theta[p,h,j] B_j.

Thus every past-to-future pair has its own coefficients, while all pairs share
the same graph support / filter bank.  Dense Laplacian primitives are retained
because METR-LA scale is small enough for the intended experiments.
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data


NodeIds = Union[Sequence[int], Tensor]


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


def as_long_tensor(idx: NodeIds, device: Optional[torch.device] = None) -> Tensor:
    if isinstance(idx, Tensor):
        return idx.to(device=device, dtype=torch.long)
    return torch.tensor([int(v) for v in idx], dtype=torch.long, device=device)


def node_ids_list(idx: NodeIds) -> List[int]:
    if isinstance(idx, Tensor):
        return [int(v) for v in idx.detach().cpu().tolist()]
    return [int(v) for v in idx]


def submatrix(M: Tensor, row_idx: NodeIds, col_idx: Optional[NodeIds] = None) -> Tensor:
    """Return M[row_idx, col_idx], preserving the order of the supplied ids."""
    r = as_long_tensor(row_idx, device=M.device)
    c = r if col_idx is None else as_long_tensor(col_idx, device=M.device)
    return M.index_select(0, r).index_select(1, c)


def symmetrize(M: Tensor) -> Tensor:
    return 0.5 * (M + M.T)


def normalize_fro(S: Tensor, eps: float = 1e-12) -> Tensor:
    """Frobenius-normalize an operator, used before polynomial/SFFA powers."""
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


def default_dataset_dir() -> Path:
    """Assume this file lives in <project_root>/revisedfilter/."""
    return Path(__file__).resolve().parents[1] / "dataset"


def load_metr_la_traffic(
    num_timesteps: int = 100,
    dataset_dir: Optional[Union[str, Path]] = None,
    adj_filename: str = "adj_METR-LA.pkl",
    h5_filename: str = "METR-LA.h5",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> TrafficData:
    """Load the METR-LA graph and clean traffic signal matrix.

    The graph handling follows the old experiment: remove self-loops, symmetrize,
    binarize, and use unit edge weights.  The signal handling also matches the old
    code: keep only rows with nonzero values at every sensor, then truncate to
    num_timesteps.
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
    N = int(A.shape[0])

    edge_index = torch.nonzero(A, as_tuple=False).T.contiguous()
    edge_weight = torch.ones(edge_index.shape[1], dtype=dtype)
    graph = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=N)

    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(N))
    graph_nx.add_edges_from(edge_index.T.detach().cpu().numpy().tolist())

    with h5py.File(str(h5_path), "r") as f:
        raw_values = f["df"]["block0_values"][:]
        # Reading axis1 keeps parity with old loader and validates the h5 layout.
        _ = pd.to_datetime(f["df"]["axis1"][:])

    valid_mask = (raw_values != 0).all(axis=1)
    clean_values = raw_values[valid_mask]
    if num_timesteps is not None:
        clean_values = clean_values[: int(num_timesteps)]
    if clean_values.shape[0] == 0:
        raise ValueError("No valid METR-LA samples remained after nonzero filtering.")

    x_full = torch.as_tensor(clean_values, dtype=dtype, device=device)
    A = A.to(device=device)
    graph.edge_index = graph.edge_index.to(device=device)
    graph.edge_attr = graph.edge_attr.to(device=device)
    return TrafficData(graph=graph, graph_nx=graph_nx, x_full=x_full, num_nodes=N, adjacency=A)


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
    """Return L_sym = I - D^{-1/2} A D^{-1/2}.

    If zero_isolated is False, isolated nodes get diagonal 1, matching the usual
    PyG normalized Laplacian behavior.  If True, isolated-node diagonals are zero,
    matching the old project's sym_zero variant.
    """
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
    """Build distance-k subgraph symmetric normalized Laplacians for k=1..k_max.

    Distances are measured in the ambient graph, while nodes are restricted to V0.
    The implementation computes shortest paths only up to k_max from V0 nodes.
    """
    if k_max < 1:
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
    """Build a union graph Laplacian from distance-k Laplacians.

    Since each distance Laplacian is normalized, we recover its support from
    negative off-diagonal entries and then re-normalize the edge union.
    """
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


def random_group_laplacians(
    graph_nx: nx.Graph,
    v0_idx: NodeIds,
    num_groups: int = 3,
    p0: float = 0.9,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    zero_isolated: bool = False,
) -> List[Tensor]:
    """Build random induced-subgraph Laplacians embedded in V0 coordinates.

    Each group samples floor(p0 * |V0|) nodes from V0, induces the ambient graph on
    them, and embeds the resulting normalized Laplacian into an identity-like V0
    matrix by leaving inactive nodes isolated.  For normalized Laplacians this is
    equivalent to building the normalized Laplacian of the padded edge set.
    """
    if num_groups <= 0:
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
    """Kron-reduce L_full onto V0 via a ridge-stabilized Schur complement.

    L_K = L_00 - L_0c (L_cc + ridge I)^dagger L_c0.
    If use_pinv=False, torch.linalg.solve is used instead of pinv.
    """
    if L_full.dim() != 2 or L_full.shape[0] != L_full.shape[1]:
        raise ValueError("L_full must be square.")
    N = int(L_full.shape[0])
    keep = as_long_tensor(v0_idx, device=L_full.device)
    if keep.numel() == 0:
        raise ValueError("v0_idx must be non-empty.")
    if torch.unique(keep).numel() != keep.numel():
        raise ValueError("v0_idx contains duplicate nodes.")

    mask = torch.ones(N, dtype=torch.bool, device=L_full.device)
    mask[keep] = False
    comp = torch.arange(N, dtype=torch.long, device=L_full.device)[mask]

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
# Truth filter and labels: y = xF^T + w
# -----------------------------------------------------------------------------


DEFAULT_TRUTH_COEFFS: Tuple[float, ...] = (-1.0, 9.0, -12.0, 4.0)


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


def make_linear_noisy_labels(
    x_full: Tensor,
    L_full: Tensor,
    coeffs: Sequence[float] = DEFAULT_TRUTH_COEFFS,
    sigma: float = 5.0,
    noise_seed: Optional[int] = None,
    normalize_base: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Build F and Y for y = xF^T + sigma * N(0, I)."""
    F = polynomial_filter_from_coeffs(L_full, coeffs=coeffs, normalize_base=normalize_base)
    X = x_full.to(device=F.device, dtype=F.dtype)
    Y = X @ F.T
    if float(sigma) != 0.0:
        if noise_seed is None:
            noise = torch.randn_like(Y)
        else:
            g = torch.Generator(device=Y.device)
            g.manual_seed(int(noise_seed))
            noise = torch.randn(Y.shape, dtype=Y.dtype, device=Y.device, generator=g)
        Y = Y + float(sigma) * noise
    return F.contiguous(), Y.contiguous()


# -----------------------------------------------------------------------------
# Basis construction
# -----------------------------------------------------------------------------


def polynomial_bases(S: Tensor, degree: int = 3, normalize_base: bool = False) -> List[Tensor]:
    """Return [I, S, ..., S^degree], optionally after Frobenius normalization."""
    if degree < 0:
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
    """Polynomial basis for several primitives with per-primitive coefficients.

    If path_order=True, primitive i is treated as having hop cost i+1 and only
    powers with (i+1)*degree <= degree_limit are kept, matching the old dk_lp
    behavior.  Identity terms are intentionally retained per primitive to match
    the old ModularFilterGenerator parameterization.
    """
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
    if num_primitives <= 0:
        raise ValueError("num_primitives must be positive.")
    if word_len < 0:
        raise ValueError("word_len must be non-negative.")
    if num_primitives == 1:
        return int(word_len + 1 if include_identity else word_len)
    if include_identity:
        return int((num_primitives ** (word_len + 1) - 1) // (num_primitives - 1))
    return int((num_primitives ** (word_len + 1) - num_primitives) // (num_primitives - 1))


def sffa_word_bases(
    primitives: Sequence[Tensor],
    word_len: int = 3,
    include_identity: bool = True,
    max_basis_size: int = 2000,
    normalize_primitives: bool = False,
) -> List[Tensor]:
    """Enumerate noncommutative word bases up to word_len.

    The bases are I and all products P_{i1}...P_{il}, l <= word_len.  This is a
    materialized dense basis list for efficient repeated training epochs.
    """
    if len(primitives) == 0:
        raise ValueError("primitives must be non-empty.")
    q = len(primitives)
    size = sffa_basis_size(q, int(word_len), include_identity=include_identity)
    if max_basis_size > 0 and size > int(max_basis_size):
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


def standard_sffa_primitives(
    induced_L: Tensor,
    distance_laps: Sequence[Tensor],
    k: int,
) -> List[Tensor]:
    """Primitives for ordinary (k,r)-SFFA.

    Degeneration rule requested for the simplified experiment:
      - k == 1: ordinary SFFA degenerates to induced-subgraph polynomial, so the
        primitive list is [induced_L].
      - k >= 2: use the first k distance-based subgraph Laplacians.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k == 1:
        return [induced_L]
    if len(distance_laps) < k:
        raise ValueError(f"Need at least {k} distance Laplacians; got {len(distance_laps)}.")
    return list(distance_laps[:k])


def kron_sffa_primitives(
    distance_laps: Sequence[Tensor],
    kron_L: Tensor,
    k: int,
) -> List[Tensor]:
    """Primitives for the Kron-enhanced (k,r)-SFFA.

    The k-matrix set contains the first k-1 distance-based normalized Laplacians
    plus one Kron Laplacian.  Thus:
      - k == 1: [kron_L], hence the model degenerates to Kron polynomial.
      - k >= 2: [distance_1, ..., distance_{k-1}, kron_L].
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k == 1:
        return [kron_L]
    if len(distance_laps) < k - 1:
        raise ValueError(f"Need at least {k - 1} distance Laplacians.")
    return [kron_L] + list(distance_laps[: k - 1])


# -----------------------------------------------------------------------------
# Linear basis fitting and direct evaluation
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BasisFitResult:
    theta: Tensor
    operator: Tensor
    train_loss: float
    num_parameters: int

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
    loss_type = str(loss_type).lower().strip()
    if loss_type not in {"mse", "mae", "maskedmape"}:
        raise ValueError("loss_type must be one of {'mse', 'mae', 'maskedmape'}.")
    return loss_type


def ridged_mape_loss_tensor(y_pred: Tensor, y_true: Tensor, mape_ridge: float = 1.0) -> Tensor:
    """Differentiable MAPE variant used for training maskedmape runs."""
    eps = float(mape_ridge)
    if eps < 0.0:
        raise ValueError("mape_ridge must be non-negative.")
    denom = torch.abs(y_true) + eps
    return torch.mean(torch.abs(y_pred - y_true) / denom)


def masked_mape_loss_tensor(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Evaluation MAPE over entries with nonzero finite targets."""
    mask = torch.isfinite(y_true) & (torch.abs(y_true) > 0)
    if not bool(mask.any().item()):
        return torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
    return torch.mean(torch.abs(y_pred[mask] - y_true[mask]) / torch.abs(y_true[mask]))


def loss_tensor(
    y_pred: Tensor,
    y_true: Tensor,
    loss_type: str = "mse",
    *,
    mape_ridge: float = 1.0,
    maskedmape_mode: str = "train",
) -> Tensor:
    loss_type = normalized_loss_type(loss_type)
    if loss_type == "mse":
        return torch.mean((y_pred - y_true) ** 2)
    if loss_type == "mae":
        return torch.mean(torch.abs(y_pred - y_true))
    mode = str(maskedmape_mode).lower().strip()
    if mode in {"train", "ridged", "ridge"}:
        return ridged_mape_loss_tensor(y_pred, y_true, mape_ridge=mape_ridge)
    if mode in {"eval", "masked"}:
        return masked_mape_loss_tensor(y_pred, y_true)
    raise ValueError("maskedmape_mode must be one of {'train', 'eval'}.")


def loss_value(
    y_pred: Tensor,
    y_true: Tensor,
    loss_type: str = "mse",
    *,
    mape_ridge: float = 1.0,
    maskedmape_mode: str = "train",
) -> float:
    return float(
        loss_tensor(
            y_pred,
            y_true,
            loss_type=loss_type,
            mape_ridge=mape_ridge,
            maskedmape_mode=maskedmape_mode,
        ).item()
    )


def fit_basis_operator(
    x_train: Tensor,
    y_train: Tensor,
    bases: Sequence[Tensor],
    epochs: int = 10,
    lr: float = 0.003,
    ridge: float = 0.0,
    optimizer_name: str = "adam",
    lbfgs_max_iter: int = 20,
    init_scale: float = 0.0,
    loss_type: str = "mse",
    mape_ridge: float = 1.0,
) -> BasisFitResult:
    """Fit H=sum_j theta_j B_j by minimizing the selected empirical loss.

    loss_type="mse" preserves the old pipeline.  loss_type="mae" changes the
    fitted data term to L1/MAE.  loss_type="maskedmape" fits a ridged MAPE
    objective with denominator |y| + mape_ridge; evaluation uses masked MAPE.
    """
    if x_train.dim() != 2 or y_train.dim() != 2:
        raise ValueError("x_train and y_train must be [T, n].")
    if x_train.shape != y_train.shape:
        raise ValueError("x_train and y_train must have the same shape.")
    loss_type = normalized_loss_type(loss_type)
    B = stack_bases(bases).to(device=x_train.device, dtype=x_train.dtype)
    m = int(B.shape[0])
    if init_scale == 0.0:
        theta = torch.zeros(m, dtype=x_train.dtype, device=x_train.device, requires_grad=True)
    else:
        theta = (float(init_scale) * torch.randn(m, dtype=x_train.dtype, device=x_train.device)).requires_grad_()

    def loss_fn() -> Tensor:
        H = combine_bases(theta, B)
        pred = x_train @ H.T
        fit = loss_tensor(pred, y_train, loss_type=loss_type, mape_ridge=mape_ridge, maskedmape_mode="train")
        if float(ridge) != 0.0:
            reg = float(ridge) * torch.mean(H * H)
            return fit + reg
        return fit

    opt_name = str(optimizer_name).lower().strip()
    if opt_name == "adam":
        opt = torch.optim.Adam([theta], lr=float(lr))
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS([theta], lr=float(lr), max_iter=int(lbfgs_max_iter), line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            opt.zero_grad()
            loss = loss_fn()
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
            mape_ridge=mape_ridge,
            maskedmape_mode="train",
        )
    return BasisFitResult(theta=theta.detach().clone(), operator=H, train_loss=train_loss, num_parameters=m)


def fit_lmmse_operator(x_train: Tensor, y_train: Tensor, ridge: float = 1e-9) -> Tensor:
    """Closed-form ridge LMMSE for y_hat = x @ H.T."""
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


def fit_lad_operator(
    x_train: Tensor,
    y_train: Tensor,
    ridge: float = 0.0,
    epochs: int = 1000,
    lr: float = 0.003,
    optimizer_name: str = "adam",
    lbfgs_max_iter: int = 20,
    init: Optional[Tensor] = None,
) -> Tensor:
    """Numerical LAD fit for y_hat = x @ H.T with optional ridge.

    The optimized objective is
        sum(|X H^T - Y|) + ridge * T * ||H||_F^2,
    which is equivalent to mean absolute error plus ridge * ||H||_F^2 up to the
    same sample-count scaling used by fit_lmmse_operator.
    """
    if x_train.dim() != 2 or y_train.dim() != 2:
        raise ValueError("x_train and y_train must be [T, n].")
    if x_train.shape != y_train.shape:
        raise ValueError("x_train and y_train must have the same shape.")

    n = int(x_train.shape[1])
    if init is None:
        H = torch.zeros((n, n), dtype=x_train.dtype, device=x_train.device, requires_grad=True)
    else:
        if init.shape != (n, n):
            raise ValueError(f"init must have shape {(n, n)}, got {tuple(init.shape)}.")
        H = init.detach().clone().to(device=x_train.device, dtype=x_train.dtype).requires_grad_()

    def objective() -> Tensor:
        pred = x_train @ H.T
        fit = torch.sum(torch.abs(pred - y_train))
        if float(ridge) != 0.0:
            fit = fit + float(ridge) * x_train.shape[0] * torch.sum(H * H)
        return fit

    opt_name = str(optimizer_name).lower().strip()
    if opt_name == "adam":
        opt = torch.optim.Adam([H], lr=float(lr))
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = objective()
            loss.backward()
            opt.step()
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS([H], lr=float(lr), max_iter=int(lbfgs_max_iter), line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        for _ in range(int(epochs)):
            opt.step(closure)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'lbfgs'.")

    return H.detach().contiguous()


def apply_operator(x: Tensor, H: Tensor) -> Tensor:
    if x.dim() != 2:
        raise ValueError("x must have shape [T, n].")
    return x @ H.to(device=x.device, dtype=x.dtype).T


def mse_loss(y_pred: Tensor, y_true: Tensor) -> float:
    return loss_value(y_pred, y_true, loss_type="mse")


def mae_loss(y_pred: Tensor, y_true: Tensor) -> float:
    return loss_value(y_pred, y_true, loss_type="mae")


def masked_mape_loss(y_pred: Tensor, y_true: Tensor) -> float:
    return loss_value(y_pred, y_true, loss_type="maskedmape", maskedmape_mode="eval")


def eval_operator_loss(
    x_test: Tensor,
    y_test: Tensor,
    H: Tensor,
    loss_type: str = "mse",
    *,
    mape_ridge: float = 1.0,
) -> float:
    with torch.no_grad():
        return loss_value(
            apply_operator(x_test, H),
            y_test,
            loss_type=loss_type,
            mape_ridge=mape_ridge,
            maskedmape_mode="eval" if normalized_loss_type(loss_type) == "maskedmape" else "train",
        )


def eval_operator_mse(x_test: Tensor, y_test: Tensor, H: Tensor) -> float:
    return eval_operator_loss(x_test, y_test, H, loss_type="mse")


def eval_operator_mae(x_test: Tensor, y_test: Tensor, H: Tensor) -> float:
    return eval_operator_loss(x_test, y_test, H, loss_type="mae")


def train_eval_basis_loss(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    bases: Sequence[Tensor],
    epochs: int = 10,
    lr: float = 0.003,
    ridge: float = 0.0,
    optimizer_name: str = "adam",
    loss_type: str = "mse",
    mape_ridge: float = 1.0,
) -> Tuple[float, BasisFitResult]:
    fit = fit_basis_operator(
        x_train=x_train,
        y_train=y_train,
        bases=bases,
        epochs=epochs,
        lr=lr,
        ridge=ridge,
        optimizer_name=optimizer_name,
        loss_type=loss_type,
        mape_ridge=mape_ridge,
    )
    test_loss = eval_operator_loss(x_test, y_test, fit.operator, loss_type=loss_type, mape_ridge=mape_ridge)
    return test_loss, fit


def train_eval_basis_mse(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    bases: Sequence[Tensor],
    epochs: int = 10,
    lr: float = 0.003,
    ridge: float = 0.0,
    optimizer_name: str = "adam",
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


def split_train_test(
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

    # Preserve V0 order as the reference local ordering.
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
        key = f"distance_{k_max}"
        if key not in self._cache:
            # Reuse the largest already-computed distance list if possible.
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
# Method basis factory for the simplified main runner
# -----------------------------------------------------------------------------


def build_method_bases(
    method: str,
    cache: PrimitiveCache,
    poly_degree: int = 3,
    sffa_k: int = 3,
    sffa_r: int = 3,
    rg_num: int = 3,
    rg_p0: float = 0.9,
    max_sffa_basis_size: int = 2000,
) -> Optional[List[Tensor]]:
    """Return trainable bases for a named method, or None for direct/LMMSE methods."""
    method = str(method).strip()

    # Match revisedfilter.main_stdlap_fin semantics: the learned coefficients are
    # applied to raw Laplacian powers/words, with no Frobenius rescaling.
    if method == "sub_lp":
        return polynomial_bases(cache.induced(), degree=poly_degree)

    # if method == "dk_lp":
    #     return multi_polynomial_bases(cache.distance_laps(sffa_k), degree=poly_degree, path_order=True)

    if method == "dk_lp":
        return multi_polynomial_bases(cache.distance_laps(sffa_k),degree=sffa_r,path_order=False)

    if method == "rg_lp":
        return multi_polynomial_bases(cache.random_groups(rg_num, rg_p0), degree=poly_degree, path_order=False)

    if method == "rg_alg":
        # Treat random-group algebra as a genuine (k,r)-SFFA: sffa_k random
        # primitives and word length sffa_r.  rg_num is kept for rg_lp only.
        return sffa_word_bases(
            cache.random_groups(sffa_k, rg_p0),
            word_len=sffa_r,
            include_identity=True,
            max_basis_size=max_sffa_basis_size,
        )

    if method == "kron_lp":
        return polynomial_bases(cache.kron(), degree=poly_degree)

    if method == "union_lp":
        return polynomial_bases(cache.union(sffa_k), degree=poly_degree)

    if method == "sffa":
        # k=1 -> induced polynomial by using a single induced primitive.
        prim = standard_sffa_primitives(cache.induced(), cache.distance_laps(max(1, sffa_k)), k=sffa_k)
        return sffa_word_bases(
            prim,
            word_len=sffa_r,
            include_identity=True,
            max_basis_size=max_sffa_basis_size,
        )

    if method in {"kron12", "kron_sffa"}:
        # k=1 -> Kron polynomial; k>=2 -> d1..d{k-1}+Kron.
        prim = kron_sffa_primitives(cache.distance_laps(max(1, sffa_k - 1)), cache.kron(), k=sffa_k)
        return sffa_word_bases(
            prim,
            word_len=sffa_r,
            include_identity=True,
            max_basis_size=max_sffa_basis_size,
        )

    return None


def direct_operator_for_method(method: str, cache: PrimitiveCache, F_full: Tensor, sffa_k: int = 3) -> Optional[Tensor]:
    """Return a non-trained operator for direct baselines, if method is direct."""
    method = str(method).strip()
    if method == "plain_trunc":
        return submatrix(F_full.to(device=cache.v0_idx.device, dtype=cache.dtype), cache.v0_idx, cache.v0_idx)
    if method == "induced_direct":
        return cache.induced()
    if method == "kron_direct":
        return cache.kron()
    if method == "union_direct":
        return cache.union(int(sffa_k))
    return None


__all__ = [
    "TrafficData",
    "PrimitiveCache",
    "BasisFitResult",
    "DEFAULT_TRUTH_COEFFS",
    "set_seed",
    "resolve_device",
    "load_metr_la_traffic",
    "full_laplacian_from_graph",
    "symmetric_normalized_laplacian_from_adjacency",
    "induced_laplacian",
    "distance_k_laplacians",
    "union_laplacian_from_distance_laplacians",
    "random_group_laplacians",
    "kron_reduced_laplacian",
    "polynomial_filter_from_coeffs",
    "make_linear_noisy_labels",
    "polynomial_bases",
    "multi_polynomial_bases",
    "sffa_basis_size",
    "sffa_word_bases",
    "standard_sffa_primitives",
    "kron_sffa_primitives",
    "fit_basis_operator",
    "fit_lmmse_operator",
    "fit_lad_operator",
    "loss_tensor",
    "loss_value",
    "mae_loss",
    "masked_mape_loss",
    "eval_operator_loss",
    "eval_operator_mse",
    "eval_operator_mae",
    "train_eval_basis_loss",
    "train_eval_basis_mse",
    "choose_v0",
    "split_train_test",
    "build_method_bases",
    "direct_operator_for_method",
]

# -----------------------------------------------------------------------------
# Horizon/window prediction utilities: [S, P, n] -> [S, H, n]
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HorizonBasisFitResult:
    """Fit result for the horizon SFL model.

    theta[p, h, j] is the coefficient of basis matrix B_j for the operator from
    input lag p to output horizon h. operators[p, h] is the corresponding dense
    node-space operator H_{p,h}; prediction uses x @ H_{p,h}.T, matching the
    row-major convention used by the one-step code.
    """

    theta: Tensor
    operators: Tensor
    train_loss: float
    num_parameters: int


@dataclass(frozen=True)
class HorizonARFitResult:
    """Node-wise autoregressive baseline on the visible subgraph.

    weights[node, p, h] maps the p-th input step of one node to its h-th future
    output.  No information is shared across different nodes.
    """

    weights: Tensor
    train_loss: float
    num_parameters: int


@dataclass(frozen=True)
class HorizonVARFitResult:
    """Unconstrained VAR baseline on the visible subgraph.

    operators[p, h, j, i] maps node i at input lag p to node j at output horizon
    h.  This is the data-dependent, structure-agnostic baseline over V0.
    """

    operators: Tensor
    train_loss: float
    num_parameters: int


def _validate_horizon_xy(x: Tensor, y: Tensor) -> Tuple[int, int, int, int]:
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("x and y must have shapes [S, P, n] and [S, H, n].")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(
            f"x and y must have matching sample and node dimensions; got {tuple(x.shape)} and {tuple(y.shape)}."
        )
    s = int(x.shape[0])
    p = int(x.shape[1])
    h = int(y.shape[1])
    n = int(x.shape[2])
    if s <= 0 or p <= 0 or h <= 0 or n <= 0:
        raise ValueError("All horizon dimensions must be positive.")
    return s, p, h, n


def combine_horizon_bases(theta: Tensor, bases_tensor: Tensor) -> Tensor:
    """Return operators[p,h,:,:] = sum_j theta[p,h,j] bases[j]."""
    if theta.dim() != 3:
        raise ValueError("theta must have shape [P, H, m].")
    if bases_tensor.dim() != 3:
        raise ValueError("bases_tensor must have shape [m, n, n].")
    if theta.shape[2] != bases_tensor.shape[0]:
        raise ValueError("theta's basis dimension does not match bases_tensor.")
    return torch.einsum(
        "phm,mij->phij",
        theta.to(device=bases_tensor.device, dtype=bases_tensor.dtype),
        bases_tensor,
    )


def apply_horizon_operators(x: Tensor, operators: Tensor) -> Tensor:
    """Apply multi-lag, multi-horizon operators.

    x has shape [S, P, n].  operators has shape [P, H, n, n], where each matrix
    follows the existing convention y = x @ H.T.  The output has shape [S, H, n].
    """
    if x.dim() != 3:
        raise ValueError("x must have shape [S, P, n].")
    if operators.dim() != 4:
        raise ValueError("operators must have shape [P, H, n, n].")
    if int(x.shape[1]) != int(operators.shape[0]):
        raise ValueError("x input length and operator input length differ.")
    if int(x.shape[2]) != int(operators.shape[2]) or int(x.shape[2]) != int(operators.shape[3]):
        raise ValueError("operator node dimensions must match x.shape[2].")
    op = operators.to(device=x.device, dtype=x.dtype)
    return torch.einsum("spi,phji->shj", x, op)


def horizon_loss_values(
    y_pred: Tensor,
    y_true: Tensor,
    loss_type: str = "mse",
    *,
    mape_ridge: float = 1.0,
    maskedmape_mode: str = "train",
) -> List[float]:
    """Return one scalar loss for each output horizon."""
    if y_pred.dim() != 3 or y_true.dim() != 3:
        raise ValueError("y_pred and y_true must have shape [S, H, n].")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape.")
    return [
        loss_value(
            y_pred[:, h, :],
            y_true[:, h, :],
            loss_type=loss_type,
            mape_ridge=mape_ridge,
            maskedmape_mode=maskedmape_mode,
        )
        for h in range(int(y_true.shape[1]))
    ]


def fit_horizon_basis_operator(
    x_train: Tensor,
    y_train: Tensor,
    bases: Sequence[Tensor],
    epochs: int = 10,
    lr: float = 0.003,
    ridge: float = 0.0,
    optimizer_name: str = "adam",
    lbfgs_max_iter: int = 20,
    init_scale: float = 0.0,
    loss_type: str = "mse",
    mape_ridge: float = 1.0,
) -> HorizonBasisFitResult:
    """Fit a horizon SFL filter bank.

    For input length P and prediction length H, the model is

        y_hat[:, h, :] = sum_p x[:, p, :] @ H_{p,h}.T,
        H_{p,h} = sum_j theta[p,h,j] B_j.

    Thus every past-to-future pair uses the same basis support but has its own
    coefficient vector.  The number of trainable coefficients is P * H * m.
    """
    _s, p, h, _n = _validate_horizon_xy(x_train, y_train)
    loss_type = normalized_loss_type(loss_type)
    B = stack_bases(bases).to(device=x_train.device, dtype=x_train.dtype)
    m = int(B.shape[0])
    if init_scale == 0.0:
        theta = torch.zeros((p, h, m), dtype=x_train.dtype, device=x_train.device, requires_grad=True)
    else:
        theta = (
            float(init_scale) * torch.randn((p, h, m), dtype=x_train.dtype, device=x_train.device)
        ).requires_grad_()

    def loss_fn() -> Tensor:
        H_ops = combine_horizon_bases(theta, B)
        pred = apply_horizon_operators(x_train, H_ops)
        fit = loss_tensor(pred, y_train, loss_type=loss_type, mape_ridge=mape_ridge, maskedmape_mode="train")
        if float(ridge) != 0.0:
            fit = fit + float(ridge) * torch.mean(H_ops * H_ops)
        return fit

    opt_name = str(optimizer_name).lower().strip()
    if opt_name == "adam":
        opt = torch.optim.Adam([theta], lr=float(lr))
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS([theta], lr=float(lr), max_iter=int(lbfgs_max_iter), line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        for _ in range(int(epochs)):
            opt.step(closure)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'lbfgs'.")

    with torch.no_grad():
        H_ops = combine_horizon_bases(theta, B).detach().clone()
        train_loss = loss_value(
            apply_horizon_operators(x_train, H_ops),
            y_train,
            loss_type=loss_type,
            mape_ridge=mape_ridge,
            maskedmape_mode="train",
        )
    return HorizonBasisFitResult(
        theta=theta.detach().clone(),
        operators=H_ops,
        train_loss=train_loss,
        num_parameters=int(theta.numel()),
    )


def eval_horizon_operator_loss(
    x_test: Tensor,
    y_test: Tensor,
    operators: Tensor,
    loss_type: str = "mse",
    *,
    mape_ridge: float = 1.0,
) -> float:
    with torch.no_grad():
        mode = "eval" if normalized_loss_type(loss_type) == "maskedmape" else "train"
        return loss_value(
            apply_horizon_operators(x_test, operators),
            y_test,
            loss_type=loss_type,
            mape_ridge=mape_ridge,
            maskedmape_mode=mode,
        )


def eval_horizon_operator_horizon_losses(
    x_test: Tensor,
    y_test: Tensor,
    operators: Tensor,
    loss_type: str = "mse",
    *,
    mape_ridge: float = 1.0,
) -> List[float]:
    with torch.no_grad():
        mode = "eval" if normalized_loss_type(loss_type) == "maskedmape" else "train"
        return horizon_loss_values(
            apply_horizon_operators(x_test, operators),
            y_test,
            loss_type=loss_type,
            mape_ridge=mape_ridge,
            maskedmape_mode=mode,
        )


def train_eval_horizon_basis_loss(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    bases: Sequence[Tensor],
    epochs: int = 10,
    lr: float = 0.003,
    ridge: float = 0.0,
    optimizer_name: str = "adam",
    loss_type: str = "mse",
    mape_ridge: float = 1.0,
) -> Tuple[float, HorizonBasisFitResult]:
    fit = fit_horizon_basis_operator(
        x_train=x_train,
        y_train=y_train,
        bases=bases,
        epochs=epochs,
        lr=lr,
        ridge=ridge,
        optimizer_name=optimizer_name,
        loss_type=loss_type,
        mape_ridge=mape_ridge,
    )
    test_loss = eval_horizon_operator_loss(
        x_test, y_test, fit.operators, loss_type=loss_type, mape_ridge=mape_ridge
    )
    return test_loss, fit


def apply_horizon_ar(x: Tensor, weights: Tensor) -> Tensor:
    """Apply node-wise AR weights with shapes x=[S,P,n], weights=[n,P,H]."""
    if x.dim() != 3:
        raise ValueError("x must have shape [S, P, n].")
    if weights.dim() != 3:
        raise ValueError("weights must have shape [n, P, H].")
    if int(x.shape[1]) != int(weights.shape[1]) or int(x.shape[2]) != int(weights.shape[0]):
        raise ValueError("AR weights must match x's input length and node dimension.")
    return torch.einsum("spn,nph->shn", x, weights.to(device=x.device, dtype=x.dtype))


def fit_horizon_ar(
    x_train: Tensor,
    y_train: Tensor,
    ridge: float = 1e-3,
    epochs: int = 1000,
    lr: float = 0.003,
    optimizer_name: str = "adam",
    lbfgs_max_iter: int = 20,
    loss_type: str = "mse",
    mape_ridge: float = 1.0,
) -> HorizonARFitResult:
    """Fit an AR baseline over V0 only.

    MSE uses independent ridge regressions for each node.  MAE and masked MAPE use
    numerical optimization so the training objective matches the requested loss.
    """
    s, p, h, n = _validate_horizon_xy(x_train, y_train)
    loss_type = normalized_loss_type(loss_type)

    if loss_type == "mse":
        weights = torch.zeros((n, p, h), dtype=x_train.dtype, device=x_train.device)
        eye = torch.eye(p, dtype=x_train.dtype, device=x_train.device)
        for node in range(n):
            X = x_train[:, :, node]
            Y = y_train[:, :, node]
            gram = X.T @ X
            rhs = X.T @ Y
            weights[node] = torch.linalg.solve(gram + float(ridge) * s * eye, rhs)
        train_loss = loss_value(apply_horizon_ar(x_train, weights), y_train, loss_type="mse")
        return HorizonARFitResult(weights=weights.contiguous(), train_loss=train_loss, num_parameters=int(weights.numel()))

    weights = torch.zeros((n, p, h), dtype=x_train.dtype, device=x_train.device, requires_grad=True)

    def objective() -> Tensor:
        pred = apply_horizon_ar(x_train, weights)
        fit = loss_tensor(pred, y_train, loss_type=loss_type, mape_ridge=mape_ridge, maskedmape_mode="train")
        if float(ridge) != 0.0:
            fit = fit + float(ridge) * torch.mean(weights * weights)
        return fit

    opt_name = str(optimizer_name).lower().strip()
    if opt_name == "adam":
        opt = torch.optim.Adam([weights], lr=float(lr))
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = objective()
            loss.backward()
            opt.step()
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS([weights], lr=float(lr), max_iter=int(lbfgs_max_iter), line_search_fn="strong_wolfe")

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
        w = weights.detach().clone()
        train_loss = loss_value(
            apply_horizon_ar(x_train, w),
            y_train,
            loss_type=loss_type,
            mape_ridge=mape_ridge,
            maskedmape_mode="train",
        )
    return HorizonARFitResult(weights=w.contiguous(), train_loss=train_loss, num_parameters=int(w.numel()))


def fit_horizon_var(
    x_train: Tensor,
    y_train: Tensor,
    ridge: float = 1e-3,
    epochs: int = 1000,
    lr: float = 0.003,
    optimizer_name: str = "adam",
    lbfgs_max_iter: int = 20,
    loss_type: str = "mse",
    mape_ridge: float = 1.0,
) -> HorizonVARFitResult:
    """Fit an unconstrained VAR baseline over V0 only.

    This is equivalent to using a full matrix basis for every past-to-future pair.
    It is structure-agnostic but data-dependent.
    """
    s, p, h, n = _validate_horizon_xy(x_train, y_train)
    loss_type = normalized_loss_type(loss_type)

    if loss_type == "mse":
        X = x_train.reshape(s, p * n)
        Y = y_train.reshape(s, h * n)
        eye = torch.eye(p * n, dtype=x_train.dtype, device=x_train.device)
        gram = X.T @ X
        rhs = X.T @ Y
        W = torch.linalg.solve(gram + float(ridge) * s * eye, rhs)
        operators = W.reshape(p, n, h, n).permute(0, 2, 3, 1).contiguous()
        train_loss = loss_value(apply_horizon_operators(x_train, operators), y_train, loss_type="mse")
        return HorizonVARFitResult(
            operators=operators,
            train_loss=train_loss,
            num_parameters=int(operators.numel()),
        )

    operators = torch.zeros((p, h, n, n), dtype=x_train.dtype, device=x_train.device, requires_grad=True)

    def objective() -> Tensor:
        pred = apply_horizon_operators(x_train, operators)
        fit = loss_tensor(pred, y_train, loss_type=loss_type, mape_ridge=mape_ridge, maskedmape_mode="train")
        if float(ridge) != 0.0:
            fit = fit + float(ridge) * torch.mean(operators * operators)
        return fit

    opt_name = str(optimizer_name).lower().strip()
    if opt_name == "adam":
        opt = torch.optim.Adam([operators], lr=float(lr))
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = objective()
            loss.backward()
            opt.step()
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS([operators], lr=float(lr), max_iter=int(lbfgs_max_iter), line_search_fn="strong_wolfe")

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
        ops = operators.detach().clone()
        train_loss = loss_value(
            apply_horizon_operators(x_train, ops),
            y_train,
            loss_type=loss_type,
            mape_ridge=mape_ridge,
            maskedmape_mode="train",
        )
    return HorizonVARFitResult(operators=ops.contiguous(), train_loss=train_loss, num_parameters=int(ops.numel()))


def persistence_prediction(x: Tensor, pred_len: int) -> Tensor:
    """Repeat the last observed input frame for every future horizon."""
    if x.dim() != 3:
        raise ValueError("x must have shape [S, P, n].")
    pred_len = int(pred_len)
    if pred_len <= 0:
        raise ValueError("pred_len must be positive.")
    return x[:, -1:, :].expand(-1, pred_len, -1)


def eval_prediction_loss(
    y_pred: Tensor,
    y_true: Tensor,
    loss_type: str = "mse",
    *,
    mape_ridge: float = 1.0,
) -> float:
    mode = "eval" if normalized_loss_type(loss_type) == "maskedmape" else "train"
    return loss_value(y_pred, y_true, loss_type=loss_type, mape_ridge=mape_ridge, maskedmape_mode=mode)


def eval_prediction_horizon_losses(
    y_pred: Tensor,
    y_true: Tensor,
    loss_type: str = "mse",
    *,
    mape_ridge: float = 1.0,
) -> List[float]:
    mode = "eval" if normalized_loss_type(loss_type) == "maskedmape" else "train"
    return horizon_loss_values(y_pred, y_true, loss_type=loss_type, mape_ridge=mape_ridge, maskedmape_mode=mode)


try:
    __all__ = list(__all__) + [
        "HorizonBasisFitResult",
        "HorizonARFitResult",
        "HorizonVARFitResult",
        "combine_horizon_bases",
        "apply_horizon_operators",
        "horizon_loss_values",
        "fit_horizon_basis_operator",
        "eval_horizon_operator_loss",
        "eval_horizon_operator_horizon_losses",
        "train_eval_horizon_basis_loss",
        "apply_horizon_ar",
        "fit_horizon_ar",
        "fit_horizon_var",
        "persistence_prediction",
        "eval_prediction_loss",
        "eval_prediction_horizon_losses",
    ]
except NameError:
    pass

