#!/usr/bin/env python3
"""Horizon traffic prediction runner for Subgraph Filter Learning on METR-LA.

Task
----
Use an input window of length P to predict H future steps on the visible node
set V0.  The default is the common traffic-forecasting setting P=12, H=3; use
--pred-lens 3,6,12 to run the usual 15/30/60-minute targets on METR-LA-like
5-minute data.

The SFL horizon model is

    y_hat[:, h, :] = sum_p x[:, p, :] @ H_{p,h}.T,
    H_{p,h} = sum_j theta[p,h,j] B_j.

Every input-lag/output-horizon pair has its own coefficient vector.  The graph
support / filter bank is shared across all P*H pairs.

Data splitting
--------------
The cleaned time axis is split sequentially into train/validation/test blocks
(default 0.7/0.1/0.2).  Windows are then constructed inside each block only, so
no input-output window crosses a split boundary.  Windows crossing deleted raw
timestamps are dropped by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import networkx as nx
import pandas as pd  # kept to validate the METR-LA h5 layout, same as old runner
import torch
from torch_geometric.data import Data

try:
    from sfl_core_horizon import (  # type: ignore
        PrimitiveCache,
        apply_horizon_ar,
        apply_horizon_operators,
        build_method_bases,
        choose_v0,
        eval_prediction_horizon_losses,
        eval_prediction_loss,
        fit_horizon_ar,
        fit_horizon_basis_operator,
        fit_horizon_var,
        full_laplacian_from_graph,
        persistence_prediction,
        resolve_device,
        set_seed,
    )
except ModuleNotFoundError:
    from revisedfilter.sfl_core_horizon import (  # type: ignore
        PrimitiveCache,
        apply_horizon_ar,
        apply_horizon_operators,
        build_method_bases,
        choose_v0,
        eval_prediction_horizon_losses,
        eval_prediction_loss,
        fit_horizon_ar,
        fit_horizon_basis_operator,
        fit_horizon_var,
        full_laplacian_from_graph,
        persistence_prediction,
        resolve_device,
        set_seed,
    )


TRAINABLE_METHODS = {
    "sub_lp",
    "dk_lp",
    "rg_lp",
    "rg_alg",
    "kron_lp",
    "union_lp",
    "sffa",
    "kron12",
    "kron_sffa",
}

# These were fixed single-matrix baselines in the one-step runner.  In this
# horizon runner they are trainable one-basis SFL models: H_{p,h}=theta[p,h] B.
SINGLE_MATRIX_METHODS = {
    "identity_direct",
    "induced_direct",
    "kron_direct",
    "union_direct",
}

BASELINE_METHODS = {
    "ar",
    "var",
    "persistence",
    "no_processing",  # compatibility alias for persistence
}

DEFAULT_METHODS = (
    "ar,var,sub_lp,dk_lp,rg_lp,rg_alg,kron_lp,union_lp,"
    "sffa,kron12,induced_direct,kron_direct,union_direct"
)

METHOD_LABELS = {
    "ar": "Node-wise AR on visible subgraph",
    "var": "Unconstrained VAR on visible subgraph",
    "persistence": "Last-observation persistence",
    "no_processing": "Last-observation persistence",
    "identity_direct": "Trainable identity support",
    "sub_lp": "Subgraph Laplacian polynomial",
    "dk_lp": "Distance-k Laplacian polynomial",
    "rg_lp": "Random-group Laplacian polynomial",
    "rg_alg": "Random-group algebra",
    "kron_lp": "Kron Laplacian polynomial",
    "union_lp": "Union Laplacian polynomial",
    "sffa": "(k,r)-SFFA",
    "kron12": "Kron (k,r)-SFFA",
    "kron_sffa": "Kron (k,r)-SFFA",
    "induced_direct": "Trainable induced-Laplacian support",
    "kron_direct": "Trainable Kron-Laplacian support",
    "union_direct": "Trainable union-Laplacian support",
}


@dataclass(frozen=True)
class TrafficGraphData:
    graph: Data
    graph_nx: nx.Graph
    num_nodes: int
    adjacency: torch.Tensor
    # Values after removing timestamps with non-finite values.
    raw_values: torch.Tensor
    # Original raw timestamp indices retained in raw_values.
    raw_indices: torch.Tensor
    # pair_is_adjacent[i] is True iff raw_values[i] and raw_values[i+1]
    # were adjacent timestamps before missing rows were removed.
    pair_is_adjacent: torch.Tensor
    raw_total_timestamps: int
    missing_timestamps: int


@dataclass(frozen=True)
class Config:
    seed: int = 42
    p_visible: float = 0.8
    methods: str = DEFAULT_METHODS

    # Horizon prediction data options.
    # If num_timesteps <= 0, use the whole raw METR-LA time axis.
    num_timesteps: int = 0
    input_len: int = 12
    pred_len: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    drop_gap_windows: bool = True

    # The training objective and reported metric are the same by design.
    loss_type: str = "mse"  # mse or mae
    mape_ridge: float = 1.0  # kept for API parity; not used when loss_type in {mse, mae}

    epochs: int = 10
    lr: float = 0.003
    ridge: float = 10.0
    sfl_ridge: float = 0.0
    optimizer: str = "adam"
    lbfgs_max_iter: int = 20

    poly_degree: int = 3
    sffa_k: int = 3
    sffa_r: int = 3
    rg_num: int = 3
    rg_p0: float = 0.9
    max_sffa_basis_size: int = 2000

    dataset_dir: str = "dataset"
    device: str = "auto"
    dtype: str = "float32"
    zero_isolated: bool = False
    kron_ridge: float = 0.0
    kron_use_pinv: bool = False

    output_csv: str = ""
    output_json: str = ""
    quiet: bool = False


def _parse_methods(text: str) -> List[str]:
    methods = [m.strip() for m in str(text).replace(";", ",").split(",") if m.strip()]
    if not methods:
        raise ValueError("At least one method must be supplied.")
    known = TRAINABLE_METHODS | SINGLE_MATRIX_METHODS | BASELINE_METHODS
    unknown = [m for m in methods if m not in known]
    if unknown:
        raise ValueError(
            f"Unknown methods for horizon prediction task: {unknown}. "
            f"Known methods: {sorted(known)}. "
            "num_lmmse and num_lad are intentionally removed; use ar/var instead."
        )
    return methods


def _parse_int_range_or_csv(text: str) -> List[int]:
    text = str(text).strip()
    if ":" in text and "," not in text:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise argparse.ArgumentTypeError("range form must be start:stop[:step]")
        return list(range(start, stop, step))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_range_or_csv(text: str) -> List[float]:
    text = str(text).strip()
    if ":" in text and "," not in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("float range form must be start:stop:step")
        start, stop, step = parts
        if step == 0:
            raise argparse.ArgumentTypeError("range step cannot be zero")
        vals: List[float] = []
        x = start
        if step > 0:
            while x <= stop + 1e-12:
                vals.append(round(x, 10))
                x += step
        else:
            while x >= stop - 1e-12:
                vals.append(round(x, 10))
                x += step
        return vals
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_split_ratios(text: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--split-ratios must have form train,val,test, e.g. 0.7,0.1,0.2")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in {"float32", "fp32", "32"}:
        return torch.float32
    if name in {"float64", "fp64", "double", "64"}:
        return torch.float64
    raise ValueError("dtype must be float32 or float64.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run horizon SFL traffic prediction experiments on METR-LA.")

    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--seeds", type=str, default="", help="Optional batch seeds, e.g. '42:47' or '42,43,44'.")
    p.add_argument("--p-visible", type=float, default=Config.p_visible)
    p.add_argument("--p-visibles", type=str, default="", help="Optional batch p values, e.g. '0.1:1.0:0.1'.")
    p.add_argument("--methods", type=str, default=Config.methods)

    p.add_argument("--num-timesteps", type=int, default=Config.num_timesteps,
                   help="Maximum raw METR-LA timestamps to load. <=0 means use all raw timestamps.")
    p.add_argument("--input-len", "--P", dest="input_len", type=int, default=Config.input_len,
                   help="Input lookback length P. Default: 12.")
    p.add_argument("--pred-len", "--H", dest="pred_len", type=int, default=Config.pred_len,
                   help="Prediction horizon H. Default: 3.")
    p.add_argument("--pred-lens", type=str, default="",
                   help="Optional batch prediction horizons, e.g. '3,6,12'. Overrides --pred-len for batching.")

    p.add_argument("--train-ratio", type=float, default=Config.train_ratio)
    p.add_argument("--val-ratio", type=float, default=Config.val_ratio)
    p.add_argument("--test-ratio", type=float, default=Config.test_ratio)
    p.add_argument("--split-ratios", type=str, default="",
                   help="Optional train,val,test split ratios, e.g. '0.7,0.1,0.2'.")
    p.add_argument("--drop-gap-windows", action=argparse.BooleanOptionalAction, default=Config.drop_gap_windows,
                   help="Drop windows whose cleaned timestamps cross a deleted/missing raw timestamp.")

    # Compatibility-only flags from the one-step runner. They are accepted but not
    # used because horizon traffic prediction is ratio-split only.
    p.add_argument("--split-mode", type=str, default="ratio", choices=["ratio"], help=argparse.SUPPRESS)
    p.add_argument("--train-steps", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--test-steps", type=int, default=None, help=argparse.SUPPRESS)

    p.add_argument("--loss-type", type=str, default=Config.loss_type, choices=["mse", "mae"],
                   help="Training/evaluation loss. The same loss is used for fitting and reporting.")
    p.add_argument("--mape-ridge", type=float, default=Config.mape_ridge, help=argparse.SUPPRESS)

    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--ridge", type=float, default=Config.ridge,
                   help="Ridge for AR/VAR MSE fits and weight penalty for AR/VAR MAE fits.")
    p.add_argument("--sfl-ridge", type=float, default=Config.sfl_ridge,
                   help="Optional Frobenius penalty on learned SFL horizon operators.")
    p.add_argument("--optimizer", type=str, default=Config.optimizer, choices=["adam", "lbfgs"])

    p.add_argument(
        "--lbfgs-max-iter",
        "--lbfgs-iter",
        dest="lbfgs_max_iter",
        type=int,
        default=Config.lbfgs_max_iter,
        help="Inner max_iter for torch.optim.LBFGS; only used when --optimizer lbfgs.",
    )

    p.add_argument("--poly-degree", type=int, default=Config.poly_degree)
    p.add_argument("--sffa-k", "--k", dest="sffa_k", type=int, default=Config.sffa_k)
    p.add_argument("--sffa-r", "--r", dest="sffa_r", type=int, default=Config.sffa_r)
    p.add_argument("--rg-num", type=int, default=Config.rg_num)
    p.add_argument("--rg-p0", type=float, default=Config.rg_p0)
    p.add_argument("--max-sffa-basis-size", type=int, default=Config.max_sffa_basis_size)

    p.add_argument("--dataset-dir", type=str, default=Config.dataset_dir)
    p.add_argument("--device", type=str, default=Config.device, choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default=Config.dtype, choices=["float32", "float64"])
    p.add_argument("--zero-isolated", action="store_true", default=Config.zero_isolated)
    p.add_argument("--kron-ridge", type=float, default=Config.kron_ridge)
    p.add_argument("--kron-use-pinv", action=argparse.BooleanOptionalAction, default=Config.kron_use_pinv)

    p.add_argument("--output-csv", type=str, default=Config.output_csv)
    p.add_argument("--output-json", type=str, default=Config.output_json)
    p.add_argument("--quiet", action="store_true", default=Config.quiet)
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    test_ratio = float(args.test_ratio)
    if str(args.split_ratios).strip():
        train_ratio, val_ratio, test_ratio = _parse_split_ratios(args.split_ratios)

    return Config(
        seed=int(args.seed),
        p_visible=float(args.p_visible),
        methods=str(args.methods),
        num_timesteps=int(args.num_timesteps),
        input_len=int(args.input_len),
        pred_len=int(args.pred_len),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        drop_gap_windows=bool(args.drop_gap_windows),
        loss_type=str(args.loss_type),
        mape_ridge=float(args.mape_ridge),
        epochs=int(args.epochs),
        lr=float(args.lr),
        ridge=float(args.ridge),
        sfl_ridge=float(args.sfl_ridge),
        optimizer=str(args.optimizer),
        lbfgs_max_iter=int(args.lbfgs_max_iter),
        poly_degree=int(args.poly_degree),
        sffa_k=int(args.sffa_k),
        sffa_r=int(args.sffa_r),
        rg_num=int(args.rg_num),
        rg_p0=float(args.rg_p0),
        max_sffa_basis_size=int(args.max_sffa_basis_size),
        dataset_dir=str(args.dataset_dir),
        device=str(args.device),
        dtype=str(args.dtype),
        zero_isolated=bool(args.zero_isolated),
        kron_ridge=float(args.kron_ridge),
        kron_use_pinv=bool(args.kron_use_pinv),
        output_csv=str(args.output_csv),
        output_json=str(args.output_json),
        quiet=bool(args.quiet),
    )


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    vals = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(v <= 0 for v in vals):
        raise ValueError("train/val/test ratios must all be positive.")
    total = sum(vals)
    if abs(total - 1.0) > 1e-8:
        vals = [v / total for v in vals]
    return vals[0], vals[1], vals[2]


def load_metr_la_graph_and_raw_values(
    dataset_dir: str,
    dtype: torch.dtype,
    device: torch.device,
    num_timesteps: int = 0,
    adj_filename: str = "adj_METR-LA.pkl",
    h5_filename: str = "METR-LA.h5",
) -> TrafficGraphData:
    root = Path(dataset_dir)
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

    A_cpu = torch.as_tensor(adj, dtype=dtype).clone()
    if A_cpu.dim() != 2 or A_cpu.shape[0] != A_cpu.shape[1]:
        raise ValueError(f"METR-LA adjacency must be square; got {tuple(A_cpu.shape)}.")
    A_cpu.fill_diagonal_(0.0)
    A_cpu = torch.maximum(A_cpu, A_cpu.T)
    A_cpu = (A_cpu > 0).to(dtype)
    n = int(A_cpu.shape[0])

    edge_index_cpu = torch.nonzero(A_cpu, as_tuple=False).T.contiguous()
    edge_weight_cpu = torch.ones(edge_index_cpu.shape[1], dtype=dtype)
    graph = Data(
        edge_index=edge_index_cpu.to(device=device),
        edge_attr=edge_weight_cpu.to(device=device),
        num_nodes=n,
    )

    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(n))
    graph_nx.add_edges_from(edge_index_cpu.T.detach().cpu().numpy().tolist())

    with h5py.File(str(h5_path), "r") as f:
        raw_values_np = f["df"]["block0_values"][:]
        _ = pd.to_datetime(f["df"]["axis1"][:])

    if raw_values_np.ndim != 2 or raw_values_np.shape[1] != n:
        raise ValueError(f"Expected raw METR-LA values with shape [T,{n}], got {raw_values_np.shape}.")

    raw_cpu = torch.as_tensor(raw_values_np, dtype=dtype)
    raw_total = int(raw_cpu.shape[0])

    # Zeros are legitimate traffic values.  Missingness is represented by
    # non-finite values, consistent with the one-step prediction runner.
    row_is_observed = torch.isfinite(raw_cpu).all(dim=1)
    raw_indices = torch.nonzero(row_is_observed, as_tuple=False).flatten()
    clean_cpu = raw_cpu.index_select(0, raw_indices)

    if int(num_timesteps) > 0:
        keep = min(int(num_timesteps), int(clean_cpu.shape[0]))
        clean_cpu = clean_cpu[:keep]
        raw_indices = raw_indices[:keep]

    if clean_cpu.shape[0] < 2:
        raise ValueError("Need at least two observed METR-LA timestamps after missing-row filtering.")

    pair_is_adjacent = (raw_indices[1:] - raw_indices[:-1]) == 1

    return TrafficGraphData(
        graph=graph,
        graph_nx=graph_nx,
        num_nodes=n,
        adjacency=A_cpu.to(device=device),
        raw_values=clean_cpu.to(device=device),
        raw_indices=raw_indices.to(device=device),
        pair_is_adjacent=pair_is_adjacent.to(device=device),
        raw_total_timestamps=raw_total,
        missing_timestamps=int((~row_is_observed).sum().item()),
    )


def build_horizon_windows_from_block(
    values: torch.Tensor,
    pair_is_adjacent: torch.Tensor,
    input_len: int,
    pred_len: int,
    *,
    drop_gap_windows: bool = True,
    global_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Build [S,P,N] -> [S,H,N] windows inside one already-split block."""
    if values.dim() != 2:
        raise ValueError("values must have shape [T, N].")
    input_len = int(input_len)
    pred_len = int(pred_len)
    if input_len <= 0 or pred_len <= 0:
        raise ValueError("input_len and pred_len must be positive.")

    block_T = int(values.shape[0])
    needed = input_len + pred_len
    if block_T < needed:
        raise ValueError(
            f"Block length {block_T} is too short for input_len={input_len}, pred_len={pred_len}."
        )

    pair_is_adjacent = pair_is_adjacent.to(device=values.device, dtype=torch.bool)
    if pair_is_adjacent.numel() != max(0, block_T - 1):
        raise ValueError("pair_is_adjacent length must be block_T - 1.")

    num_candidates = block_T - needed + 1
    starts = torch.arange(num_candidates, device=values.device, dtype=torch.long)
    total_gap_windows = 0

    if drop_gap_windows:
        edge_span = needed - 1
        if edge_span > 0:
            valid = torch.ones(num_candidates, device=values.device, dtype=torch.bool)
            for offset in range(edge_span):
                valid = valid & pair_is_adjacent[offset : offset + num_candidates]
            total_gap_windows = int((~valid).sum().item())
            starts = starts[valid]

    if starts.numel() == 0:
        raise ValueError(
            f"No valid windows remained in block with length={block_T}, input_len={input_len}, pred_len={pred_len}."
        )

    x_offsets = torch.arange(input_len, device=values.device, dtype=torch.long)
    y_offsets = torch.arange(input_len, input_len + pred_len, device=values.device, dtype=torch.long)
    x_idx = starts[:, None] + x_offsets[None, :]
    y_idx = starts[:, None] + y_offsets[None, :]

    x = values[x_idx]
    y = values[y_idx]
    global_starts = starts + int(global_offset)
    meta = {
        "candidate_windows": int(num_candidates),
        "windows": int(starts.numel()),
        "dropped_gap_windows": int(total_gap_windows),
        "gap_pairs_in_block": int((~pair_is_adjacent).sum().item()),
        "first_window_start": int(global_starts[0].item()),
        "last_window_start": int(global_starts[-1].item()),
    }
    return x, y, global_starts, meta


def make_horizon_train_val_test(
    values: torch.Tensor,
    pair_is_adjacent: torch.Tensor,
    input_len: int,
    pred_len: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    drop_gap_windows: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    if values.dim() != 2:
        raise ValueError("values must have shape [T, N].")
    T = int(values.shape[0])
    if pair_is_adjacent.numel() != T - 1:
        raise ValueError("pair_is_adjacent length must be values.shape[0] - 1.")

    train_ratio, val_ratio, test_ratio = validate_split_ratios(train_ratio, val_ratio, test_ratio)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    train_end = max(train_end, 0)
    val_end = max(val_end, train_end)
    val_end = min(val_end, T)

    needed = int(input_len) + int(pred_len)
    block_lengths = {
        "train": train_end,
        "val": val_end - train_end,
        "test": T - val_end,
    }
    too_short = {k: v for k, v in block_lengths.items() if v < needed}
    if too_short:
        raise ValueError(
            f"Each split block must contain at least input_len + pred_len = {needed} timestamps; "
            f"too short: {too_short}."
        )

    def block(a: int, b: int, name: str):
        vals = values[a:b]
        pairs = pair_is_adjacent[a : max(a, b - 1)]
        x, y, starts, meta = build_horizon_windows_from_block(
            vals,
            pairs,
            input_len=input_len,
            pred_len=pred_len,
            drop_gap_windows=drop_gap_windows,
            global_offset=a,
        )
        return x, y, starts, {f"{name}_{k}": v for k, v in meta.items()}

    x_train, y_train, _train_starts, train_meta = block(0, train_end, "train")
    x_val, y_val, _val_starts, val_meta = block(train_end, val_end, "val")
    x_test, y_test, _test_starts, test_meta = block(val_end, T, "test")

    meta: Dict[str, int] = {
        "clean_T": T,
        "input_len": int(input_len),
        "pred_len": int(pred_len),
        "train_end": int(train_end),
        "val_end": int(val_end),
        "test_end": int(T),
        "train_block_T": int(block_lengths["train"]),
        "val_block_T": int(block_lengths["val"]),
        "test_block_T": int(block_lengths["test"]),
        "adjacent_pairs_total": int(pair_is_adjacent.sum().item()),
        "gap_pairs_total": int((~pair_is_adjacent).sum().item()),
        "drop_gap_windows": int(bool(drop_gap_windows)),
    }
    meta.update(train_meta)
    meta.update(val_meta)
    meta.update(test_meta)
    return x_train, y_train, x_val, y_val, x_test, y_test, meta


def load_base_objects(cfg: Config):
    device = resolve_device(cfg.device)
    dtype = _torch_dtype(cfg.dtype)
    data = load_metr_la_graph_and_raw_values(
        num_timesteps=cfg.num_timesteps,
        dataset_dir=cfg.dataset_dir,
        dtype=dtype,
        device=device,
    )
    L_full = full_laplacian_from_graph(
        data.graph,
        dtype=dtype,
        device=device,
        zero_isolated=cfg.zero_isolated,
    )
    return data, L_full, device, dtype


def single_matrix_bases_for_prediction(method: str, cache: PrimitiveCache, sffa_k: int = 3) -> Optional[List[torch.Tensor]]:
    method = str(method).strip()
    if method == "identity_direct":
        return [torch.eye(cache.n0, dtype=cache.dtype, device=cache.v0_idx.device)]
    if method == "induced_direct":
        return [cache.induced()]
    if method == "kron_direct":
        return [cache.kron()]
    if method == "union_direct":
        return [cache.union(int(sffa_k))]
    return None


def set_record_losses(
    row: Dict[str, object],
    cfg: Config,
    *,
    y_train_pred: torch.Tensor,
    y_train: torch.Tensor,
    y_val_pred: torch.Tensor,
    y_val: torch.Tensor,
    y_test_pred: torch.Tensor,
    y_test: torch.Tensor,
) -> None:
    specs = [
        ("train", y_train_pred, y_train),
        ("val", y_val_pred, y_val),
        ("test", y_test_pred, y_test),
    ]
    for split, pred, target in specs:
        loss = eval_prediction_loss(pred, target, loss_type=cfg.loss_type, mape_ridge=cfg.mape_ridge)
        horizon_losses = eval_prediction_horizon_losses(pred, target, loss_type=cfg.loss_type, mape_ridge=cfg.mape_ridge)
        if split == "test":
            row["loss"] = loss
            row[cfg.loss_type] = loss
            for i, v in enumerate(horizon_losses, start=1):
                row[f"loss_h{i}"] = v
        else:
            row[f"{split}_loss"] = loss
            row[f"{split}_{cfg.loss_type}"] = loss
            for i, v in enumerate(horizon_losses, start=1):
                row[f"{split}_loss_h{i}"] = v


def run_one(cfg: Config, data: TrafficGraphData, L_full: torch.Tensor, device: torch.device, dtype: torch.dtype) -> List[Dict[str, object]]:
    set_seed(cfg.seed, deterministic=True)
    methods = _parse_methods(cfg.methods)

    v0_idx = choose_v0(data.num_nodes, cfg.p_visible, device=device)
    x_full = data.raw_values.to(device=device, dtype=dtype)
    x_sub = x_full.index_select(1, v0_idx)

    x_train, y_train, x_val, y_val, x_test, y_test, split_meta = make_horizon_train_val_test(
        values=x_sub,
        pair_is_adjacent=data.pair_is_adjacent,
        input_len=cfg.input_len,
        pred_len=cfg.pred_len,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        drop_gap_windows=cfg.drop_gap_windows,
    )

    cache = PrimitiveCache(
        A_full=data.adjacency,
        L_full=L_full,
        graph_nx=data.graph_nx,
        v0_idx=v0_idx,
        dtype=dtype,
        device=device,
        zero_isolated=cfg.zero_isolated,
        kron_ridge=cfg.kron_ridge,
        kron_use_pinv=cfg.kron_use_pinv,
    )

    train_ratio, val_ratio, test_ratio = validate_split_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)

    if not cfg.quiet:
        print(f"\n===== horizon seed={cfg.seed}, p_visible={cfg.p_visible:.4g}, P={cfg.input_len}, H={cfg.pred_len} =====")
        print(
            f"N={data.num_nodes} n0={cache.n0} raw_T={data.raw_total_timestamps} "
            f"clean_T={split_meta['clean_T']} missing_rows={data.missing_timestamps} "
            f"split_ratios={train_ratio:.4g}/{val_ratio:.4g}/{test_ratio:.4g}"
        )
        print(
            f"windows train/val/test={split_meta['train_windows']}/"
            f"{split_meta['val_windows']}/{split_meta['test_windows']} | "
            f"dropped_gap_windows train/val/test={split_meta['train_dropped_gap_windows']}/"
            f"{split_meta['val_dropped_gap_windows']}/{split_meta['test_dropped_gap_windows']}"
        )
        print(
            f"loss_type={cfg.loss_type} | poly_degree={cfg.poly_degree} "
            f"| sffa_k={cfg.sffa_k} sffa_r={cfg.sffa_r} | rg_p0={cfg.rg_p0:g} "
            f"| ridge={cfg.ridge:g} | sfl_ridge={cfg.sfl_ridge:g}"
        )
        print("-" * 112)

    records: List[Dict[str, object]] = []
    for method in methods:
        dispatch_method = "persistence" if method == "no_processing" else method
        row: Dict[str, object] = {
            "seed": cfg.seed,
            "p_visible": cfg.p_visible,
            "n0": cache.n0,
            "input_len": cfg.input_len,
            "pred_len": cfg.pred_len,
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "loss_type": cfg.loss_type,
            "loss": None,
            "train_loss": None,
            "val_loss": None,
            "mse": None,
            "train_mse": None,
            "val_mse": None,
            "mae": None,
            "train_mae": None,
            "val_mae": None,
            "num_parameters": 0,
            "status": "ok",
            "error": "",
            "raw_T": data.raw_total_timestamps,
            "missing_timestamps": data.missing_timestamps,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            **split_meta,
        }

        try:
            if dispatch_method == "persistence":
                y_train_pred = persistence_prediction(x_train, cfg.pred_len)
                y_val_pred = persistence_prediction(x_val, cfg.pred_len)
                y_test_pred = persistence_prediction(x_test, cfg.pred_len)
                row["num_parameters"] = 0

            elif dispatch_method == "ar":
                fit = fit_horizon_ar(
                    x_train=x_train,
                    y_train=y_train,
                    ridge=cfg.ridge,
                    epochs=cfg.epochs,
                    lr=cfg.lr,
                    optimizer_name=cfg.optimizer,
                    lbfgs_max_iter=cfg.lbfgs_max_iter,
                    loss_type=cfg.loss_type,
                    mape_ridge=cfg.mape_ridge,
                )
                y_train_pred = apply_horizon_ar(x_train, fit.weights)
                y_val_pred = apply_horizon_ar(x_val, fit.weights)
                y_test_pred = apply_horizon_ar(x_test, fit.weights)
                row["num_parameters"] = fit.num_parameters

            elif dispatch_method == "var":
                fit = fit_horizon_var(
                    x_train=x_train,
                    y_train=y_train,
                    ridge=cfg.ridge,
                    epochs=cfg.epochs,
                    lr=cfg.lr,
                    optimizer_name=cfg.optimizer,
                    lbfgs_max_iter=cfg.lbfgs_max_iter,
                    loss_type=cfg.loss_type,
                    mape_ridge=cfg.mape_ridge,
                )
                y_train_pred = apply_horizon_operators(x_train, fit.operators)
                y_val_pred = apply_horizon_operators(x_val, fit.operators)
                y_test_pred = apply_horizon_operators(x_test, fit.operators)
                row["num_parameters"] = fit.num_parameters

            elif dispatch_method in SINGLE_MATRIX_METHODS or dispatch_method in TRAINABLE_METHODS:
                if dispatch_method in SINGLE_MATRIX_METHODS:
                    bases = single_matrix_bases_for_prediction(dispatch_method, cache=cache, sffa_k=cfg.sffa_k)
                else:
                    bases = build_method_bases(
                        method=dispatch_method,
                        cache=cache,
                        poly_degree=cfg.poly_degree,
                        sffa_k=cfg.sffa_k,
                        sffa_r=cfg.sffa_r,
                        rg_num=cfg.rg_num,
                        rg_p0=cfg.rg_p0,
                        max_sffa_basis_size=cfg.max_sffa_basis_size,
                    )
                if bases is None:
                    raise RuntimeError(f"No bases were constructed for method={dispatch_method}.")
                fit = fit_horizon_basis_operator(
                    x_train=x_train,
                    y_train=y_train,
                    bases=bases,
                    epochs=cfg.epochs,
                    lr=cfg.lr,
                    ridge=cfg.sfl_ridge,
                    optimizer_name=cfg.optimizer,
                    lbfgs_max_iter=cfg.lbfgs_max_iter,
                    loss_type=cfg.loss_type,
                    mape_ridge=cfg.mape_ridge,
                )
                y_train_pred = apply_horizon_operators(x_train, fit.operators)
                y_val_pred = apply_horizon_operators(x_val, fit.operators)
                y_test_pred = apply_horizon_operators(x_test, fit.operators)
                row["num_parameters"] = fit.num_parameters

            else:
                raise RuntimeError(f"Unreachable method dispatch: {method}")

            set_record_losses(
                row,
                cfg,
                y_train_pred=y_train_pred,
                y_train=y_train,
                y_val_pred=y_val_pred,
                y_val=y_val,
                y_test_pred=y_test_pred,
                y_test=y_test,
            )

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)

        records.append(row)

        if not cfg.quiet:
            if row["status"] == "ok":
                print(
                    f"{method:16s} | test_{cfg.loss_type}={float(row['loss']):.6f} "
                    f"| val_{cfg.loss_type}={float(row['val_loss']):.6f} "
                    f"| train_{cfg.loss_type}={float(row['train_loss']):.6f} "
                    f"| params={int(row['num_parameters']):8d}"
                )
            else:
                print(f"{method:16s} | FAILED | {row['error']}")

    if not cfg.quiet:
        print("-" * 112)
    return records


def csv_fieldnames(records: Sequence[Dict[str, object]]) -> List[str]:
    max_h = 0
    for r in records:
        try:
            max_h = max(max_h, int(r.get("pred_len", 0)))
        except Exception:
            pass

    base = [
        "seed", "p_visible", "n0", "input_len", "pred_len", "method", "label",
        "loss_type", "loss", "train_loss", "val_loss",
        "mse", "train_mse", "val_mse", "mae", "train_mae", "val_mae",
        "num_parameters", "status", "error", "raw_T", "missing_timestamps",
        "clean_T", "train_ratio", "val_ratio", "test_ratio",
        "train_end", "val_end", "test_end",
        "train_block_T", "val_block_T", "test_block_T",
        "adjacent_pairs_total", "gap_pairs_total", "drop_gap_windows",
        "train_candidate_windows", "train_windows", "train_dropped_gap_windows", "train_gap_pairs_in_block",
        "train_first_window_start", "train_last_window_start",
        "val_candidate_windows", "val_windows", "val_dropped_gap_windows", "val_gap_pairs_in_block",
        "val_first_window_start", "val_last_window_start",
        "test_candidate_windows", "test_windows", "test_dropped_gap_windows", "test_gap_pairs_in_block",
        "test_first_window_start", "test_last_window_start",
    ]
    horizon = []
    for i in range(1, max_h + 1):
        horizon.extend([f"loss_h{i}", f"val_loss_h{i}", f"train_loss_h{i}"])
    return base + horizon


def write_csv(path: str, records: Sequence[Dict[str, object]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(records)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path: str, cfg: Config, records: Sequence[Dict[str, object]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(cfg), "records": list(records)}
    with out.open("w") as f:
        json.dump(payload, f, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg0 = config_from_args(args)

    # Validate once before loading data.
    validate_split_ratios(cfg0.train_ratio, cfg0.val_ratio, cfg0.test_ratio)
    if cfg0.input_len <= 0 or cfg0.pred_len <= 0:
        raise ValueError("input_len and pred_len must be positive.")

    seeds = _parse_int_range_or_csv(args.seeds) if args.seeds else [cfg0.seed]
    p_values = _parse_float_range_or_csv(args.p_visibles) if args.p_visibles else [cfg0.p_visible]
    pred_lens = _parse_int_range_or_csv(args.pred_lens) if args.pred_lens else [cfg0.pred_len]

    data, L_full, device, dtype = load_base_objects(cfg0)

    all_records: List[Dict[str, object]] = []
    base_dict = asdict(cfg0)
    for pred_len in pred_lens:
        for seed in seeds:
            for p_visible in p_values:
                cfg = Config(**{**base_dict, "seed": int(seed), "p_visible": float(p_visible), "pred_len": int(pred_len)})
                all_records.extend(run_one(cfg, data=data, L_full=L_full, device=device, dtype=dtype))

    write_csv(cfg0.output_csv, all_records)
    write_json(cfg0.output_json, cfg0, all_records)

    failed = [r for r in all_records if r.get("status") != "ok"]
    if failed:
        print(f"\nFailed methods/runs: {len(failed)}")
        for r in failed[:20]:
            print(
                f"seed={r['seed']} p={r['p_visible']} P={r['input_len']} H={r['pred_len']} "
                f"method={r['method']} error={r['error']}"
            )
        if len(failed) > 20:
            print(f"... {len(failed) - 20} more failures omitted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
