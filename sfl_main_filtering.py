#!/usr/bin/env python3
"""Filtering runner for switchable Subgraph Filter Learning experiments.

This is the companion runner for ``sfl_core_filtering.py``.  It keeps the old
single-time-step filtering experiment structure, but exposes the new switches:

    truth:      linear Laplacian polynomial, Volterra-Hadamard, or explicit Hadamard nonlinear truth
    noise:      Gaussian, uniform, Rademacher, or skewed bounded sub-Gaussian noise
    loss:       MSE, Huber, log-cosh, or smooth pinball, tied between training and test evaluation
    optimizer:  LBFGS by default, Adam optional

Only the filtering methods requested for the revised experiment are enabled:

    sub_lp, kron_lp, union_lp, dk_lap_poly, rg_alg, sffa, num_lmmse

The removed direct/truncated and Kron-SFFA variants are intentionally rejected by
argument parsing.  METR-LA rows with NaN/Inf are treated as missing; zero entries
are kept as valid traffic signals.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from sfl_core_filtering import (  # type: ignore
        CLOSED_FORM_METHODS,
        DEFAULT_METHODS,
        DEFAULT_TRUTH_COEFFS,
        DEFAULT_VOLTERRA_QUADRATIC_COEFFS,
        METHOD_LABELS,
        TRAINABLE_METHODS,
        PrimitiveCache,
        build_method_bases,
        choose_v0,
        eval_operator_loss,
        fit_lmmse_operator,
        full_laplacian_from_graph,
        load_metr_la_traffic,
        make_noisy_labels,
        resolve_device,
        run_seed_sequence,
        set_seed,
        split_train_test_with_meta,
        train_eval_basis_loss,
    )
except ModuleNotFoundError:
    from revisedfilter.sfl_core_filtering_skewloss_hadamard import (  # type: ignore
        CLOSED_FORM_METHODS,
        DEFAULT_METHODS,
        DEFAULT_TRUTH_COEFFS,
        DEFAULT_VOLTERRA_QUADRATIC_COEFFS,
        METHOD_LABELS,
        TRAINABLE_METHODS,
        PrimitiveCache,
        build_method_bases,
        choose_v0,
        eval_operator_loss,
        fit_lmmse_operator,
        full_laplacian_from_graph,
        load_metr_la_traffic,
        make_noisy_labels,
        resolve_device,
        run_seed_sequence,
        set_seed,
        split_train_test_with_meta,
        train_eval_basis_loss,
    )


REMOVED_METHODS = {
    "plain_trunc",
    "truncated",
    "induced_direct",
    "kron_direct",
    "union_direct",
    "rg_lp",
    "kron12",
    "kron_sffa",
}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    runs: int = 1
    p_visible: float = 0.8
    methods: str = DEFAULT_METHODS

    # METR-LA sample splitting.  In number mode, the loader only needs
    # train_samples + test_samples complete rows.  In ratio mode it loads all
    # complete rows and splits them sequentially.
    split_mode: str = "number"  # number or ratio
    train_samples: int = 50
    test_samples: int = 50
    train_ratio: float = 0.7

    # Truth f.
    truth_type: str = "linear_poly"  # linear_poly/poly3, volterra, or hadamard
    truth_coeffs: Tuple[float, ...] = DEFAULT_TRUTH_COEFFS
    volterra_quadratic_coeffs: Tuple[float, ...] = DEFAULT_VOLTERRA_QUADRATIC_COEFFS
    volterra_lambda: float = 1.0
    nonlinear_scale: float = 0.0
    center_nonlinear_input: bool = True
    normalize_truth_base: bool = False

    # Additive noise.
    add_noise: bool = True
    sigma: float = 5.0
    noise_type: str = "gaussian"  # gaussian, uniform, rademacher, skew_bernoulli

    # Training/evaluation loss.  The same loss is used for fitting trainable
    # methods and for test evaluation.  num_lmmse remains MSE-trained but is
    # evaluated under the selected loss.
    loss_type: str = "mse"  # mse, huber, logcosh, or smooth_pinball
    huber_delta: float = 1.0
    tau_pinball: float = 0.8

    # Optimization for learned basis methods.
    optimizer: str = "lbfgs"
    epochs: int = 20
    lbfgs_max_iter: int = 50
    lr: float = 0.5
    sfl_ridge: float = 0.0

    # Numerical LMMSE ridge.
    ridge: float = 10.0

    # Filter-bank parameters.
    k_sffa: int = 3
    r_sffa: int = 3
    r_poly: int = 3
    k_union: int = 3
    rg_p0: float = 0.9
    max_sffa_basis_size: int = 2000

    # Graph/data options.
    dataset_dir: str = "dataset"
    device: str = "auto"
    dtype: str = "float32"
    zero_isolated: bool = False
    kron_ridge: float = 0.0
    kron_use_pinv: bool = False

    # Output.
    output_csv: str = ""
    output_raw_csv: str = ""
    output_json: str = ""
    quiet: bool = False


def _parse_csv_floats(text: str) -> Tuple[float, ...]:
    vals: List[float] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one float")
    return tuple(vals)


def _parse_methods(text: str) -> List[str]:
    methods = [m.strip() for m in str(text).replace(";", ",").split(",") if m.strip()]
    methods = ["dk_lap_poly" if m == "dk_lp" else m for m in methods]
    if not methods:
        raise ValueError("At least one method must be supplied.")
    known = set(TRAINABLE_METHODS) | set(CLOSED_FORM_METHODS)
    unknown = [m for m in methods if m not in known]
    if unknown:
        removed = [m for m in unknown if m in REMOVED_METHODS]
        truly_unknown = [m for m in unknown if m not in REMOVED_METHODS]
        parts = []
        if removed:
            parts.append(f"removed methods: {removed}")
        if truly_unknown:
            parts.append(f"unknown methods: {truly_unknown}")
        parts.append(f"enabled methods: {sorted(known)}")
        raise ValueError("; ".join(parts))
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


def _torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in {"float32", "fp32", "32"}:
        return torch.float32
    if name in {"float64", "fp64", "double", "64"}:
        return torch.float64
    raise ValueError("dtype must be float32 or float64.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run filtering SFL experiments on METR-LA.")

    p.add_argument("--seed", type=int, default=Config.seed,
                   help="First seed. With --runs R, seeds are seed, ..., seed+R-1.")
    p.add_argument("--runs", type=int, default=Config.runs,
                   help="Number of consecutive seeds to run and average.")
    p.add_argument("--p-visible", type=float, default=Config.p_visible)
    p.add_argument("--p-visibles", type=str, default="",
                   help="Optional batch p values, e.g. '0.2,0.5,0.8' or '0.1:1.0:0.1'.")
    p.add_argument("--methods", type=str, default=Config.methods)

    p.add_argument("--split-mode", type=str, default=Config.split_mode, choices=["number", "ratio"])
    p.add_argument("--train-samples", "--train-steps", dest="train_samples", type=int, default=Config.train_samples)
    p.add_argument("--test-samples", "--test-steps", dest="test_samples", type=int, default=Config.test_samples)
    p.add_argument("--train-ratio", type=float, default=Config.train_ratio,
                   help="Sequential train ratio used only when --split-mode ratio.")

    p.add_argument("--truth-type", type=str, default=Config.truth_type,
                   choices=["linear_poly", "poly3", "volterra", "hadamard"])
    p.add_argument("--truth-coeffs", type=_parse_csv_floats, default=Config.truth_coeffs,
                   help="Coefficients for linear_poly truth, e.g. '-1,9,-12,4'.")
    p.add_argument("--volterra-quadratic-coeffs", type=_parse_csv_floats,
                   default=Config.volterra_quadratic_coeffs,
                   help="Quadratic polynomial coefficients. For volterra this is B; for hadamard this is B3. Default: 0,0,1.")
    p.add_argument("--volterra-lambda", type=float, default=Config.volterra_lambda,
                   help="Overall strength lambda of the nonlinear truth channel.")
    p.add_argument("--nonlinear-scale", type=float, default=Config.nonlinear_scale,
                   help="Scale s in the nonlinear tanh. <=0 uses per-sample RMS.")
    p.add_argument("--center-nonlinear-input", action=argparse.BooleanOptionalAction,
                   default=Config.center_nonlinear_input)
    p.add_argument("--normalize-truth-base", action=argparse.BooleanOptionalAction,
                   default=Config.normalize_truth_base)

    p.add_argument("--noise-type", type=str, default=Config.noise_type,
                   choices=["gaussian", "uniform", "rademacher", "skew_bernoulli", "skew", "skewed"])
    p.add_argument("--sigma", type=float, default=Config.sigma)
    p.add_argument("--add-noise", action=argparse.BooleanOptionalAction, default=Config.add_noise)
    p.add_argument("--add-wgn", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-wgn", action="store_true",
                   help="Compatibility alias: disable additive noise, regardless of --noise-type.")

    p.add_argument(
        "--loss-type",
        type=str,
        default=Config.loss_type,
        choices=["mse", "huber", "logcosh", "log_cosh", "smooth_pinball", "pinball"],
    )
    p.add_argument("--huber-delta", type=float, default=Config.huber_delta,
                   help="Smoothing scale for Huber, log-cosh, and smooth pinball losses.")
    p.add_argument("--tau-pinball", "--tau_pinball", dest="tau_pinball", type=float,
                   default=Config.tau_pinball,
                   help="Quantile/asymmetry parameter for smooth pinball. Default 0.8; positive residual asymptotic slope is tau.")

    p.add_argument("--optimizer", type=str, default=Config.optimizer, choices=["lbfgs", "adam"])
    p.add_argument("--epochs", type=int, default=Config.epochs,
                   help="Outer optimization iterations. Default: 20.")
    p.add_argument("--lbfgs-max-iter", "--lbfgs-iter", dest="lbfgs_max_iter", type=int,
                   default=Config.lbfgs_max_iter,
                   help="Inner LBFGS max_iter. Default: 50.")
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--sfl-ridge", type=float, default=Config.sfl_ridge,
                   help="Optional Frobenius penalty for learned basis methods.")
    p.add_argument("--ridge", type=float, default=Config.ridge,
                   help="Ridge regularization for numerical LMMSE.")

    p.add_argument("--k-sffa", "--sffa-k", "--k", dest="k_sffa", type=int, default=Config.k_sffa)
    p.add_argument("--r-sffa", "--sffa-r", "--r", dest="r_sffa", type=int, default=Config.r_sffa)
    p.add_argument("--r-poly", "--poly-degree", dest="r_poly", type=int, default=Config.r_poly)
    p.add_argument("--k-union", dest="k_union", type=int, default=Config.k_union)
    p.add_argument("--rg-p0", type=float, default=Config.rg_p0)
    p.add_argument("--max-sffa-basis-size", type=int, default=Config.max_sffa_basis_size)

    p.add_argument("--dataset-dir", type=str, default=Config.dataset_dir)
    p.add_argument("--device", type=str, default=Config.device, choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default=Config.dtype, choices=["float32", "float64"])
    p.add_argument("--zero-isolated", action="store_true", default=Config.zero_isolated)
    p.add_argument("--kron-ridge", type=float, default=Config.kron_ridge)
    p.add_argument("--kron-use-pinv", action=argparse.BooleanOptionalAction, default=Config.kron_use_pinv)

    p.add_argument("--output-csv", type=str, default=Config.output_csv,
                   help="Write run-averaged summary records to CSV.")
    p.add_argument("--output-raw-csv", type=str, default=Config.output_raw_csv,
                   help="Write per-run records to CSV.")
    p.add_argument("--output-json", type=str, default=Config.output_json,
                   help="Write config, raw records, and summary records to JSON.")
    p.add_argument("--quiet", action="store_true", default=Config.quiet)
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    add_noise = (bool(args.add_noise) or bool(args.add_wgn)) and not bool(args.no_wgn)
    return Config(
        seed=int(args.seed),
        runs=int(args.runs),
        p_visible=float(args.p_visible),
        methods=str(args.methods),
        split_mode=str(args.split_mode),
        train_samples=int(args.train_samples),
        test_samples=int(args.test_samples),
        train_ratio=float(args.train_ratio),
        truth_type=str(args.truth_type),
        truth_coeffs=tuple(float(v) for v in args.truth_coeffs),
        volterra_quadratic_coeffs=tuple(float(v) for v in args.volterra_quadratic_coeffs),
        volterra_lambda=float(args.volterra_lambda),
        nonlinear_scale=float(args.nonlinear_scale),
        center_nonlinear_input=bool(args.center_nonlinear_input),
        normalize_truth_base=bool(args.normalize_truth_base),
        add_noise=add_noise,
        sigma=float(args.sigma),
        noise_type=str(args.noise_type),
        loss_type=str(args.loss_type),
        huber_delta=float(args.huber_delta),
        tau_pinball=float(args.tau_pinball),
        optimizer=str(args.optimizer),
        epochs=int(args.epochs),
        lbfgs_max_iter=int(args.lbfgs_max_iter),
        lr=float(args.lr),
        sfl_ridge=float(args.sfl_ridge),
        ridge=float(args.ridge),
        k_sffa=int(args.k_sffa),
        r_sffa=int(args.r_sffa),
        r_poly=int(args.r_poly),
        k_union=int(args.k_union),
        rg_p0=float(args.rg_p0),
        max_sffa_basis_size=int(args.max_sffa_basis_size),
        dataset_dir=str(args.dataset_dir),
        device=str(args.device),
        dtype=str(args.dtype),
        zero_isolated=bool(args.zero_isolated),
        kron_ridge=float(args.kron_ridge),
        kron_use_pinv=bool(args.kron_use_pinv),
        output_csv=str(args.output_csv),
        output_raw_csv=str(args.output_raw_csv),
        output_json=str(args.output_json),
        quiet=bool(args.quiet),
    )


def _loader_timesteps_for_cfg(cfg: Config) -> Optional[int]:
    if str(cfg.split_mode).lower().strip() == "number":
        return int(cfg.train_samples) + int(cfg.test_samples)
    return None


def load_base_objects(cfg: Config):
    device = resolve_device(cfg.device)
    dtype = _torch_dtype(cfg.dtype)
    data = load_metr_la_traffic(
        num_timesteps=_loader_timesteps_for_cfg(cfg),
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


def _base_record(cfg: Config, method: str, cache: PrimitiveCache, split_meta: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "seed": int(cfg.seed),
        "p_visible": float(cfg.p_visible),
        "n0": int(cache.n0),
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "loss_type": cfg.loss_type,
        "tau_pinball": float(cfg.tau_pinball),
        "loss": None,
        "train_loss": None,
        "num_parameters": 0,
        "status": "ok",
        "error": "",
        "truth_type": cfg.truth_type,
        "noise_type": cfg.noise_type if cfg.add_noise and float(cfg.sigma) != 0.0 else "none",
        "sigma": float(cfg.sigma) if cfg.add_noise else 0.0,
        "split_mode": split_meta.get("split_mode", cfg.split_mode),
        "clean_T": split_meta.get("clean_T", ""),
        "train_samples": split_meta.get("train_samples", ""),
        "test_samples": split_meta.get("test_samples", ""),
        "train_ratio": split_meta.get("train_ratio", ""),
        "r_poly": int(cfg.r_poly),
        "k_union": int(cfg.k_union),
        "k_sffa": int(cfg.k_sffa),
        "r_sffa": int(cfg.r_sffa),
        "optimizer": cfg.optimizer,
    }
    return row


def run_one(cfg: Config, data, L_full: torch.Tensor, device: torch.device, dtype: torch.dtype) -> List[Dict[str, object]]:
    """Run one seed/p_visible filtering experiment and return one record per method."""
    set_seed(cfg.seed, deterministic=True)
    methods = _parse_methods(cfg.methods)

    truth = make_noisy_labels(
        x_full=data.x_full.to(device=device, dtype=dtype),
        L_full=L_full.to(device=device, dtype=dtype),
        truth_type=cfg.truth_type,
        coeffs=cfg.truth_coeffs,
        quadratic_coeffs=cfg.volterra_quadratic_coeffs,
        volterra_lambda=cfg.volterra_lambda,
        nonlinear_scale=cfg.nonlinear_scale,
        center_nonlinear_input=cfg.center_nonlinear_input,
        normalize_base=cfg.normalize_truth_base,
        sigma=cfg.sigma,
        noise_type=cfg.noise_type,
        noise_seed=None,
        add_noise=cfg.add_noise,
    )

    v0_idx = choose_v0(data.num_nodes, cfg.p_visible, device=device)
    x_sub = data.x_full.to(device=device, dtype=dtype).index_select(1, v0_idx)
    y_sub = truth.y_full.to(device=device, dtype=dtype).index_select(1, v0_idx)

    x_train, y_train, x_test, y_test, split_meta = split_train_test_with_meta(
        x_sub,
        y_sub,
        train_steps=cfg.train_samples,
        test_steps=cfg.test_samples,
        split_mode=cfg.split_mode,
        train_ratio=cfg.train_ratio,
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

    if not cfg.quiet:
        print(f"\n===== filtering seed={cfg.seed}, p_visible={cfg.p_visible:.4g} =====")
        print(
            f"N={data.num_nodes} n0={cache.n0} raw_T={data.raw_total_timestamps} "
            f"clean_T={split_meta['clean_T']} missing_rows={data.missing_timestamps} "
            f"split={split_meta['split_mode']} train={split_meta['train_samples']} "
            f"test={split_meta['test_samples']}"
        )
        print(
            f"loss={cfg.loss_type} huber_delta={cfg.huber_delta:g} tau_pinball={cfg.tau_pinball:g} | "
            f"optimizer={cfg.optimizer} epochs={cfg.epochs} lr={cfg.lr:g} "
            f"lbfgs_iter={cfg.lbfgs_max_iter}"
        )
        print(
            f"r_poly={cfg.r_poly} k_union={cfg.k_union} "
            f"k_sffa={cfg.k_sffa} r_sffa={cfg.r_sffa} rg_p0={cfg.rg_p0:g}"
        )
        print(truth.summary)
        print("-" * 104)

    records: List[Dict[str, object]] = []

    for method in methods:
        row = _base_record(cfg, method, cache, split_meta)
        try:
            if method in CLOSED_FORM_METHODS:
                H = fit_lmmse_operator(x_train=x_train, y_train=y_train, ridge=cfg.ridge)
                test_loss = eval_operator_loss(
                    x_test,
                    y_test,
                    H,
                    loss_type=cfg.loss_type,
                    huber_delta=cfg.huber_delta,
                    tau_pinball=cfg.tau_pinball,
                )
                train_loss = eval_operator_loss(
                    x_train,
                    y_train,
                    H,
                    loss_type=cfg.loss_type,
                    huber_delta=cfg.huber_delta,
                    tau_pinball=cfg.tau_pinball,
                )
                row["loss"] = test_loss
                row["train_loss"] = train_loss
                row["num_parameters"] = int(H.numel())
                row["trained_objective"] = "mse_closed_form"
                row["ridge"] = float(cfg.ridge)

            elif method in TRAINABLE_METHODS:
                bases = build_method_bases(
                    method=method,
                    cache=cache,
                    r_poly=cfg.r_poly,
                    k_union=cfg.k_union,
                    k_sffa=cfg.k_sffa,
                    r_sffa=cfg.r_sffa,
                    rg_p0=cfg.rg_p0,
                    max_sffa_basis_size=cfg.max_sffa_basis_size,
                )
                if bases is None:
                    raise RuntimeError(f"No bases were constructed for method={method}.")
                test_loss, fit = train_eval_basis_loss(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    bases=bases,
                    epochs=cfg.epochs,
                    lr=cfg.lr,
                    ridge=cfg.sfl_ridge,
                    optimizer_name=cfg.optimizer,
                    lbfgs_max_iter=cfg.lbfgs_max_iter,
                    loss_type=cfg.loss_type,
                    huber_delta=cfg.huber_delta,
                    tau_pinball=cfg.tau_pinball,
                )
                row["loss"] = test_loss
                row["train_loss"] = fit.train_loss
                row["num_parameters"] = int(fit.num_parameters)
                row["trained_objective"] = cfg.loss_type
                row["ridge"] = float(cfg.sfl_ridge)

            else:
                raise RuntimeError(f"Unreachable method dispatch: {method}")

            row[cfg.loss_type] = row["loss"]
            row[f"train_{cfg.loss_type}"] = row["train_loss"]

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)

        records.append(row)

        if not cfg.quiet:
            if row["status"] == "ok":
                train_txt = "" if row["train_loss"] is None else f" train={float(row['train_loss']):.6g}"
                print(
                    f"{method:16s} | {cfg.loss_type}={float(row['loss']):.6g}"
                    f"{train_txt} | params={int(row['num_parameters']):6d}"
                )
            else:
                print(f"{method:16s} | FAILED | {row['error']}")

    if not cfg.quiet:
        print("-" * 104)
    return records


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return float((sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5)


def summarize_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[float, str], List[Dict[str, object]]] = {}
    for r in records:
        key = (float(r.get("p_visible", 0.0)), str(r.get("method", "")))
        groups.setdefault(key, []).append(r)

    summary: List[Dict[str, object]] = []
    for (p_visible, method), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ok = [r for r in rows if r.get("status") == "ok" and r.get("loss") is not None]
        losses = [float(r["loss"]) for r in ok]
        train_losses = [float(r["train_loss"]) for r in ok if r.get("train_loss") is not None]
        first = rows[0]
        out: Dict[str, object] = {
            "p_visible": p_visible,
            "method": method,
            "label": first.get("label", method),
            "loss_type": first.get("loss_type", ""),
            "tau_pinball": first.get("tau_pinball", ""),
            "loss_mean": _mean(losses),
            "loss_std": _std(losses),
            "train_loss_mean": _mean(train_losses),
            "train_loss_std": _std(train_losses),
            "ok_runs": len(ok),
            "failed_runs": len(rows) - len(ok),
            "total_runs": len(rows),
            "num_parameters": ok[0].get("num_parameters", first.get("num_parameters", "")) if ok else "",
            "n0_mean": _mean([float(r["n0"]) for r in ok]) if ok else float("nan"),
            "truth_type": first.get("truth_type", ""),
            "noise_type": first.get("noise_type", ""),
            "sigma": first.get("sigma", ""),
            "split_mode": first.get("split_mode", ""),
            "train_samples": first.get("train_samples", ""),
            "test_samples": first.get("test_samples", ""),
            "train_ratio": first.get("train_ratio", ""),
            "r_poly": first.get("r_poly", ""),
            "k_union": first.get("k_union", ""),
            "k_sffa": first.get("k_sffa", ""),
            "r_sffa": first.get("r_sffa", ""),
        }
        summary.append(out)
    return summary


def write_csv(path: str, records: Sequence[Dict[str, object]], *, summary: bool) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if summary:
        fieldnames = [
            "p_visible",
            "method",
            "label",
            "loss_type",
            "tau_pinball",
            "loss_mean",
            "loss_std",
            "train_loss_mean",
            "train_loss_std",
            "ok_runs",
            "failed_runs",
            "total_runs",
            "num_parameters",
            "n0_mean",
            "truth_type",
            "noise_type",
            "sigma",
            "split_mode",
            "train_samples",
            "test_samples",
            "train_ratio",
            "r_poly",
            "k_union",
            "k_sffa",
            "r_sffa",
        ]
    else:
        fieldnames = [
            "seed",
            "p_visible",
            "n0",
            "method",
            "label",
            "loss_type",
            "tau_pinball",
            "loss",
            "train_loss",
            "mse",
            "huber",
            "logcosh",
            "smooth_pinball",
            "pinball",
            "train_mse",
            "train_huber",
            "train_logcosh",
            "train_smooth_pinball",
            "train_pinball",
            "trained_objective",
            "ridge",
            "num_parameters",
            "status",
            "error",
            "truth_type",
            "noise_type",
            "sigma",
            "split_mode",
            "clean_T",
            "train_samples",
            "test_samples",
            "train_ratio",
            "r_poly",
            "k_union",
            "k_sffa",
            "r_sffa",
            "optimizer",
        ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path: str, cfg: Config, records: Sequence[Dict[str, object]], summary_records: Sequence[Dict[str, object]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "records": list(records),
        "summary_records": list(summary_records),
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)


def print_summary(summary_records: Sequence[Dict[str, object]], quiet: bool = False) -> None:
    if quiet or not summary_records:
        return
    print("\n===== run-averaged summary =====")
    for r in summary_records:
        loss_type = str(r.get("loss_type", "loss"))
        mean = float(r.get("loss_mean", float("nan")))
        std = float(r.get("loss_std", 0.0))
        ok = int(r.get("ok_runs", 0))
        total = int(r.get("total_runs", 0))
        print(
            f"p={float(r['p_visible']):.4g} {str(r['method']):16s} | "
            f"{loss_type}_mean={mean:.6g} std={std:.6g} | ok={ok}/{total}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg0 = config_from_args(args)
    if cfg0.runs <= 0:
        raise ValueError("--runs must be positive.")

    p_values = _parse_float_range_or_csv(args.p_visibles) if args.p_visibles else [cfg0.p_visible]
    run_seeds = run_seed_sequence(cfg0.seed, cfg0.runs)

    data, L_full, device, dtype = load_base_objects(cfg0)

    all_records: List[Dict[str, object]] = []
    for p_visible in p_values:
        for seed in run_seeds:
            cfg = Config(**{**asdict(cfg0), "seed": int(seed), "p_visible": float(p_visible)})
            all_records.extend(run_one(cfg, data=data, L_full=L_full, device=device, dtype=dtype))

    summary_records = summarize_records(all_records)
    print_summary(summary_records, quiet=cfg0.quiet)

    write_csv(cfg0.output_csv, summary_records, summary=True)
    write_csv(cfg0.output_raw_csv, all_records, summary=False)
    write_json(cfg0.output_json, cfg0, all_records, summary_records)

    failed = [r for r in all_records if r.get("status") != "ok"]
    if failed:
        print(f"\nFailed methods/runs: {len(failed)}")
        for r in failed[:20]:
            print(f"seed={r['seed']} p={r['p_visible']} method={r['method']} error={r['error']}")
        if len(failed) > 20:
            print(f"... {len(failed) - 20} more failures omitted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
