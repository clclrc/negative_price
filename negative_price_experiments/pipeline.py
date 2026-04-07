from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_MODELS,
    FINAL_TEST_RANGE,
    FINAL_TRAIN_RANGE,
    WALK_FORWARD_FOLDS,
    AdaptBudget,
    ExperimentConfig,
    TransferConfig,
    build_default_experiment_configs,
)
from .data import TabularScaler, prepare_experiment_data
from .metrics import add_month_column, compute_binary_metrics, find_best_threshold_f1, summarize_prediction_frame
from .models import (
    DependencyUnavailableError,
    apply_probability_calibrator,
    fit_logistic_regression,
    fit_lightgbm_classifier,
    fit_majority_baseline,
    fit_catboost_classifier,
    fit_probability_calibrator,
    fit_sequence_final,
    fit_xgboost_classifier,
    fit_xgboost_final,
    load_sequence_model,
    predict_catboost,
    predict_lightgbm,
    predict_logistic_regression,
    predict_majority,
    predict_sequence_model,
    predict_xgboost,
    require_torch,
    train_sequence_model,
)
from .progress import ProgressReporter, estimate_remaining_seconds, format_duration, format_metric


LOGISTIC_CANDIDATES = (0.1, 1.0, 10.0)
XGBOOST_CANDIDATES = (
    {"max_depth": 4, "learning_rate": 0.05},
    {"max_depth": 4, "learning_rate": 0.1},
    {"max_depth": 6, "learning_rate": 0.05},
    {"max_depth": 6, "learning_rate": 0.1},
)
LIGHTGBM_CANDIDATES = (
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300},
    {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 200},
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 400},
    {"num_leaves": 63, "learning_rate": 0.1, "n_estimators": 250},
)
CATBOOST_CANDIDATES = (
    {"depth": 6, "learning_rate": 0.05, "n_estimators": 300},
    {"depth": 6, "learning_rate": 0.1, "n_estimators": 200},
    {"depth": 8, "learning_rate": 0.05, "n_estimators": 400},
    {"depth": 8, "learning_rate": 0.1, "n_estimators": 250},
)
XGBOOST_WEIGHTED_CALIBRATED_CANDIDATES = (
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "calibration": "sigmoid"},
    {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 200, "calibration": "sigmoid"},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 400, "calibration": "isotonic"},
    {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 250, "calibration": "isotonic"},
)
HYBRID_SEQUENCE_MODELS = (
    "GRUHybrid",
    "GRUHybridAttn",
    "GRUHybridGated",
    "GRUHybridGatedMultiTask",
    "GRUMultiMarketHybrid",
    "GraphTemporalHybrid",
)
MULTI_MARKET_SEQUENCE_MODELS = (
    "GRUMultiMarket",
    "GRUMultiMarketHybrid",
    "GRUMultiMarketTargetAttn",
    "GRUMultiMarketTemporalAttn",
    "GraphTemporal",
    "GraphTemporalHybrid",
)


def run_experiment(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    folds=None,
    final_train_range=None,
    final_test_range=None,
    skip_unavailable_models: bool = False,
    reporter: ProgressReporter | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir).resolve() / config.name
    if config.meta_kind == "late_fusion":
        return _run_late_fusion_experiment(
            config,
            output_path=output_path,
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
    if config.meta_kind == "stacking":
        return _run_stacking_experiment(
            config,
            output_path=output_path,
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
    if config.meta_kind == "cross_seed_ensemble":
        return _run_cross_seed_ensemble_experiment(
            config,
            output_path=output_path,
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
    if config.meta_kind == "calibration":
        return _run_calibration_experiment(
            config,
            output_path=output_path,
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
    if config.repeat_random_seeds:
        return _run_repeated_seed_experiment(
            config,
            output_path=output_path,
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log = output_path / "progress.log"
    reporter = (reporter or ProgressReporter()).with_log_file(progress_log)
    folds = tuple(folds or WALK_FORWARD_FOLDS)
    final_train_range = final_train_range or FINAL_TRAIN_RANGE
    final_test_range = final_test_range or FINAL_TEST_RANGE

    experiment_started_at = reporter.now()
    reporter.log(
        (config.name,),
        f"experiment started | data={config.data_path} | output={output_path}",
    )
    prep_started_at = reporter.now()
    reporter.log((config.name,), "data prep started")
    prepared = prepare_experiment_data(config)
    reporter.log(
        (config.name,),
        (
            f"data prep completed | samples={len(prepared.sample_manifest)} "
            f"| countries={len(prepared.country_panels)} "
            f"| elapsed={format_duration(reporter.now() - prep_started_at)}"
        ),
    )
    prepared.sample_manifest.to_csv(output_path / "sample_manifest.csv", index=False)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    chosen_records: dict[str, dict[str, object]] = {}

    models = tuple(config.models or DEFAULT_MODELS)
    model_loop_started_at = reporter.now()
    for model_index, model_name in enumerate(models, start=1):
        model_started_at = reporter.now()
        reporter.log(
            (config.name, model_name),
            f"model {model_index}/{len(models)} validation started",
        )
        try:
            chosen_records[model_name], model_metrics, model_predictions = _evaluate_model_across_folds(
                prepared,
                config,
                model_name,
                folds,
                reporter=reporter,
            )
        except DependencyUnavailableError:
            if skip_unavailable_models:
                reporter.log(
                    (config.name, model_name),
                    (
                        f"model {model_index}/{len(models)} skipped "
                        f"| reason=missing dependency "
                        f"| elapsed={format_duration(reporter.now() - model_started_at)} "
                        f"| eta={format_duration(estimate_remaining_seconds(loop_started_at=model_loop_started_at, completed_steps=model_index, total_steps=len(models), now=reporter.now()))}"
                    ),
                )
                continue
            raise
        metrics_rows.extend(model_metrics)
        prediction_frames.extend(model_predictions)
        reporter.log(
            (config.name, model_name),
            (
                f"model {model_index}/{len(models)} validation completed "
                f"| selected={chosen_records[model_name]['candidate']} "
                f"| elapsed={format_duration(reporter.now() - model_started_at)} "
                f"| eta={format_duration(estimate_remaining_seconds(loop_started_at=model_loop_started_at, completed_steps=model_index, total_steps=len(models), now=reporter.now()))}"
            ),
        )

    train_samples = prepared.select_samples(final_train_range)
    test_samples = prepared.select_samples(final_test_range)
    if train_samples.empty or test_samples.empty:
        raise RuntimeError(f"{config.name} has empty final train/test split.")

    final_predictions: list[pd.DataFrame] = []
    final_items = tuple(chosen_records.items())
    final_loop_started_at = reporter.now()
    for final_index, (model_name, chosen) in enumerate(final_items, start=1):
        final_started_at = reporter.now()
        reporter.log(
            (config.name, model_name),
            (
                f"final stage {final_index}/{len(final_items)} started "
                f"| train_samples={len(train_samples)} | test_samples={len(test_samples)}"
            ),
        )
        frame, metrics = _fit_and_score_final_model(
            prepared,
            config,
            model_name,
            chosen,
            train_samples,
            test_samples,
            reporter=reporter,
        )
        prediction_frames.append(frame)
        final_predictions.append(frame)
        metrics_rows.append(metrics)
        reporter.log_step(
            (config.name, model_name),
            label="final stage",
            index=final_index,
            total=len(final_items),
            loop_started_at=final_loop_started_at,
            step_started_at=final_started_at,
            extra=f"pr_auc={format_metric(metrics['pr_auc'])}",
        )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions.to_csv(output_path / "predictions.csv", index=False)

    metrics_summary = pd.DataFrame(metrics_rows)
    metrics_summary.to_csv(output_path / "metrics_summary.csv", index=False)

    if final_predictions:
        final_prediction_frame = pd.concat(final_predictions, ignore_index=True)
        country_metrics = summarize_prediction_frame(
            final_prediction_frame,
            group_cols=["experiment", "model", "split", "country"],
            min_positive=20,
        )
        country_metrics.to_csv(output_path / "country_metrics.csv", index=False)

        monthly_metrics = summarize_prediction_frame(
            add_month_column(final_prediction_frame),
            group_cols=["experiment", "model", "split", "month"],
        )
        monthly_metrics.to_csv(output_path / "monthly_metrics.csv", index=False)

    reporter.log(
        (config.name,),
        (
            f"experiment completed | elapsed={format_duration(reporter.now() - experiment_started_at)} "
            f"| artifacts={output_path}"
        ),
    )

    return {
        "sample_manifest": output_path / "sample_manifest.csv",
        "metrics_summary": output_path / "metrics_summary.csv",
        "predictions": output_path / "predictions.csv",
        "progress_log": progress_log,
    }


def _resolve_meta_member_configs(config: ExperimentConfig) -> tuple[ExperimentConfig, ...]:
    configs = build_default_experiment_configs(config.data_path)
    missing = [name for name in config.meta_members if name not in configs]
    if missing:
        raise ValueError(f"Meta experiment {config.name} references unknown members: {missing}")
    return tuple(configs[name] for name in config.meta_members)


def _meta_metric_score(metrics: pd.DataFrame, *, split: str, metric: str) -> float:
    if metrics.empty or metric not in metrics.columns:
        return float("nan")
    subset = metrics.loc[metrics["split"].astype(str) == split].copy()
    if "seed_aggregation" in subset.columns:
        subset = subset.loc[subset["seed_aggregation"].fillna("raw") == "raw"].copy()
    values = pd.to_numeric(subset[metric], errors="coerce")
    if values.empty:
        return float("nan")
    return float(values.mean())


def _normalize_member_weights(raw_scores: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    valid = {
        name: score
        for name, score in raw_scores.items()
        if np.isfinite(score) and score > 0.0
    }
    if not valid:
        if not raw_scores:
            return normalized
        equal_weight = 1.0 / float(len(raw_scores))
        return {name: equal_weight for name in raw_scores}
    total = float(sum(valid.values()))
    for name in raw_scores:
        normalized[name] = float(valid.get(name, 0.0) / total)
    return normalized


def _member_weight_label(weights: dict[str, float]) -> str:
    return ";".join(f"{name}:{weight:.4f}" for name, weight in weights.items())


def _selected_artifact_candidate(artifacts: dict[str, Path]) -> str | None:
    metrics = pd.read_csv(artifacts["metrics_summary"])
    if metrics.empty or "candidate" not in metrics.columns:
        return None
    test_rows = metrics.loc[metrics["split"].astype(str) == "test"].copy()
    if "seed_aggregation" in test_rows.columns:
        test_rows = test_rows.loc[test_rows["seed_aggregation"].fillna("raw") == "raw"].copy()
    if not test_rows.empty:
        return str(test_rows.iloc[0]["candidate"])
    val_rows = metrics.loc[metrics["split"].astype(str) == "val"].copy()
    if "seed_aggregation" in val_rows.columns:
        val_rows = val_rows.loc[val_rows["seed_aggregation"].fillna("raw") == "raw"].copy()
    if val_rows.empty:
        return None
    by_candidate = (
        val_rows.groupby("candidate", dropna=False)["pr_auc"]
        .mean()
        .sort_values(ascending=False, kind="mergesort")
    )
    if by_candidate.empty:
        return None
    return str(by_candidate.index[0])


def _read_prediction_subset(artifacts: dict[str, Path], *, split: str) -> pd.DataFrame:
    predictions = pd.read_csv(artifacts["predictions"])
    subset = predictions.loc[predictions["split"].astype(str) == split].copy()
    selected_candidate = _selected_artifact_candidate(artifacts)
    if selected_candidate is not None and "candidate" in subset.columns:
        candidate_subset = subset.loc[subset["candidate"].astype(str) == selected_candidate].copy()
        if not candidate_subset.empty:
            subset = candidate_subset
    return subset


def _prediction_merge_key_columns(subset: pd.DataFrame) -> list[str]:
    key_cols = [
        column
        for column in ("country", "anchor_time", "target_time", "y_true", "split", "fold", "protocol")
        if column in subset.columns
    ]
    if key_cols:
        return key_cols
    return [
        column
        for column in subset.columns
        if column not in {"y_prob", "seed", "threshold", "candidate", "experiment", "model"}
    ]


def _collapse_seed_averaged_predictions(subset: pd.DataFrame) -> pd.DataFrame:
    if subset.empty:
        return subset
    if "seed" not in subset.columns or subset["seed"].dropna().empty:
        return subset.drop(columns=["seed"], errors="ignore").copy()
    group_cols = _prediction_merge_key_columns(subset)
    aggregated = (
        subset.groupby(group_cols, dropna=False, sort=False)["y_prob"]
        .mean()
        .reset_index()
    )
    return aggregated


def _seed_member_label(experiment_name: str, seed: object) -> str:
    if pd.isna(seed):
        return experiment_name
    if isinstance(seed, (int, np.integer)):
        return f"{experiment_name}_SEED_{int(seed)}"
    if isinstance(seed, (float, np.floating)) and float(seed).is_integer():
        return f"{experiment_name}_SEED_{int(seed)}"
    return f"{experiment_name}_SEED_{seed}"


def _seed_member_scores(
    artifacts: dict[str, Path],
    *,
    split: str,
    metric: str,
    experiment_name: str,
) -> dict[str, float]:
    metrics = pd.read_csv(artifacts["metrics_summary"])
    subset = metrics.loc[metrics["split"].astype(str) == split].copy()
    if "seed_aggregation" in subset.columns:
        subset = subset.loc[subset["seed_aggregation"].fillna("raw") == "raw"].copy()
    if subset.empty or metric not in subset.columns:
        return {experiment_name: float("nan")}
    if "seed" not in subset.columns or subset["seed"].dropna().empty:
        values = pd.to_numeric(subset[metric], errors="coerce")
        return {experiment_name: float(values.mean()) if not values.empty else float("nan")}
    scores: dict[str, float] = {}
    grouped = subset.groupby("seed", dropna=False)[metric].mean()
    for seed, value in grouped.items():
        scores[_seed_member_label(experiment_name, seed)] = float(value)
    return scores


def _seed_member_prediction_frames(
    artifacts: dict[str, Path],
    *,
    split: str,
    experiment_name: str,
) -> dict[str, pd.DataFrame]:
    subset = _read_prediction_subset(artifacts, split=split)
    if subset.empty:
        return {}
    if "seed" not in subset.columns or subset["seed"].dropna().empty:
        return {experiment_name: subset.drop(columns=["seed"], errors="ignore").copy()}
    frames: dict[str, pd.DataFrame] = {}
    for seed, frame in subset.groupby("seed", dropna=False):
        frames[_seed_member_label(experiment_name, seed)] = frame.drop(columns=["seed"], errors="ignore").copy()
    return frames


def _merge_member_prediction_frames(member_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not member_frames:
        return pd.DataFrame()
    merged: pd.DataFrame | None = None
    key_cols: list[str] = []
    for member_name, frame in member_frames.items():
        member = frame.copy()
        key_cols = _prediction_merge_key_columns(member)
        renamed = member[key_cols + ["y_prob"]].rename(columns={"y_prob": f"y_prob__{member_name}"})
        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on=key_cols, how="inner", validate="one_to_one")
    return merged if merged is not None else pd.DataFrame()


def _weighted_member_probability(merged: pd.DataFrame, member_weights: dict[str, float]) -> np.ndarray:
    probs = np.zeros(len(merged), dtype=np.float32)
    for member_name, weight in member_weights.items():
        column = f"y_prob__{member_name}"
        if column not in merged.columns:
            continue
        probs += merged[column].to_numpy(dtype=np.float32) * np.float32(weight)
    return probs


def _build_meta_prediction_frame(
    merged: pd.DataFrame,
    y_prob: np.ndarray,
    *,
    experiment: str,
    model: str,
    split: str,
    threshold: float,
    candidate: str,
    extra_columns: dict[str, object] | None = None,
) -> pd.DataFrame:
    frame = merged[["country", "anchor_time", "target_time", "y_true"]].copy()
    frame["y_prob"] = y_prob.astype(np.float32)
    frame["split"] = split
    frame["experiment"] = experiment
    frame["model"] = model
    frame["candidate"] = candidate
    frame["threshold"] = threshold
    if "fold" in merged.columns:
        frame["fold"] = merged["fold"]
    if "protocol" in merged.columns:
        frame["protocol"] = merged["protocol"]
    for column, value in (extra_columns or {}).items():
        frame[column] = value
    return frame


def _write_meta_summary_outputs(
    output_path: Path,
    *,
    sample_manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics_rows: list[dict[str, object]],
) -> dict[str, Path]:
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log = output_path / "progress.log"
    if not progress_log.exists():
        progress_log.write_text("", encoding="utf-8")
    sample_manifest.to_csv(output_path / "sample_manifest.csv", index=False)
    predictions.to_csv(output_path / "predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output_path / "metrics_summary.csv", index=False)
    test_predictions = predictions.loc[predictions["split"].astype(str) == "test"].copy()
    if not test_predictions.empty:
        summarize_prediction_frame(
            test_predictions,
            group_cols=["experiment", "model", "split", "country"],
            min_positive=20,
        ).to_csv(output_path / "country_metrics.csv", index=False)
        summarize_prediction_frame(
            add_month_column(test_predictions),
            group_cols=["experiment", "model", "split", "month"],
        ).to_csv(output_path / "monthly_metrics.csv", index=False)
    return {
        "sample_manifest": output_path / "sample_manifest.csv",
        "metrics_summary": output_path / "metrics_summary.csv",
        "predictions": output_path / "predictions.csv",
        "progress_log": progress_log,
    }


def _build_late_fusion_artifacts(
    config: ExperimentConfig,
    *,
    output_path: Path,
    member_artifacts: dict[str, dict[str, Path]],
) -> dict[str, Path]:
    if not member_artifacts:
        raise RuntimeError(f"{config.name} did not receive any member artifacts.")
    output_path.mkdir(parents=True, exist_ok=True)
    first_artifacts = next(iter(member_artifacts.values()))
    sample_manifest = pd.read_csv(first_artifacts["sample_manifest"])

    raw_scores = {
        member_name: _meta_metric_score(pd.read_csv(artifacts["metrics_summary"]), split="val", metric="pr_auc")
        for member_name, artifacts in member_artifacts.items()
    }
    member_weights = _normalize_member_weights(raw_scores)
    member_weight_label = _member_weight_label(member_weights)
    pd.DataFrame(
        [
            {
                "experiment": config.name,
                "member_experiment": member_name,
                "val_pr_auc": raw_scores.get(member_name),
                "weight": member_weights.get(member_name),
            }
            for member_name in member_artifacts
        ]
    ).to_csv(output_path / "member_weights.csv", index=False)

    val_merged = _merge_member_prediction_frames(
        {
            member_name: _collapse_seed_averaged_predictions(_read_prediction_subset(artifacts, split="val"))
            for member_name, artifacts in member_artifacts.items()
        }
    )
    test_merged = _merge_member_prediction_frames(
        {
            member_name: _collapse_seed_averaged_predictions(_read_prediction_subset(artifacts, split="test"))
            for member_name, artifacts in member_artifacts.items()
        }
    )
    if val_merged.empty or test_merged.empty:
        raise RuntimeError(f"{config.name} could not build late-fusion predictions from empty member outputs.")

    val_prob = _weighted_member_probability(val_merged, member_weights)
    threshold = find_best_threshold_f1(val_merged["y_true"].to_numpy(dtype=int), val_prob)
    test_prob = _weighted_member_probability(test_merged, member_weights)

    extra = {
        "fusion_strategy": "val_pr_auc_weighted",
        "member_weights": member_weight_label,
    }
    val_frame = _build_meta_prediction_frame(
        val_merged,
        val_prob,
        experiment=config.name,
        model="LateFusion",
        split="val",
        threshold=threshold,
        candidate="+".join(config.meta_members),
        extra_columns=extra,
    )
    test_frame = _build_meta_prediction_frame(
        test_merged,
        test_prob,
        experiment=config.name,
        model="LateFusion",
        split="test",
        threshold=threshold,
        candidate="+".join(config.meta_members),
        extra_columns=extra,
    )

    val_metrics = _metrics_row(
        val_frame,
        experiment=config.name,
        model="LateFusion",
        split="val",
        candidate="+".join(config.meta_members),
    )
    test_metrics = _metrics_row(
        test_frame,
        experiment=config.name,
        model="LateFusion",
        split="test",
        candidate="+".join(config.meta_members),
    )
    for row in (val_metrics, test_metrics):
        row["fusion_strategy"] = "val_pr_auc_weighted"
        row["member_weights"] = member_weight_label

    artifacts = _write_meta_summary_outputs(
        output_path,
        sample_manifest=sample_manifest,
        predictions=pd.concat([val_frame, test_frame], ignore_index=True),
        metrics_rows=[val_metrics, test_metrics],
    )
    artifacts["member_weights"] = output_path / "member_weights.csv"
    return artifacts


def _build_stacking_artifacts(
    config: ExperimentConfig,
    *,
    output_path: Path,
    member_artifacts: dict[str, dict[str, Path]],
) -> dict[str, Path]:
    if not member_artifacts:
        raise RuntimeError(f"{config.name} did not receive any member artifacts.")
    output_path.mkdir(parents=True, exist_ok=True)
    first_artifacts = next(iter(member_artifacts.values()))
    sample_manifest = pd.read_csv(first_artifacts["sample_manifest"])

    val_merged = _merge_member_prediction_frames(
        {
            member_name: _collapse_seed_averaged_predictions(_read_prediction_subset(artifacts, split="val"))
            for member_name, artifacts in member_artifacts.items()
        }
    )
    test_merged = _merge_member_prediction_frames(
        {
            member_name: _collapse_seed_averaged_predictions(_read_prediction_subset(artifacts, split="test"))
            for member_name, artifacts in member_artifacts.items()
        }
    )
    if val_merged.empty or test_merged.empty:
        raise RuntimeError(f"{config.name} could not build stacking predictions from empty member outputs.")

    feature_columns = [f"y_prob__{member_name}" for member_name in member_artifacts]
    val_X = val_merged[feature_columns].to_numpy(dtype=np.float32)
    test_X = test_merged[feature_columns].to_numpy(dtype=np.float32)
    val_y = val_merged["y_true"].to_numpy(dtype=int)
    stacker_strategy = "logistic_regression"
    stacker_c = 1.0
    if np.unique(val_y).size < 2:
        stacker_strategy = "mean_fallback"
        val_prob = val_X.mean(axis=1).astype(np.float32)
        test_prob = test_X.mean(axis=1).astype(np.float32)
        coefficient_rows = [
            {
                "experiment": config.name,
                "feature": member_name,
                "coefficient": 1.0 / float(len(member_artifacts)),
                "intercept": 0.0,
                "stacker_strategy": stacker_strategy,
                "stacker_C": pd.NA,
            }
            for member_name in member_artifacts
        ]
    else:
        stacker = fit_logistic_regression(val_X, val_y, C=stacker_c, seed=config.random_seed)
        val_prob = predict_logistic_regression(stacker, val_X)
        test_prob = predict_logistic_regression(stacker, test_X)
        coefficient_rows = [
            {
                "experiment": config.name,
                "feature": member_name,
                "coefficient": float(stacker.coef_[0][feature_index]),
                "intercept": float(stacker.intercept_[0]),
                "stacker_strategy": stacker_strategy,
                "stacker_C": stacker_c,
            }
            for feature_index, member_name in enumerate(member_artifacts)
        ]
    pd.DataFrame(coefficient_rows).to_csv(output_path / "stacking_coefficients.csv", index=False)

    threshold = find_best_threshold_f1(val_y, val_prob)
    extra = {
        "stacker_strategy": stacker_strategy,
        "stacker_C": stacker_c,
    }
    candidate = "+".join(config.meta_members)
    val_frame = _build_meta_prediction_frame(
        val_merged,
        val_prob,
        experiment=config.name,
        model="StackingLogisticRegression",
        split="val",
        threshold=threshold,
        candidate=candidate,
        extra_columns=extra,
    )
    test_frame = _build_meta_prediction_frame(
        test_merged,
        test_prob,
        experiment=config.name,
        model="StackingLogisticRegression",
        split="test",
        threshold=threshold,
        candidate=candidate,
        extra_columns=extra,
    )

    val_metrics = _metrics_row(
        val_frame,
        experiment=config.name,
        model="StackingLogisticRegression",
        split="val",
        candidate=candidate,
    )
    test_metrics = _metrics_row(
        test_frame,
        experiment=config.name,
        model="StackingLogisticRegression",
        split="test",
        candidate=candidate,
    )
    for row in (val_metrics, test_metrics):
        row["stacker_strategy"] = stacker_strategy
        row["stacker_C"] = stacker_c

    artifacts = _write_meta_summary_outputs(
        output_path,
        sample_manifest=sample_manifest,
        predictions=pd.concat([val_frame, test_frame], ignore_index=True),
        metrics_rows=[val_metrics, test_metrics],
    )
    artifacts["stacking_coefficients"] = output_path / "stacking_coefficients.csv"
    return artifacts


def _build_cross_seed_ensemble_artifacts(
    config: ExperimentConfig,
    *,
    output_path: Path,
    member_artifacts: dict[str, dict[str, Path]],
) -> dict[str, Path]:
    if not member_artifacts:
        raise RuntimeError(f"{config.name} did not receive any member artifacts.")
    output_path.mkdir(parents=True, exist_ok=True)
    first_artifacts = next(iter(member_artifacts.values()))
    sample_manifest = pd.read_csv(first_artifacts["sample_manifest"])

    raw_scores: dict[str, float] = {}
    val_frames: dict[str, pd.DataFrame] = {}
    test_frames: dict[str, pd.DataFrame] = {}
    for experiment_name, artifacts in member_artifacts.items():
        raw_scores.update(_seed_member_scores(artifacts, split="val", metric="pr_auc", experiment_name=experiment_name))
        val_frames.update(_seed_member_prediction_frames(artifacts, split="val", experiment_name=experiment_name))
        test_frames.update(_seed_member_prediction_frames(artifacts, split="test", experiment_name=experiment_name))
    member_weights = _normalize_member_weights(raw_scores)
    member_weight_label = _member_weight_label(member_weights)
    pd.DataFrame(
        [
            {
                "experiment": config.name,
                "member_experiment": member_name,
                "val_pr_auc": raw_scores.get(member_name),
                "weight": member_weights.get(member_name),
            }
            for member_name in val_frames
        ]
    ).to_csv(output_path / "member_weights.csv", index=False)

    val_merged = _merge_member_prediction_frames(val_frames)
    test_merged = _merge_member_prediction_frames(test_frames)
    if val_merged.empty or test_merged.empty:
        raise RuntimeError(f"{config.name} could not build cross-seed predictions from empty member outputs.")

    val_prob = _weighted_member_probability(val_merged, member_weights)
    threshold = find_best_threshold_f1(val_merged["y_true"].to_numpy(dtype=int), val_prob)
    test_prob = _weighted_member_probability(test_merged, member_weights)
    extra = {
        "fusion_strategy": "cross_seed_val_pr_auc_weighted",
        "member_weights": member_weight_label,
    }
    candidate = "+".join(config.meta_members)
    val_frame = _build_meta_prediction_frame(
        val_merged,
        val_prob,
        experiment=config.name,
        model="CrossSeedEnsemble",
        split="val",
        threshold=threshold,
        candidate=candidate,
        extra_columns=extra,
    )
    test_frame = _build_meta_prediction_frame(
        test_merged,
        test_prob,
        experiment=config.name,
        model="CrossSeedEnsemble",
        split="test",
        threshold=threshold,
        candidate=candidate,
        extra_columns=extra,
    )
    val_metrics = _metrics_row(
        val_frame,
        experiment=config.name,
        model="CrossSeedEnsemble",
        split="val",
        candidate=candidate,
    )
    test_metrics = _metrics_row(
        test_frame,
        experiment=config.name,
        model="CrossSeedEnsemble",
        split="test",
        candidate=candidate,
    )
    for row in (val_metrics, test_metrics):
        row["fusion_strategy"] = "cross_seed_val_pr_auc_weighted"
        row["member_weights"] = member_weight_label

    artifacts = _write_meta_summary_outputs(
        output_path,
        sample_manifest=sample_manifest,
        predictions=pd.concat([val_frame, test_frame], ignore_index=True),
        metrics_rows=[val_metrics, test_metrics],
    )
    artifacts["member_weights"] = output_path / "member_weights.csv"
    return artifacts


def _run_late_fusion_experiment(
    config: ExperimentConfig,
    *,
    output_path: Path,
    folds,
    final_train_range,
    final_test_range,
    skip_unavailable_models: bool,
    reporter: ProgressReporter | None,
) -> dict[str, Path]:
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log = output_path / "progress.log"
    reporter = (reporter or ProgressReporter()).with_log_file(progress_log)
    experiment_started_at = reporter.now()
    reporter.log(
        (config.name,),
        f"late-fusion experiment started | members={config.meta_members} | data={config.data_path} | output={output_path}",
    )

    member_artifacts: dict[str, dict[str, Path]] = {}
    member_configs = _resolve_meta_member_configs(config)
    member_loop_started_at = reporter.now()
    for member_index, member_config in enumerate(member_configs, start=1):
        member_started_at = reporter.now()
        reporter.log(
            (config.name, member_config.name),
            f"member {member_index}/{len(member_configs)} started",
        )
        member_artifacts[member_config.name] = run_experiment(
            member_config,
            output_dir=output_path / "member_runs",
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
        reporter.log_step(
            (config.name, member_config.name),
            label="member",
            index=member_index,
            total=len(member_configs),
            loop_started_at=member_loop_started_at,
            step_started_at=member_started_at,
            extra="completed",
        )

    artifacts = _build_late_fusion_artifacts(config, output_path=output_path, member_artifacts=member_artifacts)
    reporter.log(
        (config.name,),
        (
            f"late-fusion experiment completed | elapsed={format_duration(reporter.now() - experiment_started_at)} "
            f"| artifacts={output_path}"
        ),
    )
    return artifacts


def _run_stacking_experiment(
    config: ExperimentConfig,
    *,
    output_path: Path,
    folds,
    final_train_range,
    final_test_range,
    skip_unavailable_models: bool,
    reporter: ProgressReporter | None,
) -> dict[str, Path]:
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log = output_path / "progress.log"
    reporter = (reporter or ProgressReporter()).with_log_file(progress_log)
    experiment_started_at = reporter.now()
    reporter.log(
        (config.name,),
        f"stacking experiment started | members={config.meta_members} | data={config.data_path} | output={output_path}",
    )

    member_artifacts: dict[str, dict[str, Path]] = {}
    member_configs = _resolve_meta_member_configs(config)
    member_loop_started_at = reporter.now()
    for member_index, member_config in enumerate(member_configs, start=1):
        member_started_at = reporter.now()
        reporter.log(
            (config.name, member_config.name),
            f"member {member_index}/{len(member_configs)} started",
        )
        member_artifacts[member_config.name] = run_experiment(
            member_config,
            output_dir=output_path / "member_runs",
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
        reporter.log_step(
            (config.name, member_config.name),
            label="member",
            index=member_index,
            total=len(member_configs),
            loop_started_at=member_loop_started_at,
            step_started_at=member_started_at,
            extra="completed",
        )

    artifacts = _build_stacking_artifacts(config, output_path=output_path, member_artifacts=member_artifacts)
    reporter.log(
        (config.name,),
        (
            f"stacking experiment completed | elapsed={format_duration(reporter.now() - experiment_started_at)} "
            f"| artifacts={output_path}"
        ),
    )
    return artifacts


def _run_cross_seed_ensemble_experiment(
    config: ExperimentConfig,
    *,
    output_path: Path,
    folds,
    final_train_range,
    final_test_range,
    skip_unavailable_models: bool,
    reporter: ProgressReporter | None,
) -> dict[str, Path]:
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log = output_path / "progress.log"
    reporter = (reporter or ProgressReporter()).with_log_file(progress_log)
    experiment_started_at = reporter.now()
    reporter.log(
        (config.name,),
        (
            f"cross-seed ensemble experiment started | members={config.meta_members} "
            f"| data={config.data_path} | output={output_path}"
        ),
    )

    member_artifacts: dict[str, dict[str, Path]] = {}
    member_configs = _resolve_meta_member_configs(config)
    member_loop_started_at = reporter.now()
    for member_index, member_config in enumerate(member_configs, start=1):
        member_started_at = reporter.now()
        reporter.log(
            (config.name, member_config.name),
            f"member {member_index}/{len(member_configs)} started",
        )
        member_artifacts[member_config.name] = run_experiment(
            member_config,
            output_dir=output_path / "member_runs",
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
        reporter.log_step(
            (config.name, member_config.name),
            label="member",
            index=member_index,
            total=len(member_configs),
            loop_started_at=member_loop_started_at,
            step_started_at=member_started_at,
            extra="completed",
        )

    artifacts = _build_cross_seed_ensemble_artifacts(config, output_path=output_path, member_artifacts=member_artifacts)
    reporter.log(
        (config.name,),
        (
            f"cross-seed ensemble experiment completed | elapsed={format_duration(reporter.now() - experiment_started_at)} "
            f"| artifacts={output_path}"
        ),
    )
    return artifacts


def _run_calibration_experiment(
    config: ExperimentConfig,
    *,
    output_path: Path,
    folds,
    final_train_range,
    final_test_range,
    skip_unavailable_models: bool,
    reporter: ProgressReporter | None,
) -> dict[str, Path]:
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log = output_path / "progress.log"
    reporter = (reporter or ProgressReporter()).with_log_file(progress_log)
    experiment_started_at = reporter.now()
    reporter.log(
        (config.name,),
        (
            f"calibration experiment started | candidates={config.meta_members} "
            f"| method={config.meta_calibration_method} | data={config.data_path} | output={output_path}"
        ),
    )

    candidate_artifacts: dict[str, dict[str, Path]] = {}
    candidate_configs = _resolve_meta_member_configs(config)
    candidate_loop_started_at = reporter.now()
    for candidate_index, candidate_config in enumerate(candidate_configs, start=1):
        candidate_started_at = reporter.now()
        reporter.log(
            (config.name, candidate_config.name),
            f"candidate {candidate_index}/{len(candidate_configs)} started",
        )
        if candidate_config.meta_kind == "late_fusion":
            nested_member_artifacts: dict[str, dict[str, Path]] = {}
            for nested_member in _resolve_meta_member_configs(candidate_config):
                if nested_member.name not in candidate_artifacts:
                    candidate_artifacts[nested_member.name] = run_experiment(
                        nested_member,
                        output_dir=output_path / "candidate_runs",
                        folds=folds,
                        final_train_range=final_train_range,
                        final_test_range=final_test_range,
                        skip_unavailable_models=skip_unavailable_models,
                        reporter=reporter,
                    )
                nested_member_artifacts[nested_member.name] = candidate_artifacts[nested_member.name]
            candidate_output_path = (output_path / "candidate_runs" / candidate_config.name).resolve()
            candidate_artifacts[candidate_config.name] = _build_late_fusion_artifacts(
                candidate_config,
                output_path=candidate_output_path,
                member_artifacts=nested_member_artifacts,
            )
        else:
            candidate_artifacts[candidate_config.name] = run_experiment(
                candidate_config,
                output_dir=output_path / "candidate_runs",
                folds=folds,
                final_train_range=final_train_range,
                final_test_range=final_test_range,
                skip_unavailable_models=skip_unavailable_models,
                reporter=reporter,
            )
        reporter.log_step(
            (config.name, candidate_config.name),
            label="candidate",
            index=candidate_index,
            total=len(candidate_configs),
            loop_started_at=candidate_loop_started_at,
            step_started_at=candidate_started_at,
            extra="completed",
        )

    candidate_scores = pd.DataFrame(
        [
            {
                "experiment": config.name,
                "candidate_experiment": candidate_name,
                "val_pr_auc": _meta_metric_score(pd.read_csv(artifacts["metrics_summary"]), split="val", metric="pr_auc"),
            }
            for candidate_name, artifacts in candidate_artifacts.items()
            if candidate_name in config.meta_members
        ]
    ).sort_values("val_pr_auc", ascending=False, kind="mergesort")
    candidate_scores.to_csv(output_path / "candidate_scores.csv", index=False)
    if candidate_scores.empty:
        raise RuntimeError(f"{config.name} did not produce any candidate scores.")

    selected_name = str(candidate_scores.iloc[0]["candidate_experiment"])
    selected_artifacts = candidate_artifacts[selected_name]
    selected_predictions = pd.read_csv(selected_artifacts["predictions"])
    val_predictions = selected_predictions.loc[selected_predictions["split"].astype(str) == "val"].copy()
    test_predictions = selected_predictions.loc[selected_predictions["split"].astype(str) == "test"].copy()
    if val_predictions.empty or test_predictions.empty:
        raise RuntimeError(f"{config.name} selected candidate {selected_name} without val/test predictions.")

    calibrator = fit_probability_calibrator(
        val_predictions["y_true"].to_numpy(dtype=int),
        val_predictions["y_prob"].to_numpy(dtype=np.float32),
        method=config.meta_calibration_method,
    )
    calibrated_val_prob = apply_probability_calibrator(calibrator, val_predictions["y_prob"].to_numpy(dtype=np.float32))
    calibrated_test_prob = apply_probability_calibrator(calibrator, test_predictions["y_prob"].to_numpy(dtype=np.float32))
    threshold = find_best_threshold_f1(val_predictions["y_true"].to_numpy(dtype=int), calibrated_val_prob)

    base_model_name = str(val_predictions["model"].iloc[0])
    extra = {
        "base_experiment": selected_name,
        "base_model": base_model_name,
        "calibration_method": calibrator.method,
    }
    val_frame = _build_meta_prediction_frame(
        val_predictions,
        calibrated_val_prob,
        experiment=config.name,
        model=f"Calibrated{base_model_name}",
        split="val",
        threshold=threshold,
        candidate=selected_name,
        extra_columns=extra,
    )
    test_frame = _build_meta_prediction_frame(
        test_predictions,
        calibrated_test_prob,
        experiment=config.name,
        model=f"Calibrated{base_model_name}",
        split="test",
        threshold=threshold,
        candidate=selected_name,
        extra_columns=extra,
    )
    val_metrics = _metrics_row(
        val_frame,
        experiment=config.name,
        model=f"Calibrated{base_model_name}",
        split="val",
        candidate=selected_name,
    )
    test_metrics = _metrics_row(
        test_frame,
        experiment=config.name,
        model=f"Calibrated{base_model_name}",
        split="test",
        candidate=selected_name,
    )
    for row in (val_metrics, test_metrics):
        row["base_experiment"] = selected_name
        row["base_model"] = base_model_name
        row["calibration_method"] = calibrator.method

    sample_manifest = pd.read_csv(selected_artifacts["sample_manifest"])
    artifacts = _write_meta_summary_outputs(
        output_path,
        sample_manifest=sample_manifest,
        predictions=pd.concat([val_frame, test_frame], ignore_index=True),
        metrics_rows=[val_metrics, test_metrics],
    )
    artifacts["candidate_scores"] = output_path / "candidate_scores.csv"
    reporter.log(
        (config.name,),
        (
            f"calibration experiment completed | selected={selected_name} | method={calibrator.method} "
            f"| elapsed={format_duration(reporter.now() - experiment_started_at)} | artifacts={output_path}"
        ),
    )
    return artifacts


def _run_repeated_seed_experiment(
    config: ExperimentConfig,
    *,
    output_path: Path,
    folds,
    final_train_range,
    final_test_range,
    skip_unavailable_models: bool,
    reporter: ProgressReporter | None,
) -> dict[str, Path]:
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log = output_path / "progress.log"
    reporter = (reporter or ProgressReporter()).with_log_file(progress_log)
    seeds = tuple(config.repeat_random_seeds)
    experiment_started_at = reporter.now()
    reporter.log(
        (config.name,),
        (
            f"repeated-seed experiment started | seeds={seeds} "
            f"| data={config.data_path} | output={output_path}"
        ),
    )

    seed_artifacts: list[tuple[int, dict[str, Path]]] = []
    seed_loop_started_at = reporter.now()
    for seed_index, seed in enumerate(seeds, start=1):
        seed_started_at = reporter.now()
        seed_config = replace(
            config,
            name=f"{config.name}_SEED_{seed}",
            random_seed=seed,
            repeat_random_seeds=(),
        )
        reporter.log(
            (config.name, f"seed={seed}"),
            f"seed {seed_index}/{len(seeds)} started",
        )
        artifacts = run_experiment(
            seed_config,
            output_dir=output_path / "seed_runs",
            folds=folds,
            final_train_range=final_train_range,
            final_test_range=final_test_range,
            skip_unavailable_models=skip_unavailable_models,
            reporter=reporter,
        )
        seed_artifacts.append((seed, artifacts))
        reporter.log_step(
            (config.name, f"seed={seed}"),
            label="seed",
            index=seed_index,
            total=len(seeds),
            loop_started_at=seed_loop_started_at,
            step_started_at=seed_started_at,
            extra="completed",
        )

    sample_manifest = pd.read_csv(seed_artifacts[0][1]["sample_manifest"])
    sample_manifest.to_csv(output_path / "sample_manifest.csv", index=False)

    metrics_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed, artifacts in seed_artifacts:
        metrics_frame = pd.read_csv(artifacts["metrics_summary"])
        metrics_frame["experiment"] = config.name
        metrics_frame["seed"] = seed
        metrics_frame["seed_aggregation"] = "raw"
        metrics_frames.append(metrics_frame)

        prediction_frame = pd.read_csv(artifacts["predictions"])
        prediction_frame["experiment"] = config.name
        prediction_frame["seed"] = seed
        prediction_frames.append(prediction_frame)

    metrics_summary = pd.concat(metrics_frames, ignore_index=True)
    aggregate_rows = _aggregate_seed_metrics(metrics_summary)
    if not aggregate_rows.empty:
        metrics_summary = pd.concat([metrics_summary, aggregate_rows], ignore_index=True, sort=False)
    metrics_summary.to_csv(output_path / "metrics_summary.csv", index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_path / "predictions.csv", index=False)

    if not predictions.empty:
        summarize_prediction_frame(
            predictions,
            group_cols=["experiment", "model", "split", "seed", "country"],
            min_positive=20,
        ).to_csv(output_path / "country_metrics.csv", index=False)
        summarize_prediction_frame(
            add_month_column(predictions),
            group_cols=["experiment", "model", "split", "seed", "month"],
        ).to_csv(output_path / "monthly_metrics.csv", index=False)

    reporter.log(
        (config.name,),
        (
            f"repeated-seed experiment completed | elapsed={format_duration(reporter.now() - experiment_started_at)} "
            f"| artifacts={output_path}"
        ),
    )
    return {
        "sample_manifest": output_path / "sample_manifest.csv",
        "metrics_summary": output_path / "metrics_summary.csv",
        "predictions": output_path / "predictions.csv",
        "progress_log": progress_log,
    }


def run_transfer_experiment(
    transfer_config: TransferConfig,
    *,
    output_dir: str | Path,
    reporter: ProgressReporter | None = None,
) -> dict[str, Path]:
    require_torch()
    output_path = Path(output_dir).resolve() / transfer_config.name
    output_path.mkdir(parents=True, exist_ok=True)
    progress_log = output_path / "progress.log"
    reporter = (reporter or ProgressReporter()).with_log_file(progress_log)
    experiment_started_at = reporter.now()
    reporter.log(
        (transfer_config.name,),
        f"experiment started | data={transfer_config.data_path} | output={output_path}",
    )

    config = ExperimentConfig(
        name="E6_public_gru",
        data_path=transfer_config.data_path,
        countries=transfer_config.all_countries,
        feature_group="public",
        window_hours=transfer_config.window_hours,
        horizon_hours=transfer_config.horizon_hours,
        models=("GRU",),
        split_strategy="transfer_v1",
        ffill_limit=transfer_config.ffill_limit,
        primary_metric=transfer_config.primary_metric,
        random_seed=transfer_config.random_seed,
        use_country_features=False,
    )
    prep_started_at = reporter.now()
    reporter.log((transfer_config.name,), "data prep started")
    prepared = prepare_experiment_data(config)
    reporter.log(
        (transfer_config.name,),
        (
            f"data prep completed | samples={len(prepared.sample_manifest)} "
            f"| countries={len(prepared.country_panels)} "
            f"| elapsed={format_duration(reporter.now() - prep_started_at)}"
        ),
    )
    prepared.sample_manifest.to_csv(output_path / "sample_manifest.csv", index=False)

    source_train = prepared.select_samples(transfer_config.pretrain_train_range, countries=transfer_config.source_countries)
    source_val = prepared.select_samples(transfer_config.pretrain_val_range, countries=transfer_config.source_countries)
    if source_train.empty or source_val.empty:
        raise RuntimeError("Source pretraining split is empty.")

    source_scaler = prepared.fit_sequence_scaler(source_train)
    source_train_ds = prepared.build_sequence_dataset(source_train, source_scaler, include_country=False)
    source_val_ds = prepared.build_sequence_dataset(source_val, source_scaler, include_country=False)
    source_train_single_class = _single_class_probability(source_train_ds.metadata["y_true"].to_numpy(dtype=int))
    pretrained_model = None
    pretrain = None
    source_started_at = reporter.now()
    reporter.log(
        (transfer_config.name, "SourcePretrain"),
        f"source pretraining started | train_samples={len(source_train_ds)} | val_samples={len(source_val_ds)}",
    )
    if source_train_single_class is None:
        pretrain = train_sequence_model(
            "GRU",
            train_dataset=source_train_ds,
            val_dataset=source_val_ds,
            use_country_embedding=False,
            num_countries=0,
            random_seed=transfer_config.random_seed,
            learning_rate=1e-3,
            max_epochs=30,
            patience=5,
            reporter=reporter,
            progress_prefix=(transfer_config.name, "SourcePretrain", "GRU"),
        )
        pretrained_model = load_sequence_model(
            "GRU",
            input_dim=source_train_ds[0]["x"].shape[1],
            use_country_embedding=False,
            num_countries=0,
            state_dict=pretrain.state_dict,
        )
        source_val_prob = predict_sequence_model(pretrained_model, source_val_ds)
    else:
        source_val_prob = predict_majority(source_train_single_class, len(source_val_ds))
        reporter.log(
            (transfer_config.name, "SourcePretrain"),
            "source pretraining used single-class fallback",
        )
    source_threshold = find_best_threshold_f1(source_val_ds.metadata["y_true"].to_numpy(dtype=int), source_val_prob)
    reporter.log(
        (transfer_config.name, "SourcePretrain"),
        (
            f"source pretraining completed | threshold={source_threshold:.4f} "
            f"| elapsed={format_duration(reporter.now() - source_started_at)}"
        ),
    )

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    country_loop_started_at = reporter.now()
    for country_index, target_country in enumerate(transfer_config.target_countries, start=1):
        test_samples = prepared.select_samples(transfer_config.target_test_range, countries=(target_country,))
        if test_samples.empty:
            reporter.log(
                (transfer_config.name, target_country),
                f"country {country_index}/{len(transfer_config.target_countries)} skipped | reason=no test samples",
            )
            continue

        country_started_at = reporter.now()
        reporter.log(
            (transfer_config.name, target_country),
            (
                f"country {country_index}/{len(transfer_config.target_countries)} started "
                f"| test_samples={len(test_samples)}"
            ),
        )
        zero_shot_test_ds = prepared.build_sequence_dataset(test_samples, source_scaler, include_country=False)
        zero_shot_started_at = reporter.now()
        reporter.log((transfer_config.name, target_country, "ZeroShot"), "evaluation started")
        if pretrained_model is None:
            zero_shot_prob = predict_majority(source_train_single_class, len(zero_shot_test_ds))
        else:
            zero_shot_prob = predict_sequence_model(pretrained_model, zero_shot_test_ds)
        zero_shot_frame = _prediction_frame(
            zero_shot_test_ds.metadata,
            zero_shot_prob,
            experiment=transfer_config.name,
            model="GRU",
            split="test",
            threshold=source_threshold,
            protocol="ZeroShot",
            candidate="source_pretrain",
        )
        prediction_frames.append(zero_shot_frame)
        metrics_rows.append(
            _metrics_row(
                zero_shot_frame,
                experiment=transfer_config.name,
                model="GRU",
                split="test",
                protocol="ZeroShot",
                target_country=target_country,
                candidate="source_pretrain",
            )
        )
        reporter.log(
            (transfer_config.name, target_country, "ZeroShot"),
            (
                f"evaluation completed | pr_auc={format_metric(metrics_rows[-1]['pr_auc'])} "
                f"| elapsed={format_duration(reporter.now() - zero_shot_started_at)}"
            ),
        )

        per_budget_metrics: dict[str, float] = {}
        budget_loop_started_at = reporter.now()
        for budget_index, budget in enumerate(transfer_config.adapt_budget, start=1):
            target_train = prepared.select_samples(budget.train_range, countries=(target_country,))
            target_val = prepared.select_samples(budget.val_range, countries=(target_country,))
            if target_train.empty or target_val.empty:
                reporter.log(
                    (transfer_config.name, target_country, budget.name),
                    f"budget {budget_index}/{len(transfer_config.adapt_budget)} skipped | reason=empty train or val split",
                )
                continue

            budget_started_at = reporter.now()
            reporter.log(
                (transfer_config.name, target_country, budget.name),
                (
                    f"budget {budget_index}/{len(transfer_config.adapt_budget)} started "
                    f"| train_samples={len(target_train)} | val_samples={len(target_val)}"
                ),
            )
            target_scaler = prepared.fit_sequence_scaler(target_train)
            target_train_ds = prepared.build_sequence_dataset(target_train, target_scaler, include_country=False)
            target_val_ds = prepared.build_sequence_dataset(target_val, target_scaler, include_country=False)
            target_test_ds = prepared.build_sequence_dataset(test_samples, target_scaler, include_country=False)
            target_train_single_class = _single_class_probability(target_train_ds.metadata["y_true"].to_numpy(dtype=int))
            target_only_started_at = reporter.now()
            reporter.log((transfer_config.name, target_country, budget.name, "TargetOnly"), "evaluation started")
            if target_train_single_class is None:
                target_only = train_sequence_model(
                    "GRU",
                    train_dataset=target_train_ds,
                    val_dataset=target_val_ds,
                    use_country_embedding=False,
                    num_countries=0,
                    random_seed=transfer_config.random_seed,
                    learning_rate=1e-3,
                    max_epochs=30,
                    patience=5,
                    reporter=reporter,
                    progress_prefix=(transfer_config.name, target_country, budget.name, "TargetOnly"),
                )
                target_only_model = load_sequence_model(
                    "GRU",
                    input_dim=target_train_ds[0]["x"].shape[1],
                    use_country_embedding=False,
                    num_countries=0,
                    state_dict=target_only.state_dict,
                )
                target_only_val_prob = predict_sequence_model(target_only_model, target_val_ds)
                target_only_prob = predict_sequence_model(target_only_model, target_test_ds)
            else:
                target_only_val_prob = predict_majority(target_train_single_class, len(target_val_ds))
                target_only_prob = predict_majority(target_train_single_class, len(target_test_ds))
                reporter.log(
                    (transfer_config.name, target_country, budget.name, "TargetOnly"),
                    "evaluation used single-class fallback",
                )
            target_only_threshold = find_best_threshold_f1(
                target_val_ds.metadata["y_true"].to_numpy(dtype=int),
                target_only_val_prob,
            )
            target_only_frame = _prediction_frame(
                target_test_ds.metadata,
                target_only_prob,
                experiment=transfer_config.name,
                model="GRU",
                split="test",
                threshold=target_only_threshold,
                protocol=f"TargetOnly-{budget.name}",
                candidate=budget.name,
            )
            prediction_frames.append(target_only_frame)
            target_only_metrics = _metrics_row(
                target_only_frame,
                experiment=transfer_config.name,
                model="GRU",
                split="test",
                protocol=f"TargetOnly-{budget.name}",
                target_country=target_country,
                candidate=budget.name,
            )
            metrics_rows.append(target_only_metrics)
            reporter.log(
                (transfer_config.name, target_country, budget.name, "TargetOnly"),
                (
                    f"evaluation completed | pr_auc={format_metric(target_only_metrics['pr_auc'])} "
                    f"| elapsed={format_duration(reporter.now() - target_only_started_at)}"
                ),
            )

            transfer_train_ds = prepared.build_sequence_dataset(target_train, source_scaler, include_country=False)
            transfer_val_ds = prepared.build_sequence_dataset(target_val, source_scaler, include_country=False)
            transfer_test_ds = prepared.build_sequence_dataset(test_samples, source_scaler, include_country=False)
            transfer_started_at = reporter.now()
            reporter.log((transfer_config.name, target_country, budget.name, "TransferFineTune"), "evaluation started")
            if target_train_single_class is None and pretrained_model is not None and pretrain is not None:
                transfer = train_sequence_model(
                    "GRU",
                    train_dataset=transfer_train_ds,
                    val_dataset=transfer_val_ds,
                    use_country_embedding=False,
                    num_countries=0,
                    random_seed=transfer_config.random_seed,
                    learning_rate=1e-4,
                    max_epochs=15,
                    patience=3,
                    init_state_dict=pretrain.state_dict,
                    reporter=reporter,
                    progress_prefix=(transfer_config.name, target_country, budget.name, "TransferFineTune"),
                )
                transfer_model = load_sequence_model(
                    "GRU",
                    input_dim=transfer_train_ds[0]["x"].shape[1],
                    use_country_embedding=False,
                    num_countries=0,
                    state_dict=transfer.state_dict,
                )
                transfer_val_prob = predict_sequence_model(transfer_model, transfer_val_ds)
                transfer_prob = predict_sequence_model(transfer_model, transfer_test_ds)
            else:
                fallback_prob = target_train_single_class if target_train_single_class is not None else source_train_single_class
                transfer_val_prob = predict_majority(fallback_prob, len(transfer_val_ds))
                transfer_prob = predict_majority(fallback_prob, len(transfer_test_ds))
                reporter.log(
                    (transfer_config.name, target_country, budget.name, "TransferFineTune"),
                    "evaluation used fallback baseline",
                )
            transfer_threshold = find_best_threshold_f1(
                transfer_val_ds.metadata["y_true"].to_numpy(dtype=int),
                transfer_val_prob,
            )
            transfer_frame = _prediction_frame(
                transfer_test_ds.metadata,
                transfer_prob,
                experiment=transfer_config.name,
                model="GRU",
                split="test",
                threshold=transfer_threshold,
                protocol=f"TransferFineTune-{budget.name}",
                candidate=budget.name,
            )
            prediction_frames.append(transfer_frame)
            transfer_metrics = _metrics_row(
                transfer_frame,
                experiment=transfer_config.name,
                model="GRU",
                split="test",
                protocol=f"TransferFineTune-{budget.name}",
                target_country=target_country,
                candidate=budget.name,
            )
            transfer_metrics["transfer_gain"] = transfer_metrics["pr_auc"] - target_only_metrics["pr_auc"]
            metrics_rows.append(transfer_metrics)
            per_budget_metrics[budget.name] = transfer_metrics["pr_auc"]
            reporter.log(
                (transfer_config.name, target_country, budget.name, "TransferFineTune"),
                (
                    f"evaluation completed | pr_auc={format_metric(transfer_metrics['pr_auc'])} "
                    f"| transfer_gain={format_metric(transfer_metrics['transfer_gain'])} "
                    f"| elapsed={format_duration(reporter.now() - transfer_started_at)}"
                ),
            )
            reporter.log_step(
                (transfer_config.name, target_country, budget.name),
                label="budget",
                index=budget_index,
                total=len(transfer_config.adapt_budget),
                loop_started_at=budget_loop_started_at,
                step_started_at=budget_started_at,
                extra=f"transfer_pr_auc={format_metric(transfer_metrics['pr_auc'])}",
            )

        if per_budget_metrics:
            metrics_rows.append(
                {
                    "experiment": transfer_config.name,
                    "model": "GRU",
                    "split": "macro_test",
                    "protocol": "summary",
                    "target_country": target_country,
                    "candidate": "macro",
                    "pr_auc": float(np.mean(list(per_budget_metrics.values()))),
                }
            )
        reporter.log_step(
            (transfer_config.name, target_country),
            label="country",
            index=country_index,
            total=len(transfer_config.target_countries),
            loop_started_at=country_loop_started_at,
            step_started_at=country_started_at,
            extra=f"budgets={len(per_budget_metrics)}",
        )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions.to_csv(output_path / "predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output_path / "metrics_summary.csv", index=False)

    if not predictions.empty:
        summarize_prediction_frame(
            predictions,
            group_cols=["experiment", "model", "protocol", "country"],
            min_positive=20,
        ).to_csv(output_path / "country_metrics.csv", index=False)
        summarize_prediction_frame(
            add_month_column(predictions),
            group_cols=["experiment", "model", "protocol", "month"],
        ).to_csv(output_path / "monthly_metrics.csv", index=False)

    reporter.log(
        (transfer_config.name,),
        (
            f"experiment completed | elapsed={format_duration(reporter.now() - experiment_started_at)} "
            f"| artifacts={output_path}"
        ),
    )

    return {
        "sample_manifest": output_path / "sample_manifest.csv",
        "metrics_summary": output_path / "metrics_summary.csv",
        "predictions": output_path / "predictions.csv",
        "progress_log": progress_log,
    }


def _evaluate_model_across_folds(
    prepared,
    config: ExperimentConfig,
    model_name: str,
    folds,
    *,
    reporter: ProgressReporter,
) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    if model_name == "Majority":
        return _evaluate_majority(prepared, config, folds, reporter=reporter)
    if model_name == "LogisticRegression":
        return _evaluate_logistic(prepared, config, folds, reporter=reporter)
    if model_name == "XGBoost":
        return _evaluate_xgboost(prepared, config, folds, reporter=reporter)
    if model_name == "LightGBM":
        return _evaluate_lightgbm(prepared, config, folds, reporter=reporter)
    if model_name == "CatBoost":
        return _evaluate_catboost(prepared, config, folds, reporter=reporter)
    if model_name == "XGBoostWeightedCalibrated":
        return _evaluate_xgboost_weighted_calibrated(prepared, config, folds, reporter=reporter)
    if model_name in ("GRU", "TCN", "PatchTST", *HYBRID_SEQUENCE_MODELS, *MULTI_MARKET_SEQUENCE_MODELS):
        return _evaluate_sequence_model(prepared, config, model_name, folds, reporter=reporter)
    raise ValueError(f"Unsupported model: {model_name}")


def _build_sequence_train_eval_datasets(
    prepared,
    config: ExperimentConfig,
    model_name: str,
    train_samples: pd.DataFrame,
    eval_samples: pd.DataFrame,
):
    scaler = prepared.fit_sequence_scaler(train_samples)
    include_multi_market = model_name in MULTI_MARKET_SEQUENCE_MODELS
    if model_name not in HYBRID_SEQUENCE_MODELS:
        train_ds = prepared.build_sequence_dataset(
            train_samples,
            scaler,
            include_country=config.use_country_features,
            include_multi_market=include_multi_market,
        )
        eval_ds = prepared.build_sequence_dataset(
            eval_samples,
            scaler,
            include_country=config.use_country_features,
            include_multi_market=include_multi_market,
        )
        return scaler, train_ds, eval_ds

    train_bundle = prepared.build_tabular_bundle(train_samples, include_country=config.use_country_features)
    eval_bundle = prepared.build_tabular_bundle(eval_samples, include_country=config.use_country_features)
    tabular_scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
    train_ds = prepared.build_sequence_dataset(
        train_samples,
        scaler,
        include_country=config.use_country_features,
        include_multi_market=include_multi_market,
        tabular_values=tabular_scaler.transform(train_bundle.X),
    )
    eval_ds = prepared.build_sequence_dataset(
        eval_samples,
        scaler,
        include_country=config.use_country_features,
        include_multi_market=include_multi_market,
        tabular_values=tabular_scaler.transform(eval_bundle.X),
    )
    return scaler, train_ds, eval_ds


def _sequence_tabular_dim(dataset) -> int:
    sample = dataset[0]
    if "tabular_x" not in sample:
        return 0
    return int(sample["tabular_x"].shape[0])


def _evaluate_majority(prepared, config: ExperimentConfig, folds, *, reporter: ProgressReporter) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    thresholds: list[float] = []
    fold_loop_started_at = reporter.now()
    for fold_index, fold in enumerate(folds, start=1):
        fold_started_at = reporter.now()
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        reporter.log(
            (config.name, "Majority", fold.name),
            f"fold {fold_index}/{len(folds)} started | train_samples={len(train)} | val_samples={len(val)}",
        )
        probability = fit_majority_baseline(train["y_true"].to_numpy(dtype=int))
        y_prob = predict_majority(probability, len(val))
        threshold = find_best_threshold_f1(val["y_true"].to_numpy(dtype=int), y_prob)
        thresholds.append(threshold)
        frame = _prediction_frame(
            val,
            y_prob,
            experiment=config.name,
            model="Majority",
            split="val",
            threshold=threshold,
            fold=fold.name,
            candidate="constant",
        )
        prediction_frames.append(frame)
        metrics_rows.append(_metrics_row(frame, experiment=config.name, model="Majority", split="val", fold=fold.name, candidate="constant"))
        reporter.log_step(
            (config.name, "Majority", fold.name),
            label="fold",
            index=fold_index,
            total=len(folds),
            loop_started_at=fold_loop_started_at,
            step_started_at=fold_started_at,
            extra=f"pr_auc={format_metric(metrics_rows[-1]['pr_auc'])}",
        )
    chosen = {"candidate": "constant", "threshold": float(np.median(thresholds)), "probability": probability}
    return chosen, metrics_rows, prediction_frames


def _evaluate_logistic(prepared, config: ExperimentConfig, folds, *, reporter: ProgressReporter) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)
    prediction_frames: list[pd.DataFrame] = []

    fold_loop_started_at = reporter.now()
    for fold_index, fold in enumerate(folds, start=1):
        fold_started_at = reporter.now()
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        reporter.log(
            (config.name, "LogisticRegression", fold.name),
            f"fold {fold_index}/{len(folds)} started | train_samples={len(train)} | val_samples={len(val)}",
        )
        train_bundle = prepared.build_tabular_bundle(train, include_country=config.use_country_features)
        val_bundle = prepared.build_tabular_bundle(val, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        val_X = scaler.transform(val_bundle.X)
        single_class_probability = _single_class_probability(train_bundle.y)
        candidate_loop_started_at = reporter.now()
        for candidate_index, candidate in enumerate(LOGISTIC_CANDIDATES, start=1):
            candidate_started_at = reporter.now()
            reporter.log(
                (config.name, "LogisticRegression", fold.name),
                f"candidate {candidate_index}/{len(LOGISTIC_CANDIDATES)} started | C={candidate}",
            )
            if single_class_probability is not None:
                y_prob = predict_majority(single_class_probability, len(val_bundle.y))
            else:
                model = fit_logistic_regression(train_X, train_bundle.y, C=candidate, seed=config.random_seed)
                y_prob = predict_logistic_regression(model, val_X)
            threshold = find_best_threshold_f1(val_bundle.y, y_prob)
            frame = _prediction_frame(
                val_bundle.metadata,
                y_prob,
                experiment=config.name,
                model="LogisticRegression",
                split="val",
                threshold=threshold,
                fold=fold.name,
                candidate=f"C={candidate}",
            )
            prediction_frames.append(frame)
            row = _metrics_row(
                frame,
                experiment=config.name,
                model="LogisticRegression",
                split="val",
                fold=fold.name,
                candidate=f"C={candidate}",
            )
            metrics_rows.append(row)
            by_candidate[f"C={candidate}"].append(row | {"threshold": threshold})
            reporter.log_step(
                (config.name, "LogisticRegression", fold.name),
                label="candidate",
                index=candidate_index,
                total=len(LOGISTIC_CANDIDATES),
                loop_started_at=candidate_loop_started_at,
                step_started_at=candidate_started_at,
                extra=f"C={candidate} | pr_auc={format_metric(row['pr_auc'])}",
            )

        reporter.log_step(
            (config.name, "LogisticRegression", fold.name),
            label="fold",
            index=fold_index,
            total=len(folds),
            loop_started_at=fold_loop_started_at,
            step_started_at=fold_started_at,
            extra="candidate_search=done",
        )

    chosen_key, chosen_rows = _pick_best_candidate(by_candidate)
    chosen = {
        "candidate": chosen_key,
        "C": float(chosen_key.split("=")[1]),
        "threshold": float(np.median([row["threshold"] for row in chosen_rows])),
    }
    return chosen, metrics_rows, prediction_frames


def _evaluate_xgboost(prepared, config: ExperimentConfig, folds, *, reporter: ProgressReporter) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)

    fold_loop_started_at = reporter.now()
    for fold_index, fold in enumerate(folds, start=1):
        fold_started_at = reporter.now()
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        reporter.log(
            (config.name, "XGBoost", fold.name),
            f"fold {fold_index}/{len(folds)} started | train_samples={len(train)} | val_samples={len(val)}",
        )
        train_bundle = prepared.build_tabular_bundle(train, include_country=config.use_country_features)
        val_bundle = prepared.build_tabular_bundle(val, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        val_X = scaler.transform(val_bundle.X)
        positives = float(train_bundle.y.sum())
        negatives = float(len(train_bundle.y) - positives)
        scale_pos_weight = negatives / max(positives, 1.0)
        single_class_probability = _single_class_probability(train_bundle.y)
        candidate_loop_started_at = reporter.now()
        for candidate_index, params in enumerate(XGBOOST_CANDIDATES, start=1):
            candidate_started_at = reporter.now()
            candidate_key = f"max_depth={params['max_depth']},lr={params['learning_rate']}"
            reporter.log(
                (config.name, "XGBoost", fold.name),
                f"candidate {candidate_index}/{len(XGBOOST_CANDIDATES)} started | {candidate_key}",
            )
            if single_class_probability is not None:
                y_prob = predict_majority(single_class_probability, len(val_bundle.y))
                best_iteration = 1
            else:
                model, best_iteration = fit_xgboost_classifier(
                    train_X,
                    train_bundle.y,
                    val_X,
                    val_bundle.y,
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    scale_pos_weight=scale_pos_weight,
                    seed=config.random_seed,
                )
                y_prob = predict_xgboost(model, val_X)
            threshold = find_best_threshold_f1(val_bundle.y, y_prob)
            frame = _prediction_frame(
                val_bundle.metadata,
                y_prob,
                experiment=config.name,
                model="XGBoost",
                split="val",
                threshold=threshold,
                fold=fold.name,
                candidate=candidate_key,
            )
            prediction_frames.append(frame)
            row = _metrics_row(
                frame,
                experiment=config.name,
                model="XGBoost",
                split="val",
                fold=fold.name,
                candidate=candidate_key,
            )
            row["best_iteration"] = best_iteration
            row["scale_pos_weight"] = scale_pos_weight
            metrics_rows.append(row)
            by_candidate[candidate_key].append(row | {"threshold": threshold})
            reporter.log_step(
                (config.name, "XGBoost", fold.name),
                label="candidate",
                index=candidate_index,
                total=len(XGBOOST_CANDIDATES),
                loop_started_at=candidate_loop_started_at,
                step_started_at=candidate_started_at,
                extra=f"{candidate_key} | pr_auc={format_metric(row['pr_auc'])}",
            )

        reporter.log_step(
            (config.name, "XGBoost", fold.name),
            label="fold",
            index=fold_index,
            total=len(folds),
            loop_started_at=fold_loop_started_at,
            step_started_at=fold_started_at,
            extra="candidate_search=done",
        )

    chosen_key, chosen_rows = _pick_best_candidate(by_candidate)
    params = {
        "max_depth": int(chosen_key.split(",")[0].split("=")[1]),
        "learning_rate": float(chosen_key.split(",")[1].split("=")[1]),
    }
    chosen = {
        "candidate": chosen_key,
        **params,
        "threshold": float(np.median([row["threshold"] for row in chosen_rows])),
        "n_estimators": int(np.median([row["best_iteration"] for row in chosen_rows])),
    }
    return chosen, metrics_rows, prediction_frames


def _scale_pos_weight(y: np.ndarray) -> float:
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    return negatives / max(positives, 1.0)


def _split_samples_for_calibration(
    samples: pd.DataFrame,
    *,
    calibration_fraction: float = 0.2,
    min_calibration_samples: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = samples.sort_values(["target_time", "country", "anchor_time"], kind="mergesort").reset_index(drop=True)
    if len(ordered) < max(min_calibration_samples * 2, 2):
        return ordered, ordered.iloc[0:0].copy()
    calibration_size = max(int(round(len(ordered) * calibration_fraction)), min_calibration_samples)
    calibration_size = min(calibration_size, len(ordered) - 1)
    split_idx = len(ordered) - calibration_size
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def _predict_weighted_calibrated_xgboost(
    prepared,
    config: ExperimentConfig,
    train_samples: pd.DataFrame,
    predict_samples: pd.DataFrame,
    *,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    calibration: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    full_train_bundle = prepared.build_tabular_bundle(train_samples, include_country=config.use_country_features)
    predict_bundle = prepared.build_tabular_bundle(predict_samples, include_country=config.use_country_features)
    single_class_probability = _single_class_probability(full_train_bundle.y)
    if single_class_probability is not None:
        return predict_bundle.metadata, predict_majority(single_class_probability, len(predict_bundle.y)), {
            "calibration": "identity",
            "scale_pos_weight": float("nan"),
        }

    base_train_samples, calibration_samples = _split_samples_for_calibration(train_samples)
    if calibration_samples.empty or calibration_samples["y_true"].nunique() < 2:
        scaler = TabularScaler(full_train_bundle.continuous_indices).fit(full_train_bundle.X)
        train_X = scaler.transform(full_train_bundle.X)
        predict_X = scaler.transform(predict_bundle.X)
        model = fit_xgboost_final(
            train_X,
            full_train_bundle.y,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            scale_pos_weight=_scale_pos_weight(full_train_bundle.y),
            seed=config.random_seed,
        )
        return predict_bundle.metadata, predict_xgboost(model, predict_X), {
            "calibration": "identity",
            "scale_pos_weight": _scale_pos_weight(full_train_bundle.y),
        }

    base_bundle = prepared.build_tabular_bundle(base_train_samples, include_country=config.use_country_features)
    calibration_bundle = prepared.build_tabular_bundle(calibration_samples, include_country=config.use_country_features)
    if _single_class_probability(base_bundle.y) is not None:
        scaler = TabularScaler(full_train_bundle.continuous_indices).fit(full_train_bundle.X)
        train_X = scaler.transform(full_train_bundle.X)
        predict_X = scaler.transform(predict_bundle.X)
        model = fit_xgboost_final(
            train_X,
            full_train_bundle.y,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            scale_pos_weight=_scale_pos_weight(full_train_bundle.y),
            seed=config.random_seed,
        )
        return predict_bundle.metadata, predict_xgboost(model, predict_X), {
            "calibration": "identity",
            "scale_pos_weight": _scale_pos_weight(full_train_bundle.y),
        }
    scaler = TabularScaler(base_bundle.continuous_indices).fit(base_bundle.X)
    base_X = scaler.transform(base_bundle.X)
    calibration_X = scaler.transform(calibration_bundle.X)
    predict_X = scaler.transform(predict_bundle.X)
    scale_pos_weight = _scale_pos_weight(base_bundle.y)
    model = fit_xgboost_final(
        base_X,
        base_bundle.y,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        seed=config.random_seed,
    )
    calibration_raw = predict_xgboost(model, calibration_X)
    calibrator = fit_probability_calibrator(calibration_bundle.y, calibration_raw, method=calibration)
    predict_raw = predict_xgboost(model, predict_X)
    return predict_bundle.metadata, apply_probability_calibrator(calibrator, predict_raw), {
        "calibration": calibrator.method,
        "scale_pos_weight": scale_pos_weight,
    }


def _evaluate_lightgbm(prepared, config: ExperimentConfig, folds, *, reporter: ProgressReporter) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)

    fold_loop_started_at = reporter.now()
    for fold_index, fold in enumerate(folds, start=1):
        fold_started_at = reporter.now()
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        reporter.log(
            (config.name, "LightGBM", fold.name),
            f"fold {fold_index}/{len(folds)} started | train_samples={len(train)} | val_samples={len(val)}",
        )
        train_bundle = prepared.build_tabular_bundle(train, include_country=config.use_country_features)
        val_bundle = prepared.build_tabular_bundle(val, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        val_X = scaler.transform(val_bundle.X)
        scale_pos_weight = _scale_pos_weight(train_bundle.y)
        single_class_probability = _single_class_probability(train_bundle.y)
        candidate_loop_started_at = reporter.now()
        for candidate_index, params in enumerate(LIGHTGBM_CANDIDATES, start=1):
            candidate_started_at = reporter.now()
            candidate_key = (
                f"num_leaves={params['num_leaves']},lr={params['learning_rate']},n_estimators={params['n_estimators']}"
            )
            reporter.log(
                (config.name, "LightGBM", fold.name),
                f"candidate {candidate_index}/{len(LIGHTGBM_CANDIDATES)} started | {candidate_key}",
            )
            if single_class_probability is not None:
                y_prob = predict_majority(single_class_probability, len(val_bundle.y))
            else:
                model = fit_lightgbm_classifier(
                    train_X,
                    train_bundle.y,
                    num_leaves=params["num_leaves"],
                    learning_rate=params["learning_rate"],
                    n_estimators=params["n_estimators"],
                    scale_pos_weight=scale_pos_weight,
                    seed=config.random_seed,
                )
                y_prob = predict_lightgbm(model, val_X)
            threshold = find_best_threshold_f1(val_bundle.y, y_prob)
            frame = _prediction_frame(
                val_bundle.metadata,
                y_prob,
                experiment=config.name,
                model="LightGBM",
                split="val",
                threshold=threshold,
                fold=fold.name,
                candidate=candidate_key,
            )
            prediction_frames.append(frame)
            row = _metrics_row(
                frame,
                experiment=config.name,
                model="LightGBM",
                split="val",
                fold=fold.name,
                candidate=candidate_key,
            )
            row["scale_pos_weight"] = scale_pos_weight
            metrics_rows.append(row)
            by_candidate[candidate_key].append(row | {"threshold": threshold})
            reporter.log_step(
                (config.name, "LightGBM", fold.name),
                label="candidate",
                index=candidate_index,
                total=len(LIGHTGBM_CANDIDATES),
                loop_started_at=candidate_loop_started_at,
                step_started_at=candidate_started_at,
                extra=f"{candidate_key} | pr_auc={format_metric(row['pr_auc'])}",
            )
        reporter.log_step(
            (config.name, "LightGBM", fold.name),
            label="fold",
            index=fold_index,
            total=len(folds),
            loop_started_at=fold_loop_started_at,
            step_started_at=fold_started_at,
            extra="candidate_search=done",
        )

    chosen_key, chosen_rows = _pick_best_candidate(by_candidate)
    parts = dict(piece.split("=") for piece in chosen_key.split(","))
    chosen = {
        "candidate": chosen_key,
        "num_leaves": int(parts["num_leaves"]),
        "learning_rate": float(parts["lr"]),
        "n_estimators": int(parts["n_estimators"]),
        "threshold": float(np.median([row["threshold"] for row in chosen_rows])),
    }
    return chosen, metrics_rows, prediction_frames


def _evaluate_catboost(prepared, config: ExperimentConfig, folds, *, reporter: ProgressReporter) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)

    fold_loop_started_at = reporter.now()
    for fold_index, fold in enumerate(folds, start=1):
        fold_started_at = reporter.now()
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        reporter.log(
            (config.name, "CatBoost", fold.name),
            f"fold {fold_index}/{len(folds)} started | train_samples={len(train)} | val_samples={len(val)}",
        )
        train_bundle = prepared.build_tabular_bundle(train, include_country=config.use_country_features)
        val_bundle = prepared.build_tabular_bundle(val, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        val_X = scaler.transform(val_bundle.X)
        scale_pos_weight = _scale_pos_weight(train_bundle.y)
        single_class_probability = _single_class_probability(train_bundle.y)
        candidate_loop_started_at = reporter.now()
        for candidate_index, params in enumerate(CATBOOST_CANDIDATES, start=1):
            candidate_started_at = reporter.now()
            candidate_key = f"depth={params['depth']},lr={params['learning_rate']},n_estimators={params['n_estimators']}"
            reporter.log(
                (config.name, "CatBoost", fold.name),
                f"candidate {candidate_index}/{len(CATBOOST_CANDIDATES)} started | {candidate_key}",
            )
            if single_class_probability is not None:
                y_prob = predict_majority(single_class_probability, len(val_bundle.y))
            else:
                model = fit_catboost_classifier(
                    train_X,
                    train_bundle.y,
                    depth=params["depth"],
                    learning_rate=params["learning_rate"],
                    n_estimators=params["n_estimators"],
                    scale_pos_weight=scale_pos_weight,
                    seed=config.random_seed,
                )
                y_prob = predict_catboost(model, val_X)
            threshold = find_best_threshold_f1(val_bundle.y, y_prob)
            frame = _prediction_frame(
                val_bundle.metadata,
                y_prob,
                experiment=config.name,
                model="CatBoost",
                split="val",
                threshold=threshold,
                fold=fold.name,
                candidate=candidate_key,
            )
            prediction_frames.append(frame)
            row = _metrics_row(
                frame,
                experiment=config.name,
                model="CatBoost",
                split="val",
                fold=fold.name,
                candidate=candidate_key,
            )
            row["scale_pos_weight"] = scale_pos_weight
            metrics_rows.append(row)
            by_candidate[candidate_key].append(row | {"threshold": threshold})
            reporter.log_step(
                (config.name, "CatBoost", fold.name),
                label="candidate",
                index=candidate_index,
                total=len(CATBOOST_CANDIDATES),
                loop_started_at=candidate_loop_started_at,
                step_started_at=candidate_started_at,
                extra=f"{candidate_key} | pr_auc={format_metric(row['pr_auc'])}",
            )
        reporter.log_step(
            (config.name, "CatBoost", fold.name),
            label="fold",
            index=fold_index,
            total=len(folds),
            loop_started_at=fold_loop_started_at,
            step_started_at=fold_started_at,
            extra="candidate_search=done",
        )

    chosen_key, chosen_rows = _pick_best_candidate(by_candidate)
    parts = dict(piece.split("=") for piece in chosen_key.split(","))
    chosen = {
        "candidate": chosen_key,
        "depth": int(parts["depth"]),
        "learning_rate": float(parts["lr"]),
        "n_estimators": int(parts["n_estimators"]),
        "threshold": float(np.median([row["threshold"] for row in chosen_rows])),
    }
    return chosen, metrics_rows, prediction_frames


def _evaluate_xgboost_weighted_calibrated(
    prepared,
    config: ExperimentConfig,
    folds,
    *,
    reporter: ProgressReporter,
) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)

    fold_loop_started_at = reporter.now()
    for fold_index, fold in enumerate(folds, start=1):
        fold_started_at = reporter.now()
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        reporter.log(
            (config.name, "XGBoostWeightedCalibrated", fold.name),
            f"fold {fold_index}/{len(folds)} started | train_samples={len(train)} | val_samples={len(val)}",
        )
        candidate_loop_started_at = reporter.now()
        for candidate_index, params in enumerate(XGBOOST_WEIGHTED_CALIBRATED_CANDIDATES, start=1):
            candidate_started_at = reporter.now()
            candidate_key = (
                f"max_depth={params['max_depth']},lr={params['learning_rate']},"
                f"n_estimators={params['n_estimators']},calibration={params['calibration']}"
            )
            reporter.log(
                (config.name, "XGBoostWeightedCalibrated", fold.name),
                f"candidate {candidate_index}/{len(XGBOOST_WEIGHTED_CALIBRATED_CANDIDATES)} started | {candidate_key}",
            )
            metadata, y_prob, fit_info = _predict_weighted_calibrated_xgboost(
                prepared,
                config,
                train,
                val,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                calibration=params["calibration"],
            )
            threshold = find_best_threshold_f1(metadata["y_true"].to_numpy(dtype=int), y_prob)
            frame = _prediction_frame(
                metadata,
                y_prob,
                experiment=config.name,
                model="XGBoostWeightedCalibrated",
                split="val",
                threshold=threshold,
                fold=fold.name,
                candidate=candidate_key,
            )
            prediction_frames.append(frame)
            row = _metrics_row(
                frame,
                experiment=config.name,
                model="XGBoostWeightedCalibrated",
                split="val",
                fold=fold.name,
                candidate=candidate_key,
            )
            row["scale_pos_weight"] = fit_info["scale_pos_weight"]
            row["calibration_method"] = fit_info["calibration"]
            metrics_rows.append(row)
            by_candidate[candidate_key].append(row | {"threshold": threshold})
            reporter.log_step(
                (config.name, "XGBoostWeightedCalibrated", fold.name),
                label="candidate",
                index=candidate_index,
                total=len(XGBOOST_WEIGHTED_CALIBRATED_CANDIDATES),
                loop_started_at=candidate_loop_started_at,
                step_started_at=candidate_started_at,
                extra=f"{candidate_key} | pr_auc={format_metric(row['pr_auc'])}",
            )
        reporter.log_step(
            (config.name, "XGBoostWeightedCalibrated", fold.name),
            label="fold",
            index=fold_index,
            total=len(folds),
            loop_started_at=fold_loop_started_at,
            step_started_at=fold_started_at,
            extra="candidate_search=done",
        )

    chosen_key, chosen_rows = _pick_best_candidate(by_candidate)
    parts = dict(piece.split("=") for piece in chosen_key.split(","))
    chosen = {
        "candidate": chosen_key,
        "max_depth": int(parts["max_depth"]),
        "learning_rate": float(parts["lr"]),
        "n_estimators": int(parts["n_estimators"]),
        "calibration": parts["calibration"],
        "threshold": float(np.median([row["threshold"] for row in chosen_rows])),
    }
    return chosen, metrics_rows, prediction_frames


def _evaluate_sequence_model(
    prepared,
    config: ExperimentConfig,
    model_name: str,
    folds,
    *,
    reporter: ProgressReporter,
) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    thresholds: list[float] = []
    best_epochs: list[int] = []

    fold_loop_started_at = reporter.now()
    for fold_index, fold in enumerate(folds, start=1):
        fold_started_at = reporter.now()
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        reporter.log(
            (config.name, model_name, fold.name),
            f"fold {fold_index}/{len(folds)} started | train_samples={len(train)} | val_samples={len(val)}",
        )
        _, train_ds, val_ds = _build_sequence_train_eval_datasets(
            prepared,
            config,
            model_name,
            train,
            val,
        )
        learning_rate = config.sequence_learning_rate
        single_class_probability = _single_class_probability(train_ds.metadata["y_true"].to_numpy(dtype=int))
        if single_class_probability is not None:
            y_prob = predict_majority(single_class_probability, len(val_ds))
            best_epoch = 1
            reporter.log(
                (config.name, model_name, fold.name),
                "fold used single-class fallback",
            )
        else:
            outcome = train_sequence_model(
                model_name,
                train_dataset=train_ds,
                val_dataset=val_ds,
                use_country_embedding=config.use_country_features,
                num_countries=len(config.countries),
                random_seed=config.random_seed,
                learning_rate=learning_rate,
                max_epochs=config.sequence_max_epochs,
                patience=config.sequence_patience,
                loss_name=config.sequence_loss,
                focal_gamma=config.focal_gamma,
                aux_target=config.sequence_aux_target,
                aux_weight=config.sequence_aux_weight,
                reporter=reporter,
                progress_prefix=(config.name, model_name, fold.name),
            )
            fitted = load_sequence_model(
                model_name,
                input_dim=train_ds[0]["x"].shape[1],
                use_country_embedding=config.use_country_features,
                num_countries=len(config.countries),
                state_dict=outcome.state_dict,
                tabular_dim=_sequence_tabular_dim(train_ds),
            )
            y_prob = predict_sequence_model(fitted, val_ds)
            best_epoch = outcome.best_epoch
        threshold = find_best_threshold_f1(val_ds.metadata["y_true"].to_numpy(dtype=int), y_prob)
        thresholds.append(threshold)
        best_epochs.append(best_epoch)
        frame = _prediction_frame(
            val_ds.metadata,
            y_prob,
            experiment=config.name,
            model=model_name,
            split="val",
            threshold=threshold,
            fold=fold.name,
            candidate="default",
        )
        prediction_frames.append(frame)
        row = _metrics_row(frame, experiment=config.name, model=model_name, split="val", fold=fold.name, candidate="default")
        row["best_epoch"] = best_epoch
        metrics_rows.append(row)
        reporter.log_step(
            (config.name, model_name, fold.name),
            label="fold",
            index=fold_index,
            total=len(folds),
            loop_started_at=fold_loop_started_at,
            step_started_at=fold_started_at,
            extra=f"best_epoch={best_epoch} | pr_auc={format_metric(row['pr_auc'])}",
        )

    chosen = {
        "candidate": "default",
        "threshold": float(np.median(thresholds)),
        "epochs": int(np.median(best_epochs)),
        "learning_rate": config.sequence_learning_rate,
        "loss_name": config.sequence_loss,
        "focal_gamma": config.focal_gamma,
        "aux_target": config.sequence_aux_target,
        "aux_weight": config.sequence_aux_weight,
    }
    return chosen, metrics_rows, prediction_frames


def _fit_and_score_final_model(
    prepared,
    config,
    model_name,
    chosen,
    train_samples,
    test_samples,
    *,
    reporter: ProgressReporter,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if model_name == "Majority":
        probability = fit_majority_baseline(train_samples["y_true"].to_numpy(dtype=int))
        y_prob = predict_majority(probability, len(test_samples))
        frame = _prediction_frame(
            test_samples,
            y_prob,
            experiment=config.name,
            model="Majority",
            split="test",
            threshold=chosen["threshold"],
            candidate="constant",
        )
        return frame, _metrics_row(frame, experiment=config.name, model="Majority", split="test", candidate="constant")

    if model_name == "LogisticRegression":
        train_bundle = prepared.build_tabular_bundle(train_samples, include_country=config.use_country_features)
        test_bundle = prepared.build_tabular_bundle(test_samples, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        test_X = scaler.transform(test_bundle.X)
        single_class_probability = _single_class_probability(train_bundle.y)
        if single_class_probability is not None:
            y_prob = predict_majority(single_class_probability, len(test_bundle.y))
        else:
            model = fit_logistic_regression(train_X, train_bundle.y, C=chosen["C"], seed=config.random_seed)
            y_prob = predict_logistic_regression(model, test_X)
        frame = _prediction_frame(
            test_bundle.metadata,
            y_prob,
            experiment=config.name,
            model=model_name,
            split="test",
            threshold=chosen["threshold"],
            candidate=chosen["candidate"],
        )
        return frame, _metrics_row(frame, experiment=config.name, model=model_name, split="test", candidate=chosen["candidate"])

    if model_name == "XGBoost":
        train_bundle = prepared.build_tabular_bundle(train_samples, include_country=config.use_country_features)
        test_bundle = prepared.build_tabular_bundle(test_samples, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        test_X = scaler.transform(test_bundle.X)
        positives = float(train_bundle.y.sum())
        negatives = float(len(train_bundle.y) - positives)
        scale_pos_weight = negatives / max(positives, 1.0)
        single_class_probability = _single_class_probability(train_bundle.y)
        if single_class_probability is not None:
            y_prob = predict_majority(single_class_probability, len(test_bundle.y))
        else:
            model = fit_xgboost_final(
                train_X,
                train_bundle.y,
                max_depth=chosen["max_depth"],
                learning_rate=chosen["learning_rate"],
                n_estimators=chosen["n_estimators"],
                scale_pos_weight=scale_pos_weight,
                seed=config.random_seed,
            )
            y_prob = predict_xgboost(model, test_X)
        frame = _prediction_frame(
            test_bundle.metadata,
            y_prob,
            experiment=config.name,
            model=model_name,
            split="test",
            threshold=chosen["threshold"],
            candidate=chosen["candidate"],
        )
        return frame, _metrics_row(frame, experiment=config.name, model=model_name, split="test", candidate=chosen["candidate"])

    if model_name == "LightGBM":
        train_bundle = prepared.build_tabular_bundle(train_samples, include_country=config.use_country_features)
        test_bundle = prepared.build_tabular_bundle(test_samples, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        test_X = scaler.transform(test_bundle.X)
        single_class_probability = _single_class_probability(train_bundle.y)
        if single_class_probability is not None:
            y_prob = predict_majority(single_class_probability, len(test_bundle.y))
        else:
            model = fit_lightgbm_classifier(
                train_X,
                train_bundle.y,
                num_leaves=chosen["num_leaves"],
                learning_rate=chosen["learning_rate"],
                n_estimators=chosen["n_estimators"],
                scale_pos_weight=_scale_pos_weight(train_bundle.y),
                seed=config.random_seed,
            )
            y_prob = predict_lightgbm(model, test_X)
        frame = _prediction_frame(
            test_bundle.metadata,
            y_prob,
            experiment=config.name,
            model=model_name,
            split="test",
            threshold=chosen["threshold"],
            candidate=chosen["candidate"],
        )
        return frame, _metrics_row(frame, experiment=config.name, model=model_name, split="test", candidate=chosen["candidate"])

    if model_name == "CatBoost":
        train_bundle = prepared.build_tabular_bundle(train_samples, include_country=config.use_country_features)
        test_bundle = prepared.build_tabular_bundle(test_samples, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        test_X = scaler.transform(test_bundle.X)
        single_class_probability = _single_class_probability(train_bundle.y)
        if single_class_probability is not None:
            y_prob = predict_majority(single_class_probability, len(test_bundle.y))
        else:
            model = fit_catboost_classifier(
                train_X,
                train_bundle.y,
                depth=chosen["depth"],
                learning_rate=chosen["learning_rate"],
                n_estimators=chosen["n_estimators"],
                scale_pos_weight=_scale_pos_weight(train_bundle.y),
                seed=config.random_seed,
            )
            y_prob = predict_catboost(model, test_X)
        frame = _prediction_frame(
            test_bundle.metadata,
            y_prob,
            experiment=config.name,
            model=model_name,
            split="test",
            threshold=chosen["threshold"],
            candidate=chosen["candidate"],
        )
        return frame, _metrics_row(frame, experiment=config.name, model=model_name, split="test", candidate=chosen["candidate"])

    if model_name == "XGBoostWeightedCalibrated":
        metadata, y_prob, _ = _predict_weighted_calibrated_xgboost(
            prepared,
            config,
            train_samples,
            test_samples,
            max_depth=chosen["max_depth"],
            learning_rate=chosen["learning_rate"],
            n_estimators=chosen["n_estimators"],
            calibration=chosen["calibration"],
        )
        frame = _prediction_frame(
            metadata,
            y_prob,
            experiment=config.name,
            model=model_name,
            split="test",
            threshold=chosen["threshold"],
            candidate=chosen["candidate"],
        )
        return frame, _metrics_row(frame, experiment=config.name, model=model_name, split="test", candidate=chosen["candidate"])

    _, train_ds, test_ds = _build_sequence_train_eval_datasets(
        prepared,
        config,
        model_name,
        train_samples,
        test_samples,
    )
    single_class_probability = _single_class_probability(train_ds.metadata["y_true"].to_numpy(dtype=int))
    if single_class_probability is not None:
        y_prob = predict_majority(single_class_probability, len(test_ds))
        reporter.log(
            (config.name, model_name, "Final"),
            "final stage used single-class fallback",
        )
    else:
        fitted = fit_sequence_final(
            model_name,
            train_dataset=train_ds,
            use_country_embedding=config.use_country_features,
            num_countries=len(config.countries),
            random_seed=config.random_seed,
            learning_rate=chosen["learning_rate"],
            epochs=chosen["epochs"],
            loss_name=chosen.get("loss_name", "bce"),
            focal_gamma=chosen.get("focal_gamma", 2.0),
            aux_target=chosen.get("aux_target"),
            aux_weight=chosen.get("aux_weight", 0.2),
            reporter=reporter,
            progress_prefix=(config.name, model_name, "Final"),
        )
        y_prob = predict_sequence_model(fitted, test_ds)
    frame = _prediction_frame(
        test_ds.metadata,
        y_prob,
        experiment=config.name,
        model=model_name,
        split="test",
        threshold=chosen["threshold"],
        candidate=chosen["candidate"],
    )
    return frame, _metrics_row(frame, experiment=config.name, model=model_name, split="test", candidate=chosen["candidate"])


def _pick_best_candidate(candidate_rows: dict[str, list[dict[str, object]]]) -> tuple[str, list[dict[str, object]]]:
    best_key = ""
    best_sort = None
    for key, rows in candidate_rows.items():
        pr_auc = float(np.nanmean([row["pr_auc"] for row in rows]))
        f1 = float(np.nanmean([row["f1"] for row in rows]))
        roc_auc = float(np.nanmean([row["roc_auc"] for row in rows]))
        sort_key = (np.nan_to_num(pr_auc, nan=-1.0), np.nan_to_num(f1, nan=-1.0), np.nan_to_num(roc_auc, nan=-1.0))
        if best_sort is None or sort_key > best_sort:
            best_sort = sort_key
            best_key = key
    if not best_key:
        raise RuntimeError("No candidate results were produced.")
    return best_key, candidate_rows[best_key]


def _single_class_probability(y: np.ndarray) -> float | None:
    unique = np.unique(y)
    if unique.size < 2:
        return float(unique[0])
    return None


def _aggregate_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "seed" not in metrics.columns:
        return pd.DataFrame()
    if "seed_aggregation" in metrics.columns:
        raw = metrics.loc[metrics["seed_aggregation"] == "raw"].copy()
    else:
        raw = metrics.copy()
    if raw.empty:
        return pd.DataFrame()

    numeric_cols = [
        column
        for column in raw.columns
        if column not in {"seed", "seed_aggregation"} and pd.api.types.is_numeric_dtype(raw[column])
    ]
    if not numeric_cols:
        return pd.DataFrame()
    group_cols = [column for column in raw.columns if column not in set(numeric_cols).union({"seed", "seed_aggregation"})]

    mean_rows = raw.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    mean_rows["seed"] = pd.NA
    mean_rows["seed_aggregation"] = "mean"

    std_rows = raw.groupby(group_cols, dropna=False)[numeric_cols].std(ddof=0).reset_index()
    std_rows["seed"] = pd.NA
    std_rows["seed_aggregation"] = "std"
    return pd.concat([mean_rows, std_rows], ignore_index=True, sort=False)


def _prediction_frame(
    metadata: pd.DataFrame,
    y_prob: np.ndarray,
    *,
    experiment: str,
    model: str,
    split: str,
    threshold: float,
    candidate: str,
    fold: str | None = None,
    protocol: str | None = None,
) -> pd.DataFrame:
    frame = metadata[["country", "anchor_time", "target_time", "y_true"]].copy()
    frame["y_prob"] = y_prob.astype(np.float32)
    frame["split"] = split
    frame["experiment"] = experiment
    frame["model"] = model
    frame["candidate"] = candidate
    frame["threshold"] = threshold
    if fold is not None:
        frame["fold"] = fold
    if protocol is not None:
        frame["protocol"] = protocol
    return frame


def _metrics_row(
    frame: pd.DataFrame,
    *,
    experiment: str,
    model: str,
    split: str,
    candidate: str,
    fold: str | None = None,
    protocol: str | None = None,
    target_country: str | None = None,
) -> dict[str, object]:
    metrics = compute_binary_metrics(
        frame["y_true"].to_numpy(dtype=int),
        frame["y_prob"].to_numpy(dtype=float),
        float(frame["threshold"].iloc[0]),
    ).to_dict()
    row: dict[str, object] = {
        "experiment": experiment,
        "model": model,
        "split": split,
        "candidate": candidate,
        **metrics,
    }
    if fold is not None:
        row["fold"] = fold
    if protocol is not None:
        row["protocol"] = protocol
    if target_country is not None:
        row["target_country"] = target_country
    return row
