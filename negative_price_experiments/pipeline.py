from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_MODELS,
    FINAL_TEST_RANGE,
    FINAL_TRAIN_RANGE,
    TABULAR_MODELS,
    WALK_FORWARD_FOLDS,
    AdaptBudget,
    ExperimentConfig,
    TransferConfig,
)
from .data import TabularScaler, prepare_experiment_data
from .metrics import add_month_column, compute_binary_metrics, find_best_threshold_f1, summarize_prediction_frame
from .models import (
    DependencyUnavailableError,
    fit_logistic_regression,
    fit_majority_baseline,
    fit_sequence_final,
    fit_xgboost_classifier,
    fit_xgboost_final,
    load_sequence_model,
    predict_logistic_regression,
    predict_majority,
    predict_sequence_model,
    predict_xgboost,
    require_torch,
    train_sequence_model,
)


LOGISTIC_CANDIDATES = (0.1, 1.0, 10.0)
XGBOOST_CANDIDATES = (
    {"max_depth": 4, "learning_rate": 0.05},
    {"max_depth": 4, "learning_rate": 0.1},
    {"max_depth": 6, "learning_rate": 0.05},
    {"max_depth": 6, "learning_rate": 0.1},
)


def run_experiment(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    folds=None,
    final_train_range=None,
    final_test_range=None,
    skip_unavailable_models: bool = False,
) -> dict[str, Path]:
    output_path = Path(output_dir).resolve() / config.name
    output_path.mkdir(parents=True, exist_ok=True)
    folds = tuple(folds or WALK_FORWARD_FOLDS)
    final_train_range = final_train_range or FINAL_TRAIN_RANGE
    final_test_range = final_test_range or FINAL_TEST_RANGE

    prepared = prepare_experiment_data(config)
    prepared.sample_manifest.to_csv(output_path / "sample_manifest.csv", index=False)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    chosen_records: dict[str, dict[str, object]] = {}

    models = tuple(config.models or DEFAULT_MODELS)
    for model_name in models:
        try:
            chosen_records[model_name], model_metrics, model_predictions = _evaluate_model_across_folds(
                prepared, config, model_name, folds
            )
        except DependencyUnavailableError:
            if skip_unavailable_models:
                continue
            raise
        metrics_rows.extend(model_metrics)
        prediction_frames.extend(model_predictions)

    train_samples = prepared.select_samples(final_train_range)
    test_samples = prepared.select_samples(final_test_range)
    if train_samples.empty or test_samples.empty:
        raise RuntimeError(f"{config.name} has empty final train/test split.")

    final_predictions: list[pd.DataFrame] = []
    for model_name, chosen in chosen_records.items():
        frame, metrics = _fit_and_score_final_model(prepared, config, model_name, chosen, train_samples, test_samples)
        prediction_frames.append(frame)
        final_predictions.append(frame)
        metrics_rows.append(metrics)

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

    return {
        "sample_manifest": output_path / "sample_manifest.csv",
        "metrics_summary": output_path / "metrics_summary.csv",
        "predictions": output_path / "predictions.csv",
    }


def run_transfer_experiment(
    transfer_config: TransferConfig,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    require_torch()
    output_path = Path(output_dir).resolve() / transfer_config.name
    output_path.mkdir(parents=True, exist_ok=True)

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
    prepared = prepare_experiment_data(config)
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
    source_threshold = find_best_threshold_f1(source_val_ds.metadata["y_true"].to_numpy(dtype=int), source_val_prob)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for target_country in transfer_config.target_countries:
        test_samples = prepared.select_samples(transfer_config.target_test_range, countries=(target_country,))
        if test_samples.empty:
            continue

        zero_shot_test_ds = prepared.build_sequence_dataset(test_samples, source_scaler, include_country=False)
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

        per_budget_metrics: dict[str, float] = {}
        for budget in transfer_config.adapt_budget:
            target_train = prepared.select_samples(budget.train_range, countries=(target_country,))
            target_val = prepared.select_samples(budget.val_range, countries=(target_country,))
            if target_train.empty or target_val.empty:
                continue

            target_scaler = prepared.fit_sequence_scaler(target_train)
            target_train_ds = prepared.build_sequence_dataset(target_train, target_scaler, include_country=False)
            target_val_ds = prepared.build_sequence_dataset(target_val, target_scaler, include_country=False)
            target_test_ds = prepared.build_sequence_dataset(test_samples, target_scaler, include_country=False)
            target_train_single_class = _single_class_probability(target_train_ds.metadata["y_true"].to_numpy(dtype=int))
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

            transfer_train_ds = prepared.build_sequence_dataset(target_train, source_scaler, include_country=False)
            transfer_val_ds = prepared.build_sequence_dataset(target_val, source_scaler, include_country=False)
            transfer_test_ds = prepared.build_sequence_dataset(test_samples, source_scaler, include_country=False)
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

    return {
        "sample_manifest": output_path / "sample_manifest.csv",
        "metrics_summary": output_path / "metrics_summary.csv",
        "predictions": output_path / "predictions.csv",
    }


def _evaluate_model_across_folds(prepared, config: ExperimentConfig, model_name: str, folds) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    if model_name == "Majority":
        return _evaluate_majority(prepared, config, folds)
    if model_name == "LogisticRegression":
        return _evaluate_logistic(prepared, config, folds)
    if model_name == "XGBoost":
        return _evaluate_xgboost(prepared, config, folds)
    if model_name in ("GRU", "TCN"):
        return _evaluate_sequence_model(prepared, config, model_name, folds)
    raise ValueError(f"Unsupported model: {model_name}")


def _evaluate_majority(prepared, config: ExperimentConfig, folds) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    thresholds: list[float] = []
    for fold in folds:
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
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
    chosen = {"candidate": "constant", "threshold": float(np.median(thresholds)), "probability": probability}
    return chosen, metrics_rows, prediction_frames


def _evaluate_logistic(prepared, config: ExperimentConfig, folds) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        train_bundle = prepared.build_tabular_bundle(train, include_country=config.use_country_features)
        val_bundle = prepared.build_tabular_bundle(val, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        val_X = scaler.transform(val_bundle.X)
        single_class_probability = _single_class_probability(train_bundle.y)
        for candidate in LOGISTIC_CANDIDATES:
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

    chosen_key, chosen_rows = _pick_best_candidate(by_candidate)
    chosen = {
        "candidate": chosen_key,
        "C": float(chosen_key.split("=")[1]),
        "threshold": float(np.median([row["threshold"] for row in chosen_rows])),
    }
    return chosen, metrics_rows, prediction_frames


def _evaluate_xgboost(prepared, config: ExperimentConfig, folds) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)

    for fold in folds:
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        train_bundle = prepared.build_tabular_bundle(train, include_country=config.use_country_features)
        val_bundle = prepared.build_tabular_bundle(val, include_country=config.use_country_features)
        scaler = TabularScaler(train_bundle.continuous_indices).fit(train_bundle.X)
        train_X = scaler.transform(train_bundle.X)
        val_X = scaler.transform(val_bundle.X)
        positives = float(train_bundle.y.sum())
        negatives = float(len(train_bundle.y) - positives)
        scale_pos_weight = negatives / max(positives, 1.0)
        single_class_probability = _single_class_probability(train_bundle.y)
        for params in XGBOOST_CANDIDATES:
            candidate_key = f"max_depth={params['max_depth']},lr={params['learning_rate']}"
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


def _evaluate_sequence_model(prepared, config: ExperimentConfig, model_name: str, folds) -> tuple[dict[str, object], list[dict[str, object]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    thresholds: list[float] = []
    best_epochs: list[int] = []

    for fold in folds:
        train = prepared.select_samples(fold.train_range)
        val = prepared.select_samples(fold.val_range)
        scaler = prepared.fit_sequence_scaler(train)
        train_ds = prepared.build_sequence_dataset(train, scaler, include_country=config.use_country_features)
        val_ds = prepared.build_sequence_dataset(val, scaler, include_country=config.use_country_features)
        learning_rate = 1e-3
        single_class_probability = _single_class_probability(train_ds.metadata["y_true"].to_numpy(dtype=int))
        if single_class_probability is not None:
            y_prob = predict_majority(single_class_probability, len(val_ds))
            best_epoch = 1
        else:
            outcome = train_sequence_model(
                model_name,
                train_dataset=train_ds,
                val_dataset=val_ds,
                use_country_embedding=config.use_country_features,
                num_countries=len(config.countries),
                random_seed=config.random_seed,
                learning_rate=learning_rate,
                max_epochs=30,
                patience=5,
            )
            fitted = load_sequence_model(
                model_name,
                input_dim=train_ds[0]["x"].shape[1],
                use_country_embedding=config.use_country_features,
                num_countries=len(config.countries),
                state_dict=outcome.state_dict,
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

    chosen = {
        "candidate": "default",
        "threshold": float(np.median(thresholds)),
        "epochs": int(np.median(best_epochs)),
        "learning_rate": 1e-3,
    }
    return chosen, metrics_rows, prediction_frames


def _fit_and_score_final_model(prepared, config, model_name, chosen, train_samples, test_samples) -> tuple[pd.DataFrame, dict[str, object]]:
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

    scaler = prepared.fit_sequence_scaler(train_samples)
    train_ds = prepared.build_sequence_dataset(train_samples, scaler, include_country=config.use_country_features)
    test_ds = prepared.build_sequence_dataset(test_samples, scaler, include_country=config.use_country_features)
    single_class_probability = _single_class_probability(train_ds.metadata["y_true"].to_numpy(dtype=int))
    if single_class_probability is not None:
        y_prob = predict_majority(single_class_probability, len(test_ds))
    else:
        fitted = fit_sequence_final(
            model_name,
            train_dataset=train_ds,
            use_country_embedding=config.use_country_features,
            num_countries=len(config.countries),
            random_seed=config.random_seed,
            learning_rate=chosen["learning_rate"],
            epochs=chosen["epochs"],
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
