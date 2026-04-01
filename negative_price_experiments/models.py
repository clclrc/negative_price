from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    xgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    DataLoader = None

from .metrics import safe_average_precision
from .progress import ProgressReporter, estimate_remaining_seconds, format_duration, format_metric, format_rate
from .runtime import get_cpu_worker_count


HAS_XGBOOST = xgb is not None
HAS_TORCH = torch is not None


class DependencyUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TorchTrainingOutcome:
    state_dict: dict[str, object]
    best_epoch: int
    best_score: float


def has_cuda() -> bool:
    return HAS_TORCH and torch.cuda.is_available()


def has_mps() -> bool:
    if not HAS_TORCH:
        return False
    mps_backend = getattr(torch.backends, "mps", None)
    return mps_backend is not None and mps_backend.is_available()


def get_preferred_torch_device() -> "torch.device":
    require_torch()
    if has_cuda():
        return torch.device("cuda")
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def get_xgboost_device() -> str:
    if has_cuda():
        return "cuda"
    return "cpu"


def require_xgboost() -> None:
    if not HAS_XGBOOST:
        raise DependencyUnavailableError(
            "xgboost is required for the XGBoost model. Install it with `python3 -m pip install xgboost`."
        )


def require_torch() -> None:
    if not HAS_TORCH:
        raise DependencyUnavailableError(
            "torch is required for GRU/TCN models. Install it with `python3 -m pip install torch`."
        )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:  # pragma: no branch - optional dependency
        torch.manual_seed(seed)


def fit_majority_baseline(y_train: np.ndarray) -> float:
    if y_train.size == 0:
        raise ValueError("Majority baseline cannot be fit on an empty target array.")
    return float(np.mean(y_train))


def predict_majority(probability: float, count: int) -> np.ndarray:
    return np.full(count, probability, dtype=np.float32)


def fit_logistic_regression(train_X: np.ndarray, train_y: np.ndarray, *, C: float, seed: int) -> LogisticRegression:
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        penalty="l2",
        solver="liblinear",
        random_state=seed,
        max_iter=1000,
    )
    model.fit(train_X, train_y)
    return model


def predict_logistic_regression(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)


def fit_xgboost_classifier(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    *,
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
    seed: int,
) -> tuple[object, int]:
    require_xgboost()
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device=get_xgboost_device(),
        n_jobs=get_cpu_worker_count(max_workers=8),
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1000,
        early_stopping_rounds=50,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
    )
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return model, 1000
    return model, int(best_iteration) + 1


def fit_xgboost_final(
    train_X: np.ndarray,
    train_y: np.ndarray,
    *,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    scale_pos_weight: float,
    seed: int,
) -> object:
    require_xgboost()
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device=get_xgboost_device(),
        n_jobs=get_cpu_worker_count(max_workers=8),
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
    )
    model.fit(train_X, train_y, verbose=False)
    return model


def predict_xgboost(model: object, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)


if HAS_TORCH:
    class GRUClassifier(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(
            self,
            *,
            input_dim: int,
            use_country_embedding: bool,
            num_countries: int,
            embedding_dim: int = 8,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.use_country_embedding = use_country_embedding
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            if use_country_embedding:
                self.country_embedding = nn.Embedding(num_countries, embedding_dim)
                classifier_in = hidden_size + embedding_dim
            else:
                self.country_embedding = None
                classifier_in = hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor, country_idx: torch.Tensor) -> torch.Tensor:
            _, hidden = self.gru(x)
            features = hidden[-1]
            if self.use_country_embedding:
                embedded = self.country_embedding(country_idx)
                features = torch.cat([features, embedded], dim=1)
            return self.classifier(features).squeeze(1)


    class TemporalBlock(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x if self.residual is None else self.residual(x)
            out = self.conv1(x)
            out = self.activation(out)
            out = self.dropout(out)
            out = self.conv2(out)
            out = self.activation(out)
            out = self.dropout(out)
            return self.activation(out + residual)


    class TCNClassifier(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(
            self,
            *,
            input_dim: int,
            use_country_embedding: bool,
            num_countries: int,
            embedding_dim: int = 8,
            channels: tuple[int, ...] = (64, 64, 128, 128),
            kernel_size: int = 3,
            dilations: tuple[int, ...] = (1, 2, 4, 8),
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_channels = input_dim
            for out_channels, dilation in zip(channels, dilations):
                layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))
                in_channels = out_channels
            self.network = nn.Sequential(*layers)
            self.use_country_embedding = use_country_embedding
            if use_country_embedding:
                self.country_embedding = nn.Embedding(num_countries, embedding_dim)
                classifier_in = in_channels + embedding_dim
            else:
                self.country_embedding = None
                classifier_in = in_channels
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor, country_idx: torch.Tensor) -> torch.Tensor:
            features = self.network(x.transpose(1, 2)).mean(dim=2)
            if self.use_country_embedding:
                embedded = self.country_embedding(country_idx)
                features = torch.cat([features, embedded], dim=1)
            return self.classifier(features).squeeze(1)


def build_sequence_model(
    model_name: str,
    *,
    input_dim: int,
    use_country_embedding: bool,
    num_countries: int,
) -> nn.Module:
    require_torch()
    if model_name == "GRU":
        return GRUClassifier(
            input_dim=input_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "TCN":
        return TCNClassifier(
            input_dim=input_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    raise ValueError(f"Unsupported sequence model: {model_name}")


def load_sequence_model(
    model_name: str,
    *,
    input_dim: int,
    use_country_embedding: bool,
    num_countries: int,
    state_dict: dict[str, object],
) -> nn.Module:
    model = build_sequence_model(
        model_name,
        input_dim=input_dim,
        use_country_embedding=use_country_embedding,
        num_countries=num_countries,
    )
    model.load_state_dict(state_dict)
    return model


def _build_loader(dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    require_torch()
    device = get_preferred_torch_device()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _configure_torch_cpu_threads(device: "torch.device") -> None:
    if device.type != "cpu":
        return
    torch.set_num_threads(get_cpu_worker_count(max_workers=8))


def _predict_with_model(model: nn.Module, dataset) -> np.ndarray:
    require_torch()
    loader = _build_loader(dataset, batch_size=256, shuffle=False)
    device = get_preferred_torch_device()
    _configure_torch_cpu_threads(device)
    model.to(device)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device), batch["country_idx"].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    if not probs:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(probs).astype(np.float32)


def train_sequence_model(
    model_name: str,
    *,
    train_dataset,
    val_dataset,
    use_country_embedding: bool,
    num_countries: int,
    random_seed: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    init_state_dict: dict[str, object] | None = None,
    reporter: ProgressReporter | None = None,
    progress_prefix: tuple[str, ...] = (),
) -> TorchTrainingOutcome:
    reporter = reporter or ProgressReporter()
    require_torch()
    set_random_seed(random_seed)
    model = build_sequence_model(
        model_name,
        input_dim=train_dataset[0]["x"].shape[1],
        use_country_embedding=use_country_embedding,
        num_countries=num_countries,
    )
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    device = get_preferred_torch_device()
    _configure_torch_cpu_threads(device)
    model.to(device)
    train_loader = _build_loader(train_dataset, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_started_at = reporter.now()
    reporter.log(
        progress_prefix,
        (
            f"sequence training start | model={model_name} | device={device.type} "
            f"| train_samples={len(train_dataset)} | val_samples={len(val_dataset)} "
            f"| batches={len(train_loader)} | max_epochs={max_epochs}"
        ),
    )

    train_y = train_dataset.metadata["y_true"].to_numpy(dtype=np.float32)
    positives = float(train_y.sum())
    negatives = float(train_y.size - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score = float("-inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        epoch_started_at = reporter.now()
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["x"].to(device), batch["country_idx"].to(device))
            loss = loss_fn(logits, batch["y"].to(device))
            loss.backward()
            optimizer.step()
            batch_size = int(batch["y"].shape[0])
            epoch_loss_sum += float(loss.item()) * batch_size
            epoch_sample_count += batch_size

        val_prob = _predict_with_model(model, val_dataset)
        val_y = val_dataset.metadata["y_true"].to_numpy(dtype=np.int8)
        score = safe_average_precision(val_y, val_prob)
        score_for_patience = score if not np.isnan(score) else float("-inf")
        if score_for_patience > best_score:
            best_score = score_for_patience
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        best_score_display = float("nan") if best_score == float("-inf") else best_score
        epoch_elapsed = reporter.now() - epoch_started_at
        total_elapsed = reporter.now() - train_started_at
        reporter.log(
            progress_prefix,
            (
                f"epoch {epoch}/{max_epochs} | train_loss={epoch_loss_sum / max(epoch_sample_count, 1):.4f} "
                f"| val_pr_auc={format_metric(score)} | best_epoch={best_epoch} "
                f"| best_pr_auc={format_metric(best_score_display)} "
                f"| epoch={format_duration(epoch_elapsed)} | elapsed={format_duration(total_elapsed)} "
                f"| speed={format_rate(epoch_sample_count, epoch_elapsed)} "
                f"| eta={format_duration(estimate_remaining_seconds(loop_started_at=train_started_at, completed_steps=epoch, total_steps=max_epochs, now=reporter.now()))}"
            ),
        )
        if epochs_without_improvement >= patience:
            reporter.log(
                progress_prefix,
                (
                    f"early stopping triggered | epoch={epoch}/{max_epochs} | patience={patience} "
                    f"| best_epoch={best_epoch} | best_pr_auc={format_metric(best_score_display)}"
                ),
            )
            break

    final_best_score = float("nan") if best_score == float("-inf") else best_score
    reporter.log(
        progress_prefix,
        (
            f"sequence training completed | best_epoch={best_epoch} "
            f"| best_pr_auc={format_metric(final_best_score)} "
            f"| elapsed={format_duration(reporter.now() - train_started_at)}"
        ),
    )

    return TorchTrainingOutcome(state_dict=best_state, best_epoch=best_epoch, best_score=best_score)


def fit_sequence_final(
    model_name: str,
    *,
    train_dataset,
    use_country_embedding: bool,
    num_countries: int,
    random_seed: int,
    learning_rate: float,
    epochs: int,
    init_state_dict: dict[str, object] | None = None,
    reporter: ProgressReporter | None = None,
    progress_prefix: tuple[str, ...] = (),
) -> nn.Module:
    reporter = reporter or ProgressReporter()
    require_torch()
    set_random_seed(random_seed)
    model = build_sequence_model(
        model_name,
        input_dim=train_dataset[0]["x"].shape[1],
        use_country_embedding=use_country_embedding,
        num_countries=num_countries,
    )
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    device = get_preferred_torch_device()
    _configure_torch_cpu_threads(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = _build_loader(train_dataset, batch_size=256, shuffle=True)
    train_started_at = reporter.now()
    reporter.log(
        progress_prefix,
        (
            f"final training start | model={model_name} | device={device.type} "
            f"| train_samples={len(train_dataset)} | batches={len(train_loader)} | epochs={max(epochs, 1)}"
        ),
    )

    train_y = train_dataset.metadata["y_true"].to_numpy(dtype=np.float32)
    positives = float(train_y.sum())
    negatives = float(train_y.size - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_epochs = max(epochs, 1)
    for epoch in range(1, total_epochs + 1):
        epoch_started_at = reporter.now()
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["x"].to(device), batch["country_idx"].to(device))
            loss = loss_fn(logits, batch["y"].to(device))
            loss.backward()
            optimizer.step()
            batch_size = int(batch["y"].shape[0])
            epoch_loss_sum += float(loss.item()) * batch_size
            epoch_sample_count += batch_size
        epoch_elapsed = reporter.now() - epoch_started_at
        reporter.log(
            progress_prefix,
            (
                f"epoch {epoch}/{total_epochs} | train_loss={epoch_loss_sum / max(epoch_sample_count, 1):.4f} "
                f"| epoch={format_duration(epoch_elapsed)} "
                f"| elapsed={format_duration(reporter.now() - train_started_at)} "
                f"| speed={format_rate(epoch_sample_count, epoch_elapsed)} "
                f"| eta={format_duration(estimate_remaining_seconds(loop_started_at=train_started_at, completed_steps=epoch, total_steps=total_epochs, now=reporter.now()))}"
            ),
        )
    reporter.log(
        progress_prefix,
        f"final training completed | elapsed={format_duration(reporter.now() - train_started_at)}",
    )
    return model


def predict_sequence_model(model: nn.Module, dataset) -> np.ndarray:
    return _predict_with_model(model, dataset)
