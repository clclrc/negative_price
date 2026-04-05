from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - environment dependent
    lgb = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - environment dependent
    CatBoostClassifier = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - environment dependent
    xgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    DataLoader = None

from .metrics import safe_average_precision
from .progress import ProgressReporter, estimate_remaining_seconds, format_duration, format_metric, format_rate
from .runtime import get_cpu_worker_count


HAS_LIGHTGBM = lgb is not None
HAS_CATBOOST = CatBoostClassifier is not None
HAS_XGBOOST = xgb is not None
HAS_TORCH = torch is not None


class DependencyUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TorchTrainingOutcome:
    state_dict: dict[str, object]
    best_epoch: int
    best_score: float


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: str
    model: object | None = None


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


def require_lightgbm() -> None:
    if not HAS_LIGHTGBM:
        raise DependencyUnavailableError(
            "lightgbm is required for the LightGBM model. Install it with `python3 -m pip install lightgbm`."
        )


def require_catboost() -> None:
    if not HAS_CATBOOST:
        raise DependencyUnavailableError(
            "catboost is required for the CatBoost model. Install it with `python3 -m pip install catboost`."
        )


def require_xgboost() -> None:
    if not HAS_XGBOOST:
        raise DependencyUnavailableError(
            "xgboost is required for the XGBoost model. Install it with `python3 -m pip install xgboost`."
        )


def require_torch() -> None:
    if not HAS_TORCH:
        raise DependencyUnavailableError(
            "torch is required for GRU/TCN/PatchTST models. Install it with `python3 -m pip install torch`."
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


def fit_lightgbm_classifier(
    train_X: np.ndarray,
    train_y: np.ndarray,
    *,
    num_leaves: int,
    learning_rate: float,
    n_estimators: int,
    scale_pos_weight: float,
    seed: int,
) -> object:
    require_lightgbm()
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=get_cpu_worker_count(max_workers=8),
        verbosity=-1,
    )
    model.fit(train_X, train_y)
    return model


def predict_lightgbm(model: object, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)


def fit_catboost_classifier(
    train_X: np.ndarray,
    train_y: np.ndarray,
    *,
    depth: int,
    learning_rate: float,
    n_estimators: int,
    scale_pos_weight: float,
    seed: int,
) -> object:
    require_catboost()
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=n_estimators,
        depth=depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_seed=seed,
        thread_count=get_cpu_worker_count(max_workers=8),
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(train_X, train_y)
    return model


def predict_catboost(model: object, X: np.ndarray) -> np.ndarray:
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


def fit_probability_calibrator(y_true: np.ndarray, y_prob: np.ndarray, *, method: str) -> ProbabilityCalibrator:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float32)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return ProbabilityCalibrator(method="identity")

    clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    if method == "sigmoid":
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)
        model.fit(logits, y_true)
        return ProbabilityCalibrator(method=method, model=model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(clipped, y_true)
        return ProbabilityCalibrator(method=method, model=model)
    if method == "identity":
        return ProbabilityCalibrator(method=method)
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_probability_calibrator(calibrator: ProbabilityCalibrator, y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob).astype(np.float32)
    clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    if calibrator.method == "identity" or calibrator.model is None:
        return clipped.astype(np.float32)
    if calibrator.method == "sigmoid":
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return calibrator.model.predict_proba(logits)[:, 1].astype(np.float32)
    if calibrator.method == "isotonic":
        return calibrator.model.predict(clipped).astype(np.float32)
    raise ValueError(f"Unsupported calibration method: {calibrator.method}")


if HAS_TORCH:
    class BinaryFocalLossWithLogits(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(self, *, gamma: float = 2.0, pos_weight: "torch.Tensor | None" = None) -> None:
            super().__init__()
            self.gamma = gamma
            self.pos_weight = pos_weight

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            targets = targets.float()
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight,
                reduction="none",
            )
            probs = torch.sigmoid(logits)
            pt = torch.where(targets > 0.5, probs, 1.0 - probs)
            focal_factor = (1.0 - pt).pow(self.gamma)
            return (focal_factor * bce).mean()


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

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
        ) -> torch.Tensor:
            _, hidden = self.gru(x)
            features = hidden[-1]
            if self.use_country_embedding:
                embedded = self.country_embedding(country_idx)
                features = torch.cat([features, embedded], dim=1)
            return self.classifier(features).squeeze(1)


    class GRUHybridClassifier(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(
            self,
            *,
            input_dim: int,
            tabular_dim: int,
            use_country_embedding: bool,
            num_countries: int,
            embedding_dim: int = 8,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if tabular_dim <= 0:
                raise ValueError("GRUHybridClassifier requires tabular_dim > 0.")
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
                sequence_dim = hidden_size + embedding_dim
            else:
                self.country_embedding = None
                sequence_dim = hidden_size
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(sequence_dim + 64, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if tabular_x is None:
                raise ValueError("GRUHybridClassifier requires tabular_x.")
            _, hidden = self.gru(x)
            sequence_features = hidden[-1]
            if self.use_country_embedding:
                embedded = self.country_embedding(country_idx)
                sequence_features = torch.cat([sequence_features, embedded], dim=1)
            tabular_features = self.tabular_encoder(tabular_x)
            features = torch.cat([sequence_features, tabular_features], dim=1)
            return self.classifier(features).squeeze(1)


    class TemporalAttentionPooling(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.score = nn.Linear(hidden_size, 1)

        def forward(self, sequence_states: torch.Tensor) -> torch.Tensor:
            weights = torch.softmax(self.score(sequence_states).squeeze(-1), dim=1)
            return torch.sum(sequence_states * weights.unsqueeze(-1), dim=1)


    class GRUHybridAttnClassifier(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(
            self,
            *,
            input_dim: int,
            tabular_dim: int,
            use_country_embedding: bool,
            num_countries: int,
            embedding_dim: int = 8,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if tabular_dim <= 0:
                raise ValueError("GRUHybridAttnClassifier requires tabular_dim > 0.")
            self.use_country_embedding = use_country_embedding
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attention_pool = TemporalAttentionPooling(hidden_size)
            if use_country_embedding:
                self.country_embedding = nn.Embedding(num_countries, embedding_dim)
                sequence_dim = hidden_size + embedding_dim
            else:
                self.country_embedding = None
                sequence_dim = hidden_size
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(sequence_dim + 64, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if tabular_x is None:
                raise ValueError("GRUHybridAttnClassifier requires tabular_x.")
            sequence_states, _ = self.gru(x)
            sequence_features = self.attention_pool(sequence_states)
            if self.use_country_embedding:
                embedded = self.country_embedding(country_idx)
                sequence_features = torch.cat([sequence_features, embedded], dim=1)
            tabular_features = self.tabular_encoder(tabular_x)
            features = torch.cat([sequence_features, tabular_features], dim=1)
            return self.classifier(features).squeeze(1)


    class GRUHybridGatedClassifier(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(
            self,
            *,
            input_dim: int,
            tabular_dim: int,
            use_country_embedding: bool,
            num_countries: int,
            embedding_dim: int = 8,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if tabular_dim <= 0:
                raise ValueError("GRUHybridGatedClassifier requires tabular_dim > 0.")
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
                sequence_dim = hidden_size + embedding_dim
            else:
                self.country_embedding = None
                sequence_dim = hidden_size
            self.sequence_projection = nn.Sequential(
                nn.Linear(sequence_dim, 64),
                nn.ReLU(),
            )
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(sequence_dim + 64, 64),
                nn.Sigmoid(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def fused_features(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor,
        ) -> torch.Tensor:
            _, hidden = self.gru(x)
            sequence_features = hidden[-1]
            if self.use_country_embedding:
                embedded = self.country_embedding(country_idx)
                sequence_features = torch.cat([sequence_features, embedded], dim=1)
            tabular_features = self.tabular_encoder(tabular_x)
            projected_sequence = self.sequence_projection(sequence_features)
            gate = self.fusion_gate(torch.cat([sequence_features, tabular_features], dim=1))
            return gate * projected_sequence + (1.0 - gate) * tabular_features

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if tabular_x is None:
                raise ValueError("GRUHybridGatedClassifier requires tabular_x.")
            fused = self.fused_features(x, country_idx, tabular_x)
            return self.classifier(fused).squeeze(1)


    class GRUHybridGatedMultiTaskClassifier(GRUHybridGatedClassifier):  # pragma: no cover - exercised through integration tests
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.aux_head = nn.Linear(64, 1)

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if tabular_x is None:
                raise ValueError("GRUHybridGatedMultiTaskClassifier requires tabular_x.")
            fused = self.fused_features(x, country_idx, tabular_x)
            logits = self.classifier(fused).squeeze(1)
            aux = self.aux_head(fused).squeeze(1)
            return logits, aux


    class DenseMarketGraphLayer(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(self, hidden_size: int, dropout: float = 0.2) -> None:
            super().__init__()
            self.query = nn.Linear(hidden_size, hidden_size, bias=False)
            self.key = nn.Linear(hidden_size, hidden_size, bias=False)
            self.value = nn.Linear(hidden_size, hidden_size, bias=False)
            self.output = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size)
            self.scale = hidden_size**0.5

        def forward(self, node_states: torch.Tensor, market_valid: torch.Tensor | None = None) -> torch.Tensor:
            query = self.query(node_states)
            key = self.key(node_states)
            value = self.value(node_states)
            scores = torch.matmul(query, key.transpose(1, 2)) / self.scale
            if market_valid is not None:
                node_mask = market_valid > 0.5
                pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
                scores = scores.masked_fill(~pair_mask, -1e9)
            attention = torch.softmax(scores, dim=-1)
            if market_valid is not None:
                valid = market_valid.unsqueeze(1)
                attention = attention * valid
                attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            updated = torch.matmul(self.dropout(attention), value)
            return self.norm(node_states + self.output(updated))


    class MultiMarketSequenceEncoder(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(
            self,
            *,
            input_dim: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
            _, hidden = self.gru(x)
            return hidden[-1]

        def encode_markets(self, market_x: torch.Tensor) -> torch.Tensor:
            batch_size, num_markets, seq_len, input_dim = market_x.shape
            flat = market_x.view(batch_size * num_markets, seq_len, input_dim)
            encoded = self.encode_sequence(flat)
            return encoded.view(batch_size, num_markets, self.hidden_size)

        @staticmethod
        def masked_mean(states: torch.Tensor, market_valid: torch.Tensor | None) -> torch.Tensor:
            if market_valid is None:
                return states.mean(dim=1)
            weights = market_valid.unsqueeze(-1)
            return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)


    class GRUMultiMarketClassifier(MultiMarketSequenceEncoder):  # pragma: no cover - exercised through integration tests
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
            super().__init__(
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.use_country_embedding = use_country_embedding
            if use_country_embedding:
                self.country_embedding = nn.Embedding(num_countries, embedding_dim)
                classifier_in = hidden_size * 2 + embedding_dim
            else:
                self.country_embedding = None
                classifier_in = hidden_size * 2
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
            market_x: torch.Tensor | None = None,
            market_valid: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if market_x is None:
                raise ValueError("GRUMultiMarketClassifier requires market_x.")
            target_features = self.encode_sequence(x)
            market_states = self.encode_markets(market_x)
            global_features = self.masked_mean(market_states, market_valid)
            features = [target_features, global_features]
            if self.use_country_embedding:
                features.append(self.country_embedding(country_idx))
            return self.classifier(torch.cat(features, dim=1)).squeeze(1)


    class GraphTemporalClassifier(MultiMarketSequenceEncoder):  # pragma: no cover - exercised through integration tests
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
            super().__init__(
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.use_country_embedding = use_country_embedding
            self.graph_layer = DenseMarketGraphLayer(hidden_size, dropout=dropout)
            if use_country_embedding:
                self.country_embedding = nn.Embedding(num_countries, embedding_dim)
                fusion_in = hidden_size * 3 + embedding_dim
            else:
                self.country_embedding = None
                fusion_in = hidden_size * 3
            self.classifier = nn.Sequential(
                nn.Linear(fusion_in, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def graph_features(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            market_x: torch.Tensor,
            market_valid: torch.Tensor | None,
        ) -> torch.Tensor:
            target_features = self.encode_sequence(x)
            market_states = self.encode_markets(market_x)
            graph_states = self.graph_layer(market_states, market_valid)
            target_graph = graph_states[torch.arange(graph_states.shape[0], device=graph_states.device), country_idx.clamp(min=0)]
            global_graph = self.masked_mean(graph_states, market_valid)
            features = [target_features, target_graph, global_graph]
            if self.use_country_embedding:
                features.append(self.country_embedding(country_idx))
            return torch.cat(features, dim=1)

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
            market_x: torch.Tensor | None = None,
            market_valid: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if market_x is None:
                raise ValueError("GraphTemporalClassifier requires market_x.")
            features = self.graph_features(x, country_idx, market_x, market_valid)
            return self.classifier(features).squeeze(1)


    class GraphTemporalHybridClassifier(GraphTemporalClassifier):  # pragma: no cover - exercised through integration tests
        def __init__(
            self,
            *,
            input_dim: int,
            tabular_dim: int,
            use_country_embedding: bool,
            num_countries: int,
            embedding_dim: int = 8,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            if tabular_dim <= 0:
                raise ValueError("GraphTemporalHybridClassifier requires tabular_dim > 0.")
            super().__init__(
                input_dim=input_dim,
                use_country_embedding=use_country_embedding,
                num_countries=num_countries,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            fusion_in = self.classifier[0].in_features
            self.graph_projection = nn.Sequential(
                nn.Linear(fusion_in, 64),
                nn.ReLU(),
            )
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(fusion_in + 64, 64),
                nn.Sigmoid(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
            market_x: torch.Tensor | None = None,
            market_valid: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if market_x is None:
                raise ValueError("GraphTemporalHybridClassifier requires market_x.")
            if tabular_x is None:
                raise ValueError("GraphTemporalHybridClassifier requires tabular_x.")
            graph_features = self.graph_features(x, country_idx, market_x, market_valid)
            projected_graph = self.graph_projection(graph_features)
            tabular_features = self.tabular_encoder(tabular_x)
            gate = self.fusion_gate(torch.cat([graph_features, tabular_features], dim=1))
            fused = gate * projected_graph + (1.0 - gate) * tabular_features
            return self.classifier(fused).squeeze(1)


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

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
        ) -> torch.Tensor:
            features = self.network(x.transpose(1, 2)).mean(dim=2)
            if self.use_country_embedding:
                embedded = self.country_embedding(country_idx)
                features = torch.cat([features, embedded], dim=1)
            return self.classifier(features).squeeze(1)


    class PatchTSTClassifier(nn.Module):  # pragma: no cover - exercised through integration tests
        def __init__(
            self,
            *,
            input_dim: int,
            use_country_embedding: bool,
            num_countries: int,
            patch_len: int = 6,
            stride: int = 6,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
            embedding_dim: int = 8,
            max_tokens: int = 64,
        ) -> None:
            super().__init__()
            self.use_country_embedding = use_country_embedding
            self.patch_len = patch_len
            self.stride = stride
            self.input_projection = nn.Linear(input_dim * patch_len, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.position_embedding = nn.Parameter(torch.zeros(1, max_tokens + 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            if use_country_embedding:
                self.country_embedding = nn.Embedding(num_countries, embedding_dim)
                classifier_in = d_model + embedding_dim
            else:
                self.country_embedding = None
                classifier_in = d_model
            self.classifier = nn.Sequential(
                nn.LayerNorm(classifier_in),
                nn.Linear(classifier_in, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def _patchify(self, x: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, feature_dim = x.shape
            if seq_len < self.patch_len:
                pad = self.patch_len - seq_len
                x = torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
                seq_len = x.shape[1]
            remainder = (seq_len - self.patch_len) % self.stride
            if remainder != 0:
                pad = self.stride - remainder
                x = torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
            patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            return patches.contiguous().view(batch_size, -1, self.patch_len * feature_dim)

        def forward(
            self,
            x: torch.Tensor,
            country_idx: torch.Tensor,
            tabular_x: torch.Tensor | None = None,
        ) -> torch.Tensor:
            patches = self._patchify(x)
            tokens = self.input_projection(patches)
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
            encoded = self.encoder(tokens)
            features = encoded[:, 0, :]
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
    tabular_dim: int = 0,
) -> nn.Module:
    require_torch()
    if model_name == "GRU":
        return GRUClassifier(
            input_dim=input_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "GRUHybrid":
        return GRUHybridClassifier(
            input_dim=input_dim,
            tabular_dim=tabular_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "GRUHybridAttn":
        return GRUHybridAttnClassifier(
            input_dim=input_dim,
            tabular_dim=tabular_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "GRUHybridGated":
        return GRUHybridGatedClassifier(
            input_dim=input_dim,
            tabular_dim=tabular_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "GRUHybridGatedMultiTask":
        return GRUHybridGatedMultiTaskClassifier(
            input_dim=input_dim,
            tabular_dim=tabular_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "GRUMultiMarket":
        return GRUMultiMarketClassifier(
            input_dim=input_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "GraphTemporal":
        return GraphTemporalClassifier(
            input_dim=input_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "GraphTemporalHybrid":
        return GraphTemporalHybridClassifier(
            input_dim=input_dim,
            tabular_dim=tabular_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "TCN":
        return TCNClassifier(
            input_dim=input_dim,
            use_country_embedding=use_country_embedding,
            num_countries=num_countries,
        )
    if model_name == "PatchTST":
        return PatchTSTClassifier(
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
    tabular_dim: int = 0,
) -> nn.Module:
    model = build_sequence_model(
        model_name,
        input_dim=input_dim,
        use_country_embedding=use_country_embedding,
        num_countries=num_countries,
        tabular_dim=tabular_dim,
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


def _split_model_outputs(output):
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def _forward_model_outputs(model: nn.Module, batch: dict[str, torch.Tensor], device: "torch.device") -> tuple["torch.Tensor", "torch.Tensor | None"]:
    kwargs = {
        "x": batch["x"].to(device),
        "country_idx": batch["country_idx"].to(device),
    }
    tabular_x = batch.get("tabular_x")
    if tabular_x is not None:
        kwargs["tabular_x"] = tabular_x.to(device)
    market_x = batch.get("market_x")
    if market_x is not None:
        kwargs["market_x"] = market_x.to(device)
    market_valid = batch.get("market_valid")
    if market_valid is not None:
        kwargs["market_valid"] = market_valid.to(device)
    return _split_model_outputs(model(**kwargs))


def _forward_model(model: nn.Module, batch: dict[str, torch.Tensor], device: "torch.device") -> "torch.Tensor":
    logits, _ = _forward_model_outputs(model, batch, device)
    return logits


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
            logits, _ = _forward_model_outputs(model, batch, device)
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
    loss_name: str = "bce",
    focal_gamma: float = 2.0,
    aux_target: str | None = None,
    aux_weight: float = 0.2,
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
        tabular_dim=int(train_dataset[0]["tabular_x"].shape[0]) if "tabular_x" in train_dataset[0] else 0,
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
            f"| batches={len(train_loader)} | max_epochs={max_epochs} | loss={loss_name}"
            f"{f' | aux_target={aux_target} | aux_weight={aux_weight:.2f}' if aux_target else ''}"
        ),
    )

    train_y = train_dataset.metadata["y_true"].to_numpy(dtype=np.float32)
    positives = float(train_y.sum())
    negatives = float(train_y.size - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    if loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == "focal":
        loss_fn = BinaryFocalLossWithLogits(gamma=focal_gamma, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unsupported sequence loss: {loss_name}")
    aux_loss_fn = nn.SmoothL1Loss() if aux_target else None
    aux_mean = 0.0
    aux_scale = 1.0
    if aux_target == "target_price":
        aux_values = train_dataset.metadata["target_price"].to_numpy(dtype=np.float32)
        aux_mean = float(np.nanmean(aux_values))
        aux_scale = float(np.nanstd(aux_values))
        if aux_scale == 0.0 or np.isnan(aux_scale):
            aux_scale = 1.0
    elif aux_target is not None:
        raise ValueError(f"Unsupported sequence auxiliary target: {aux_target}")

    best_score = float("-inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        epoch_started_at = reporter.now()
        model.train()
        epoch_loss_sum = 0.0
        epoch_aux_loss_sum = 0.0
        epoch_sample_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, aux_pred = _forward_model_outputs(model, batch, device)
            loss = loss_fn(logits, batch["y"].to(device))
            if aux_loss_fn is not None:
                if aux_pred is None or "aux_y" not in batch:
                    raise ValueError("Auxiliary training requires both auxiliary model output and batch aux_y values.")
                aux_target_tensor = (batch["aux_y"].to(device) - aux_mean) / aux_scale
                aux_loss = aux_loss_fn(aux_pred, aux_target_tensor)
                loss = loss + (aux_weight * aux_loss)
                epoch_aux_loss_sum += float(aux_loss.item()) * int(batch["y"].shape[0])
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
                f"{f'| aux_loss={epoch_aux_loss_sum / max(epoch_sample_count, 1):.4f} ' if aux_loss_fn is not None else ''}"
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
    loss_name: str = "bce",
    focal_gamma: float = 2.0,
    aux_target: str | None = None,
    aux_weight: float = 0.2,
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
        tabular_dim=int(train_dataset[0]["tabular_x"].shape[0]) if "tabular_x" in train_dataset[0] else 0,
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
            f"| train_samples={len(train_dataset)} | batches={len(train_loader)} "
            f"| epochs={max(epochs, 1)} | loss={loss_name}"
            f"{f' | aux_target={aux_target} | aux_weight={aux_weight:.2f}' if aux_target else ''}"
        ),
    )

    train_y = train_dataset.metadata["y_true"].to_numpy(dtype=np.float32)
    positives = float(train_y.sum())
    negatives = float(train_y.size - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    if loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == "focal":
        loss_fn = BinaryFocalLossWithLogits(gamma=focal_gamma, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unsupported sequence loss: {loss_name}")
    aux_loss_fn = nn.SmoothL1Loss() if aux_target else None
    aux_mean = 0.0
    aux_scale = 1.0
    if aux_target == "target_price":
        aux_values = train_dataset.metadata["target_price"].to_numpy(dtype=np.float32)
        aux_mean = float(np.nanmean(aux_values))
        aux_scale = float(np.nanstd(aux_values))
        if aux_scale == 0.0 or np.isnan(aux_scale):
            aux_scale = 1.0
    elif aux_target is not None:
        raise ValueError(f"Unsupported sequence auxiliary target: {aux_target}")

    total_epochs = max(epochs, 1)
    for epoch in range(1, total_epochs + 1):
        epoch_started_at = reporter.now()
        model.train()
        epoch_loss_sum = 0.0
        epoch_aux_loss_sum = 0.0
        epoch_sample_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, aux_pred = _forward_model_outputs(model, batch, device)
            loss = loss_fn(logits, batch["y"].to(device))
            if aux_loss_fn is not None:
                if aux_pred is None or "aux_y" not in batch:
                    raise ValueError("Auxiliary training requires both auxiliary model output and batch aux_y values.")
                aux_target_tensor = (batch["aux_y"].to(device) - aux_mean) / aux_scale
                aux_loss = aux_loss_fn(aux_pred, aux_target_tensor)
                loss = loss + (aux_weight * aux_loss)
                epoch_aux_loss_sum += float(aux_loss.item()) * int(batch["y"].shape[0])
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
                f"{f'| aux_loss={epoch_aux_loss_sum / max(epoch_sample_count, 1):.4f} ' if aux_loss_fn is not None else ''}"
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
