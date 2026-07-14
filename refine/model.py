from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from .metrics import classification_metrics


@dataclass
class ReFINEConfig:
    keep_frac: float = 0.8
    repeats: int = 100
    max_iter: int = 100
    n_splits: int = 5
    auc_stop: float = 0.5
    random_state: int | None = 42
    batch_size: int = 100_000
    n_estimators: int = 500
    max_depth: int | None = 6
    n_jobs: int = -1


def build_rf(config: ReFINEConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )


def predict_proba_in_batches(model: RandomForestClassifier, x: np.ndarray, batch_size: int) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size]
        if len(batch):
            scores.append(model.predict_proba(batch)[:, 1])
    return np.concatenate(scores) if scores else np.array([])


def run_refine_rf(
    pos_features: pd.DataFrame,
    neg_features: pd.DataFrame,
    config: ReFINEConfig,
    metadata: pd.DataFrame | None = None,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Run ReFINE using the Random Forest setup from the paper.

    Returns one retained-negative table per iteration and a long CV metrics table.
    """
    if not 0 < config.keep_frac < 1:
        raise ValueError("keep_frac must be between 0 and 1.")
    if len(pos_features) == 0 or len(neg_features) == 0:
        raise ValueError("Positive and negative feature tables must be non-empty.")

    pos = pos_features.reset_index(drop=True)
    neg = neg_features.copy()
    rng = np.random.default_rng(config.random_state)
    retained_by_iter: list[pd.DataFrame] = []
    cv_rows: list[dict[str, float | int]] = []

    for iteration in range(config.max_iter):
        if len(neg) <= len(pos):
            break

        all_scores = []
        auc_sum = 0.0
        neg_x = neg.to_numpy()

        for repeat in range(config.repeats):
            sampled_idx = rng.choice(neg.index.to_numpy(), size=len(pos), replace=False)
            sampled_neg = neg.loc[sampled_idx]
            x = pd.concat([pos, sampled_neg.reset_index(drop=True)], axis=0).to_numpy()
            y = np.repeat([1, 0], len(pos))

            y_score = np.array([])
            y_true = np.array([])
            kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=None if config.random_state is None else config.random_state + repeat)
            for train_index, test_index in kf.split(x):
                model = build_rf(config)
                model.fit(x[train_index], y[train_index])
                y_score = np.hstack([y_score, model.predict_proba(x[test_index])[:, 1]])
                y_true = np.hstack([y_true, y[test_index]])

            metrics = classification_metrics(y_true, y_score)
            metrics.update({"iteration": iteration, "repeat": repeat})
            cv_rows.append(metrics)
            auc_sum += float(metrics["auc"])

            model = build_rf(config)
            model.fit(x, y)
            scores = pd.DataFrame(
                {"proba": predict_proba_in_batches(model, neg_x, config.batch_size)},
                index=neg.index,
            )
            scores = scores.loc[~scores.index.isin(sampled_idx)]
            all_scores.append(scores)

        mean_auc = auc_sum / config.repeats
        ranked = pd.concat(all_scores).groupby(level=0).mean().sort_values("proba", ascending=False)
        remove_n = int(len(ranked) * (1 - config.keep_frac))
        if remove_n <= 0:
            break

        retained_scores = ranked.iloc[:-remove_n].copy()
        retained_scores["iter"] = iteration
        if metadata is not None:
            retained = metadata.loc[retained_scores.index].copy()
            retained["proba"] = retained_scores["proba"].values
            retained["iter"] = iteration
        else:
            retained = retained_scores
        retained_by_iter.append(retained)

        neg = neg.loc[retained_scores.index]
        if mean_auc < config.auc_stop:
            break

    return retained_by_iter, pd.DataFrame(cv_rows)

