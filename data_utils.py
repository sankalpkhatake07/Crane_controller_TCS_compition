"""
Shared dataset utilities for crane LSTM training and inference.

Centralising the episode split logic guarantees that train_lstm.py,
train_risk_lstm.py, and early_warning.py all operate on identical
splits, preventing any accidental data leakage.
"""

import numpy as np


# Reproducible split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# Test ratio is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.15


def get_episode_splits(episode_ids, seed=42):
    """
    Shuffle and split episode IDs into train / val / test sets.

    Entire episodes stay within a single split to prevent leakage
    between overlapping sliding windows.

    Args:
        episode_ids : array-like of episode ID values
        seed        : random seed for reproducible shuffle

    Returns:
        train_ids, val_ids, test_ids : numpy arrays of episode IDs
    """

    episode_ids = np.array(sorted(episode_ids))

    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    n = len(episode_ids)
    train_end = int(TRAIN_RATIO * n)
    val_end   = int((TRAIN_RATIO + VAL_RATIO) * n)

    train_ids = episode_ids[:train_end]
    val_ids   = episode_ids[train_end:val_end]
    test_ids  = episode_ids[val_end:]

    return train_ids, val_ids, test_ids


def build_episode_store(split_df, feature_columns, target_column,
                        feature_mean, feature_std,
                        target_mean, target_std):
    """
    Pre-compute and normalise all episodes in a split dataframe.

    Converting episodes to NumPy arrays once avoids repeated
    pandas overhead during Dataset.__getitem__ calls.

    Args:
        split_df        : DataFrame for one split (train/val/test)
        feature_columns : list of input feature column names
        target_column   : name of the target column
        feature_mean    : np.ndarray of per-feature means (train-fit)
        feature_std     : np.ndarray of per-feature stds  (train-fit)
        target_mean     : float, target mean (train-fit)
        target_std      : float, target std  (train-fit)

    Returns:
        dict mapping episode_id (int) ->
            {"features": np.ndarray float32,
             "target":   np.ndarray float32}
    """

    store = {}

    for episode_id, group in split_df.groupby(
        "episode_id",
        sort=False
    ):
        group = group.sort_values("step")

        features = (
            group[feature_columns]
            .to_numpy(dtype=np.float32)
        )

        target = (
            group[target_column]
            .to_numpy(dtype=np.float32)
        )

        # Z-score normalisation (train statistics only)
        features = (features - feature_mean) / feature_std
        target   = (target   - target_mean)  / target_std

        store[int(episode_id)] = {
            "features": features.astype(np.float32),
            "target":   target.astype(np.float32)
        }

    return store


def fit_scalers(train_df, feature_columns, target_column):
    """
    Compute mean and std from training data only.

    Returns:
        feature_mean, feature_std, target_mean, target_std
    """

    feature_mean = (
        train_df[feature_columns].mean().to_numpy(dtype=np.float32)
    )
    feature_std = (
        train_df[feature_columns].std().to_numpy(dtype=np.float32)
    )
    feature_std = np.where(
        feature_std < 1e-8, 1.0, feature_std
    ).astype(np.float32)

    target_mean = float(train_df[target_column].mean())
    target_std  = float(train_df[target_column].std())
    if target_std < 1e-8:
        target_std = 1.0

    return feature_mean, feature_std, target_mean, target_std
