import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import CraneRiskLSTM
from data_utils import get_episode_splits, build_episode_store, fit_scalers


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_FILE = "crane_sensor_dataset.csv"

MODEL_FILE  = "crane_risk_lstm_model.pth"
SCALER_FILE = "crane_risk_lstm_scalers.pkl"

SEED = 42

INPUT_STEPS    = 250   # 5 seconds history
FORECAST_STEPS = 50    # next 1 second forecast

FEATURE_COLUMNS = [
    "wind_velocity",
    "wind_force",
    "structural_tilt",
    "tilt_rate",
    "tilt_acceleration",
    "load_mass",
    "trolley_position",
    "swing_angle",
    "swing_angular_velocity",
    "stability_margin"
]

TARGET_COLUMN = "margin_ratio"

# LSTM architecture (upgraded from hidden=64)
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2

# Training
BATCH_SIZE    = 64
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3

# Early stopping
PATIENCE = 10

# Stride: 2 for train (speed), 1 for val/test (exact evaluation)
TRAIN_STRIDE = 2
EVAL_STRIDE  = 1


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 72)
print("CRANE FUTURE STABILITY-RISK FORECASTING - LSTM")
print("=" * 72)
print(f"\nDevice: {device}")
print(f"LSTM hidden size: {HIDDEN_SIZE}")
print(f"Max epochs: {NUM_EPOCHS}  |  Early stopping patience: {PATIENCE}")


# =========================================================
# LOAD AND SORT DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)
df = df.sort_values(["episode_id", "step"]).reset_index(drop=True)

print(f"Rows loaded    : {len(df):,}")
print(f"Episodes loaded: {df['episode_id'].nunique()}")


# =========================================================
# CREATE PHYSICS-DEFINED TARGET
# =========================================================

df[TARGET_COLUMN] = df["stability_margin"] / df["restoring_moment"]

print("\nMargin-ratio target range:")
print(f"  Min : {df[TARGET_COLUMN].min():.6f}")
print(f"  Max : {df[TARGET_COLUMN].max():.6f}")
print(f"  Mean: {df[TARGET_COLUMN].mean():.6f}")


# =========================================================
# EPISODE SPLIT (via shared utility)
# =========================================================

train_ids, val_ids, test_ids = get_episode_splits(
    df["episode_id"].unique(),
    seed=SEED
)

print("\nEpisode split:")
print(f"  Train     : {len(train_ids)}")
print(f"  Validation: {len(val_ids)}")
print(f"  Test      : {len(test_ids)}")

train_df = df[df["episode_id"].isin(train_ids)].copy()
val_df   = df[df["episode_id"].isin(val_ids)].copy()
test_df  = df[df["episode_id"].isin(test_ids)].copy()


# =========================================================
# FIT SCALERS ON TRAINING DATA ONLY
# =========================================================

feature_mean, feature_std, target_mean, target_std = fit_scalers(
    train_df, FEATURE_COLUMNS, TARGET_COLUMN
)


# =========================================================
# SAVE SCALERS
# =========================================================

scaler_data = {
    "feature_columns": FEATURE_COLUMNS,
    "target_column":   TARGET_COLUMN,
    "feature_mean":    feature_mean,
    "feature_std":     feature_std,
    "target_mean":     target_mean,
    "target_std":      target_std,
    "input_steps":     INPUT_STEPS,
    "forecast_steps":  FORECAST_STEPS,
    "dt":              0.02
}

with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler_data, f)

print(f"\nScaler saved: {SCALER_FILE}")


# =========================================================
# BUILD EPISODE STORES
# =========================================================

print("\nBuilding episode stores...")

train_store = build_episode_store(
    train_df, FEATURE_COLUMNS, TARGET_COLUMN,
    feature_mean, feature_std, target_mean, target_std
)

val_store = build_episode_store(
    val_df, FEATURE_COLUMNS, TARGET_COLUMN,
    feature_mean, feature_std, target_mean, target_std
)

test_store = build_episode_store(
    test_df, FEATURE_COLUMNS, TARGET_COLUMN,
    feature_mean, feature_std, target_mean, target_std
)


# =========================================================
# SLIDING WINDOW DATASET
# =========================================================

class CraneRiskWindowDataset(Dataset):

    def __init__(self, episode_store, input_steps, forecast_steps, stride=1):
        self.episode_store  = episode_store
        self.input_steps    = input_steps
        self.forecast_steps = forecast_steps
        self.window_index   = []

        for episode_id, ep in episode_store.items():
            max_start = len(ep["target"]) - input_steps - forecast_steps
            if max_start < 0:
                continue
            for s in range(0, max_start + 1, stride):
                self.window_index.append((episode_id, s))

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        episode_id, start = self.window_index[idx]
        ep = self.episode_store[episode_id]
        end     = start + self.input_steps
        tgt_end = end   + self.forecast_steps
        x = ep["features"][start:end]
        y = ep["target"][end:tgt_end]
        return torch.from_numpy(x), torch.from_numpy(y)


train_dataset = CraneRiskWindowDataset(train_store, INPUT_STEPS, FORECAST_STEPS, TRAIN_STRIDE)
val_dataset   = CraneRiskWindowDataset(val_store,   INPUT_STEPS, FORECAST_STEPS, EVAL_STRIDE)
test_dataset  = CraneRiskWindowDataset(test_store,  INPUT_STEPS, FORECAST_STEPS, EVAL_STRIDE)

print("\nWindow counts:")
print(f"  Train     : {len(train_dataset):,}")
print(f"  Validation: {len(val_dataset):,}")
print(f"  Test      : {len(test_dataset):,}")


# =========================================================
# DATA LOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)


# =========================================================
# MODEL
# =========================================================

model = CraneRiskLSTM(
    input_size=len(FEATURE_COLUMNS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    forecast_steps=FORECAST_STEPS,
    dropout=DROPOUT
).to(device)

print("\nModel:")
print(model)


# =========================================================
# LOSS, OPTIMIZER, SCHEDULER
# =========================================================

def huber_mse_loss(pred, target, alpha=0.1):
    """MSE + alpha * MAE (robust to rare tipping outliers)."""
    return (
        nn.functional.mse_loss(pred, target)
        + alpha * nn.functional.l1_loss(pred, target)
    )


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# =========================================================
# TRAIN / EVAL FUNCTIONS
# =========================================================

def train_one_epoch():
    model.train()
    total_loss, n = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = huber_mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n          += x.size(0)
    return total_loss / n


def evaluate_loss(loader):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = huber_mse_loss(model(x), y)
            total_loss += loss.item() * x.size(0)
            n          += x.size(0)
    return total_loss / n


# =========================================================
# TRAINING LOOP WITH EARLY STOPPING
# =========================================================

print("\n" + "=" * 72)
print("RISK LSTM TRAINING STARTED")
print("=" * 72)

best_val_loss     = float("inf")
epochs_no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):

    train_loss = train_one_epoch()
    val_loss   = evaluate_loss(val_loader)
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train: {train_loss:.6f} | "
        f"Val: {val_loss:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        epochs_no_improve = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "input_size":       len(FEATURE_COLUMNS),
            "hidden_size":      HIDDEN_SIZE,
            "num_layers":       NUM_LAYERS,
            "forecast_steps":   FORECAST_STEPS,
            "dropout":          DROPOUT,
            "feature_columns":  FEATURE_COLUMNS,
            "target_column":    TARGET_COLUMN
        }, MODEL_FILE)

        print(f"  -> Best risk model saved (val={best_val_loss:.6f})")

    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break


# =========================================================
# FINAL TEST EVALUATION
# =========================================================

checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss = evaluate_loss(test_loader)

print("\n" + "=" * 72)
print("RISK LSTM TRAINING COMPLETE")
print("=" * 72)
print(f"Best validation loss : {best_val_loss:.6f}")
print(f"Test loss            : {test_loss:.6f}")
print(f"Model saved          : {MODEL_FILE}")
print(f"Scaler saved         : {SCALER_FILE}")
print("\nForecast meaning:")
print("  Output index 24 = predicted margin ratio at +500 ms")
print("  Output index 49 = predicted margin ratio at +1.0 s")
print("\nPhysics interpretation:")
print("  margin_ratio > 0 = restoring reserve remains")
print("  margin_ratio = 0 = tipping threshold")
print("  margin_ratio < 0 = predicted threshold crossing")
