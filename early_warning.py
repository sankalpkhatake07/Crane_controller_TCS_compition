import pickle
import random

import numpy as np
import pandas as pd
import torch

from models import CraneRiskLSTM
from data_utils import get_episode_splits


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_FILE = "crane_sensor_dataset.csv"

# Switched to risk LSTM (predicts margin_ratio directly)
MODEL_FILE  = "crane_risk_lstm_model.pth"
SCALER_FILE = "crane_risk_lstm_scalers.pkl"

OUTPUT_FILE = "early_warning_results.csv"
EVENT_FILE  = "early_warning_events.csv"

SEED = 42

DT             = 0.02
INPUT_STEPS    = 250    # 5 seconds history
CHECKPOINT_INDEX = 24   # +500 ms = 25 future steps
INFERENCE_STRIDE = 5    # evaluate every 5 steps = 100 ms
BATCH_SIZE     = 256


# =========================================================
# WARNING THRESHOLDS
#
# Applied to the predicted margin_ratio at +500 ms.
# These are prototype thresholds for simulation validation.
# NOT certified real-crane safety thresholds.
# =========================================================

WARNING_MARGIN_RATIO  = 0.35
CRITICAL_MARGIN_RATIO = 0.20
IMMINENT_MARGIN_RATIO = 0.08

# Predicted absolute structural tilt at +500 ms
WARNING_TILT_RAD  = np.radians(2.5)
CRITICAL_TILT_RAD = np.radians(3.5)
IMMINENT_TILT_RAD = np.radians(4.5)

# Predicted tilt growth over next 500 ms
WARNING_GROWTH_RAD  = np.radians(0.20)
CRITICAL_GROWTH_RAD = np.radians(0.40)
IMMINENT_GROWTH_RAD = np.radians(0.70)

# Current absolute tilt rate
WARNING_TILT_RATE  = 0.015
CRITICAL_TILT_RATE = 0.030
IMMINENT_TILT_RATE = 0.050


# =========================================================
# HYSTERESIS PARAMETERS
#
# Prevents noisy single-step risk level flickering.
# Escalation requires N consecutive high-risk inferences.
# De-escalation requires M consecutive safe inferences.
# =========================================================

ESCALATE_AFTER   = 2   # consecutive above-threshold steps to escalate
DEESCALATE_AFTER = 5   # consecutive below-threshold steps to de-escalate


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
print("500 ms EARLY-WARNING RISK LAYER  [Risk LSTM + Hysteresis]")
print("=" * 72)
print(f"\nDevice: {device}")
print(f"Hysteresis: escalate after {ESCALATE_AFTER}, de-escalate after {DEESCALATE_AFTER} steps")


# =========================================================
# LOAD SCALERS
# =========================================================

print("\nLoading scalers...")

with open(SCALER_FILE, "rb") as f:
    scaler_data = pickle.load(f)

FEATURE_COLUMNS = scaler_data["feature_columns"]
TARGET_COLUMN   = scaler_data["target_column"]

feature_mean = np.asarray(scaler_data["feature_mean"], dtype=np.float32)
feature_std  = np.asarray(scaler_data["feature_std"],  dtype=np.float32)
target_mean  = float(scaler_data["target_mean"])
target_std   = float(scaler_data["target_std"])
INPUT_STEPS  = int(scaler_data["input_steps"])
FORECAST_STEPS = int(scaler_data["forecast_steps"])

print(f"Input steps   : {INPUT_STEPS}")
print(f"Forecast steps: {FORECAST_STEPS}")
print(f"Target column : {TARGET_COLUMN}")


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading trained Risk LSTM...")

checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)

model = CraneRiskLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    forecast_steps=checkpoint["forecast_steps"],
    dropout=checkpoint["dropout"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Risk LSTM loaded successfully.")


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)
df = df.sort_values(["episode_id", "step"]).reset_index(drop=True)

print(f"Rows    : {len(df):,}")
print(f"Episodes: {df['episode_id'].nunique()}")


# =========================================================
# RECREATE EXACT TEST SPLIT
# Must match train_risk_lstm.py via shared data_utils
# =========================================================

_, _, test_ids = get_episode_splits(
    df["episode_id"].unique(),
    seed=SEED
)

test_df = df[df["episode_id"].isin(test_ids)].copy()

print(f"\nHeld-out test episodes: {len(test_ids)}")
print(f"Held-out test rows    : {len(test_df):,}")


# =========================================================
# RISK CLASSIFICATION
# =========================================================

RISK_ORDER = {
    "SAFE":            0,
    "WARNING":         1,
    "CRITICAL":        2,
    "IMMINENT_TIPPING": 3
}


def classify_risk(
    predicted_margin_ratio_500ms,
    current_margin_ratio,
    current_tilt,
    current_tilt_rate
):
    """
    Hybrid risk classifier.

    Uses the Risk LSTM's predicted margin_ratio at +500 ms as
    the primary signal (instead of the tilt LSTM). This reduces
    prediction chain error since the classifier already operates
    in margin_ratio space.

    Args:
        predicted_margin_ratio_500ms : LSTM forecast at +500 ms
        current_margin_ratio         : current observed margin ratio
        current_tilt                 : current structural tilt (rad)
        current_tilt_rate            : current tilt rate (rad/s)

    Returns:
        (risk_level str, predicted_margin_drop float)
    """

    abs_tilt_rate = abs(current_tilt_rate)

    # Predicted drop in margin ratio over next 500 ms
    predicted_margin_drop = (
        current_margin_ratio
        - predicted_margin_ratio_500ms
    )

    # --------------------------------------------------
    # Immediate physical threshold crossing
    # --------------------------------------------------

    if current_margin_ratio <= 0.0:
        return "IMMINENT_TIPPING", predicted_margin_drop

    # --------------------------------------------------
    # IMMINENT — require severe evidence from 2+ signals
    # --------------------------------------------------

    imminent_score = 0

    if current_margin_ratio <= IMMINENT_MARGIN_RATIO:
        imminent_score += 1

    if predicted_margin_ratio_500ms <= IMMINENT_MARGIN_RATIO:
        imminent_score += 1

    if predicted_margin_drop >= (WARNING_MARGIN_RATIO - IMMINENT_MARGIN_RATIO):
        imminent_score += 1

    if abs_tilt_rate >= IMMINENT_TILT_RATE:
        imminent_score += 1

    if abs(current_tilt) >= IMMINENT_TILT_RAD:
        imminent_score += 1

    if imminent_score >= 2:
        return "IMMINENT_TIPPING", predicted_margin_drop

    # --------------------------------------------------
    # CRITICAL
    # --------------------------------------------------

    critical_score = 0

    if current_margin_ratio <= CRITICAL_MARGIN_RATIO:
        critical_score += 1

    if predicted_margin_ratio_500ms <= CRITICAL_MARGIN_RATIO:
        critical_score += 1

    if predicted_margin_drop >= (WARNING_MARGIN_RATIO - CRITICAL_MARGIN_RATIO):
        critical_score += 1

    if abs_tilt_rate >= CRITICAL_TILT_RATE:
        critical_score += 1

    if abs(current_tilt) >= CRITICAL_TILT_RAD:
        critical_score += 1

    if critical_score >= 2:
        return "CRITICAL", predicted_margin_drop

    # --------------------------------------------------
    # WARNING
    # --------------------------------------------------

    warning_score = 0

    if current_margin_ratio <= WARNING_MARGIN_RATIO:
        warning_score += 1

    if predicted_margin_ratio_500ms <= WARNING_MARGIN_RATIO:
        warning_score += 1

    if predicted_margin_drop > 0.0:
        warning_score += 1

    if abs_tilt_rate >= WARNING_TILT_RATE:
        warning_score += 1

    if abs(current_tilt) >= WARNING_TILT_RAD:
        warning_score += 1

    if warning_score >= 2:
        return "WARNING", predicted_margin_drop

    return "SAFE", predicted_margin_drop


# =========================================================
# HYSTERESIS STATE TRACKER
# =========================================================

class RiskHysteresis:
    """
    Smooths risk level transitions to prevent single-step flickering.

    Escalation   : requires ESCALATE_AFTER consecutive inferences
                   at or above the new level.
    De-escalation: requires DEESCALATE_AFTER consecutive inferences
                   below the current level.
    """

    def __init__(self, escalate_after=2, deescalate_after=5):
        self.escalate_after   = escalate_after
        self.deescalate_after = deescalate_after
        self.current_level    = "SAFE"
        self.pending_level    = "SAFE"
        self.pending_count    = 0
        self.lower_count      = 0

    def reset(self):
        self.current_level = "SAFE"
        self.pending_level = "SAFE"
        self.pending_count = 0
        self.lower_count   = 0

    def update(self, raw_level):
        raw_score     = RISK_ORDER[raw_level]
        current_score = RISK_ORDER[self.current_level]

        if raw_score > current_score:
            # Escalation path
            if raw_level == self.pending_level:
                self.pending_count += 1
            else:
                self.pending_level = raw_level
                self.pending_count = 1
            self.lower_count = 0

            if self.pending_count >= self.escalate_after:
                self.current_level = self.pending_level
                self.pending_count = 0

        elif raw_score < current_score:
            # De-escalation path
            self.lower_count   += 1
            self.pending_count  = 0

            if self.lower_count >= self.deescalate_after:
                self.current_level = raw_level
                self.lower_count   = 0

        else:
            # Same level — reset both counters
            self.pending_count = 0
            self.lower_count   = 0

        return self.current_level


# =========================================================
# BUILD INFERENCE WINDOWS
# =========================================================

print("\nBuilding inference windows...")

inference_entries = []
episode_groups    = {}

for episode_id, group in test_df.groupby("episode_id", sort=False):
    group = group.sort_values("step").reset_index(drop=True)
    episode_groups[int(episode_id)] = group
    episode_length = len(group)

    for current_index in range(
        INPUT_STEPS - 1,
        episode_length,
        INFERENCE_STRIDE
    ):
        start_index = current_index - INPUT_STEPS + 1
        inference_entries.append(
            (int(episode_id), start_index, current_index)
        )

print(f"Inference points: {len(inference_entries):,}")


# =========================================================
# RUN RISK LSTM INFERENCE
# =========================================================

print("\nRunning warning inference...")

result_rows = []

with torch.no_grad():

    for batch_start in range(0, len(inference_entries), BATCH_SIZE):

        batch = inference_entries[batch_start : batch_start + BATCH_SIZE]

        x_list = []
        for (episode_id, start_index, current_index) in batch:
            group = episode_groups[episode_id]
            raw = (
                group[FEATURE_COLUMNS]
                .iloc[start_index : current_index + 1]
                .to_numpy(dtype=np.float32)
            )
            x_list.append(((raw - feature_mean) / feature_std).astype(np.float32))

        x_tensor = torch.from_numpy(np.stack(x_list)).to(device)

        # Predicted normalised margin_ratio sequence
        pred_norm  = model(x_tensor).cpu().numpy()
        pred_ratio = pred_norm * target_std + target_mean   # de-normalise

        for local_i, (episode_id, start_index, current_index) in enumerate(batch):
            group = episode_groups[episode_id]
            row   = group.iloc[current_index]

            predicted_ratio_500ms = float(pred_ratio[local_i, CHECKPOINT_INDEX])
            predicted_ratio_1s    = float(pred_ratio[local_i, -1])

            current_tilt       = float(row["structural_tilt"])
            current_tilt_rate  = float(row["tilt_rate"])
            stability_margin   = float(row["stability_margin"])
            restoring_moment   = float(row["restoring_moment"])
            current_margin_ratio = stability_margin / restoring_moment

            raw_level, margin_drop = classify_risk(
                predicted_ratio_500ms,
                current_margin_ratio,
                current_tilt,
                current_tilt_rate
            )

            result_rows.append({
                "episode_id":                  episode_id,
                "scenario":                    row["scenario"],
                "current_step":                int(row["step"]),
                "current_time":                float(row["time"]),
                "current_tilt_rad":            current_tilt,
                "current_tilt_deg":            np.degrees(current_tilt),
                "current_tilt_rate":           current_tilt_rate,
                "current_margin_ratio":        current_margin_ratio,
                "predicted_margin_ratio_500ms": predicted_ratio_500ms,
                "predicted_margin_ratio_1s":   predicted_ratio_1s,
                "predicted_margin_drop_500ms": margin_drop,
                "stability_margin":            stability_margin,
                "restoring_moment":            restoring_moment,
                "raw_risk_level":              raw_level,
                "raw_risk_score":              RISK_ORDER[raw_level]
            })

        processed = min(batch_start + BATCH_SIZE, len(inference_entries))
        if processed % 2000 < BATCH_SIZE or processed == len(inference_entries):
            print(f"Processed {processed:,}/{len(inference_entries):,}")


# =========================================================
# APPLY HYSTERESIS PER EPISODE
# =========================================================

print("\nApplying per-episode hysteresis...")

results_df = pd.DataFrame(result_rows)
results_df = results_df.sort_values(["episode_id", "current_step"]).reset_index(drop=True)

smoothed_levels  = []
smoothed_scores  = []

hysteresis = RiskHysteresis(ESCALATE_AFTER, DEESCALATE_AFTER)

prev_episode = None

for _, row in results_df.iterrows():
    ep = int(row["episode_id"])
    if ep != prev_episode:
        hysteresis.reset()
        prev_episode = ep

    smoothed = hysteresis.update(row["raw_risk_level"])
    smoothed_levels.append(smoothed)
    smoothed_scores.append(RISK_ORDER[smoothed])

results_df["risk_level"] = smoothed_levels
results_df["risk_score"] = smoothed_scores


# =========================================================
# SAVE POINTWISE RESULTS
# =========================================================

results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nPointwise results saved: {OUTPUT_FILE}")


# =========================================================
# EVENT-LEVEL TIPPING EVALUATION
# =========================================================

print("\n" + "=" * 72)
print("EVENT-LEVEL EARLY-WARNING EVALUATION")
print("=" * 72)

event_rows          = []
tipping_episode_ids = []

for episode_id, group in test_df.groupby("episode_id"):
    tipping_rows = group[group["terminated"] == 1]
    if len(tipping_rows) == 0:
        continue

    tipping_episode_ids.append(int(episode_id))
    tipping_row  = tipping_rows.iloc[0]
    tipping_time = float(tipping_row["time"])
    tipping_step = int(tipping_row["step"])

    ep_preds = results_df[results_df["episode_id"] == episode_id].copy()

    def first_time_at_level(min_score):
        subset = ep_preds[
            (ep_preds["risk_score"] >= min_score) &
            (ep_preds["current_time"] < tipping_time)
        ]
        if len(subset) > 0:
            t = float(subset.iloc[0]["current_time"])
            return t, tipping_time - t
        return np.nan, np.nan

    w_time, w_lead = first_time_at_level(1)
    c_time, c_lead = first_time_at_level(2)
    i_time, i_lead = first_time_at_level(3)

    event_rows.append({
        "episode_id":                    int(episode_id),
        "scenario":                      group.iloc[0]["scenario"],
        "tipping_step":                  tipping_step,
        "tipping_time":                  tipping_time,
        "first_warning_time":            w_time,
        "warning_lead_time_s":           w_lead,
        "first_critical_time":           c_time,
        "critical_lead_time_s":          c_lead,
        "first_imminent_time":           i_time,
        "imminent_lead_time_s":          i_lead,
        "warning_detected":              int(not np.isnan(w_lead)),
        "critical_detected":             int(not np.isnan(c_lead)),
        "imminent_detected":             int(not np.isnan(i_lead)),
        "warning_at_least_500ms_early":  int(not np.isnan(w_lead) and w_lead >= 0.5),
        "critical_at_least_500ms_early": int(not np.isnan(c_lead) and c_lead >= 0.5),
        "imminent_at_least_500ms_early": int(not np.isnan(i_lead) and i_lead >= 0.5),
    })


events_df = pd.DataFrame(event_rows)
events_df.to_csv(EVENT_FILE, index=False)


# =========================================================
# PRINT RISK DISTRIBUTION
# =========================================================

print("\nRisk distribution (smoothed):")
print(results_df["risk_level"].value_counts())


# =========================================================
# TIPPING EVENT SUMMARY
# =========================================================

n_tip = len(events_df)
print(f"\nTipping episodes in test set: {n_tip}")

if n_tip > 0:

    def _sum(col):
        return int(events_df[col].sum())

    print("\nDetection before tipping:")
    print(f"  WARNING or higher  : {_sum('warning_detected')}/{n_tip}")
    print(f"  CRITICAL or higher : {_sum('critical_detected')}/{n_tip}")
    print(f"  IMMINENT           : {_sum('imminent_detected')}/{n_tip}")

    print("\nDetected at least 500 ms early:")
    print(f"  WARNING or higher  : {_sum('warning_at_least_500ms_early')}/{n_tip}")
    print(f"  CRITICAL or higher : {_sum('critical_at_least_500ms_early')}/{n_tip}")
    print(f"  IMMINENT           : {_sum('imminent_at_least_500ms_early')}/{n_tip}")

    print("\nLead-time statistics:")
    for label, col in [
        ("WARNING",  "warning_lead_time_s"),
        ("CRITICAL", "critical_lead_time_s"),
        ("IMMINENT", "imminent_lead_time_s")
    ]:
        valid = events_df[col].dropna()
        if len(valid) > 0:
            print(f"  {label} mean lead: {valid.mean():.3f} s")
            print(f"  {label} min lead : {valid.min():.3f} s")


# =========================================================
# FALSE-ALARM ANALYSIS
# =========================================================

print("\n" + "-" * 72)
print("NON-TIPPING EPISODE FALSE-ALARM ANALYSIS")
print("-" * 72)

all_test_ids     = set(int(x) for x in test_ids)
tipping_set      = set(tipping_episode_ids)
non_tipping_ids  = all_test_ids - tipping_set

fa_warning  = 0
fa_critical = 0
fa_imminent = 0

for ep_id in non_tipping_ids:
    ep_preds = results_df[results_df["episode_id"] == ep_id]
    if (ep_preds["risk_score"] >= 1).any():
        fa_warning  += 1
    if (ep_preds["risk_score"] >= 2).any():
        fa_critical += 1
    if (ep_preds["risk_score"] >= 3).any():
        fa_imminent += 1

n_non = len(non_tipping_ids)
print(f"Non-tipping test episodes       : {n_non}")
print(f"Episodes with WARNING false alarm  : {fa_warning}")
print(f"Episodes with CRITICAL false alarm : {fa_critical}")
print(f"Episodes with IMMINENT false alarm : {fa_imminent}")


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 72)
print("EARLY-WARNING EVALUATION COMPLETE")
print("=" * 72)
print(f"Pointwise output : {OUTPUT_FILE}")
print(f"Event output     : {EVENT_FILE}")
print("\nNote: thresholds are prototype simulation values,")
print("not certified crane safety limits.")
