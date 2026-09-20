import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_FILE = "crane_sensor_dataset.csv"
MODEL_FILE = "crane_risk_lstm_model.pth"
SCALER_FILE = "crane_risk_lstm_scalers.pkl"

POINT_OUTPUT_FILE = "risk_lstm_predictions.csv"
EVENT_OUTPUT_FILE = "risk_lstm_event_evaluation.csv"

SEED = 42

DT = 0.02

# Run prediction every 100 ms
INFERENCE_STRIDE = 5

BATCH_SIZE = 256

# Forecast checkpoints
INDEX_500MS = 24
INDEX_1S = 49


# =========================================================
# WARNING THRESHOLDS
#
# margin_ratio:
# > 0 means restoring reserve remains
# = 0 means tipping threshold
# < 0 means threshold crossed
#
# These warning bands are prototype simulation thresholds.
# =========================================================

WARNING_THRESHOLD = 0.15
CRITICAL_THRESHOLD = 0.05
TIPPING_THRESHOLD = 0.0


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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 74)
print("RISK LSTM - 500 ms EARLY TIPPING EVALUATION")
print("=" * 74)

print(f"\nDevice: {device}")


# =========================================================
# LOAD SCALERS
# =========================================================

print("\nLoading risk scalers...")

with open(
    SCALER_FILE,
    "rb"
) as file:

    scaler_data = pickle.load(file)


FEATURE_COLUMNS = scaler_data[
    "feature_columns"
]

feature_mean = np.asarray(
    scaler_data["feature_mean"],
    dtype=np.float32
)

feature_std = np.asarray(
    scaler_data["feature_std"],
    dtype=np.float32
)

target_mean = float(
    scaler_data["target_mean"]
)

target_std = float(
    scaler_data["target_std"]
)

INPUT_STEPS = int(
    scaler_data["input_steps"]
)

FORECAST_STEPS = int(
    scaler_data["forecast_steps"]
)


print(
    f"Input history: "
    f"{INPUT_STEPS} steps"
)

print(
    f"Forecast horizon: "
    f"{FORECAST_STEPS} steps"
)


# =========================================================
# MODEL DEFINITION
#
# Must exactly match train_risk_lstm.py
# =========================================================

class CraneRiskLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        forecast_steps,
        dropout
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.forecast_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                forecast_steps
            )
        )


    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        last_hidden = (
            lstm_output[:, -1, :]
        )

        return self.forecast_head(
            last_hidden
        )


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading trained Risk LSTM...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model = CraneRiskLSTM(

    input_size=checkpoint[
        "input_size"
    ],

    hidden_size=checkpoint[
        "hidden_size"
    ],

    num_layers=checkpoint[
        "num_layers"
    ],

    forecast_steps=checkpoint[
        "forecast_steps"
    ],

    dropout=checkpoint[
        "dropout"
    ]

).to(device)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print("Risk LSTM loaded successfully.")


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_FILE
)

df = df.sort_values(
    [
        "episode_id",
        "step"
    ]
).reset_index(
    drop=True
)


# Physics-defined target
df["margin_ratio"] = (
    df["stability_margin"]
    / df["restoring_moment"]
)


print(
    f"Rows loaded: "
    f"{len(df):,}"
)

print(
    f"Episodes loaded: "
    f"{df['episode_id'].nunique()}"
)


# =========================================================
# RECREATE EXACT TEST SPLIT
#
# Must match train_risk_lstm.py
# =========================================================

episode_ids = np.array(
    sorted(
        df["episode_id"]
        .unique()
    )
)

rng = np.random.default_rng(
    SEED
)

rng.shuffle(
    episode_ids
)

num_episodes = len(
    episode_ids
)

train_end = int(
    0.70 * num_episodes
)

val_end = int(
    0.85 * num_episodes
)

test_ids = episode_ids[
    val_end:
]


test_df = df[
    df["episode_id"].isin(
        test_ids
    )
].copy()


print("\nHeld-out test split:")

print(
    f"  Test episodes: "
    f"{len(test_ids)}"
)

print(
    f"  Test rows: "
    f"{len(test_df):,}"
)


# =========================================================
# BUILD EPISODE STORES
# =========================================================

print("\nBuilding test episode stores...")

episode_store = {}

for episode_id, group in test_df.groupby(
    "episode_id",
    sort=False
):

    group = group.sort_values(
        "step"
    ).reset_index(
        drop=True
    )

    features_raw = (
        group[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    features_normalized = (
        features_raw
        - feature_mean
    ) / feature_std

    episode_store[
        int(episode_id)
    ] = {
        "group":
            group,

        "features":
            features_normalized.astype(
                np.float32
            )
    }


# =========================================================
# BUILD INFERENCE POINTS
#
# We require 250 past steps.
# Prediction is performed every 100 ms.
# =========================================================

inference_entries = []

for (
    episode_id,
    episode_data
) in episode_store.items():

    group = episode_data[
        "group"
    ]

    episode_length = len(
        group
    )

    for current_index in range(
        INPUT_STEPS - 1,
        episode_length,
        INFERENCE_STRIDE
    ):

        start_index = (
            current_index
            - INPUT_STEPS
            + 1
        )

        inference_entries.append(
            (
                episode_id,
                start_index,
                current_index
            )
        )


print(
    f"\nInference points: "
    f"{len(inference_entries):,}"
)

if len(inference_entries) == 0:

    raise RuntimeError(
        "No valid inference points found."
    )


# =========================================================
# METRIC ACCUMULATORS
#
# Only evaluate forecast MAE where actual future data exists.
# =========================================================

sum_abs_error_500ms = 0.0
count_500ms = 0

sum_abs_error_1s = 0.0
count_1s = 0

sum_abs_error_trajectory = 0.0
count_trajectory = 0


# =========================================================
# RUN INFERENCE
# =========================================================

print("\nRunning Risk LSTM inference...")

result_rows = []


with torch.no_grad():

    for batch_start in range(
        0,
        len(inference_entries),
        BATCH_SIZE
    ):

        batch_entries = inference_entries[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        x_list = []

        for (
            episode_id,
            start_index,
            current_index
        ) in batch_entries:

            features = episode_store[
                episode_id
            ]["features"]

            x = features[
                start_index:
                current_index + 1
            ]

            x_list.append(x)


        x_batch = np.stack(
            x_list
        ).astype(
            np.float32
        )


        x_tensor = torch.from_numpy(
            x_batch
        ).to(device)


        predicted_normalized = model(
            x_tensor
        )


        predicted_normalized = (
            predicted_normalized
            .cpu()
            .numpy()
        )


        # Convert back to physical margin ratio
        predicted_margin = (
            predicted_normalized
            * target_std
            + target_mean
        )


        # -------------------------------------------------
        # PROCESS EACH FORECAST
        # -------------------------------------------------

        for local_index, (
            episode_id,
            start_index,
            current_index
        ) in enumerate(
            batch_entries
        ):

            group = episode_store[
                episode_id
            ]["group"]

            current_row = group.iloc[
                current_index
            ]

            current_margin = float(
                current_row[
                    "margin_ratio"
                ]
            )

            predicted_500ms = float(
                predicted_margin[
                    local_index,
                    INDEX_500MS
                ]
            )

            predicted_1s = float(
                predicted_margin[
                    local_index,
                    INDEX_1S
                ]
            )

            predicted_min_1s = float(
                np.min(
                    predicted_margin[
                        local_index
                    ]
                )
            )

            predicted_min_index = int(
                np.argmin(
                    predicted_margin[
                        local_index
                    ]
                )
            )

            predicted_min_horizon_s = (
                predicted_min_index + 1
            ) * DT


            # ---------------------------------------------
            # Risk classification from predicted future
            # ---------------------------------------------

            if predicted_min_1s <= TIPPING_THRESHOLD:

                risk_level = (
                    "PREDICTED_TIPPING"
                )

                risk_score = 3

            elif predicted_min_1s <= CRITICAL_THRESHOLD:

                risk_level = "CRITICAL"
                risk_score = 2

            elif predicted_min_1s <= WARNING_THRESHOLD:

                risk_level = "WARNING"
                risk_score = 1

            else:

                risk_level = "SAFE"
                risk_score = 0


            # ---------------------------------------------
            # Actual +500 ms target if available
            # ---------------------------------------------

            actual_index_500ms = (
                current_index
                + 25
            )

            actual_500ms = np.nan
            error_500ms = np.nan

            if actual_index_500ms < len(
                group
            ):

                actual_500ms = float(
                    group.iloc[
                        actual_index_500ms
                    ]["margin_ratio"]
                )

                error_500ms = abs(
                    predicted_500ms
                    - actual_500ms
                )

                sum_abs_error_500ms += (
                    error_500ms
                )

                count_500ms += 1


            # ---------------------------------------------
            # Actual +1 second target if available
            # ---------------------------------------------

            actual_index_1s = (
                current_index
                + 50
            )

            actual_1s = np.nan
            error_1s = np.nan

            if actual_index_1s < len(
                group
            ):

                actual_1s = float(
                    group.iloc[
                        actual_index_1s
                    ]["margin_ratio"]
                )

                error_1s = abs(
                    predicted_1s
                    - actual_1s
                )

                sum_abs_error_1s += (
                    error_1s
                )

                count_1s += 1


            # ---------------------------------------------
            # Full trajectory MAE where complete future
            # 50-step target exists
            # ---------------------------------------------

            trajectory_start = (
                current_index + 1
            )

            trajectory_end = (
                trajectory_start
                + FORECAST_STEPS
            )

            trajectory_mae = np.nan

            if trajectory_end <= len(
                group
            ):

                actual_trajectory = (
                    group[
                        "margin_ratio"
                    ]
                    .iloc[
                        trajectory_start:
                        trajectory_end
                    ]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

                predicted_trajectory = (
                    predicted_margin[
                        local_index
                    ]
                )

                absolute_errors = np.abs(
                    predicted_trajectory
                    - actual_trajectory
                )

                trajectory_mae = float(
                    absolute_errors.mean()
                )

                sum_abs_error_trajectory += float(
                    absolute_errors.sum()
                )

                count_trajectory += int(
                    absolute_errors.size
                )


            result_rows.append({

                "episode_id":
                    episode_id,

                "scenario":
                    current_row[
                        "scenario"
                    ],

                "current_step":
                    int(
                        current_row[
                            "step"
                        ]
                    ),

                "current_time":
                    float(
                        current_row[
                            "time"
                        ]
                    ),

                "current_margin_ratio":
                    current_margin,

                "predicted_margin_500ms":
                    predicted_500ms,

                "actual_margin_500ms":
                    actual_500ms,

                "absolute_error_500ms":
                    error_500ms,

                "predicted_margin_1s":
                    predicted_1s,

                "actual_margin_1s":
                    actual_1s,

                "absolute_error_1s":
                    error_1s,

                "predicted_min_margin_1s":
                    predicted_min_1s,

                "predicted_min_horizon_s":
                    predicted_min_horizon_s,

                "trajectory_mae":
                    trajectory_mae,

                "risk_level":
                    risk_level,

                "risk_score":
                    risk_score
            })


        processed = min(
            batch_start + BATCH_SIZE,
            len(inference_entries)
        )

        if (
            processed % 2000 < BATCH_SIZE
            or processed
            == len(inference_entries)
        ):

            print(
                f"Processed "
                f"{processed:,}/"
                f"{len(inference_entries):,}"
            )


# =========================================================
# CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    result_rows
)

results_df.to_csv(
    POINT_OUTPUT_FILE,
    index=False
)


# =========================================================
# PHYSICAL TARGET METRICS
# =========================================================

mae_500ms = (
    sum_abs_error_500ms
    / count_500ms
    if count_500ms > 0
    else np.nan
)

mae_1s = (
    sum_abs_error_1s
    / count_1s
    if count_1s > 0
    else np.nan
)

trajectory_mae = (
    sum_abs_error_trajectory
    / count_trajectory
    if count_trajectory > 0
    else np.nan
)


print("\n" + "=" * 74)
print("RISK FORECAST ACCURACY")
print("=" * 74)

print(
    f"\n+500 ms margin-ratio MAE: "
    f"{mae_500ms:.8f}"
)

print(
    f"+1.0 s margin-ratio MAE: "
    f"{mae_1s:.8f}"
)

print(
    f"Full next-1-second trajectory MAE: "
    f"{trajectory_mae:.8f}"
)


# =========================================================
# RISK DISTRIBUTION
# =========================================================

print("\nRisk distribution:")

print(
    results_df[
        "risk_level"
    ].value_counts()
)


# =========================================================
# EVENT-LEVEL EVALUATION
# =========================================================

print("\n" + "=" * 74)
print("EVENT-LEVEL TIPPING EVALUATION")
print("=" * 74)

event_rows = []

tipping_episode_ids = []


for episode_id, group in test_df.groupby(
    "episode_id"
):

    group = group.sort_values(
        "step"
    )

    tipping_rows = group[
        group["margin_ratio"] <= 0.0
    ]

    if len(tipping_rows) == 0:
        continue


    tipping_episode_ids.append(
        int(episode_id)
    )

    first_tip = tipping_rows.iloc[0]

    tipping_time = float(
        first_tip["time"]
    )

    tipping_step = int(
        first_tip["step"]
    )


    episode_predictions = results_df[
        results_df["episode_id"]
        == episode_id
    ].sort_values(
        "current_time"
    )


    # Only predictions before actual tipping
    pre_tip_predictions = (
        episode_predictions[
            episode_predictions[
                "current_time"
            ] < tipping_time
        ]
    )


    # ---------------------------------------------
    # WARNING OR HIGHER
    # ---------------------------------------------

    warning_rows = pre_tip_predictions[
        pre_tip_predictions[
            "risk_score"
        ] >= 1
    ]


    # ---------------------------------------------
    # CRITICAL OR HIGHER
    # ---------------------------------------------

    critical_rows = pre_tip_predictions[
        pre_tip_predictions[
            "risk_score"
        ] >= 2
    ]


    # ---------------------------------------------
    # DIRECT PREDICTED TIPPING
    # ---------------------------------------------

    tipping_prediction_rows = (
        pre_tip_predictions[
            pre_tip_predictions[
                "risk_score"
            ] >= 3
        ]
    )


    def first_detection_info(
        detection_rows
    ):

        if len(detection_rows) == 0:

            return (
                np.nan,
                np.nan
            )

        first_time = float(
            detection_rows.iloc[
                0
            ]["current_time"]
        )

        lead_time = (
            tipping_time
            - first_time
        )

        return (
            first_time,
            lead_time
        )


    (
        first_warning_time,
        warning_lead
    ) = first_detection_info(
        warning_rows
    )


    (
        first_critical_time,
        critical_lead
    ) = first_detection_info(
        critical_rows
    )


    (
        first_predicted_tip_time,
        predicted_tip_lead
    ) = first_detection_info(
        tipping_prediction_rows
    )


    event_rows.append({

        "episode_id":
            int(episode_id),

        "scenario":
            group.iloc[0][
                "scenario"
            ],

        "tipping_step":
            tipping_step,

        "tipping_time":
            tipping_time,

        "first_warning_time":
            first_warning_time,

        "warning_lead_time_s":
            warning_lead,

        "first_critical_time":
            first_critical_time,

        "critical_lead_time_s":
            critical_lead,

        "first_predicted_tipping_time":
            first_predicted_tip_time,

        "predicted_tipping_lead_time_s":
            predicted_tip_lead,

        "warning_detected":
            int(
                not np.isnan(
                    warning_lead
                )
            ),

        "critical_detected":
            int(
                not np.isnan(
                    critical_lead
                )
            ),

        "predicted_tipping_detected":
            int(
                not np.isnan(
                    predicted_tip_lead
                )
            ),

        "warning_500ms_early":
            int(
                (
                    not np.isnan(
                        warning_lead
                    )
                )
                and
                (
                    warning_lead >= 0.5
                )
            ),

        "critical_500ms_early":
            int(
                (
                    not np.isnan(
                        critical_lead
                    )
                )
                and
                (
                    critical_lead >= 0.5
                )
            ),

        "predicted_tipping_500ms_early":
            int(
                (
                    not np.isnan(
                        predicted_tip_lead
                    )
                )
                and
                (
                    predicted_tip_lead >= 0.5
                )
            )
    })


events_df = pd.DataFrame(
    event_rows
)

events_df.to_csv(
    EVENT_OUTPUT_FILE,
    index=False
)


num_tipping = len(
    events_df
)


print(
    f"\nTipping episodes in held-out test set: "
    f"{num_tipping}"
)


if num_tipping > 0:

    warning_detected = int(
        events_df[
            "warning_detected"
        ].sum()
    )

    critical_detected = int(
        events_df[
            "critical_detected"
        ].sum()
    )

    predicted_tip_detected = int(
        events_df[
            "predicted_tipping_detected"
        ].sum()
    )


    warning_500 = int(
        events_df[
            "warning_500ms_early"
        ].sum()
    )

    critical_500 = int(
        events_df[
            "critical_500ms_early"
        ].sum()
    )

    predicted_tip_500 = int(
        events_df[
            "predicted_tipping_500ms_early"
        ].sum()
    )


    print("\nDetection before actual tipping:")

    print(
        f"  WARNING or higher: "
        f"{warning_detected}/{num_tipping}"
    )

    print(
        f"  CRITICAL or higher: "
        f"{critical_detected}/{num_tipping}"
    )

    print(
        f"  Predicted threshold crossing: "
        f"{predicted_tip_detected}/{num_tipping}"
    )


    print("\nDetected at least 500 ms early:")

    print(
        f"  WARNING or higher: "
        f"{warning_500}/{num_tipping}"
    )

    print(
        f"  CRITICAL or higher: "
        f"{critical_500}/{num_tipping}"
    )

    print(
        f"  Predicted threshold crossing: "
        f"{predicted_tip_500}/{num_tipping}"
    )


    # ---------------------------------------------
    # LEAD TIME STATISTICS
    # ---------------------------------------------

    print("\nLead-time statistics:")

    for label, column in [

        (
            "WARNING",
            "warning_lead_time_s"
        ),

        (
            "CRITICAL",
            "critical_lead_time_s"
        ),

        (
            "PREDICTED TIPPING",
            "predicted_tipping_lead_time_s"
        )
    ]:

        valid = events_df[
            column
        ].dropna()

        if len(valid) > 0:

            print(
                f"  {label} mean lead: "
                f"{valid.mean():.3f} s"
            )

            print(
                f"  {label} minimum lead: "
                f"{valid.min():.3f} s"
            )

            print(
                f"  {label} maximum lead: "
                f"{valid.max():.3f} s"
            )


# =========================================================
# FALSE-ALARM ANALYSIS
# =========================================================

print("\n" + "-" * 74)
print("NON-TIPPING EPISODE FALSE-ALARM ANALYSIS")
print("-" * 74)


all_test_ids = set(
    int(x)
    for x in test_ids
)

tipping_set = set(
    tipping_episode_ids
)

non_tipping_ids = (
    all_test_ids
    - tipping_set
)


false_warning = 0
false_critical = 0
false_predicted_tipping = 0


for episode_id in non_tipping_ids:

    episode_predictions = results_df[
        results_df["episode_id"]
        == episode_id
    ]


    if (
        episode_predictions[
            "risk_score"
        ] >= 1
    ).any():

        false_warning += 1


    if (
        episode_predictions[
            "risk_score"
        ] >= 2
    ).any():

        false_critical += 1


    if (
        episode_predictions[
            "risk_score"
        ] >= 3
    ).any():

        false_predicted_tipping += 1


num_non_tipping = len(
    non_tipping_ids
)


print(
    f"Non-tipping test episodes: "
    f"{num_non_tipping}"
)

print(
    f"Episodes with WARNING false alarm: "
    f"{false_warning}"
)

print(
    f"Episodes with CRITICAL false alarm: "
    f"{false_critical}"
)

print(
    f"Episodes with predicted-tipping false alarm: "
    f"{false_predicted_tipping}"
)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 74)
print("FINAL RISK LSTM EVALUATION COMPLETE")
print("=" * 74)

print(
    f"Point predictions saved: "
    f"{POINT_OUTPUT_FILE}"
)

print(
    f"Event evaluation saved: "
    f"{EVENT_OUTPUT_FILE}"
)

print(
    "\nInterpretation:"
)

print(
    "A direct predicted-threshold-crossing event means "
    "the LSTM forecasted margin_ratio <= 0 "
    "within its next 1-second trajectory."
)

print(
    "This remains simulation-domain prototype evidence, "
    "not certified real-crane safety validation."
)