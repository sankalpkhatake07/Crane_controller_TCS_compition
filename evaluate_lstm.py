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
MODEL_FILE = "crane_lstm_model.pth"
SCALER_FILE = "crane_lstm_scalers.pkl"

OUTPUT_METRICS_FILE = "lstm_evaluation_metrics.csv"
OUTPUT_EXAMPLES_FILE = "lstm_forecast_examples.csv"

SEED = 42

BATCH_SIZE = 256

# Number of example trajectories to save
NUM_EXAMPLES = 5


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

print("=" * 70)
print("CRANE LSTM PHYSICAL-UNIT EVALUATION")
print("=" * 70)

print(f"\nDevice: {device}")


# =========================================================
# LOAD SCALERS
# =========================================================

print("\nLoading scaler data...")

with open(
    SCALER_FILE,
    "rb"
) as file:

    scaler_data = pickle.load(
        file
    )

FEATURE_COLUMNS = scaler_data[
    "feature_columns"
]

TARGET_COLUMN = scaler_data[
    "target_column"
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
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_FILE
)

df = df.sort_values(
    by=[
        "episode_id",
        "step"
    ]
).reset_index(
    drop=True
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
# RECREATE EXACT EPISODE SPLIT
#
# Must match train_lstm.py exactly.
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

train_ids = episode_ids[
    :train_end
]

val_ids = episode_ids[
    train_end:val_end
]

test_ids = episode_ids[
    val_end:
]

test_df = df[
    df["episode_id"].isin(
        test_ids
    )
].copy()

print("\nTest split:")

print(
    f"Test episodes: "
    f"{len(test_ids)}"
)

print(
    f"Test rows: "
    f"{len(test_df):,}"
)


# =========================================================
# MODEL DEFINITION
#
# Must match train_lstm.py exactly.
# =========================================================

class CraneTiltLSTM(nn.Module):

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


    def forward(
        self,
        x
    ):

        lstm_output, _ = self.lstm(
            x
        )

        last_hidden = (
            lstm_output[:, -1, :]
        )

        forecast = self.forecast_head(
            last_hidden
        )

        return forecast


# =========================================================
# LOAD CHECKPOINT
# =========================================================

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model = CraneTiltLSTM(
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
).to(
    device
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print("Model loaded successfully.")


# =========================================================
# BUILD TEST EPISODE STORE
# =========================================================

print("\nBuilding test episode store...")

episode_store = {}

for episode_id, group in test_df.groupby(
    "episode_id",
    sort=False
):

    group = group.sort_values(
        "step"
    )

    features_raw = (
        group[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    target_raw = (
        group[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    steps_raw = (
        group[
            "step"
        ]
        .to_numpy()
    )

    times_raw = (
        group[
            "time"
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
        "features":
            features_normalized.astype(
                np.float32
            ),

        "target_raw":
            target_raw,

        "steps":
            steps_raw,

        "times":
            times_raw
    }


# =========================================================
# BUILD WINDOW INDEX
#
# Use same stride = 2 as training.
# =========================================================

WINDOW_STRIDE = 2

window_index = []

for (
    episode_id,
    episode_data
) in episode_store.items():

    episode_length = len(
        episode_data[
            "target_raw"
        ]
    )

    max_start = (
        episode_length
        - INPUT_STEPS
        - FORECAST_STEPS
    )

    if max_start < 0:
        continue

    for start_index in range(
        0,
        max_start + 1,
        WINDOW_STRIDE
    ):

        window_index.append(
            (
                episode_id,
                start_index
            )
        )

print(
    f"Evaluation windows: "
    f"{len(window_index):,}"
)

if len(window_index) == 0:
    raise RuntimeError(
        "No valid test windows found."
    )


# =========================================================
# METRIC ACCUMULATORS
# =========================================================

sum_abs_error_all = 0.0
count_all = 0

sum_abs_error_500ms = 0.0
count_500ms = 0

sum_abs_error_1s = 0.0
count_1s = 0

sum_squared_error_all = 0.0

# For example trajectory selection
example_records = []


# =========================================================
# EVALUATE IN BATCHES
# =========================================================

print("\nRunning physical-unit evaluation...")

with torch.no_grad():

    for batch_start in range(
        0,
        len(window_index),
        BATCH_SIZE
    ):

        batch_entries = window_index[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        x_list = []
        actual_list = []

        for (
            episode_id,
            start_index
        ) in batch_entries:

            episode_data = (
                episode_store[
                    episode_id
                ]
            )

            input_end = (
                start_index
                + INPUT_STEPS
            )

            target_end = (
                input_end
                + FORECAST_STEPS
            )

            x = (
                episode_data[
                    "features"
                ][
                    start_index:
                    input_end
                ]
            )

            actual = (
                episode_data[
                    "target_raw"
                ][
                    input_end:
                    target_end
                ]
            )

            x_list.append(
                x
            )

            actual_list.append(
                actual
            )

        x_batch = np.stack(
            x_list
        ).astype(
            np.float32
        )

        actual_batch = np.stack(
            actual_list
        ).astype(
            np.float32
        )

        x_tensor = torch.from_numpy(
            x_batch
        ).to(
            device
        )

        predicted_normalized = model(
            x_tensor
        )

        predicted_normalized = (
            predicted_normalized
            .cpu()
            .numpy()
        )

        # -----------------------------------------
        # Convert predictions back to radians
        # -----------------------------------------

        predicted_radians = (
            predicted_normalized
            * target_std
            + target_mean
        )

        # Actual target is already raw radians
        actual_radians = actual_batch

        # -----------------------------------------
        # Absolute error
        # -----------------------------------------

        absolute_error = np.abs(
            predicted_radians
            - actual_radians
        )

        squared_error = (
            predicted_radians
            - actual_radians
        ) ** 2

        # Full 1-second trajectory
        sum_abs_error_all += float(
            absolute_error.sum()
        )

        sum_squared_error_all += float(
            squared_error.sum()
        )

        count_all += int(
            absolute_error.size
        )

        # -----------------------------------------
        # +500 ms checkpoint
        #
        # index 24 corresponds to 25 × 20 ms
        # -----------------------------------------

        error_500ms = absolute_error[
            :,
            24
        ]

        sum_abs_error_500ms += float(
            error_500ms.sum()
        )

        count_500ms += int(
            error_500ms.size
        )

        # -----------------------------------------
        # +1 second checkpoint
        #
        # index 49 corresponds to 50 × 20 ms
        # -----------------------------------------

        error_1s = absolute_error[
            :,
            49
        ]

        sum_abs_error_1s += float(
            error_1s.sum()
        )

        count_1s += int(
            error_1s.size
        )

        # -----------------------------------------
        # Save a few example trajectories
        # -----------------------------------------

        if len(example_records) < NUM_EXAMPLES:

            needed = (
                NUM_EXAMPLES
                - len(example_records)
            )

            take_count = min(
                needed,
                len(batch_entries)
            )

            for local_index in range(
                take_count
            ):

                (
                    episode_id,
                    start_index
                ) = batch_entries[
                    local_index
                ]

                example_records.append({
                    "episode_id":
                        episode_id,

                    "start_index":
                        start_index,

                    "predicted":
                        predicted_radians[
                            local_index
                        ].copy(),

                    "actual":
                        actual_radians[
                            local_index
                        ].copy()
                })

        # Progress
        processed = min(
            batch_start + BATCH_SIZE,
            len(window_index)
        )

        if (
            processed % 5000 < BATCH_SIZE
            or processed == len(window_index)
        ):
            print(
                f"Processed "
                f"{processed:,}/"
                f"{len(window_index):,} "
                f"windows"
            )


# =========================================================
# FINAL METRICS IN RADIANS
# =========================================================

mae_all_rad = (
    sum_abs_error_all
    / count_all
)

rmse_all_rad = np.sqrt(
    sum_squared_error_all
    / count_all
)

mae_500ms_rad = (
    sum_abs_error_500ms
    / count_500ms
)

mae_1s_rad = (
    sum_abs_error_1s
    / count_1s
)


# =========================================================
# CONVERT TO DEGREES
# =========================================================

mae_all_deg = np.degrees(
    mae_all_rad
)

rmse_all_deg = np.degrees(
    rmse_all_rad
)

mae_500ms_deg = np.degrees(
    mae_500ms_rad
)

mae_1s_deg = np.degrees(
    mae_1s_rad
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 70)
print("PHYSICAL-UNIT EVALUATION RESULTS")
print("=" * 70)

print(
    "\nFull next-1-second trajectory:"
)

print(
    f"  MAE:  "
    f"{mae_all_rad:.8f} rad"
)

print(
    f"  MAE:  "
    f"{mae_all_deg:.6f} deg"
)

print(
    f"  RMSE: "
    f"{rmse_all_rad:.8f} rad"
)

print(
    f"  RMSE: "
    f"{rmse_all_deg:.6f} deg"
)

print(
    "\n+500 ms checkpoint:"
)

print(
    f"  MAE: "
    f"{mae_500ms_rad:.8f} rad"
)

print(
    f"  MAE: "
    f"{mae_500ms_deg:.6f} deg"
)

print(
    "\n+1.0 s checkpoint:"
)

print(
    f"  MAE: "
    f"{mae_1s_rad:.8f} rad"
)

print(
    f"  MAE: "
    f"{mae_1s_deg:.6f} deg"
)


# =========================================================
# SAVE METRICS CSV
# =========================================================

metrics_df = pd.DataFrame([
    {
        "metric":
            "trajectory_mae",

        "horizon":
            "next_1_second_all_50_steps",

        "radians":
            mae_all_rad,

        "degrees":
            mae_all_deg
    },

    {
        "metric":
            "trajectory_rmse",

        "horizon":
            "next_1_second_all_50_steps",

        "radians":
            rmse_all_rad,

        "degrees":
            rmse_all_deg
    },

    {
        "metric":
            "mae",

        "horizon":
            "500_ms",

        "radians":
            mae_500ms_rad,

        "degrees":
            mae_500ms_deg
    },

    {
        "metric":
            "mae",

        "horizon":
            "1_second",

        "radians":
            mae_1s_rad,

        "degrees":
            mae_1s_deg
    }
])

metrics_df.to_csv(
    OUTPUT_METRICS_FILE,
    index=False
)


# =========================================================
# SAVE EXAMPLE TRAJECTORIES CSV
# =========================================================

example_rows = []

for example_number, record in enumerate(
    example_records,
    start=1
):

    for forecast_index in range(
        FORECAST_STEPS
    ):

        horizon_seconds = (
            (forecast_index + 1)
            * 0.02
        )

        actual_rad = float(
            record[
                "actual"
            ][
                forecast_index
            ]
        )

        predicted_rad = float(
            record[
                "predicted"
            ][
                forecast_index
            ]
        )

        error_rad = abs(
            predicted_rad
            - actual_rad
        )

        example_rows.append({
            "example":
                example_number,

            "episode_id":
                record[
                    "episode_id"
                ],

            "window_start_index":
                record[
                    "start_index"
                ],

            "forecast_step":
                forecast_index + 1,

            "horizon_seconds":
                horizon_seconds,

            "actual_tilt_rad":
                actual_rad,

            "predicted_tilt_rad":
                predicted_rad,

            "absolute_error_rad":
                error_rad,

            "actual_tilt_deg":
                np.degrees(
                    actual_rad
                ),

            "predicted_tilt_deg":
                np.degrees(
                    predicted_rad
                ),

            "absolute_error_deg":
                np.degrees(
                    error_rad
                )
        })

examples_df = pd.DataFrame(
    example_rows
)

examples_df.to_csv(
    OUTPUT_EXAMPLES_FILE,
    index=False
)


# =========================================================
# PRINT COMPACT EXAMPLES
# =========================================================

print("\n" + "-" * 70)
print("EXAMPLE PREDICTED VS ACTUAL TRAJECTORIES")
print("-" * 70)

for example_number in range(
    1,
    NUM_EXAMPLES + 1
):

    example_df = examples_df[
        examples_df["example"]
        == example_number
    ]

    if len(example_df) == 0:
        continue

    row_500ms = example_df.iloc[
        24
    ]

    row_1s = example_df.iloc[
        49
    ]

    print(
        f"\nExample {example_number}"
    )

    print(
        f"  Episode ID: "
        f"{int(row_500ms['episode_id'])}"
    )

    print(
        "  +500 ms | "
        f"Actual: "
        f"{row_500ms['actual_tilt_deg']:.6f} deg | "
        f"Predicted: "
        f"{row_500ms['predicted_tilt_deg']:.6f} deg | "
        f"Error: "
        f"{row_500ms['absolute_error_deg']:.6f} deg"
    )

    print(
        "  +1.0 s  | "
        f"Actual: "
        f"{row_1s['actual_tilt_deg']:.6f} deg | "
        f"Predicted: "
        f"{row_1s['predicted_tilt_deg']:.6f} deg | "
        f"Error: "
        f"{row_1s['absolute_error_deg']:.6f} deg"
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Metrics saved: "
    f"{OUTPUT_METRICS_FILE}"
)

print(
    f"Examples saved: "
    f"{OUTPUT_EXAMPLES_FILE}"
)