import pandas as pd
import numpy as np


DATASET_FILE = "crane_sensor_dataset.csv"

INPUT_STEPS = 250      # 5 seconds
FORECAST_STEPS = 50    # next 1 second
REQUIRED_STEPS = INPUT_STEPS + FORECAST_STEPS


print("=" * 70)
print("PHASE 2 DATASET QUALITY CHECK")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATASET_FILE)

print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Total episodes: {df['episode_id'].nunique()}")


# ---------------------------------------------------------
# 2. Scenario distribution
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("SCENARIO DISTRIBUTION")
print("-" * 70)

scenario_counts = (
    df.groupby("scenario")["episode_id"]
    .nunique()
    .sort_index()
)

print(scenario_counts)


# ---------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()

missing = missing[
    missing > 0
]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)


# ---------------------------------------------------------
# 4. Duplicate sensor rows
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("DUPLICATE CHECK")
print("-" * 70)

duplicate_count = df.duplicated(
    subset=[
        "episode_id",
        "step"
    ]
).sum()

print(
    f"Duplicate episode-step rows: "
    f"{duplicate_count}"
)


# ---------------------------------------------------------
# 5. Episode lengths
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("EPISODE LENGTHS")
print("-" * 70)

episode_lengths = (
    df.groupby("episode_id")
    .size()
)

print(
    f"Minimum episode length: "
    f"{episode_lengths.min()} steps"
)

print(
    f"Maximum episode length: "
    f"{episode_lengths.max()} steps"
)

print(
    f"Mean episode length: "
    f"{episode_lengths.mean():.2f} steps"
)

print(
    f"Median episode length: "
    f"{episode_lengths.median():.2f} steps"
)


# ---------------------------------------------------------
# 6. Tipping analysis
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TIPPING ANALYSIS")
print("-" * 70)

tipping_rows = df[
    df["terminated"] == 1
]

tipping_episodes = (
    tipping_rows["episode_id"]
    .nunique()
)

print(
    f"Tipping episodes: "
    f"{tipping_episodes}"
)

print(
    f"Non-tipping episodes: "
    f"{df['episode_id'].nunique() - tipping_episodes}"
)

if tipping_episodes > 0:

    tipping_steps = (
        tipping_rows.groupby("episode_id")["step"]
        .min()
    )

    print(
        f"Earliest tipping step: "
        f"{tipping_steps.min()}"
    )

    print(
        f"Latest tipping step: "
        f"{tipping_steps.max()}"
    )

    print(
        f"Mean tipping step: "
        f"{tipping_steps.mean():.2f}"
    )


# ---------------------------------------------------------
# 7. Can episodes create LSTM windows?
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("LSTM WINDOW ELIGIBILITY")
print("-" * 70)

eligible_episodes = (
    episode_lengths >= REQUIRED_STEPS
)

num_eligible = int(
    eligible_episodes.sum()
)

num_ineligible = int(
    (~eligible_episodes).sum()
)

print(
    f"Required minimum length: "
    f"{REQUIRED_STEPS} steps"
)

print(
    f"Eligible episodes: "
    f"{num_eligible}"
)

print(
    f"Ineligible episodes: "
    f"{num_ineligible}"
)


# ---------------------------------------------------------
# 8. Count possible sliding windows
#
# For episode length N:
# windows = N - 250 - 50 + 1
# ---------------------------------------------------------

possible_windows = (
    episode_lengths
    - INPUT_STEPS
    - FORECAST_STEPS
    + 1
)

possible_windows = possible_windows.clip(
    lower=0
)

total_windows = int(
    possible_windows.sum()
)

print(
    f"Total possible LSTM windows: "
    f"{total_windows:,}"
)


# ---------------------------------------------------------
# 9. Windows by scenario
# ---------------------------------------------------------

episode_scenarios = (
    df.groupby("episode_id")["scenario"]
    .first()
)

window_table = pd.DataFrame({
    "scenario": episode_scenarios,
    "windows": possible_windows
})

windows_by_scenario = (
    window_table
    .groupby("scenario")["windows"]
    .sum()
)

print("\nPossible windows by scenario:")
print(windows_by_scenario)


# ---------------------------------------------------------
# 10. Sensor ranges
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("KEY SENSOR RANGES")
print("-" * 70)

sensor_columns = [
    "wind_velocity",
    "load_mass",
    "trolley_position",
    "swing_angle",
    "structural_tilt",
    "tilt_rate",
    "stability_margin"
]

summary = df[
    sensor_columns
].agg([
    "min",
    "max",
    "mean",
    "std"
])

print(summary.T)


# ---------------------------------------------------------
# 11. Infinite values
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("INFINITE VALUE CHECK")
print("-" * 70)

numeric_df = df.select_dtypes(
    include=[np.number]
)

infinite_count = np.isinf(
    numeric_df.to_numpy()
).sum()

print(
    f"Infinite numeric values: "
    f"{infinite_count}"
)


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("QUALITY CHECK COMPLETE")
print("=" * 70)

if (
    len(missing) == 0
    and duplicate_count == 0
    and infinite_count == 0
    and total_windows > 0
):
    print(
        "Dataset passed basic integrity checks."
    )
else:
    print(
        "Dataset needs attention before LSTM training."
    )
    