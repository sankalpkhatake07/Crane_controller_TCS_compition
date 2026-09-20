import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from crane_env import CraneStabilityEnv


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = os.path.join(
    "models",
    "crane_ppo_best_tracked.zip"
)

OUTPUT_FILE = (
    "ppo_copilot_evaluation.csv"
)

NUM_SEEDS = 50

BASE_SEED = 1000

MAX_STEPS = 1000


# =========================================================
# CHALLENGE CONDITIONS
#
# We test multiple operating conditions.
# Each controller receives exactly the same conditions.
# =========================================================

CHALLENGES = [

    {
        "name": "moderate",
        "trolley_position": 35.0,
        "load_mass": 7000.0
    },

    {
        "name": "high_risk",
        "trolley_position": 42.0,
        "load_mass": 9000.0
    },

    {
        "name": "severe",
        "trolley_position": 46.0,
        "load_mass": 10000.0
    }
]


# =========================================================
# LOAD PPO MODEL
# =========================================================

print("=" * 78)
print("PHASE 3 - AI CO-PILOT CONTROLLED EVALUATION")
print("=" * 78)

print(
    f"\nLoading PPO model: "
    f"{MODEL_PATH}"
)

model = PPO.load(
    MODEL_PATH,
    device="cpu"
)

print(
    "PPO AI Co-Pilot loaded successfully."
)


# =========================================================
# CONTROLLER DEFINITIONS
# =========================================================

def hold_controller(
    observation
):

    return np.array(
        [0.0],
        dtype=np.float32
    )


def inward_controller(
    observation
):

    return np.array(
        [-1.0],
        dtype=np.float32
    )


def ppo_controller(
    observation
):

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    return np.asarray(
        action,
        dtype=np.float32
    ).reshape(1)


CONTROLLERS = {
    "HOLD":
        hold_controller,

    "INWARD":
        inward_controller,

    "PPO_AI":
        ppo_controller
}


# =========================================================
# RUN ONE EPISODE
# =========================================================

def run_episode(
    controller_name,
    controller_function,
    challenge,
    seed
):

    env = CraneStabilityEnv()

    observation, reset_info = (
        env.reset(
            seed=seed
        )
    )

    # -------------------------------------------------
    # Force identical challenge condition
    # -------------------------------------------------

    env.state[0] = float(
        challenge[
            "trolley_position"
        ]
    )

    env.state[1] = float(
        challenge[
            "load_mass"
        ]
    )

    # -------------------------------------------------
    # Recalculate state-dependent physics
    #
    # One HOLD step updates stability margin,
    # tilt target and other derived quantities.
    # This step is identical for all controllers.
    # -------------------------------------------------

    (
        observation,
        warmup_reward,
        terminated,
        truncated,
        info
    ) = env.step(
        np.array(
            [0.0],
            dtype=np.float32
        )
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    total_reward = 0.0

    minimum_margin = float(
        info["stability_margin"]
    )

    minimum_margin_ratio = float(
        info["margin_ratio"]
    )

    maximum_abs_tilt = abs(
        float(
            info["structural_tilt"]
        )
    )

    maximum_abs_tilt_rate = abs(
        float(
            info["tilt_rate"]
        )
    )

    maximum_abs_swing = abs(
        float(
            info["swing_angle"]
        )
    )

    total_action_magnitude = 0.0

    action_changes = 0

    previous_action = None

    final_info = info

    steps_completed = 0

    tipped = bool(
        terminated
    )

    # -------------------------------------------------
    # Controller loop
    # -------------------------------------------------

    if not terminated:

        for step in range(
            MAX_STEPS
        ):

            action = controller_function(
                observation
            )

            action = np.asarray(
                action,
                dtype=np.float32
            ).reshape(1)

            action = np.clip(
                action,
                -1.0,
                1.0
            ).astype(
                np.float32
            )

            # -----------------------------------------
            # Action metrics
            # -----------------------------------------

            action_value = float(
                action[0]
            )

            total_action_magnitude += abs(
                action_value
            )

            if previous_action is not None:

                if abs(
                    action_value
                    - previous_action
                ) > 0.05:

                    action_changes += 1

            previous_action = (
                action_value
            )

            # -----------------------------------------
            # Environment step
            # -----------------------------------------

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(
                action
            )

            total_reward += float(
                reward
            )

            minimum_margin = min(
                minimum_margin,
                float(
                    info[
                        "stability_margin"
                    ]
                )
            )

            minimum_margin_ratio = min(
                minimum_margin_ratio,
                float(
                    info[
                        "margin_ratio"
                    ]
                )
            )

            maximum_abs_tilt = max(
                maximum_abs_tilt,
                abs(
                    float(
                        info[
                            "structural_tilt"
                        ]
                    )
                )
            )

            maximum_abs_tilt_rate = max(
                maximum_abs_tilt_rate,
                abs(
                    float(
                        info[
                            "tilt_rate"
                        ]
                    )
                )
            )

            maximum_abs_swing = max(
                maximum_abs_swing,
                abs(
                    float(
                        info[
                            "swing_angle"
                        ]
                    )
                )
            )

            final_info = info

            steps_completed = (
                step + 1
            )

            if terminated:

                tipped = True
                break

            if truncated:
                break

    # -------------------------------------------------
    # Final metrics
    # -------------------------------------------------

    mean_action_magnitude = (
        total_action_magnitude
        / steps_completed
        if steps_completed > 0
        else 0.0
    )

    result = {

        "controller":
            controller_name,

        "challenge":
            challenge["name"],

        "seed":
            seed,

        "initial_trolley_position":
            challenge[
                "trolley_position"
            ],

        "load_mass":
            challenge[
                "load_mass"
            ],

        "tipped":
            int(
                tipped
            ),

        "steps_completed":
            steps_completed,

        "survival_time_s":
            float(
                final_info[
                    "simulation_time"
                ]
            ),

        "total_reward":
            total_reward,

        "minimum_margin":
            minimum_margin,

        "minimum_margin_ratio":
            minimum_margin_ratio,

        "maximum_abs_tilt":
            maximum_abs_tilt,

        "maximum_abs_tilt_rate":
            maximum_abs_tilt_rate,

        "maximum_abs_swing":
            maximum_abs_swing,

        "final_margin":
            float(
                final_info[
                    "stability_margin"
                ]
            ),

        "final_margin_ratio":
            float(
                final_info[
                    "margin_ratio"
                ]
            ),

        "final_trolley_position":
            float(
                final_info[
                    "trolley_position"
                ]
            ),

        "mean_action_magnitude":
            mean_action_magnitude,

        "action_changes":
            action_changes,

        "final_status":
            final_info[
                "status"
            ]
    }

    env.close()

    return result


# =========================================================
# RUN CONTROLLED EXPERIMENT
# =========================================================

results = []

total_runs = (
    len(CHALLENGES)
    * NUM_SEEDS
    * len(CONTROLLERS)
)

completed_runs = 0


print(
    "\nEvaluation configuration:"
)

print(
    f"  Challenges: "
    f"{len(CHALLENGES)}"
)

print(
    f"  Seeds per challenge: "
    f"{NUM_SEEDS}"
)

print(
    f"  Controllers: "
    f"{len(CONTROLLERS)}"
)

print(
    f"  Total runs: "
    f"{total_runs}"
)


for challenge in CHALLENGES:

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"Challenge: "
        f"{challenge['name']}"
    )

    print(
        f"Trolley: "
        f"{challenge['trolley_position']} m"
    )

    print(
        f"Load: "
        f"{challenge['load_mass']} kg"
    )


    for seed_offset in range(
        NUM_SEEDS
    ):

        seed = (
            BASE_SEED
            + seed_offset
        )


        for (
            controller_name,
            controller_function
        ) in CONTROLLERS.items():

            result = run_episode(

                controller_name=
                    controller_name,

                controller_function=
                    controller_function,

                challenge=
                    challenge,

                seed=
                    seed
            )

            results.append(
                result
            )

            completed_runs += 1


        if (
            (seed_offset + 1) % 10
            == 0
        ):

            print(
                f"Completed seeds: "
                f"{seed_offset + 1}/"
                f"{NUM_SEEDS}"
            )


# =========================================================
# SAVE RAW RESULTS
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# OVERALL SUMMARY
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "OVERALL CONTROLLER COMPARISON"
)

print(
    "=" * 78
)


overall_summary = (
    results_df
    .groupby(
        "controller"
    )
    .agg(

        episodes=(
            "tipped",
            "count"
        ),

        tipping_events=(
            "tipped",
            "sum"
        ),

        tipping_rate=(
            "tipped",
            "mean"
        ),

        mean_reward=(
            "total_reward",
            "mean"
        ),

        mean_min_margin_ratio=(
            "minimum_margin_ratio",
            "mean"
        ),

        worst_margin_ratio=(
            "minimum_margin_ratio",
            "min"
        ),

        mean_max_tilt=(
            "maximum_abs_tilt",
            "mean"
        ),

        mean_action_magnitude=(
            "mean_action_magnitude",
            "mean"
        ),

        mean_action_changes=(
            "action_changes",
            "mean"
        )
    )
    .reset_index()
)


overall_summary[
    "tipping_rate_percent"
] = (
    100.0
    * overall_summary[
        "tipping_rate"
    ]
)


for _, row in (
    overall_summary.iterrows()
):

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"Controller: "
        f"{row['controller']}"
    )

    print(
        f"Episodes: "
        f"{int(row['episodes'])}"
    )

    print(
        f"Tipping events: "
        f"{int(row['tipping_events'])}"
    )

    print(
        f"Tipping rate: "
        f"{row['tipping_rate_percent']:.2f}%"
    )

    print(
        f"Mean total reward: "
        f"{row['mean_reward']:.4f}"
    )

    print(
        f"Mean minimum margin ratio: "
        f"{row['mean_min_margin_ratio']:.6f}"
    )

    print(
        f"Worst margin ratio: "
        f"{row['worst_margin_ratio']:.6f}"
    )

    print(
        f"Mean maximum |tilt|: "
        f"{row['mean_max_tilt']:.6f} rad"
    )

    print(
        f"Mean action magnitude: "
        f"{row['mean_action_magnitude']:.6f}"
    )

    print(
        f"Mean action changes: "
        f"{row['mean_action_changes']:.2f}"
    )


# =========================================================
# CHALLENGE-WISE SUMMARY
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "CHALLENGE-WISE TIPPING COMPARISON"
)

print(
    "=" * 78
)


challenge_summary = (
    results_df
    .groupby(
        [
            "challenge",
            "controller"
        ]
    )
    .agg(

        episodes=(
            "tipped",
            "count"
        ),

        tipping_events=(
            "tipped",
            "sum"
        ),

        tipping_rate=(
            "tipped",
            "mean"
        ),

        mean_min_margin_ratio=(
            "minimum_margin_ratio",
            "mean"
        ),

        mean_reward=(
            "total_reward",
            "mean"
        )
    )
    .reset_index()
)


challenge_summary[
    "tipping_rate_percent"
] = (
    100.0
    * challenge_summary[
        "tipping_rate"
    ]
)


for challenge in CHALLENGES:

    challenge_name = (
        challenge["name"]
    )

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"Challenge: "
        f"{challenge_name}"
    )


    subset = challenge_summary[
        challenge_summary[
            "challenge"
        ] == challenge_name
    ]


    for _, row in subset.iterrows():

        print(
            f"{row['controller']:<10} "
            f"| Tipping: "
            f"{int(row['tipping_events'])}/"
            f"{int(row['episodes'])} "
            f"| Rate: "
            f"{row['tipping_rate_percent']:.2f}% "
            f"| Mean min ratio: "
            f"{row['mean_min_margin_ratio']:.6f} "
            f"| Reward: "
            f"{row['mean_reward']:.2f}"
        )


# =========================================================
# PAIRED PPO VS HOLD ANALYSIS
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "PAIRED PPO AI VS HOLD ANALYSIS"
)

print(
    "=" * 78
)


paired = results_df.pivot_table(

    index=[
        "challenge",
        "seed"
    ],

    columns="controller",

    values=[
        "tipped",
        "minimum_margin_ratio",
        "total_reward"
    ]
)


ppo_prevented_tip = 0
ppo_caused_tip = 0

if (
    ("tipped", "HOLD")
    in paired.columns
    and
    ("tipped", "PPO_AI")
    in paired.columns
):

    ppo_prevented_tip = int(
        (
            (
                paired[
                    ("tipped", "HOLD")
                ] == 1
            )
            &
            (
                paired[
                    ("tipped", "PPO_AI")
                ] == 0
            )
        ).sum()
    )

    ppo_caused_tip = int(
        (
            (
                paired[
                    ("tipped", "HOLD")
                ] == 0
            )
            &
            (
                paired[
                    ("tipped", "PPO_AI")
                ] == 1
            )
        ).sum()
    )


print(
    f"\nCases where HOLD tipped "
    f"but PPO survived: "
    f"{ppo_prevented_tip}"
)

print(
    f"Cases where HOLD survived "
    f"but PPO tipped: "
    f"{ppo_caused_tip}"
)


if (
    ("minimum_margin_ratio", "HOLD")
    in paired.columns
    and
    ("minimum_margin_ratio", "PPO_AI")
    in paired.columns
):

    margin_improvement = (
        paired[
            (
                "minimum_margin_ratio",
                "PPO_AI"
            )
        ]
        -
        paired[
            (
                "minimum_margin_ratio",
                "HOLD"
            )
        ]
    )

    print(
        f"Mean PPO improvement in "
        f"minimum margin ratio vs HOLD: "
        f"{margin_improvement.mean():.6f}"
    )

    print(
        f"Paired cases where PPO had "
        f"better minimum margin: "
        f"{int((margin_improvement > 0).sum())}/"
        f"{len(margin_improvement)}"
    )


# =========================================================
# FINAL VERDICT
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "PHASE 3 EVALUATION COMPLETE"
)

print(
    "=" * 78
)

print(
    f"\nRaw results saved: "
    f"{OUTPUT_FILE}"
)

print(
    "\nInterpretation:"
)

print(
    "The strongest evidence is not training reward alone."
)

print(
    "Look for lower tipping rate and better paired "
    "minimum stability margin under identical seeds."
)

print(
    "PPO should outperform HOLD on challenge episodes "
    "to demonstrate active mitigation."
)