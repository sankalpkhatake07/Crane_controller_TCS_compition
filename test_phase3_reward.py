import numpy as np

from crane_env import CraneStabilityEnv


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42
TEST_STEPS = 300

ACTIONS = {
    "INWARD": np.array(
        [-1.0],
        dtype=np.float32
    ),

    "HOLD": np.array(
        [0.0],
        dtype=np.float32
    ),

    "OUTWARD": np.array(
        [1.0],
        dtype=np.float32
    )
}


# =========================================================
# RUN ONE ACTION STRATEGY
# =========================================================

def run_strategy(
    strategy_name,
    action
):

    env = CraneStabilityEnv()

    observation, reset_info = env.reset(
        seed=SEED
    )

    # -------------------------------------------------
    # Force a more demanding crane configuration
    #
    # Same initial condition for every strategy.
    # -------------------------------------------------

    env.state[0] = 42.0      # trolley far outward
    env.state[1] = 9000.0    # heavier load

    # Recalculate approximate starting margin by taking
    # one HOLD step before measuring strategy results.
    observation, _, terminated, truncated, info = (
        env.step(
            np.array(
                [0.0],
                dtype=np.float32
            )
        )
    )

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

    maximum_abs_swing = abs(
        float(
            info["swing_angle"]
        )
    )

    final_info = info

    steps_completed = 0

    # -------------------------------------------------
    # Run fixed strategy
    # -------------------------------------------------

    for step in range(
        TEST_STEPS
    ):

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

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

        if terminated or truncated:
            break

    env.close()

    return {
        "strategy":
            strategy_name,

        "steps":
            steps_completed,

        "total_reward":
            total_reward,

        "final_status":
            final_info[
                "status"
            ],

        "final_trolley_position":
            float(
                final_info[
                    "trolley_position"
                ]
            ),

        "minimum_margin":
            minimum_margin,

        "minimum_margin_ratio":
            minimum_margin_ratio,

        "maximum_abs_tilt":
            maximum_abs_tilt,

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
            )
    }


# =========================================================
# RUN ALL STRATEGIES
# =========================================================

print("=" * 78)
print("PHASE 3 REWARD DIAGNOSTIC")
print("=" * 78)

print(
    "\nSame seed, same demanding initial state, "
    "different trolley strategies."
)

results = []

for (
    strategy_name,
    action
) in ACTIONS.items():

    result = run_strategy(
        strategy_name,
        action
    )

    results.append(
        result
    )


# =========================================================
# PRINT RESULTS
# =========================================================

for result in results:

    print("\n" + "-" * 78)

    print(
        f"Strategy: "
        f"{result['strategy']}"
    )

    print(
        f"Steps completed: "
        f"{result['steps']}"
    )

    print(
        f"Total reward: "
        f"{result['total_reward']:.4f}"
    )

    print(
        f"Final status: "
        f"{result['final_status']}"
    )

    print(
        f"Final trolley position: "
        f"{result['final_trolley_position']:.4f} m"
    )

    print(
        f"Minimum stability margin: "
        f"{result['minimum_margin']:,.2f} N.m"
    )

    print(
        f"Minimum margin ratio: "
        f"{result['minimum_margin_ratio']:.6f}"
    )

    print(
        f"Maximum |tilt|: "
        f"{result['maximum_abs_tilt']:.6f} rad"
    )

    print(
        f"Maximum |swing|: "
        f"{result['maximum_abs_swing']:.6f} rad"
    )

    print(
        f"Final stability margin: "
        f"{result['final_margin']:,.2f} N.m"
    )

    print(
        f"Final margin ratio: "
        f"{result['final_margin_ratio']:.6f}"
    )


# =========================================================
# RANK BY TOTAL REWARD
# =========================================================

ranked = sorted(
    results,
    key=lambda item: item[
        "total_reward"
    ],
    reverse=True
)

print("\n" + "=" * 78)
print("REWARD RANKING")
print("=" * 78)

for rank, result in enumerate(
    ranked,
    start=1
):

    print(
        f"{rank}. "
        f"{result['strategy']:<8} "
        f"| Reward: "
        f"{result['total_reward']:.4f} "
        f"| Min ratio: "
        f"{result['minimum_margin_ratio']:.6f} "
        f"| Status: "
        f"{result['final_status']}"
    )


print("\nExpected physical tendency:")

print(
    "INWARD should generally improve load leverage "
    "and stability reserve."
)

print(
    "OUTWARD should generally reduce stability reserve."
)

print(
    "The reward ranking should reflect that tendency "
    "under the same wind realization."
)