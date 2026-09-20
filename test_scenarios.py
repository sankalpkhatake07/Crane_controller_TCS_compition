from crane_env import CraneStabilityEnv
import numpy as np


def run_scenario(
    name,
    load_mass,
    trolley_position,
    base_wind,
    max_steps=1000
):
    print("\n" + "=" * 75)
    print(f"SCENARIO: {name}")
    print("=" * 75)

    env = CraneStabilityEnv()

    # Configure wind before reset
    env.wind_model.base_wind = base_wind

    obs, info = env.reset(seed=42)

    # Set scenario conditions
    env.state[0] = trolley_position
    env.state[1] = load_mass
    env.state[2] = base_wind

    # Neutral trolley action
    action = np.array(
        [0.0],
        dtype=np.float32
    )

    minimum_margin = float("inf")
    maximum_wind = 0.0
    maximum_swing = 0.0

    for step in range(max_steps):

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        minimum_margin = min(
            minimum_margin,
            info["stability_margin"]
        )

        maximum_wind = max(
            maximum_wind,
            abs(info["wind_velocity"])
        )

        maximum_swing = max(
            maximum_swing,
            abs(info["swing_angle"])
        )

        # Print every second
        if env.current_step % 50 == 0:
            print(
                f"Time: {info['simulation_time']:5.2f} s | "
                f"Wind: {info['wind_velocity']:7.2f} m/s | "
                f"Swing: {info['swing_angle']:8.4f} rad | "
                f"Margin: {info['stability_margin']:12,.2f} N.m"
            )

        if terminated:
            print("\n>>> TIPPING THRESHOLD CROSSED <<<")
            print(
                f"Time: {info['simulation_time']:.2f} s"
            )
            print(
                f"Wind: {info['wind_velocity']:.2f} m/s"
            )
            print(
                f"Overturning moment: "
                f"{info['overturning_moment']:,.2f} N.m"
            )
            print(
                f"Restoring moment: "
                f"{info['restoring_moment']:,.2f} N.m"
            )
            print(
                f"Stability margin: "
                f"{info['stability_margin']:,.2f} N.m"
            )
            break

        if truncated:
            break

    print("\n--- Scenario Summary ---")
    print(f"Final status: {info['status']}")
    print(f"Maximum |wind|: {maximum_wind:.2f} m/s")
    print(f"Maximum |swing|: {maximum_swing:.4f} rad")
    print(f"Minimum margin: {minimum_margin:,.2f} N.m")

    env.close()


# -------------------------------------------------
# 1. SAFE SCENARIO
# -------------------------------------------------

run_scenario(
    name="SAFE",
    load_mass=5000.0,
    trolley_position=20.0,
    base_wind=5.0
)


# -------------------------------------------------
# 2. CRITICAL SCENARIO
# -------------------------------------------------

run_scenario(
    name="CRITICAL",
    load_mass=9000.0,
    trolley_position=42.0,
    base_wind=12.0
)


# -------------------------------------------------
# 3. TIPPING SCENARIO
# -------------------------------------------------

run_scenario(
    name="TIPPING",
    load_mass=12000.0,
    trolley_position=48.0,
    base_wind=18.0
)