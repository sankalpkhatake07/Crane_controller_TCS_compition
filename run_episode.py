from crane_env import CraneStabilityEnv
import numpy as np


# ---------------------------------
# Create environment
# ---------------------------------
env = CraneStabilityEnv()

# Fixed seed = reproducible experiment
obs, info = env.reset(seed=42)

print("=" * 75)
print("CRANE STABILITY SIMULATION")
print("=" * 75)

print(
    f"Initial wind: {obs[2]:.3f} m/s | "
    f"Trolley: {obs[0]:.3f} m"
)

print("-" * 75)


# ---------------------------------
# Neutral action
# 0.0 = no trolley movement
# ---------------------------------
action = np.array(
    [0.0],
    dtype=np.float32
)


previous_gust_active = False


# ---------------------------------
# Run full episode
# ---------------------------------
while True:

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    current_step = env.current_step
    current_gust_active = info["gust_active"]


    # ---------------------------------
    # Detect gust start
    # ---------------------------------
    if (
        current_gust_active
        and not previous_gust_active
    ):
        print("\n>>> GUST STARTED <<<")

        print(
            f"Time: {info['simulation_time']:.2f} s | "
            f"Wind: {info['wind_velocity']:.3f} m/s | "
            f"Force: {info['wind_force']:.2f} N"
        )


    # ---------------------------------
    # Detect gust end
    # ---------------------------------
    if (
        not current_gust_active
        and previous_gust_active
    ):
        print("\n>>> GUST ENDED <<<")

        print(
            f"Time: {info['simulation_time']:.2f} s | "
            f"Wind: {info['wind_velocity']:.3f} m/s"
        )


    # ---------------------------------
    # Print every 50 steps
    #
    # 50 × 0.02 = 1 second
    # ---------------------------------
    if current_step % 50 == 0:

        print(
            f"\nTime: {info['simulation_time']:5.2f} s | "
            f"Wind: {info['wind_velocity']:7.3f} m/s"
        )

        print(
            f"Swing angle: "
            f"{info['swing_angle']: .6f} rad | "
            f"Swing velocity: "
            f"{info['swing_angular_velocity']: .6f} rad/s"
        )

        print(
            f"Overturning moment: "
            f"{info['overturning_moment']:,.2f} N.m"
        )

        print(
            f"Stability margin: "
            f"{info['stability_margin']:,.2f} N.m | "
            f"Reward: {reward:.4f}"
        )

        print(
            f"Status: {info['status']}"
        )


    # ---------------------------------
    # Save gust state
    # ---------------------------------
    previous_gust_active = current_gust_active


    # ---------------------------------
    # Crane tipped
    # ---------------------------------
    if terminated:

        print("\n" + "!" * 75)
        print("CRANE TIPPING THRESHOLD CROSSED")
        print("!" * 75)

        print(
            f"Time: "
            f"{info['simulation_time']:.2f} s"
        )

        print(
            f"Wind: "
            f"{info['wind_velocity']:.3f} m/s"
        )

        print(
            f"Stability margin: "
            f"{info['stability_margin']:,.2f} N.m"
        )

        break


    # ---------------------------------
    # 20-second time limit reached
    # ---------------------------------
    if truncated:

        print("\n" + "=" * 75)
        print("EPISODE COMPLETED")
        print("=" * 75)

        print(
            f"Total simulation time: "
            f"{info['simulation_time']:.2f} s"
        )

        print(
            "Crane did not cross the "
            "tipping threshold."
        )

        break


env.close()