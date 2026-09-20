import csv
import os
import numpy as np

from crane_env import CraneStabilityEnv


# =========================================================
# DATASET CONFIGURATION
# =========================================================

OUTPUT_FILE = "crane_sensor_dataset.csv"

# Increased from 300 → 1000 episodes for better LSTM coverage
NUM_EPISODES = 1000

MAX_STEPS_PER_EPISODE = 1000   # 1000 × 0.02 s = 20 seconds

MASTER_SEED = 2026

# Fraction of episodes to run with the trained PPO agent.
# Set to 0.0 if no model exists yet (first run).
PPO_EPISODE_FRACTION = 0.0   # set to 0.4 after first PPO training run
PPO_MODEL_PATH = os.path.join("models", "crane_ppo_best_tracked.zip")


# =========================================================
# SCENARIO MIX
#
# 4 scenarios (safe / critical / high_risk / extreme_wind)
# generate diverse tipping-precursor examples.
# =========================================================

SCENARIOS = {

    "safe": {
        "probability": 0.35,

        "load_mass_range":          (2500.0, 6500.0),
        "trolley_position_range":   (8.0,   30.0),
        "base_wind_range":          (2.0,    9.0),
        "turbulence_range":         (0.5,    2.0),
        "gust_probability_range":   (0.002,  0.010),
        "gust_min_range":           (4.0,    8.0),
        "gust_max_range":           (10.0,  18.0)
    },

    "critical": {
        "probability": 0.35,

        "load_mass_range":          (6500.0, 10000.0),
        "trolley_position_range":   (30.0,   44.0),
        "base_wind_range":          (8.0,   16.0),
        "turbulence_range":         (1.0,    3.0),
        "gust_probability_range":   (0.006,  0.020),
        "gust_min_range":           (7.0,   12.0),
        "gust_max_range":           (14.0,  25.0)
    },

    "high_risk": {
        "probability": 0.20,

        "load_mass_range":          (8500.0, 11500.0),
        "trolley_position_range":   (38.0,   47.0),
        "base_wind_range":          (10.0,  18.0),
        "turbulence_range":         (1.5,    4.0),
        "gust_probability_range":   (0.010,  0.030),
        "gust_min_range":           (8.0,   14.0),
        "gust_max_range":           (18.0,  30.0)
    },

    # New scenario: sudden extreme wind gusts.
    # Rare but critical — forces LSTM to learn
    # the fast-onset tipping signature.
    "extreme_wind": {
        "probability": 0.10,

        "load_mass_range":          (5000.0, 10000.0),
        "trolley_position_range":   (20.0,   45.0),
        "base_wind_range":          (12.0,  20.0),
        "turbulence_range":         (2.0,    5.0),
        "gust_probability_range":   (0.020,  0.050),
        "gust_min_range":           (15.0,  25.0),
        "gust_max_range":           (30.0,  50.0)
    }
}


# =========================================================
# CSV COLUMNS
# =========================================================

FIELDNAMES = [
    "episode_id",
    "scenario",
    "controller",
    "step",
    "time",

    "wind_velocity",
    "wind_force",

    "load_mass",

    "trolley_position",
    "trolley_extension_ratio",
    "trolley_action",

    "swing_angle",
    "swing_angular_velocity",

    "structural_tilt",
    "tilt_rate",
    "tilt_acceleration",
    "target_tilt",

    "load_offset",

    "overturning_moment",
    "restoring_moment",
    "stability_margin",

    "gust_active",
    "gust_steps_remaining",

    "reward",
    "status",
    "terminated",
    "truncated"
]


# =========================================================
# HELPERS
# =========================================================

def choose_scenario(rng):
    names = list(SCENARIOS.keys())
    probs = [SCENARIOS[n]["probability"] for n in names]
    return rng.choice(names, p=probs)


def sample_range(rng, r):
    return float(rng.uniform(r[0], r[1]))


def configure_episode(env, rng, scenario_name):
    cfg = SCENARIOS[scenario_name]

    load_mass         = sample_range(rng, cfg["load_mass_range"])
    trolley_position  = sample_range(rng, cfg["trolley_position_range"])
    base_wind         = sample_range(rng, cfg["base_wind_range"])
    turbulence        = sample_range(rng, cfg["turbulence_range"])
    gust_prob         = sample_range(rng, cfg["gust_probability_range"])
    gust_min          = sample_range(rng, cfg["gust_min_range"])
    gust_max          = sample_range(rng, cfg["gust_max_range"])
    gust_max          = max(gust_max, gust_min + 1.0)

    env.wind_model.base_wind           = base_wind
    env.wind_model.turbulence_strength = turbulence
    env.wind_model.gust_probability    = gust_prob
    env.wind_model.gust_min            = gust_min
    env.wind_model.gust_max            = gust_max
    env.wind_model.max_wind            = 50.0

    return {
        "load_mass":        load_mass,
        "trolley_position": trolley_position,
        "base_wind":        base_wind
    }


def sample_new_action(rng):
    action_value = float(rng.uniform(-1.0, 1.0))
    hold_steps   = int(rng.integers(10, 76))
    return action_value, hold_steps


# =========================================================
# PPO CONTROLLER (loaded lazily)
# =========================================================

_ppo_model = None

def get_ppo_controller():
    """Lazy-load the PPO model once and return a callable."""
    global _ppo_model
    if _ppo_model is None:
        from stable_baselines3 import PPO
        _ppo_model = PPO.load(PPO_MODEL_PATH, device="cpu")
        print(f"PPO model loaded: {PPO_MODEL_PATH}")
    return _ppo_model


# =========================================================
# MAIN DATA GENERATION
# =========================================================

def main():

    rng = np.random.default_rng(MASTER_SEED)

    # Decide which episodes use PPO vs random controller
    ppo_available = (
        PPO_EPISODE_FRACTION > 0.0
        and os.path.exists(PPO_MODEL_PATH)
    )

    if ppo_available:
        ppo_episode_count = int(NUM_EPISODES * PPO_EPISODE_FRACTION)
        ppo_episode_ids   = set(
            rng.choice(NUM_EPISODES, size=ppo_episode_count, replace=False).tolist()
        )
        print(f"PPO controller: {ppo_episode_count} episodes")
    else:
        ppo_episode_ids = set()
        if PPO_EPISODE_FRACTION > 0.0:
            print(f"WARNING: PPO model not found at {PPO_MODEL_PATH}. Using random controller for all episodes.")

    env = CraneStabilityEnv()

    total_rows       = 0
    tipping_episodes = 0
    scenario_counts  = {name: 0 for name in SCENARIOS}
    controller_counts = {"random": 0, "ppo": 0}

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for episode_id in range(NUM_EPISODES):

            # ------------------------------------------
            # Assign scenario and controller
            # ------------------------------------------

            scenario_name = choose_scenario(rng)
            scenario_counts[scenario_name] += 1

            use_ppo      = episode_id in ppo_episode_ids
            controller   = "ppo" if use_ppo else "random"
            controller_counts[controller] += 1

            # ------------------------------------------
            # Configure environment
            # ------------------------------------------

            episode_config = configure_episode(env, rng, scenario_name)

            episode_seed = MASTER_SEED + episode_id
            obs, _ = env.reset(seed=episode_seed)

            env.state[0] = episode_config["trolley_position"]
            env.state[1] = episode_config["load_mass"]
            env.state[2] = episode_config["base_wind"]
            env.state[3] = 0.0
            env.state[4] = 0.0
            env.state[5] = 0.0
            env.state[6] = 0.0

            # ------------------------------------------
            # Initialise action for random controller
            # ------------------------------------------

            action_value          = 0.0
            action_steps_remaining = 0

            episode_tipped = False
            episode_rows   = 0

            # ------------------------------------------
            # TIMESTEP LOOP
            # ------------------------------------------

            for _ in range(MAX_STEPS_PER_EPISODE):

                if use_ppo:
                    ppo = get_ppo_controller()
                    action_arr, _ = ppo.predict(obs, deterministic=True)
                    action_value  = float(
                        np.clip(action_arr[0], -1.0, 1.0)
                    )
                else:
                    if action_steps_remaining <= 0:
                        action_value, action_steps_remaining = sample_new_action(rng)
                    action_steps_remaining -= 1

                action = np.array([action_value], dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)

                writer.writerow({
                    "episode_id":              episode_id,
                    "scenario":                scenario_name,
                    "controller":              controller,
                    "step":                    env.current_step,
                    "time":                    info["simulation_time"],
                    "wind_velocity":           info["wind_velocity"],
                    "wind_force":              info["wind_force"],
                    "load_mass":               float(obs[1]),
                    "trolley_position":        info["trolley_position"],
                    "trolley_extension_ratio": info["trolley_extension_ratio"],
                    "trolley_action":          action_value,
                    "swing_angle":             info["swing_angle"],
                    "swing_angular_velocity":  info["swing_angular_velocity"],
                    "structural_tilt":         info["structural_tilt"],
                    "tilt_rate":               info["tilt_rate"],
                    "tilt_acceleration":       info["tilt_acceleration"],
                    "target_tilt":             info["target_tilt"],
                    "load_offset":             info["load_offset"],
                    "overturning_moment":      info["overturning_moment"],
                    "restoring_moment":        info["restoring_moment"],
                    "stability_margin":        info["stability_margin"],
                    "gust_active":             int(info["gust_active"]),
                    "gust_steps_remaining":    info["gust_steps_remaining"],
                    "reward":                  reward,
                    "status":                  info["status"],
                    "terminated":              int(terminated),
                    "truncated":               int(truncated)
                })

                episode_rows += 1
                total_rows   += 1

                if terminated:
                    episode_tipped = True
                    tipping_episodes += 1
                    break

                if truncated:
                    break

            if (episode_id + 1) % 50 == 0 or episode_id == 0:
                print(
                    f"Episode {episode_id + 1:4d}/{NUM_EPISODES} | "
                    f"Scenario: {scenario_name:12s} | "
                    f"Controller: {controller:6s} | "
                    f"Rows: {episode_rows:4d} | "
                    f"Tipped: {episode_tipped}"
                )

    env.close()

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output file     : {OUTPUT_FILE}")
    print(f"Total episodes  : {NUM_EPISODES}")
    print(f"Total rows      : {total_rows:,}")
    print(f"Tipping episodes: {tipping_episodes}")
    print(f"File size       : {file_size_mb:.2f} MB")

    print("\nScenario counts:")
    for name, count in scenario_counts.items():
        print(f"  {name:12s}: {count}")

    print("\nController counts:")
    for name, count in controller_counts.items():
        print(f"  {name:6s}: {count}")

    print("\nTiming:")
    print("  Simulator timestep : 0.02 s")
    print("  LSTM input history : 5.0 s = 250 steps")
    print("  Forecast horizon   : 1.0 s = 50 steps")
    print("  500 ms checkpoint  : 25 future steps")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
