import os
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from crane_env import CraneStabilityEnv


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

# Increased from 200k → 1M for proper convergence
TOTAL_TIMESTEPS = 1_000_000

# Parallel environments: 8× more data per wall-clock second
NUM_ENVS = 8

MODEL_DIR      = "models"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "ppo_logs"
EVAL_LOG_DIR   = os.path.join(LOG_DIR, "eval")

FINAL_MODEL_PATH       = os.path.join(MODEL_DIR, "crane_ppo_agent")
BEST_TRACKED_MODEL_PATH = os.path.join(MODEL_DIR, "crane_ppo_best_tracked")
VECNORM_PATH           = os.path.join(MODEL_DIR, "crane_ppo_vecnorm.pkl")

for d in [MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, EVAL_LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================
# REPRODUCIBILITY
# =========================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# LINEAR LEARNING-RATE SCHEDULE
#
# Decays from initial_lr → final_lr over training.
# Encourages exploration early and fine-tuning late.
# =========================================================

INITIAL_LR = 3e-4
FINAL_LR   = 1e-5

def linear_lr_schedule(progress_remaining: float) -> float:
    """
    progress_remaining goes from 1.0 (start) to 0.0 (end).
    Returns learning rate for the current training progress.
    """
    return FINAL_LR + progress_remaining * (INITIAL_LR - FINAL_LR)


# =========================================================
# ENVIRONMENT FACTORIES
# =========================================================

def make_train_env(rank: int = 0):
    def _init():
        env = CraneStabilityEnv()
        env = Monitor(
            env,
            filename=os.path.join(LOG_DIR, f"monitor_{rank}.csv"),
            info_keywords=("status", "margin_ratio")
        )
        return env
    return _init


def make_eval_env():
    env = CraneStabilityEnv()
    env = Monitor(env)
    return env


# =========================================================
# VECTOR TRAINING ENVIRONMENT
#
# SubprocVecEnv runs each env in a separate process for true
# parallelism. Falls back to DummyVecEnv if multiprocessing
# is unavailable (e.g. on Windows without __main__ guard).
# =========================================================

def build_train_env():
    fns = [make_train_env(rank=i) for i in range(NUM_ENVS)]
    try:
        vec_env = SubprocVecEnv(fns, start_method="spawn")
    except Exception:
        vec_env = DummyVecEnv(fns)

    # VecNormalize: normalises observations AND rewards online.
    # This is the single most impactful change for PPO stability
    # when physical quantities span vastly different scales
    # (e.g. stability_margin up to ±1e7 vs swing_angle ~0.01 rad).
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    return vec_env


# =========================================================
# CUSTOM TRAINING CALLBACK
# =========================================================

class CraneTrainingCallback(BaseCallback):

    def __init__(self, print_freq=50_000, verbose=1):
        super().__init__(verbose)
        self.print_freq       = print_freq
        self.episode_count    = 0
        self.tipping_count    = 0
        self.recent_rewards   = []
        self.recent_lengths   = []
        self.best_mean_reward = -np.inf

    def _on_step(self):

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i]
            self.episode_count += 1

            if info.get("status") == "tipping":
                self.tipping_count += 1

            if "episode" in info:
                ep = info["episode"]
                self.recent_rewards.append(float(ep["r"]))
                self.recent_lengths.append(int(ep["l"]))
                self.recent_rewards = self.recent_rewards[-50:]
                self.recent_lengths = self.recent_lengths[-50:]

        if self.num_timesteps % self.print_freq == 0:

            print("\n" + "=" * 72)
            print(f"PPO TRAINING PROGRESS @ {self.num_timesteps:,} steps")
            print("=" * 72)
            print(f"Episodes completed : {self.episode_count}")
            print(f"Tipping episodes   : {self.tipping_count}")

            if self.recent_rewards:
                mean_r = float(np.mean(self.recent_rewards))
                mean_l = float(np.mean(self.recent_lengths))
                print(f"Recent mean reward : {mean_r:.4f}")
                print(f"Recent mean length : {mean_l:.2f}")

            if self.episode_count > 0:
                rate = 100.0 * self.tipping_count / self.episode_count
                print(f"Tipping rate       : {rate:.2f}%")

        return True


# =========================================================
# CALLBACKS
# =========================================================

training_callback = CraneTrainingCallback(print_freq=50_000)

checkpoint_callback = CheckpointCallback(
    save_freq=max(100_000 // NUM_ENVS, 1),
    save_path=CHECKPOINT_DIR,
    name_prefix="crane_ppo"
)

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,   # do NOT normalise rewards in eval env
    clip_obs=10.0,
    training=False       # eval env stats are updated from train env
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=EVAL_LOG_DIR,
    eval_freq=max(50_000 // NUM_ENVS, 1),
    n_eval_episodes=20,
    deterministic=True,
    render=False,
    verbose=1
)


# =========================================================
# BUILD ENVIRONMENTS
# =========================================================

env = build_train_env()


# =========================================================
# PPO MODEL
#
# Key changes vs previous version:
#   - VecNormalize handles obs/reward scaling (see above)
#   - Larger network [256, 256] captures more complex patterns
#   - Linear LR schedule: aggressive early, precise late
#   - Larger effective batch: n_steps × NUM_ENVS = 16,384
#   - Slightly higher entropy early (annealed via schedule)
# =========================================================

print("=" * 72)
print("PPO CRANE CONTROLLER — IMPROVED TRAINING")
print("=" * 72)
print("\nTraining configuration:")
print(f"  Device          : {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
print(f"  Parallel envs   : {NUM_ENVS}")
print(f"  Effective batch : {2048 * NUM_ENVS:,} steps/update")
print(f"  LR schedule     : {INITIAL_LR} → {FINAL_LR} (linear)")
print(f"  VecNormalize    : enabled (obs + reward)")
print( "  Network         : pi=[256,256], vf=[256,256]")


model = PPO(

    policy="MlpPolicy",

    env=env,

    learning_rate=linear_lr_schedule,

    n_steps=2048,         # steps per env per update

    batch_size=256,       # mini-batch size (was 64)

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,        # slight exploration incentive

    vf_coef=0.5,

    max_grad_norm=0.5,

    policy_kwargs={
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256]
        },
        "ortho_init": True   # orthogonal weight initialisation
    },

    verbose=1,

    seed=SEED,

    device="auto"
)


# =========================================================
# TRAIN
# =========================================================

print("\nStarting PPO training...")
print("Do not close the terminal while training.")


try:

    model.learn(

        total_timesteps=TOTAL_TIMESTEPS,

        callback=[
            training_callback,
            checkpoint_callback,
            eval_callback
        ],

        progress_bar=False,

        reset_num_timesteps=True
    )


except KeyboardInterrupt:

    print("\nTraining interrupted. Saving checkpoint...")

    model.save(os.path.join(MODEL_DIR, "crane_ppo_interrupted"))
    env.save(os.path.join(MODEL_DIR, "crane_ppo_interrupted_vecnorm.pkl"))

    env.close()
    eval_env.close()
    raise


# =========================================================
# SAVE FINAL MODEL + NORMALISATION STATS
# =========================================================

model.save(FINAL_MODEL_PATH)

# VecNormalize running statistics MUST be saved alongside
# the model weights — inference will be wrong without them.
env.save(VECNORM_PATH)

print("\n" + "=" * 72)
print("PPO TRAINING COMPLETE")
print("=" * 72)
print(f"Final model saved     : {FINAL_MODEL_PATH}.zip")
print(f"VecNormalize stats    : {VECNORM_PATH}")
print(f"Best model (eval)     : {MODEL_DIR}/best_model.zip")
print(f"Episodes completed    : {training_callback.episode_count}")
print(f"Tipping episodes      : {training_callback.tipping_count}")

if training_callback.episode_count > 0:
    rate = 100.0 * training_callback.tipping_count / training_callback.episode_count
    print(f"Final tipping rate    : {rate:.2f}%")

if training_callback.recent_rewards:
    print(f"Final mean reward     : {np.mean(training_callback.recent_rewards):.4f}")

print("\nIMPORTANT: When loading this model for evaluation, also load")
print(f"the VecNormalize stats from: {VECNORM_PATH}")

env.close()
eval_env.close()
