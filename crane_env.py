import gymnasium as gym
from gymnasium import spaces
import numpy as np

from physics import (
    calculate_wind_force,
    calculate_pendulum_acceleration,
    update_pendulum_state,
    calculate_load_horizontal_offset,
    calculate_overturning_moment,
    calculate_restoring_moment,
    calculate_stability_margin
)

from wind import WindGustModel


class CraneStabilityEnv(gym.Env):
    """
    Custom 2D Crane Stability Environment.

    Phase 1 features:
    - Continuous trolley control
    - Explicit boom geometry
    - Time-varying wind
    - Random wind gusts
    - Hanging-load pendulum physics
    - Overturning moment
    - Restoring moment
    - Stability margin
    - Dynamic structural tilt
    - Tipping detection

    Phase 3 additions:
    - Multi-objective DRL reward function
    - Stability reserve reward
    - Structural tilt penalty
    - Tilt-rate penalty
    - Load-swing penalty
    - Action-energy penalty
    - Near-tipping danger penalty
    - Active recovery bonus
    - Stable / warning / critical / tipping states
    """

    metadata = {
        "render_modes": []
    }

    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        # Observation vector:
        #
        # 0 = trolley_position
        # 1 = load_mass
        # 2 = wind_velocity
        # 3 = swing_angle
        # 4 = swing_angular_velocity
        # 5 = structural_tilt
        # 6 = tilt_rate
        # 7 = stability_margin
        # -------------------------------------------------

        self.observation_space = spaces.Box(
            low=np.array([
                0.0,
                100.0,
                -50.0,
                -np.pi / 2,
                -10.0,
                -0.5,
                -5.0,
                -1e7
            ], dtype=np.float32),

            high=np.array([
                50.0,
                50000.0,
                50.0,
                np.pi / 2,
                10.0,
                0.5,
                5.0,
                1e7
            ], dtype=np.float32),

            dtype=np.float32
        )

        # -------------------------------------------------
        # Continuous action space
        #
        # -1.0 = move trolley inward
        #  0.0 = hold position
        # +1.0 = move trolley outward
        # -------------------------------------------------

        self.action_space = spaces.Box(
            low=np.array(
                [-1.0],
                dtype=np.float32
            ),

            high=np.array(
                [1.0],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        self.state = None

        # -------------------------------------------------
        # Crane physical parameters
        # -------------------------------------------------

        self.crane_height = 50.0
        self.boom_length = 50.0
        self.cable_length = 15.0
        self.crane_mass = 100000.0
        self.base_width = 10.0

        # -------------------------------------------------
        # Simulation timestep
        # -------------------------------------------------

        self.dt = 0.02

        # -------------------------------------------------
        # Dynamic structural tilt model
        # -------------------------------------------------

        self.tilt_natural_frequency = 2.0
        self.tilt_damping_ratio = 0.7

        self.max_target_tilt = 0.08

        # -------------------------------------------------
        # Trolley dynamics
        # -------------------------------------------------

        self.trolley_speed = 0.5

        # -------------------------------------------------
        # Time-varying wind generator
        # -------------------------------------------------

        self.wind_model = WindGustModel(
            base_wind=5.0,
            turbulence_strength=1.5,
            gust_probability=0.01,
            gust_min=8.0,
            gust_max=20.0,
            max_wind=50.0
        )

        # -------------------------------------------------
        # Episode tracking
        # -------------------------------------------------

        self.current_step = 0

        # 1000 × 0.02 s = 20 seconds
        self.max_steps = 1000


    def reset(self, seed=None, options=None):
        """
        Reset crane simulation with randomised initial conditions.

        Randomising start state prevents the agent from overfitting
        to a single fixed configuration and improves generalisation
        to the challenge evaluation conditions.
        """

        super().reset(seed=seed)

        # -------------------------------------------------
        # Reset episode counter
        # -------------------------------------------------

        self.current_step = 0

        # -------------------------------------------------
        # Reset wind model
        # -------------------------------------------------

        initial_wind = self.wind_model.reset()

        # -------------------------------------------------
        # Randomised initial crane condition
        #
        # trolley_position : uniform [5, 45] m
        # load_mass        : uniform [1000, 10000] kg
        # swing_angle      : small perturbation [-0.05, 0.05] rad
        #
        # Using self.np_random (seeded by super().reset)
        # ensures reproducibility when seed is provided.
        # -------------------------------------------------

        initial_trolley_position = float(
            self.np_random.uniform(5.0, 45.0)
        )

        initial_load_mass = float(
            self.np_random.uniform(1000.0, 10000.0)
        )

        initial_swing_angle = float(
            self.np_random.uniform(-0.05, 0.05)
        )

        # -------------------------------------------------
        # Calculate initial wind force
        # -------------------------------------------------

        initial_wind_force = (
            calculate_wind_force(
                initial_wind
            )
        )

        # -------------------------------------------------
        # Calculate initial load offset
        # -------------------------------------------------

        initial_load_offset = (
            calculate_load_horizontal_offset(
                initial_trolley_position,
                self.cable_length,
                initial_swing_angle
            )
        )

        # -------------------------------------------------
        # Calculate initial overturning moment
        # -------------------------------------------------

        initial_overturning_moment = (
            calculate_overturning_moment(
                initial_load_mass,
                initial_load_offset,
                initial_wind_force,
                self.crane_height
            )
        )

        # -------------------------------------------------
        # Calculate initial restoring moment
        # -------------------------------------------------

        initial_restoring_moment = (
            calculate_restoring_moment(
                self.crane_mass,
                self.base_width
            )
        )

        # -------------------------------------------------
        # Calculate physically consistent initial margin
        # -------------------------------------------------

        initial_stability_margin = (
            calculate_stability_margin(
                initial_restoring_moment,
                initial_overturning_moment
            )
        )

        # -------------------------------------------------
        # Initial state
        # -------------------------------------------------

        self.state = np.array([
            initial_trolley_position,
            initial_load_mass,
            initial_wind,
            0.0,
            0.0,
            0.0,
            0.0,
            initial_stability_margin
        ], dtype=np.float32)

        observation = self.state.copy()

        # -------------------------------------------------
        # Initial margin ratio
        # -------------------------------------------------

        initial_margin_ratio = (
            initial_stability_margin
            / initial_restoring_moment
        )

        info = {
            "status": "safe",

            "simulation_time": 0.0,

            "wind_velocity": float(
                initial_wind
            ),

            "boom_length": float(
                self.boom_length
            ),

            "trolley_position": float(
                initial_trolley_position
            ),

            "trolley_extension_ratio": float(
                initial_trolley_position
                / self.boom_length
            ),

            "stability_margin": float(
                initial_stability_margin
            ),

            "restoring_moment": float(
                initial_restoring_moment
            ),

            "margin_ratio": float(
                initial_margin_ratio
            )
        }

        return (
            observation,
            info
        )


    def step(self, action):
        """
        Execute one 20 ms physics timestep.
        """

        # -------------------------------------------------
        # Validate and clip action
        # -------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float32
        ).reshape(-1)

        if action.size != 1:
            raise ValueError(
                "Action must contain exactly one value."
            )

        action_value = float(
            np.clip(
                action[0],
                -1.0,
                1.0
            )
        )

        # -------------------------------------------------
        # Advance episode time
        # -------------------------------------------------

        self.current_step += 1

        # -------------------------------------------------
        # 1. Read current state
        # -------------------------------------------------

        trolley_position = float(
            self.state[0]
        )

        load_mass = float(
            self.state[1]
        )

        swing_angle = float(
            self.state[3]
        )

        angular_velocity = float(
            self.state[4]
        )

        previous_tilt = float(
            self.state[5]
        )

        previous_tilt_rate = float(
            self.state[6]
        )

        # IMPORTANT:
        # Save old stability margin BEFORE state update.
        # This is required for the recovery bonus.
        previous_stability_margin = float(
            self.state[7]
        )

        # -------------------------------------------------
        # 2. Generate time-varying wind
        # -------------------------------------------------

        wind_velocity = (
            self.wind_model.update(
                rng=self.np_random,
                dt=self.dt
            )
        )

        # -------------------------------------------------
        # 3. Apply trolley action
        # -------------------------------------------------

        trolley_position += (
            action_value
            * self.trolley_speed
            * self.dt
        )

        trolley_position = float(
            np.clip(
                trolley_position,
                0.0,
                self.boom_length
            )
        )

        # -------------------------------------------------
        # 4. Calculate wind force
        # -------------------------------------------------

        wind_force = (
            calculate_wind_force(
                wind_velocity
            )
        )

        # -------------------------------------------------
        # 5. Pendulum physics
        # -------------------------------------------------

        angular_acceleration = (
            calculate_pendulum_acceleration(
                swing_angle,
                angular_velocity,
                self.cable_length,
                load_mass,
                wind_force
            )
        )

        (
            new_swing_angle,
            new_angular_velocity
        ) = update_pendulum_state(
            swing_angle,
            angular_velocity,
            angular_acceleration,
            self.dt
        )

        # -------------------------------------------------
        # Keep pendulum state inside declared bounds
        # -------------------------------------------------

        new_swing_angle = float(
            np.clip(
                new_swing_angle,
                -np.pi / 2,
                np.pi / 2
            )
        )

        new_angular_velocity = float(
            np.clip(
                new_angular_velocity,
                -10.0,
                10.0
            )
        )

        # -------------------------------------------------
        # 6. Effective horizontal load position
        # -------------------------------------------------

        load_offset = (
            calculate_load_horizontal_offset(
                trolley_position,
                self.cable_length,
                new_swing_angle
            )
        )

        # -------------------------------------------------
        # 7. Crane moments
        # -------------------------------------------------

        overturning_moment = (
            calculate_overturning_moment(
                load_mass,
                load_offset,
                wind_force,
                self.crane_height
            )
        )

        restoring_moment = (
            calculate_restoring_moment(
                self.crane_mass,
                self.base_width
            )
        )

        stability_margin = (
            calculate_stability_margin(
                restoring_moment,
                overturning_moment
            )
        )

        # -------------------------------------------------
        # 8. Dynamic structural tilt physics
        # -------------------------------------------------

        moment_ratio = (
            overturning_moment
            / restoring_moment
        )

        target_tilt = float(
            np.clip(
                moment_ratio
                * self.max_target_tilt,
                -0.5,
                0.5
            )
        )

        omega_n = (
            self.tilt_natural_frequency
        )

        zeta = (
            self.tilt_damping_ratio
        )

        tilt_acceleration = (
            (omega_n ** 2)
            * (
                target_tilt
                - previous_tilt
            )
            -
            (
                2.0
                * zeta
                * omega_n
                * previous_tilt_rate
            )
        )

        # -------------------------------------------------
        # Semi-implicit Euler integration
        # -------------------------------------------------

        new_tilt_rate = (
            previous_tilt_rate
            + tilt_acceleration
            * self.dt
        )

        new_tilt_rate = float(
            np.clip(
                new_tilt_rate,
                -5.0,
                5.0
            )
        )

        new_structural_tilt = (
            previous_tilt
            + new_tilt_rate
            * self.dt
        )

        new_structural_tilt = float(
            np.clip(
                new_structural_tilt,
                -0.5,
                0.5
            )
        )

        # -------------------------------------------------
        # 9. Calculate margin ratios BEFORE state update
        # -------------------------------------------------

        margin_ratio = (
            stability_margin
            / restoring_moment
        )

        previous_margin_ratio = (
            previous_stability_margin
            / restoring_moment
        )

        margin_improvement = (
            margin_ratio
            - previous_margin_ratio
        )

        # -------------------------------------------------
        # 10. Tipping condition
        # -------------------------------------------------

        terminated = bool(
            stability_margin <= 0.0
        )

        # -------------------------------------------------
        # 11. Episode time limit
        # -------------------------------------------------

        truncated = bool(
            self.current_step
            >= self.max_steps
        )

        # -------------------------------------------------
        # 12. Phase 3 DRL reward function
        # -------------------------------------------------

        clipped_margin_ratio = float(
            np.clip(
                margin_ratio,
                -1.0,
                1.0
            )
        )

        # Recovery bonus cap (prevent single-step dominance)
        RECOVERY_BONUS_CAP = 2.0

        abs_tilt = abs(
            new_structural_tilt
        )

        abs_tilt_rate = abs(
            new_tilt_rate
        )

        abs_swing = abs(
            new_swing_angle
        )

        # ---------------------------------------------
        # A. Positive stability reward
        # ---------------------------------------------

        stability_reward = (
            2.0
            * clipped_margin_ratio
        )

        # ---------------------------------------------
        # B. Structural tilt penalty
        # ---------------------------------------------

        tilt_penalty = (
            1.5
            * (
                abs_tilt
                / self.max_target_tilt
            )
        )

        # ---------------------------------------------
        # C. Tilt-rate penalty
        # ---------------------------------------------

        tilt_rate_penalty = (
            0.5
            * abs_tilt_rate
        )

        # ---------------------------------------------
        # D. Load-swing penalty
        # ---------------------------------------------

        swing_penalty = (
            0.25
            * abs_swing
        )

        # ---------------------------------------------
        # E. Action-energy penalty
        # ---------------------------------------------

        action_penalty = (
            0.05
            * abs(action_value)
        )

        # ---------------------------------------------
        # F. Near-tipping danger penalties
        # ---------------------------------------------

        danger_penalty = 0.0

        if margin_ratio < 0.30:
            danger_penalty += 1.0

        if margin_ratio < 0.20:
            danger_penalty += 2.0

        if margin_ratio < 0.10:
            danger_penalty += 5.0

        if margin_ratio < 0.05:
            danger_penalty += 10.0

        # ---------------------------------------------
        # G. Active recovery bonus
        #
        # Reward actions that improve stability reserve.
        # Capped to prevent a single large step from
        # dominating all other reward components.
        # ---------------------------------------------

        recovery_bonus = min(
            10.0 * max(margin_improvement, 0.0),
            RECOVERY_BONUS_CAP
        )

        # ---------------------------------------------
        # H. Final reward
        # ---------------------------------------------

        if terminated:

            reward = -100.0
            status = "tipping"

        else:

            reward = float(
                stability_reward
                - tilt_penalty
                - tilt_rate_penalty
                - swing_penalty
                - action_penalty
                - danger_penalty
                + recovery_bonus
            )

            # Note: hard reward clipping removed.
            # VecNormalize handles reward scaling
            # during PPO training.

            if margin_ratio < 0.10:
                status = "critical"

            elif margin_ratio < 0.25:
                status = "warning"

            else:
                status = "stable"

        # -------------------------------------------------
        # 13. Update complete state
        #
        # IMPORTANT:
        # State update occurs AFTER reward calculations
        # requiring the previous stability margin.
        # -------------------------------------------------

        self.state = np.array([
            trolley_position,
            load_mass,
            wind_velocity,
            new_swing_angle,
            new_angular_velocity,
            new_structural_tilt,
            new_tilt_rate,
            stability_margin
        ], dtype=np.float32)

        # -------------------------------------------------
        # 14. Simulation time
        # -------------------------------------------------

        simulation_time = (
            self.current_step
            * self.dt
        )

        # -------------------------------------------------
        # 15. Trolley extension ratio
        # -------------------------------------------------

        trolley_extension_ratio = (
            trolley_position
            / self.boom_length
        )

        # -------------------------------------------------
        # 16. Debug and DRL information
        # -------------------------------------------------

        info = {
            "status": status,

            "simulation_time": float(
                simulation_time
            ),

            "boom_length": float(
                self.boom_length
            ),

            "trolley_position": float(
                trolley_position
            ),

            "trolley_extension_ratio": float(
                trolley_extension_ratio
            ),

            "wind_velocity": float(
                wind_velocity
            ),

            "wind_force": float(
                wind_force
            ),

            "gust_active": bool(
                self.wind_model
                .gust_steps_remaining > 0
            ),

            "gust_steps_remaining": int(
                self.wind_model
                .gust_steps_remaining
            ),

            "load_offset": float(
                load_offset
            ),

            "swing_angle": float(
                new_swing_angle
            ),

            "swing_angular_velocity": float(
                new_angular_velocity
            ),

            "structural_tilt": float(
                new_structural_tilt
            ),

            "tilt_rate": float(
                new_tilt_rate
            ),

            "tilt_acceleration": float(
                tilt_acceleration
            ),

            "target_tilt": float(
                target_tilt
            ),

            "overturning_moment": float(
                overturning_moment
            ),

            "restoring_moment": float(
                restoring_moment
            ),

            "stability_margin": float(
                stability_margin
            ),

            # -----------------------------------------
            # Phase 3 risk information
            # -----------------------------------------

            "margin_ratio": float(
                margin_ratio
            ),

            "previous_margin_ratio": float(
                previous_margin_ratio
            ),

            "margin_improvement": float(
                margin_improvement
            ),

            # -----------------------------------------
            # Phase 3 reward components
            # -----------------------------------------

            "stability_reward": float(
                stability_reward
            ),

            "tilt_penalty": float(
                tilt_penalty
            ),

            "tilt_rate_penalty": float(
                tilt_rate_penalty
            ),

            "swing_penalty": float(
                swing_penalty
            ),

            "action_penalty": float(
                action_penalty
            ),

            "danger_penalty": float(
                danger_penalty
            ),

            "recovery_bonus": float(
                recovery_bonus
            ),

            "action_value": float(
                action_value
            ),

            "reward": float(
                reward
            )
        }

        observation = (
            self.state.copy()
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )