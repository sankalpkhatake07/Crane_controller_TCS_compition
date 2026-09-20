import numpy as np


class WindGustModel:
    """
    Time-varying wind model for crane simulation.

    Components:
    1. Base wind with Ornstein-Uhlenbeck mean-reversion
    2. Correlated random turbulence (no unbounded drift)
    3. Sudden temporary gust events

    Uses Gymnasium's seeded random generator
    for reproducible experiments.

    Ornstein-Uhlenbeck update:
        dW = theta * (mu - W) * dt + sigma * sqrt(dt) * eps
    where:
        theta = mean-reversion speed (default 0.5)
        mu    = base_wind (long-run mean)
        sigma = turbulence_strength
    """

    def __init__(
        self,
        base_wind=5.0,
        turbulence_strength=1.5,
        gust_probability=0.01,
        gust_min=8.0,
        gust_max=20.0,
        max_wind=50.0,
        ou_theta=0.5
    ):
        self.base_wind = base_wind
        self.turbulence_strength = turbulence_strength
        self.gust_probability = gust_probability

        self.gust_min = gust_min
        self.gust_max = gust_max
        self.max_wind = max_wind

        # Mean-reversion speed for OU process
        self.ou_theta = ou_theta

        self.current_wind = base_wind
        self.active_gust = 0.0
        self.gust_steps_remaining = 0


    def reset(self):
        """
        Reset wind model to initial condition.
        """

        self.current_wind = self.base_wind
        self.active_gust = 0.0
        self.gust_steps_remaining = 0

        return self.current_wind


    def update(self, rng, dt):
        """
        Advance wind simulation by one timestep.

        Args:
            rng:
                Gymnasium seeded random generator

            dt:
                Simulation timestep in seconds

        Returns:
            float:
                Updated signed wind velocity (m/s)
        """

        # -----------------------------
        # 1. Ornstein-Uhlenbeck turbulence
        #
        # Mean-reversion prevents the wind from
        # drifting to unrealistic extreme values
        # over long episodes.
        # -----------------------------
        random_noise = rng.normal(
            loc=0.0,
            scale=self.turbulence_strength
        )

        mean_reversion = (
            self.ou_theta
            * (self.base_wind - self.current_wind)
            * dt
        )

        diffusion = (
            random_noise * np.sqrt(dt)
        )

        self.current_wind += (
            mean_reversion + diffusion
        )


        # -----------------------------
        # 2. Possibly trigger new gust
        # -----------------------------
        if self.gust_steps_remaining <= 0:

            if rng.random() < self.gust_probability:

                gust_magnitude = rng.uniform(
                    self.gust_min,
                    self.gust_max
                )

                # Wind can attack from either side
                gust_direction = rng.choice(
                    [-1.0, 1.0]
                )

                self.active_gust = (
                    gust_direction
                    * gust_magnitude
                )

                # Gust lasts 0.3 to 1.5 seconds
                gust_duration = rng.uniform(
                    0.3,
                    1.5
                )

                self.gust_steps_remaining = max(
                    1,
                    int(gust_duration / dt)
                )


        # -----------------------------
        # 3. Apply active gust
        # -----------------------------
        if self.gust_steps_remaining > 0:

            gust_wind = self.active_gust

            self.gust_steps_remaining -= 1

        else:
            gust_wind = 0.0
            self.active_gust = 0.0


        # -----------------------------
        # 4. Total wind
        # -----------------------------
        total_wind = (
            self.current_wind
            + gust_wind
        )

        total_wind = np.clip(
            total_wind,
            -self.max_wind,
            self.max_wind
        )

        return float(total_wind)