import numpy as np

from wind import WindGustModel


# Reproducible random generator
rng = np.random.default_rng(42)

wind_model = WindGustModel(
    base_wind=5.0,
    turbulence_strength=1.5,
    gust_probability=0.05
)

dt = 0.02

wind_model.reset()

print("Starting wind simulation...\n")


for step in range(200):

    wind_velocity = wind_model.update(
        rng=rng,
        dt=dt
    )

    # Print every 10 simulation steps
    if step % 10 == 0:

        time_seconds = step * dt

        print(
            f"Time: {time_seconds:4.2f} s | "
            f"Wind: {wind_velocity:7.3f} m/s | "
            f"Gust remaining: "
            f"{wind_model.gust_steps_remaining}"
        )