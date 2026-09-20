from physics import (
    calculate_wind_force,
    calculate_pendulum_acceleration,
    update_pendulum_state,
    calculate_load_horizontal_offset,
    calculate_overturning_moment,
    calculate_restoring_moment,
    calculate_stability_margin
)


# -----------------------------
# Test crane parameters
# -----------------------------
wind_velocity = 20.0       # m/s
load_mass = 5000.0         # kg
cable_length = 15.0        # m
crane_height = 50.0        # m
trolley_position = 20.0    # m

crane_mass = 100000.0      # kg
base_width = 10.0          # m

swing_angle = 0.0          # radians
angular_velocity = 0.0     # rad/s

dt = 0.02                  # 20 ms simulation timestep


# -----------------------------
# Wind force
# -----------------------------
wind_force = calculate_wind_force(
    wind_velocity
)

print("Wind force:", wind_force, "N")


# -----------------------------
# Pendulum acceleration
# -----------------------------
angular_acceleration = calculate_pendulum_acceleration(
    swing_angle,
    angular_velocity,
    cable_length,
    load_mass,
    wind_force
)

print(
    "Angular acceleration:",
    angular_acceleration,
    "rad/s^2"
)


# -----------------------------
# Update pendulum
# -----------------------------
new_angle, new_velocity = update_pendulum_state(
    swing_angle,
    angular_velocity,
    angular_acceleration,
    dt
)

print("New swing angle:", new_angle, "rad")
print("New angular velocity:", new_velocity, "rad/s")


# -----------------------------
# Load horizontal offset
# -----------------------------
load_offset = calculate_load_horizontal_offset(
    trolley_position,
    cable_length,
    new_angle
)

print("Load horizontal offset:", load_offset, "m")


# -----------------------------
# Moments
# -----------------------------
overturning_moment = calculate_overturning_moment(
    load_mass,
    load_offset,
    wind_force,
    crane_height
)

restoring_moment = calculate_restoring_moment(
    crane_mass,
    base_width
)

stability_margin = calculate_stability_margin(
    restoring_moment,
    overturning_moment
)

print("Overturning moment:", overturning_moment, "N.m")
print("Restoring moment:", restoring_moment, "N.m")
print("Stability margin:", stability_margin, "N.m")


# -----------------------------
# Tipping decision
# -----------------------------
if stability_margin <= 0:
    print("STATUS: TIPPING DANGER")
else:
    print("STATUS: STABLE")