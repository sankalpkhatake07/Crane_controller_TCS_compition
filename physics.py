import numpy as np


# -----------------------------
# Physical constants
# -----------------------------
GRAVITY = 9.81          # m/s^2
AIR_DENSITY = 1.225     # kg/m^3


def calculate_wind_force(
    wind_velocity,
    drag_coefficient=1.2,
    exposed_area=10.0
):
    """
    Calculate lateral aerodynamic wind force.

    Formula:
        F = 0.5 * rho * Cd * A * v * |v|

    Using v * |v| preserves wind direction:
        positive velocity -> positive force
        negative velocity -> negative force
    """

    wind_force = (
        0.5
        * AIR_DENSITY
        * drag_coefficient
        * exposed_area
        * wind_velocity
        * abs(wind_velocity)
    )

    return wind_force


def calculate_pendulum_acceleration(
    swing_angle,
    angular_velocity,
    cable_length,
    load_mass,
    wind_force,
    damping_coefficient=0.15
):
    """
    Forced damped pendulum model.

    theta_ddot =
        -(g / L) * sin(theta)
        -(c / (m * L^2)) * theta_dot
        +(F_wind / (m * L)) * cos(theta)
    """

    gravity_term = (
        -(GRAVITY / cable_length)
        * np.sin(swing_angle)
    )

    damping_term = (
        -(damping_coefficient /
          (load_mass * cable_length ** 2))
        * angular_velocity
    )

    wind_term = (
        (wind_force /
         (load_mass * cable_length))
        * np.cos(swing_angle)
    )

    angular_acceleration = (
        gravity_term
        + damping_term
        + wind_term
    )

    return angular_acceleration


def update_pendulum_state(
    swing_angle,
    angular_velocity,
    angular_acceleration,
    dt
):
    """
    Semi-implicit Euler integration.

    First update angular velocity,
    then update swing angle.
    """

    new_angular_velocity = (
        angular_velocity
        + angular_acceleration * dt
    )

    new_swing_angle = (
        swing_angle
        + new_angular_velocity * dt
    )

    return new_swing_angle, new_angular_velocity


def calculate_load_horizontal_offset(
    trolley_position,
    cable_length,
    swing_angle
):
    """
    Effective horizontal load position
    relative to crane center.
    """

    load_offset = (
        trolley_position
        + cable_length * np.sin(swing_angle)
    )

    return load_offset


def calculate_overturning_moment(
    load_mass,
    load_horizontal_offset,
    wind_force,
    crane_height
):
    """
    Total simplified overturning moment.

    Load moment:
        M_load = m * g * x

    Wind moment:
        M_wind = F_wind * h
    """

    load_moment = (
        load_mass
        * GRAVITY
        * load_horizontal_offset
    )

    wind_moment = (
        wind_force
        * crane_height
    )

    total_overturning_moment = (
        load_moment
        + wind_moment
    )

    return total_overturning_moment


def calculate_restoring_moment(
    crane_mass,
    base_width
):
    """
    Simplified restoring moment:

        M_restore = M_crane * g * (B / 2)
    """

    restoring_moment = (
        crane_mass
        * GRAVITY
        * (base_width / 2.0)
    )

    return restoring_moment


def calculate_stability_margin(
    restoring_moment,
    overturning_moment
):
    """
    Positive margin -> stable
    Zero/negative -> tipping threshold crossed
    """

    stability_margin = (
        restoring_moment
        - abs(overturning_moment)
    )

    return stability_margin