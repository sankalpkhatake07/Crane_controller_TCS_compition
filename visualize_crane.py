import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from crane_env import CraneStabilityEnv


# =========================================================
# CREATE ENVIRONMENT
# =========================================================

env = CraneStabilityEnv()

# Fixed seed for reproducible demonstration
obs, info = env.reset(seed=42)

# Neutral action:
# trolley does not intentionally move
action = np.array(
    [0.0],
    dtype=np.float32
)


# =========================================================
# FIGURE SETUP
# =========================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.set_title(
    "Agentic AI Crane Co-Pilot - 2D Physics Simulation",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlim(
    -20,
    env.boom_length + 15
)

ax.set_ylim(
    -5,
    env.crane_height + 15
)

ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.set_xlabel(
    "Horizontal Position (m)"
)

ax.set_ylabel(
    "Height (m)"
)

ax.grid(
    True,
    alpha=0.3
)


# =========================================================
# GROUND
# =========================================================

ground_line, = ax.plot(
    [-20, env.boom_length + 15],
    [0, 0],
    linewidth=3
)


# =========================================================
# CRANE OBJECTS
# =========================================================

# Crane mast
mast_line, = ax.plot(
    [],
    [],
    linewidth=6
)

# Boom
boom_line, = ax.plot(
    [],
    [],
    linewidth=4
)

# Cable
cable_line, = ax.plot(
    [],
    [],
    linewidth=2
)

# Trolley
trolley_point, = ax.plot(
    [],
    [],
    marker="s",
    markersize=10
)

# Hanging load
load_point, = ax.plot(
    [],
    [],
    marker="o",
    markersize=14
)

# Crane base
base_line, = ax.plot(
    [],
    [],
    linewidth=8
)


# =========================================================
# WIND ARROW
# =========================================================

wind_arrow = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=3
    )
)


# =========================================================
# INFORMATION TEXT
# =========================================================

info_text = ax.text(
    0.02,
    0.97,
    "",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=10,
    family="monospace",
    bbox=dict(
        boxstyle="round",
        alpha=0.85
    )
)


status_text = ax.text(
    0.5,
    0.93,
    "",
    transform=ax.transAxes,
    horizontalalignment="center",
    fontsize=14,
    fontweight="bold"
)


# =========================================================
# INITIALIZATION FUNCTION
# =========================================================

def init():

    mast_line.set_data(
        [],
        []
    )

    boom_line.set_data(
        [],
        []
    )

    cable_line.set_data(
        [],
        []
    )

    trolley_point.set_data(
        [],
        []
    )

    load_point.set_data(
        [],
        []
    )

    base_line.set_data(
        [],
        []
    )

    info_text.set_text("")
    status_text.set_text("")

    return (
        mast_line,
        boom_line,
        cable_line,
        trolley_point,
        load_point,
        base_line,
        info_text,
        status_text
    )


# =========================================================
# ANIMATION UPDATE
# =========================================================

def update(frame):

    global obs, info

    # -----------------------------------------------------
    # Advance physics simulation
    # -----------------------------------------------------

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    # -----------------------------------------------------
    # Read state
    # -----------------------------------------------------

    trolley_position = float(obs[0])
    wind_velocity = float(obs[2])
    swing_angle = float(obs[3])
    structural_tilt = float(obs[5])

    # -----------------------------------------------------
    # Crane geometry
    # -----------------------------------------------------

    # Base pivot
    base_x = 0.0
    base_y = 0.0

    # Mast top moves slightly due to structural tilt
    mast_top_x = (
        env.crane_height
        * np.sin(structural_tilt)
    )

    mast_top_y = (
        env.crane_height
        * np.cos(structural_tilt)
    )

    # Boom follows structural tilt
    boom_end_x = (
        mast_top_x
        + env.boom_length
        * np.cos(structural_tilt)
    )

    boom_end_y = (
        mast_top_y
        + env.boom_length
        * np.sin(structural_tilt)
    )

    # Trolley position along boom
    trolley_x = (
        mast_top_x
        + trolley_position
        * np.cos(structural_tilt)
    )

    trolley_y = (
        mast_top_y
        + trolley_position
        * np.sin(structural_tilt)
    )

    # -----------------------------------------------------
    # Hanging load geometry
    # -----------------------------------------------------

    load_x = (
        trolley_x
        + env.cable_length
        * np.sin(swing_angle)
    )

    load_y = (
        trolley_y
        - env.cable_length
        * np.cos(swing_angle)
    )

    # -----------------------------------------------------
    # Update crane drawing
    # -----------------------------------------------------

    mast_line.set_data(
        [base_x, mast_top_x],
        [base_y, mast_top_y]
    )

    boom_line.set_data(
        [mast_top_x, boom_end_x],
        [mast_top_y, boom_end_y]
    )

    base_line.set_data(
        [
            -env.base_width / 2,
            env.base_width / 2
        ],
        [0, 0]
    )

    trolley_point.set_data(
        [trolley_x],
        [trolley_y]
    )

    cable_line.set_data(
        [trolley_x, load_x],
        [trolley_y, load_y]
    )

    load_point.set_data(
        [load_x],
        [load_y]
    )

    # -----------------------------------------------------
    # Update wind arrow
    # -----------------------------------------------------

    arrow_start_x = -15.0
    arrow_start_y = env.crane_height * 0.70

    # Scale only for visualization
    wind_visual_scale = 0.5

    arrow_end_x = (
        arrow_start_x
        + wind_velocity
        * wind_visual_scale
    )

    wind_arrow.xy = (
        arrow_end_x,
        arrow_start_y
    )

    wind_arrow.set_position(
        (
            arrow_start_x,
            arrow_start_y
        )
    )

    # -----------------------------------------------------
    # Status classification for visualization
    # -----------------------------------------------------

    stability_margin = info[
        "stability_margin"
    ]

    restoring_moment = info[
        "restoring_moment"
    ]

    margin_ratio = (
        stability_margin
        / restoring_moment
    )

    if terminated:
        visual_status = "TIPPING"

    elif margin_ratio < 0.20:
        visual_status = "CRITICAL"

    elif margin_ratio < 0.50:
        visual_status = "WARNING"

    else:
        visual_status = "STABLE"

    # -----------------------------------------------------
    # Information panel
    # -----------------------------------------------------

    info_text.set_text(
        f"Time:          "
        f"{info['simulation_time']:6.2f} s\n"

        f"Wind:          "
        f"{wind_velocity:6.2f} m/s\n"

        f"Wind Force:    "
        f"{info['wind_force']:9.2f} N\n"

        f"Trolley:       "
        f"{trolley_position:6.2f} m\n"

        f"Boom Length:   "
        f"{info['boom_length']:6.2f} m\n"

        f"Extension:     "
        f"{info['trolley_extension_ratio'] * 100:6.2f} %\n"

        f"Swing Angle:   "
        f"{np.degrees(swing_angle):6.2f} deg\n"

        f"Struct. Tilt:  "
        f"{np.degrees(structural_tilt):6.3f} deg\n"

        f"Margin:        "
        f"{stability_margin / 1e6:6.3f} MN.m\n"

        f"Reward:        "
        f"{reward:6.3f}"
    )

    status_text.set_text(
        f"STATUS: {visual_status}"
    )

    # -----------------------------------------------------
    # Stop or reset at episode end
    # -----------------------------------------------------

    if terminated or truncated:

        obs, info = env.reset(
            seed=42
        )

    return (
        mast_line,
        boom_line,
        cable_line,
        trolley_point,
        load_point,
        base_line,
        info_text,
        status_text
    )


# =========================================================
# CREATE ANIMATION
# =========================================================

animation = FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=20,
    blit=False,
    cache_frame_data=False
)


# =========================================================
# SHOW WINDOW
# =========================================================

plt.tight_layout()
plt.show()


# =========================================================
# CLEANUP
# =========================================================

env.close()