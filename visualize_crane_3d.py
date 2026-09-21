"""
Crane Copilot: 3D Physics Simulation & AI Safety Visualizer
============================================================
Provides real-time 3D visualization of the crane stability environment.
Renders the 3D lattice tower mast, boom, counter-jib, counterweights,
traversing trolley, 3D hanging load pendulum, dynamic structural tilt,
wind vector, and stability telemetry.

Usage:
    python visualize_crane_3d.py          # Native 3D Matplotlib simulation
    python visualize_crane_3d.py --web    # Launch high-fidelity WebGL 3D simulator
"""

import sys
import os
import argparse
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib.widgets as widgets

from crane_env import CraneStabilityEnv
from physics import (
    calculate_wind_force,
    calculate_pendulum_acceleration,
    update_pendulum_state,
    calculate_overturning_moment,
    calculate_restoring_moment,
    calculate_stability_margin
)


def launch_web_3d():
    """Launch the rich WebGL 3D simulation in the default browser."""
    html_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "visualize_crane_3d.html")
    )
    if os.path.exists(html_path):
        print(f"Launching Crane Copilot 3D WebGL Simulation: {html_path}")
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
    else:
        print(f"Error: 3D HTML file not found at {html_path}")


class Crane3DVisualizer:
    def __init__(self):
        self.env = CraneStabilityEnv()
        self.obs, self.info = self.env.reset(seed=42)

        self.action = np.array([0.0], dtype=np.float32)
        self.paused = False
        self.copilot_active = True
        self.current_scenario = "SAFE"

        # 3D Sway perturbation state
        self.swing_angle_y = 0.0
        self.swing_velocity_y = 0.0

        # Figure & 3D Axes setup
        self.fig = plt.figure(figsize=(14, 9), facecolor="#0b0f19")
        self.fig.canvas.manager.set_window_title(
            "Crane Copilot — 3D Physics & Stability Simulation"
        )

        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#0b0f19")
        self.ax.set_box_aspect((1.2, 0.7, 1.0))

        # View angle
        self.ax.view_init(elev=22, azim=-60)

        # Plot elements references
        self.mast_lines = []
        self.boom_lines = []
        self.counter_boom_lines = []
        self.cable_line = None
        self.load_mesh = None
        self.trolley_point = None
        self.wind_quiver = None
        self.hud_text = None

        self._setup_scene()
        self._setup_controls()

    def _setup_scene(self):
        """Configure 3D limits, ground grid, and coordinate colors."""
        self.ax.set_xlim(-25, 60)
        self.ax.set_ylim(-35, 35)
        self.ax.set_zlim(0, 70)

        self.ax.set_xlabel("X: Boom Reach (m)", color="#94a3b8", labelpad=10)
        self.ax.set_ylabel("Y: Crosswind Sway (m)", color="#94a3b8", labelpad=10)
        self.ax.set_zlabel("Z: Height (m)", color="#94a3b8", labelpad=10)

        self.ax.tick_params(colors="#64748b")
        self.ax.grid(True, color="#1e293b", linestyle="--", alpha=0.5)

        # Draw Ground Grid
        gx = np.linspace(-25, 60, 18)
        gy = np.linspace(-35, 35, 15)
        for x in gx:
            self.ax.plot([x, x], [-35, 35], [0, 0], color="#1e293b", lw=0.6)
        for y in gy:
            self.ax.plot([-25, 60], [y, y], [0, 0], color="#1e293b", lw=0.6)

        # Concrete foundation slab
        bw = self.env.base_width / 2.0
        base_x = [-bw, bw, bw, -bw, -bw]
        base_y = [-bw, -bw, bw, bw, -bw]
        self.ax.plot(base_x, base_y, [0, 0, 0, 0, 0], color="#64748b", lw=3.0)

        # Cable line
        self.cable_line, = self.ax.plot([], [], [], color="#e2e8f0", lw=2.0)

        # Trolley point
        self.trolley_point, = self.ax.plot(
            [], [], [], marker="s", markersize=9, color="#f59e0b"
        )

        # Suspended load point / box
        self.load_point, = self.ax.plot(
            [], [], [], marker="o", markersize=14, color="#38bdf8"
        )

        # Telemetry HUD on figure
        self.hud_text = self.fig.text(
            0.02,
            0.78,
            "",
            fontsize=10,
            family="monospace",
            color="#f1f5f9",
            bbox=dict(boxstyle="round,pad=0.6", fc="#0f172a", ec="#334155", alpha=0.9),
        )

        self.status_text = self.fig.text(
            0.02,
            0.94,
            "STATUS: STABLE",
            fontsize=13,
            fontweight="bold",
            color="#22c55e",
            bbox=dict(boxstyle="round,pad=0.4", fc="#0f172a", ec="#22c55e", alpha=0.9),
        )

        # Instructions banner
        self.fig.text(
            0.02,
            0.02,
            "KEYS: [Left/Right] Trolley | [C] AI Copilot | [Space] Pause | [S] Safe | [W] Critical | [T] Tipping | [B] Open WebGL 3D",
            fontsize=9,
            color="#94a3b8",
        )

    def _setup_controls(self):
        """Keyboard interaction handlers."""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event):
        if event.key == "left":
            self.action[0] = -1.0
        elif event.key == "right":
            self.action[0] = 1.0
        elif event.key == "up":
            self.env.wind_model.base_wind += 2.0
        elif event.key == "down":
            self.env.wind_model.base_wind -= 2.0
        elif event.key == " ":
            self.paused = not self.paused
        elif event.key in ("c", "C"):
            self.copilot_active = not self.copilot_active
        elif event.key in ("s", "S"):
            self.set_scenario("SAFE")
        elif event.key in ("w", "W"):
            self.set_scenario("CRITICAL")
        elif event.key in ("t", "T"):
            self.set_scenario("TIPPING")
        elif event.key in ("b", "B"):
            launch_web_3d()

    def set_scenario(self, scenario):
        """Apply predefined scenario parameters."""
        self.current_scenario = scenario
        self.env.reset(seed=42)
        if scenario == "SAFE":
            self.env.state[0] = 20.0
            self.env.state[1] = 5000.0
            self.env.wind_model.base_wind = 5.0
        elif scenario == "CRITICAL":
            self.env.state[0] = 42.0
            self.env.state[1] = 8000.0
            self.env.wind_model.base_wind = 12.0
        elif scenario == "TIPPING":
            self.env.state[0] = 48.0
            self.env.state[1] = 18000.0
            self.env.wind_model.base_wind = 25.0
            self.copilot_active = False

        self.env.state[2] = self.env.wind_model.base_wind
        self.obs = self.env.state.copy()
        self.paused = False

    def _draw_lattice_truss(self, p1, p2, width, color="#f59e0b", lw=1.2):
        """Draw a simplified 3D truss box between two 3D points."""
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-4:
            return []

        # Coordinate frame along beam
        u = vec / length
        # Arbitrary normal
        n1 = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(u, n1)
        v = v / np.linalg.norm(v)
        w = np.cross(u, v)

        hw = width / 2.0
        offsets = [-hw * v - hw * w, hw * v - hw * w, hw * v + hw * w, -hw * v + hw * w]

        lines = []
        for off in offsets:
            c1 = p1 + off
            c2 = p2 + off
            line, = self.ax.plot(
                [c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], color=color, lw=lw
            )
            lines.append(line)
        return lines

    def update(self, frame):
        """Animation update step."""
        if not self.paused:
            # AI Copilot logic
            if self.copilot_active:
                margin_ratio = self.obs[7] / (
                    self.env.crane_mass * 9.81 * (self.env.base_width / 2.0)
                )
                if margin_ratio < 0.30:
                    self.action[0] = -1.0  # Retract trolley
                elif abs(self.obs[4]) > 0.05:
                    self.action[0] = -np.sign(self.obs[4]) * 0.4
                else:
                    self.action[0] = 0.0

            # Step physics
            self.obs, reward, terminated, truncated, self.info = self.env.step(self.action)

            # Lateral sway model
            wind_lateral_accel = - (9.81 / self.env.cable_length) * np.sin(self.swing_angle_y) - 0.2 * self.swing_velocity_y
            self.swing_velocity_y += wind_lateral_accel * self.env.dt
            self.swing_angle_y += self.swing_velocity_y * self.env.dt

            if terminated:
                self.paused = True

        # Read physics states
        trolley_x = float(self.obs[0])
        load_mass = float(self.obs[1])
        wind_v = float(self.obs[2])
        swing_x = float(self.obs[3])
        swing_rate_x = float(self.obs[4])
        structural_tilt = float(self.obs[5])
        stability_margin = float(self.obs[7])
        restoring_m = float(self.info.get("restoring_moment", 4905000.0))
        status = self.info.get("status", "stable")

        # ── Clear previous structural lines ──
        for l in self.mast_lines:
            l.remove()
        self.mast_lines.clear()

        for l in self.boom_lines:
            l.remove()
        self.boom_lines.clear()

        for l in self.counter_boom_lines:
            l.remove()
        self.counter_boom_lines.clear()

        # Mast deflection angle & top point
        mast_height = self.env.crane_height
        mast_top_x = -mast_height * np.sin(structural_tilt)
        mast_top_z = mast_height * np.cos(structural_tilt)

        p_base = np.array([0.0, 0.0, 0.0])
        p_top = np.array([mast_top_x, 0.0, mast_top_z])

        # Draw 3D Mast
        self.mast_lines = self._draw_lattice_truss(
            p_base, p_top, width=2.4, color="#f59e0b", lw=2.0
        )

        # Boom (X direction)
        boom_len = self.env.boom_length
        p_boom_tip = p_top + np.array([boom_len * np.cos(structural_tilt), 0.0, boom_len * np.sin(structural_tilt)])
        self.boom_lines = self._draw_lattice_truss(
            p_top, p_boom_tip, width=1.6, color="#f59e0b", lw=1.5
        )

        # Counter-Boom (negative X direction)
        c_len = 16.0
        p_counter_tip = p_top + np.array([-c_len * np.cos(structural_tilt), 0.0, -c_len * np.sin(structural_tilt)])
        self.counter_boom_lines = self._draw_lattice_truss(
            p_top, p_counter_tip, width=1.6, color="#475569", lw=1.5
        )

        # Counterweight block
        self.counter_boom_lines.extend(
            self._draw_lattice_truss(
                p_counter_tip + np.array([1.0, 0, 0]),
                p_counter_tip + np.array([4.0, 0, 0]),
                width=2.5,
                color="#64748b",
                lw=3.0
            )
        )

        # Trolley position in 3D
        trolley_3d_x = mast_top_x + trolley_x * np.cos(structural_tilt)
        trolley_3d_z = mast_top_z + trolley_x * np.sin(structural_tilt)
        self.trolley_point.set_data([trolley_3d_x], [0.0])
        self.trolley_point.set_3d_properties([trolley_3d_z])

        # Load position in 3D with spherical pendulum
        cable_l = self.env.cable_length
        load_3d_x = trolley_3d_x + cable_l * np.sin(swing_x)
        load_3d_y = cable_l * np.sin(self.swing_angle_y)
        load_3d_z = trolley_3d_z - cable_l * np.cos(swing_x)

        self.load_point.set_data([load_3d_x], [load_3d_y])
        self.load_point.set_3d_properties([load_3d_z])

        # Cable line
        self.cable_line.set_data([trolley_3d_x, load_3d_x], [0.0, load_3d_y])
        self.cable_line.set_3d_properties([trolley_3d_z, load_3d_z])

        # Update HUD Text
        sim_time = self.info.get("simulation_time", 0.0)
        overturn_m = restoring_m - stability_margin
        ratio = (stability_margin / restoring_m) * 100.0

        copilot_str = "ACTIVE (AUTO)" if self.copilot_active else "MANUAL (OFF)"
        self.hud_text.set_text(
            f"SIM TIME:         {sim_time:5.2f} s\n"
            f"WIND VELOCITY:    {wind_v:5.2f} m/s\n"
            f"TROLLEY POSITION: {trolley_x:5.2f} m ({trolley_x/boom_len*100:4.1f}%)\n"
            f"PAYLOAD MASS:     {load_mass:7.0f} kg\n"
            f"SWING ANGLE:      {swing_x*180/np.pi:5.2f} deg\n"
            f"STRUCTURAL TILT:  {structural_tilt*180/np.pi:5.2f} deg\n"
            f"OVERTURNING M:    {overturn_m:11,.0f} N.m\n"
            f"STABILITY MARGIN: {stability_margin:11,.0f} N.m\n"
            f"RESERVE RATIO:    {ratio:5.1f} %\n"
            f"AI CO-PILOT:      {copilot_str}"
        )

        # Status badge update
        status_colors = {
            "stable": "#22c55e",
            "warning": "#f59e0b",
            "critical": "#f97316",
            "tipping": "#ef4444",
        }
        col = status_colors.get(status, "#22c55e")
        self.status_text.set_text(f"STATUS: {status.upper()}")
        self.status_text.set_color(col)
        self.status_text.get_bbox_patch().set_edgecolor(col)

        return [self.trolley_point, self.load_point, self.cable_line, self.hud_text, self.status_text]


def main():
    parser = argparse.ArgumentParser(description="Crane Copilot 3D Physics Simulation")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the interactive Three.js / WebGL 3D simulation in your web browser",
    )
    args = parser.parse_args()

    if args.web:
        launch_web_3d()
        return

    print("=================================================================")
    print("CRANE COPILOT: 3D PHYSICS & STABILITY SIMULATION")
    print("=================================================================")
    print("Controls:")
    print("  [Left] / [Right] Arrow : Move trolley inward / outward")
    print("  [Up] / [Down] Arrow    : Adjust base wind velocity")
    print("  [C]                    : Toggle AI Co-Pilot (Autonomous Stabilization)")
    print("  [Space]                : Pause / Resume simulation")
    print("  [S]                    : Load SAFE Scenario")
    print("  [W]                    : Load CRITICAL Scenario")
    print("  [T]                    : Load TIPPING Scenario")
    print("  [B]                    : Open high-fidelity WebGL 3D simulator in browser")
    print("=================================================================\n")

    app = Crane3DVisualizer()
    anim = FuncAnimation(app.fig, app.update, interval=20, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
