import tkinter as tk
import math
import numpy as np

from crane_env import CraneStabilityEnv


class CraneVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Agentic AI Crane Co-Pilot - 2D Physics Simulation"
        )

        self.width = 1200
        self.height = 760

        # ---------------------------------------------
        # Top control panel
        # ---------------------------------------------
        self.control_frame = tk.Frame(
            root,
            bg="#17202A",
            pady=8
        )
        self.control_frame.pack(fill="x")

        tk.Label(
            self.control_frame,
            text="SCENARIO CONTROL:",
            bg="#17202A",
            fg="white",
            font=("Segoe UI", 11, "bold")
        ).pack(side="left", padx=(15, 10))

        tk.Button(
            self.control_frame,
            text="SAFE",
            command=lambda: self.set_scenario("SAFE"),
            bg="#1E8449",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=12
        ).pack(side="left", padx=5)

        tk.Button(
            self.control_frame,
            text="CRITICAL",
            command=lambda: self.set_scenario("CRITICAL"),
            bg="#D68910",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=12
        ).pack(side="left", padx=5)

        tk.Button(
            self.control_frame,
            text="TIPPING",
            command=lambda: self.set_scenario("TIPPING"),
            bg="#C0392B",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=12
        ).pack(side="left", padx=5)

        tk.Button(
            self.control_frame,
            text="PAUSE / RESUME",
            command=self.toggle_pause,
            bg="#2E86C1",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=15
        ).pack(side="left", padx=20)

        # ---------------------------------------------
        # Live parameter controls
        # ---------------------------------------------
        self.parameter_frame = tk.Frame(
            root,
            bg="#273746",
            pady=5
        )
        self.parameter_frame.pack(fill="x")

        self.control_values = {}
        controls = [
            ("Load kg", "load_mass", 1000, 20000, 5000),
            ("Trolley m", "trolley_position", 0, 50, 20),
            ("Base wind", "base_wind", -30, 30, 5),
            ("Turbulence", "turbulence_strength", 0, 10, 1.5),
            ("Gust chance", "gust_probability", 0, 0.1, 0.01),
            ("Gust max", "gust_max", 0, 50, 20),
            ("Height m", "crane_height", 10, 80, 50),
            ("Boom m", "boom_length", 10, 80, 50),
            ("Cable m", "cable_length", 2, 30, 15),
            ("Trolley speed", "trolley_speed", 0.1, 5, 0.5),
            ("Trolley ctl", "action", -1, 1, 0),
        ]

        for column, (label, key, minimum, maximum, value) in enumerate(controls):
            frame = tk.Frame(self.parameter_frame, bg="#273746")
            frame.grid(row=column // 6, column=column % 6, padx=3)
            tk.Label(
                frame,
                text=label,
                bg="#273746",
                fg="#D5DBDB",
                font=("Segoe UI", 8)
            ).pack()
            resolution = 0.01 if key in {"turbulence_strength", "gust_probability"} else 0.1
            if key in {"load_mass", "trolley_position", "boom_length", "cable_length"}:
                resolution = 1
            scale = tk.Scale(
                frame,
                from_=minimum,
                to=maximum,
                resolution=resolution,
                orient="horizontal",
                length=105,
                showvalue=True,
                variable=tk.DoubleVar(value=value),
                bg="#273746",
                fg="white",
                troughcolor="#566573",
                highlightthickness=0
            )
            scale.pack()
            self.control_values[key] = scale

        tk.Button(
            self.parameter_frame,
            text="APPLY",
            command=self.apply_parameters,
            bg="#148F77",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            width=9
        ).grid(row=1, column=6, padx=8, pady=12)

        # ---------------------------------------------
        # Canvas
        # ---------------------------------------------
        self.canvas = tk.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#101820"
        )
        self.canvas.pack()

        # ---------------------------------------------
        # Environment
        # ---------------------------------------------
        self.env = CraneStabilityEnv()

        self.current_scenario = "SAFE"

        self.obs, self.info = self.env.reset(
            seed=42
        )

        self.action = np.array(
            [0.0],
            dtype=np.float32
        )

        # ---------------------------------------------
        # Animation settings
        # ---------------------------------------------
        self.running = True
        self.paused = False
        self.frame_delay_ms = 20

        self.scale = 9.0

        self.origin_x = 300
        self.origin_y = 650

        self.update_animation()

    def apply_parameters(self):
        """Apply the live controls and restart the current episode."""
        self.env.crane_height = float(self.control_values["crane_height"].get())
        self.env.boom_length = float(self.control_values["boom_length"].get())
        self.env.cable_length = float(self.control_values["cable_length"].get())
        self.env.trolley_speed = float(self.control_values["trolley_speed"].get())
        self.env.wind_model.base_wind = float(self.control_values["base_wind"].get())
        self.env.wind_model.turbulence_strength = float(self.control_values["turbulence_strength"].get())
        self.env.wind_model.gust_probability = float(self.control_values["gust_probability"].get())
        self.env.wind_model.gust_max = float(self.control_values["gust_max"].get())
        self.action[0] = float(self.control_values["action"].get())

        self.env.reset(seed=42)
        self.env.state[0] = float(self.control_values["trolley_position"].get())
        self.env.state[1] = float(self.control_values["load_mass"].get())
        self.env.state[2] = self.env.wind_model.base_wind
        self.obs = self.env.state.copy()
        self.info = {}
        self.current_scenario = "CUSTOM"
        self.paused = False


    def world_to_screen(self, x, y):
        screen_x = self.origin_x + x * self.scale
        screen_y = self.origin_y - y * self.scale

        return screen_x, screen_y


    def draw_arrow(
        self,
        x1,
        y1,
        x2,
        y2,
        color,
        width=4
    ):
        self.canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill=color,
            width=width,
            arrow=tk.LAST,
            arrowshape=(14, 18, 6)
        )


    def toggle_pause(self):
        self.paused = not self.paused


    def set_scenario(self, scenario):
        """
        Load one of the three validated
        prototype scenarios.
        """

        self.current_scenario = scenario

        # ---------------------------------------------
        # Scenario parameters
        # ---------------------------------------------
        if scenario == "SAFE":
            load_mass = 5000.0
            trolley_position = 20.0
            base_wind = 5.0

        elif scenario == "CRITICAL":
            load_mass = 9000.0
            trolley_position = 42.0
            base_wind = 12.0

        elif scenario == "TIPPING":
            load_mass = 12000.0
            trolley_position = 48.0
            base_wind = 18.0

        else:
            return

        # Configure wind model before reset
        self.env.wind_model.base_wind = base_wind

        # Reset complete episode
        self.obs, self.info = self.env.reset(
            seed=42
        )

        # Apply scenario state
        self.env.state[0] = trolley_position
        self.env.state[1] = load_mass
        self.env.state[2] = base_wind

        # Keep local observation synchronized
        self.obs = self.env.state.copy()

        # Resume automatically
        self.paused = False


    def update_animation(self):

        if not self.running:
            return

        # ---------------------------------------------
        # Advance simulation only when not paused
        # ---------------------------------------------
        if not self.paused:

            (
                self.obs,
                reward,
                terminated,
                truncated,
                self.info
            ) = self.env.step(
                self.action
            )

        else:
            reward = 0.0
            terminated = False
            truncated = False

        # ---------------------------------------------
        # Read state
        # ---------------------------------------------
        trolley_position = float(
            self.obs[0]
        )

        load_mass = float(
            self.obs[1]
        )

        wind_velocity = float(
            self.obs[2]
        )

        swing_angle = float(
            self.obs[3]
        )

        structural_tilt = float(
            self.obs[5]
        )

        # ---------------------------------------------
        # Crane geometry
        # ---------------------------------------------
        base_x = 0.0
        base_y = 0.0

        mast_top_x = (
            self.env.crane_height
            * math.sin(structural_tilt)
        )

        mast_top_y = (
            self.env.crane_height
            * math.cos(structural_tilt)
        )

        boom_end_x = (
            mast_top_x
            + self.env.boom_length
            * math.cos(structural_tilt)
        )

        boom_end_y = (
            mast_top_y
            + self.env.boom_length
            * math.sin(structural_tilt)
        )

        trolley_x = (
            mast_top_x
            + trolley_position
            * math.cos(structural_tilt)
        )

        trolley_y = (
            mast_top_y
            + trolley_position
            * math.sin(structural_tilt)
        )

        load_x = (
            trolley_x
            + self.env.cable_length
            * math.sin(swing_angle)
        )

        load_y = (
            trolley_y
            - self.env.cable_length
            * math.cos(swing_angle)
        )

        # ---------------------------------------------
        # Convert to screen
        # ---------------------------------------------
        bx, by = self.world_to_screen(
            base_x,
            base_y
        )

        mx, my = self.world_to_screen(
            mast_top_x,
            mast_top_y
        )

        ex, ey = self.world_to_screen(
            boom_end_x,
            boom_end_y
        )

        tx, ty = self.world_to_screen(
            trolley_x,
            trolley_y
        )

        lx, ly = self.world_to_screen(
            load_x,
            load_y
        )

        # ---------------------------------------------
        # Clear frame
        # ---------------------------------------------
        self.canvas.delete("all")

        # ---------------------------------------------
        # Title
        # ---------------------------------------------
        self.canvas.create_text(
            self.width / 2,
            28,
            text="AGENTIC AI CRANE CO-PILOT",
            fill="white",
            font=("Segoe UI", 20, "bold")
        )

        self.canvas.create_text(
            self.width / 2,
            58,
            text=(
                "Live 2D Physics Environment | "
                f"Scenario: {self.current_scenario}"
            ),
            fill="#AAB7B8",
            font=("Segoe UI", 11)
        )

        # ---------------------------------------------
        # Ground
        # ---------------------------------------------
        self.canvas.create_line(
            30,
            self.origin_y,
            self.width - 30,
            self.origin_y,
            fill="#7F8C8D",
            width=5
        )

        for x in range(
            40,
            self.width - 40,
            40
        ):
            self.canvas.create_line(
                x,
                self.origin_y,
                x + 15,
                self.origin_y + 10,
                fill="#566573",
                width=2
            )

        # ---------------------------------------------
        # Base
        # ---------------------------------------------
        base_half_pixels = (
            self.env.base_width
            * self.scale
            / 2
        )

        self.canvas.create_line(
            bx - base_half_pixels,
            by,
            bx + base_half_pixels,
            by,
            fill="#F4D03F",
            width=14
        )

        # ---------------------------------------------
        # Mast
        # ---------------------------------------------
        self.canvas.create_line(
            bx,
            by,
            mx,
            my,
            fill="#F39C12",
            width=10
        )

        # ---------------------------------------------
        # Boom
        # ---------------------------------------------
        self.canvas.create_line(
            mx,
            my,
            ex,
            ey,
            fill="#F1C40F",
            width=7
        )

        # ---------------------------------------------
        # Trolley
        # ---------------------------------------------
        trolley_size = 8

        self.canvas.create_rectangle(
            tx - trolley_size,
            ty - trolley_size,
            tx + trolley_size,
            ty + trolley_size,
            fill="#3498DB",
            outline="white",
            width=2
        )

        # ---------------------------------------------
        # Cable
        # ---------------------------------------------
        self.canvas.create_line(
            tx,
            ty,
            lx,
            ly,
            fill="#ECF0F1",
            width=3
        )

        # ---------------------------------------------
        # Load
        # ---------------------------------------------
        load_size = 13

        self.canvas.create_oval(
            lx - load_size,
            ly - load_size,
            lx + load_size,
            ly + load_size,
            fill="#E74C3C",
            outline="white",
            width=2
        )

        # ---------------------------------------------
        # Wind arrow
        # ---------------------------------------------
        wind_start_x = 60
        wind_start_y = 220

        wind_scale = 6.0

        wind_end_x = (
            wind_start_x
            + wind_velocity * wind_scale
        )

        self.draw_arrow(
            wind_start_x,
            wind_start_y,
            wind_end_x,
            wind_start_y,
            "#5DADE2",
            width=5
        )

        self.canvas.create_text(
            wind_start_x,
            wind_start_y - 25,
            anchor="w",
            text=f"WIND {wind_velocity:.2f} m/s",
            fill="#85C1E9",
            font=("Segoe UI", 12, "bold")
        )

        # ---------------------------------------------
        # Stability values
        # ---------------------------------------------
        stability_margin = float(
            self.info["stability_margin"]
        )

        restoring_moment = float(
            self.info["restoring_moment"]
        )

        margin_ratio = (
            stability_margin
            / restoring_moment
        )

        if terminated:
            visual_status = "TIPPING"
            status_color = "#FF1744"

        elif margin_ratio < 0.20:
            visual_status = "CRITICAL"
            status_color = "#FF3D00"

        elif margin_ratio < 0.50:
            visual_status = "WARNING"
            status_color = "#FFC107"

        else:
            visual_status = "STABLE"
            status_color = "#00E676"

        # ---------------------------------------------
        # Status box
        # ---------------------------------------------
        self.canvas.create_rectangle(
            875,
            90,
            1165,
            150,
            fill="#17202A",
            outline=status_color,
            width=3
        )

        self.canvas.create_text(
            1020,
            120,
            text=f"STATUS: {visual_status}",
            fill=status_color,
            font=("Segoe UI", 18, "bold")
        )

        # ---------------------------------------------
        # Information panel
        # ---------------------------------------------
        self.canvas.create_rectangle(
            855,
            175,
            1175,
            585,
            fill="#17202A",
            outline="#5D6D7E",
            width=2
        )

        info_lines = [
            (
                "Scenario",
                self.current_scenario
            ),
            (
                "Simulation Time",
                f"{self.info['simulation_time']:.2f} s"
            ),
            (
                "Load Mass",
                f"{load_mass:.0f} kg"
            ),
            (
                "Wind Velocity",
                f"{wind_velocity:.2f} m/s"
            ),
            (
                "Wind Force",
                f"{self.info['wind_force']:.2f} N"
            ),
            (
                "Boom Length",
                f"{self.info['boom_length']:.2f} m"
            ),
            (
                "Trolley Position",
                f"{trolley_position:.2f} m"
            ),
            (
                "Boom Extension",
                f"{self.info['trolley_extension_ratio'] * 100:.1f} %"
            ),
            (
                "Swing Angle",
                f"{math.degrees(swing_angle):.2f} deg"
            ),
            (
                "Structural Tilt",
                f"{math.degrees(structural_tilt):.3f} deg"
            ),
            (
                "Stability Margin",
                f"{stability_margin / 1e6:.3f} MN.m"
            ),
            (
                "Reward",
                f"{reward:.3f}"
            )
        ]

        y_position = 200

        for label, value in info_lines:

            self.canvas.create_text(
                875,
                y_position,
                anchor="w",
                text=label,
                fill="#AAB7B8",
                font=("Segoe UI", 10)
            )

            self.canvas.create_text(
                1150,
                y_position,
                anchor="e",
                text=value,
                fill="white",
                font=("Consolas", 10, "bold")
            )

            y_position += 31

        # ---------------------------------------------
        # Gust indicator
        # ---------------------------------------------
        if self.info["gust_active"]:

            self.canvas.create_text(
                600,
                100,
                text="WIND GUST ACTIVE",
                fill="#FF7043",
                font=("Segoe UI", 16, "bold")
            )

        # ---------------------------------------------
        # Crane labels
        # ---------------------------------------------
        self.canvas.create_text(
            mx,
            my - 18,
            text="Mast Top",
            fill="white",
            font=("Segoe UI", 9)
        )

        self.canvas.create_text(
            tx,
            ty - 20,
            text="Trolley",
            fill="#85C1E9",
            font=("Segoe UI", 9, "bold")
        )

        self.canvas.create_text(
            lx,
            ly + 25,
            text="Load",
            fill="#F1948A",
            font=("Segoe UI", 9, "bold")
        )

        # ---------------------------------------------
        # Pause indicator
        # ---------------------------------------------
        if self.paused:

            self.canvas.create_text(
                600,
                140,
                text="SIMULATION PAUSED",
                fill="#F4D03F",
                font=("Segoe UI", 16, "bold")
            )

        # ---------------------------------------------
        # Tipping overlay
        # ---------------------------------------------
        if terminated:

            self.canvas.create_rectangle(
                350,
                300,
                800,
                410,
                fill="#641E16",
                outline="#FF1744",
                width=5
            )

            self.canvas.create_text(
                575,
                340,
                text="TIPPING THRESHOLD CROSSED",
                fill="white",
                font=("Segoe UI", 20, "bold")
            )

            self.canvas.create_text(
                575,
                380,
                text=(
                    f"Margin: "
                    f"{stability_margin / 1e6:.3f} MN.m"
                ),
                fill="#FFCDD2",
                font=("Consolas", 14, "bold")
            )

        # ---------------------------------------------
        # Episode handling
        # ---------------------------------------------
        if terminated:

            # Freeze tipping result.
            # User can choose another scenario button.
            self.paused = True

        elif truncated:

            # Restart same scenario after 1 second
            self.root.after(
                1000,
                lambda: self.set_scenario(
                    self.current_scenario
                )
            )

            return

        # ---------------------------------------------
        # Next frame
        # ---------------------------------------------
        self.root.after(
            self.frame_delay_ms,
            self.update_animation
        )


    def close(self):
        self.running = False
        self.env.close()
        self.root.destroy()


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = CraneVisualizer(
        root
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        app.close
    )

    root.mainloop()