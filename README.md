# Crane Copilot — AI Safety & 3D Physics Simulation

An AI-assisted crane safety, predictive forecasting, and autonomous stabilization copilot prototype for early warning detection, reinforcement learning control, and high-fidelity 3D simulation analysis.

---

## 🌐 3D Physics Simulation

The application features a complete **3D Physics Simulation** suite:

### 1. Interactive 3D WebGL / Three.js Visualizer
Launch directly in your web browser:
```powershell
python visualize_crane_3d.py --web
```
Or open [`visualize_crane_3d.html`](./visualize_crane_3d.html) directly in any browser.

**Features:**
- **3D Rigged Lattice Crane**: 4-chord vertical mast with lattice cross-bracing, triangular 50m boom/jib, counter-jib with concrete ballast counterweights, slewing turntable, and operator cabin.
- **3D Physics & Dynamics**: Spherical pendulum swing ($\theta_x, \theta_y$), dynamic structural mast tilt flex ($\phi_x, \phi_y$), wind force vector, overturning/restoring moments, and dynamic tipping collapse simulation.
- **Atmospheric Model**: Animated 3D particle wind field, direction arrows, and live gust generator.
- **Autonomous AI Co-Pilot**: Active stabilization mode that automatically counter-steers the trolley inward during high moment states, provides anti-sway damping, and prevents tipping accidents.
- **Multi-Camera Views**: Orbit Overview, Operator Cockpit View, Hook/Load Follow, Base Tilt Inspector, Profile Elevation, and Top-Down.
- **Telemetry HUD & Sound**: Analog/digital moment gauges, stability reserve percentages, rolling strip charts, and Web Audio API synthesized warning sirens.

### 2. Native Python 3D Simulator
Run directly via the terminal:
```powershell
python visualize_crane_3d.py
```
**Interactive Controls:**
- `[Left]` / `[Right]` Arrow: Move trolley inward / outward
- `[Up]` / `[Down]` Arrow: Adjust wind speed
- `[C]`: Toggle AI Co-Pilot (Autonomous Stabilization)
- `[Space]`: Pause / Resume simulation
- `[S]`: Load SAFE Scenario
- `[W]`: Load CRITICAL Scenario
- `[T]`: Load TIPPING Scenario
- `[B]`: Launch browser-based WebGL 3D simulator

---

## 📦 Project Architecture & Pipeline

```text
Crane Parameters & Wind Model (Base + OU Turbulence + Gusts)
                    │
                    ▼
          Crane Physics Engine (physics.py)
                    │
                    ▼
          CraneStabilityEnv (crane_env.py)
           │         │         │
           │         │         ├──► 3D WebGL / Python Simulation (visualize_crane_3d.*)
           │         │         ├──► 2D Tkinter Panel (visualize_crane_tk.py)
           │         │         └──► Matplotlib Animation (visualize_crane.py)
           │         │
           │         └──► PPO Control Training & Evaluation (train_ppo.py)
           │
           └──► Sensor Dataset Generation (generate_dataset.py)
                     │
                     ▼
              LSTM Forecasting (train_lstm.py, train_risk_lstm.py)
                     │
                     ▼
           Early Warning Risk Classifier (early_warning.py)
```

---

## 🚀 Quickstart & Execution Guide

### 1. Verification & Physics Checks
```powershell
python validate_env.py
python test_physics.py
python test_scenarios.py
python test_phase3_reward.py
```

### 2. Run Simulations
```powershell
# Interactive 3D WebGL Simulation (Three.js)
python visualize_crane_3d.py --web

# Native Python 3D Simulation
python visualize_crane_3d.py

# 2D Tkinter Interactive Tuning Panel
python visualize_crane_tk.py

# Deterministic Console Simulation
python run_episode.py
```

### 3. Data & ML Pipelines
```powershell
# Generate sensor dataset
python generate_dataset.py

# Inspect dataset
python inspect_dataset.py

# Train & evaluate structural tilt LSTM
python train_lstm.py
python evaluate_lstm.py

# Train & evaluate stability risk LSTM
python train_risk_lstm.py
python evaluate_risk_warning.py

# Train & evaluate PPO copilot
python train_ppo.py
python evaluate_ppo_copilot.py
```

### 4. Interactive Website
```powershell
python -m http.server 8000
# Open http://localhost:8000/crane-copilot-project-website.html
```

---

## 📁 Repository File Map

| File | Description |
|---|---|
| [`visualize_crane_3d.html`](./visualize_crane_3d.html) | Interactive 3D WebGL Three.js Crane Simulation with HUD & Audio |
| [`visualize_crane_3d.py`](./visualize_crane_3d.py) | Python 3D Matplotlib visualizer and 3D web launcher |
| [`crane_env.py`](./crane_env.py) | Gymnasium crane stability physics environment |
| [`physics.py`](./physics.py) | Aerodynamic wind force, moments, and pendulum equations |
| [`wind.py`](./wind.py) | Mean-reverting Ornstein-Uhlenbeck wind and gust model |
| [`models.py`](./models.py) | Shared PyTorch LSTM neural network architectures |
| [`data_utils.py`](./data_utils.py) | Dataset sequence windowing and scaling utilities |
| [`generate_dataset.py`](./generate_dataset.py) | Sensor data recording pipeline |
| [`train_lstm.py`](./train_lstm.py) | Structural tilt forecasting model trainer |
| [`train_risk_lstm.py`](./train_risk_lstm.py) | Stability risk forecasting model trainer |
| [`train_ppo.py`](./train_ppo.py) | Reinforcement learning PPO agent trainer |
| [`early_warning.py`](./early_warning.py) | Multi-tier early warning and risk hysteresis classification |
| [`evaluate_ppo_copilot.py`](./evaluate_ppo_copilot.py) | PPO policy benchmark against HOLD and INWARD baselines |
| [`visualize_crane_tk.py`](./visualize_crane_tk.py) | Tkinter interactive 2D parameter tuning panel |
| [`visualize_crane.py`](./visualize_crane.py) | Matplotlib 2D animation visualizer |
| [`crane-copilot-project-website.html`](./crane-copilot-project-website.html) | Project overview showcase website |
| [`PROJECT_PIPELINE.md`](./PROJECT_PIPELINE.md) | Comprehensive engineering and physics pipeline reference |

























Application is running
The Crane Copilot 3D WebGL application is running at:

http://localhost:8000/visualize_crane_3d.html

It loaded successfully with:

HTTP 200 OK
Interactive 3D crane simulation
SAFE status
AI Copilot monitoring enabled
Scenario presets and live telemetry controls available
The local server is still running in the background.

How to run the application
# Open PowerShell in the project folder:

Set-Location "G:\My Drive\Projects\TCS_Innovent_2026\Prototype\crane-copilot\crane-copilot"


# Start the web application:
& "C:\Users\SANKALP\AppData\Local\Programs\Python\Python314\python.exe" -m http.server 8000

# Then open:
http://localhost:8000/visualize_crane_3d.html

# Alternatively, open the static project website:
http://localhost:8000/crane-copilot-project-website.html

## Other application modes
# Native Python 3D simulator
& "C:\Users\SANKALP\AppData\Local\Programs\Python\Python314\python.exe" visualize_crane_3d.py

# Tkinter control panel
& "C:\Users\SANKALP\AppData\Local\Programs\Python\Python314\python.exe" visualize_crane_tk.py

# Console simulation
& "C:\Users\SANKALP\AppData\Local\Programs\Python\Python314\python.exe" run_episode.py

# Verification performed
# Environment validation passed:


The 20-second console simulation also completed successfully without crossing the tipping threshold.

The documented pytest command ran, but pytest found 0 discoverable test functions in the repository:


More detailed execution instructions are available in README.md and PROJECT_PIPELINE.md.