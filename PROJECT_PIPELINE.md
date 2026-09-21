# Crane Copilot: Project Pipeline and Feature Reference

## 1. Project purpose

Crane Copilot is a Python-based crane safety prototype. It combines:

- A Gymnasium-compatible crane physics environment.
- Time-varying wind and gust simulation.
- Hanging-load pendulum and structural-tilt modelling.
- Reinforcement-learning control experiments.
- LSTM forecasting for future crane conditions.
- Early-warning risk classification.
- Console, Matplotlib, and Tkinter visualisation.

The project is a simulation and research prototype. It does not connect directly
to physical crane hardware.

## 2. End-to-end pipeline

```text
Crane parameters and random seed
              |
              v
      Wind and gust model
              |
              v
       Crane physics engine
              |
              v
      CraneStabilityEnv
       |       |       |
       |       |       +--> Console episode simulation
       |       |
       |       +----------> PPO training and evaluation
       |
       +------------------> Sensor dataset generation
                                |
                                v
                         LSTM forecasting
                                |
                                v
                       Risk / early-warning layer
                                |
                                v
                Metrics, CSV results, and visualisations
```

## 3. Core runtime flow

### 3.1 Environment initialisation

[`crane_env.py`](./crane_env.py) creates `CraneStabilityEnv`, including:

- Crane height, boom length, cable length, crane mass, and base width.
- Trolley speed and continuous trolley action space.
- Simulation timestep and episode duration.
- Wind model configuration.
- Observation and action spaces.

The default action is a one-value continuous control:

```text
-1.0  move trolley inward
 0.0  hold position
 1.0  move trolley outward
```

The observation contains trolley position, load mass, wind velocity, swing
angle, swing velocity, structural tilt, tilt rate, and stability margin.

### 3.2 Environment reset

`reset(seed=...)`:

1. Seeds the Gymnasium random generator.
2. Resets wind and gust state.
3. Randomises the initial trolley position, load mass, and swing angle.
4. Calculates the initial wind force and stability values.
5. Returns the initial observation and diagnostic information.

### 3.3 Environment step

Each `step(action)`:

1. Clips and applies the trolley action.
2. Updates the wind and gust model.
3. Calculates wind force.
4. Updates pendulum swing state.
5. Calculates load offset and overturning moment.
6. Calculates restoring moment and stability margin.
7. Updates structural tilt.
8. Calculates the multi-objective reward.
9. Returns observation, reward, termination state, truncation state, and metrics.

An episode terminates when the tipping threshold is crossed and truncates at
the configured simulation time limit.

## 4. Physics and wind pipeline

[`physics.py`](./physics.py) contains the reusable calculations for:

- Wind force.
- Pendulum acceleration and state updates.
- Load horizontal offset.
- Overturning moment.
- Restoring moment.
- Stability margin.

[`wind.py`](./wind.py) provides:

- Base wind.
- Mean-reverting turbulence.
- Random gust probability.
- Gust minimum and maximum strength.
- Wind direction changes.
- Maximum wind clipping.

## 5. Data and machine-learning pipeline

### 5.1 Dataset generation

[`generate_dataset.py`](./generate_dataset.py) runs simulation episodes and
stores crane sensor data in:

- [`crane_sensor_dataset.csv`](./crane_sensor_dataset.csv)

The generated records can be inspected with
[`inspect_dataset.py`](./inspect_dataset.py).

### 5.2 LSTM forecasting

[`train_lstm.py`](./train_lstm.py) trains the forecasting model from sensor
sequences. The model and scalers are stored in:

- [`crane_lstm_model.pth`](./crane_lstm_model.pth)
- [`crane_lstm_scalers.pkl`](./crane_lstm_scalers.pkl)

[`evaluate_lstm.py`](./evaluate_lstm.py) evaluates forecasts and writes:

- [`lstm_evaluation_metrics.csv`](./lstm_evaluation_metrics.csv)
- [`lstm_forecast_examples.csv`](./lstm_forecast_examples.csv)

### 5.3 PPO control pipeline

[`train_ppo.py`](./train_ppo.py) trains a PPO policy against the crane
environment. [`evaluate_ppo_copilot.py`](./evaluate_ppo_copilot.py) evaluates
the trained policy and records control results in:

- [`ppo_copilot_evaluation.csv`](./ppo_copilot_evaluation.csv)
- [`ppo_logs/`](./ppo_logs/)

### 5.4 Risk and early-warning pipeline

[`train_risk_lstm.py`](./train_risk_lstm.py) trains the risk forecasting
model. The resulting files are:

- [`crane_risk_lstm_model.pth`](./crane_risk_lstm_model.pth)
- [`crane_risk_lstm_scalers.pkl`](./crane_risk_lstm_scalers.pkl)

[`evaluate_risk_warning.py`](./evaluate_risk_warning.py) evaluates warning
performance and writes:

- [`risk_lstm_predictions.csv`](./risk_lstm_predictions.csv)
- [`risk_lstm_event_evaluation.csv`](./risk_lstm_event_evaluation.csv)

[`early_warning.py`](./early_warning.py) provides the early-warning analysis
and event-level output:

- [`early_warning_events.csv`](./early_warning_events.csv)
- [`early_warning_results.csv`](./early_warning_results.csv)

## 6. User-facing features

### 6.1 Console simulation

Run:

```powershell
& "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" run_episode.py
```

The console reports:

- Simulation time.
- Wind velocity and wind force.
- Gust start and end events.
- Swing angle and angular velocity.
- Overturning moment.
- Stability margin.
- Reward and stability status.

### 6.2 Matplotlib animation

Run:

```powershell
& "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" visualize_crane.py
```

This displays the crane mast, boom, trolley, cable, load, wind arrow,
stability metrics, and live status.

### 6.3 Interactive Tkinter control panel

Run:

```powershell
& "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" visualize_crane_tk.py
```

The interactive panel supports:

- SAFE, CRITICAL, and TIPPING presets.
- Pause and resume.
- Load mass.
- Trolley position.
- Base wind.
- Turbulence strength.
- Gust probability.
- Maximum gust strength.
- Crane height.
- Boom length.
- Cable length.
- Trolley speed.
- Trolley control action.
- Applying a custom configuration and restarting the episode.

After changing sliders, click **APPLY**. Custom settings are applied to the
next simulation episode.

### 6.4 Interactive 3D WebGL / Three.js simulation

Open [`visualize_crane_3d.html`](./visualize_crane_3d.html) directly in any web browser, or launch via:

```powershell
& "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" visualize_crane_3d.py --web
```

The 3D WebGL simulation includes:
- Fully rigged 3D lattice tower crane (mast, boom, counter-jib, ballast, cabin, trolley, cables, hook block, and cargo container).
- Dynamic 3D physics: spherical pendulum swing ($\theta_x, \theta_y$), structural mast deflection/tilt ($\phi$), moment calculations, and tipping collapse animations.
- 3D wind particle streamline field with live gust generator.
- Multi-camera viewpoints: Orbit Overview, Operator Cockpit View, Hook Follow, Base Tilt Inspector, Profile Elevation, and Top-Down.
- Autonomous AI Copilot mode: Active real-time counter-steering and anti-sway trolley modulation.
- Audio safety sirens and synthesized warning beeps via Web Audio API.
- Real-time telemetry HUD with analog/digital gauges and rolling strip charts.

### 6.5 Native Python 3D simulation

Run:

```powershell
& "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" visualize_crane_3d.py
```

Features:
- Renders 3D lattice truss mast, boom, counter-jib, and 3D swinging pendulum directly using `mpl_toolkits.mplot3d`.
- Real-time interactive keyboard controls (`[Left]`/`[Right]` trolley, `[Up]`/`[Down]` wind, `[C]` AI Copilot, `[S]` Safe, `[W]` Critical, `[T]` Tipping, `[B]` launch Web 3D).

### 6.6 Static project website

[`crane-copilot-project-website.html`](./crane-copilot-project-website.html)
is a browser presentation of the project with integrated links to the 3D simulation.

To serve it locally:

```powershell
Set-Location "G:\My Drive\Projects\TCS_Innovent_2026\Prototype\crane-copilot\crane-copilot"
python -m http.server 8000
```

Open:

```text
http://localhost:8000/crane-copilot-project-website.html
```

## 7. Recommended execution order

For a fresh experiment:

1. Run environment checks:

   ```powershell
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" validate_env.py
   ```

2. Run the deterministic console simulation:

   ```powershell
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" run_episode.py
   ```

3. Generate or refresh sensor data:

   ```powershell
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" generate_dataset.py
   ```

4. Inspect the generated dataset:

   ```powershell
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" inspect_dataset.py
   ```

5. Train and evaluate the forecasting model:

   ```powershell
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" train_lstm.py
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" evaluate_lstm.py
   ```

6. Train and evaluate the PPO controller:

   ```powershell
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" train_ppo.py
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" evaluate_ppo_copilot.py
   ```

7. Train and evaluate risk warnings:

   ```powershell
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" train_risk_lstm.py
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" evaluate_risk_warning.py
   ```

8. Launch an interactive view:

   ```powershell
   & "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" visualize_crane_tk.py
   ```

## 8. Validation and tests

The repository includes focused checks for:

- Environment behaviour: [`test_env.py`](./test_env.py)
- Physics: [`test_physics.py`](./test_physics.py)
- Wind: [`test_wind.py`](./test_wind.py)
- Scenarios: [`test_scenarios.py`](./test_scenarios.py)
- Reward behaviour: [`test_phase3_reward.py`](./test_phase3_reward.py)

Run the test suite with:

```powershell
& "C:/Users/SANKALP/AppData/Local/Programs/Python/Python314/python.exe" -m pytest
```

## 9. Main configuration points

The primary physical defaults are defined in `CraneStabilityEnv.__init__()`:

| Parameter | Default |
|---|---:|
| Crane height | 50 m |
| Boom length | 50 m |
| Cable length | 15 m |
| Crane mass | 100,000 kg |
| Base width | 10 m |
| Timestep | 0.02 s |
| Trolley speed | 0.5 m/s |
| Episode duration | 20 s |
| Base wind | 5 m/s |
| Turbulence strength | 1.5 |
| Gust probability | 0.01 per step |
| Gust range | 8–20 m/s |
| Maximum wind | 50 m/s |

The Tkinter visualizer exposes the operational parameters that are useful to
change interactively. Structural constants and reward design remain in the
environment code so that experiments remain reproducible and auditable.

## 10. Outputs and reproducibility

Use a fixed seed such as `seed=42` when comparing experiments. Keep the
following together when recording a result:

- Configuration values.
- Random seed.
- Generated dataset version.
- Model checkpoint.
- Evaluation CSV.
- Scenario and environment status.

This makes it possible to distinguish a real model or physics change from a
different random initial condition.
