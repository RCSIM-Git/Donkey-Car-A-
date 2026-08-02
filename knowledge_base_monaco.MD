# 🏎️ Monaco Autonomous Racing: Complete Knowledge Base

This document serves as the "source of truth" for the autonomous racing pipeline on the Monaco track. It documents the evolution from waypoint mapping to unified Reinforcement Learning.

## 1. Project Overview & Pipeline
The project follows a three-stage evolutionary path to achieve sub-22s lap times:

| Phase | Method | Goal | Core Technology |
| :--- | :--- | :--- | :--- |
| **I. Mapping** | Voronoi / PD | Generate Racing Line | A* Search, Waypoint CSV |
| **II. Expert (BC)** | Behavioral Cloning | Neural Driving Mimicry | CNN (Keras/PyTorch), Manual Data |
| **III. Optimization** | RL (PPO) | Surpassing Human Expert | Stable Baselines3, Weight Injection |

---

## 2. Technical Architecture

### 🧠 Neural Network Parity
To allow Behavioral Cloning (BC) weights to be used in PPO, we maintained strict architectural identity:
- **Feature Extractor**: 4-Layer Convolutional Neural Network (CNN).
  - Layer 1: 24 filters, 5x5 kernel, stride 2.
  - Layer 2: 32 filters, 5x5 kernel, stride 2.
  - Layer 3: 64 filters, 5x5 kernel, stride 2.
  - Layer 4: 64 filters, 3x3 kernel, stride 1.
- **Decision Head (MLP)**:
  - Dense Layer: 100 units (ReLU).
  - Dense Layer: 50 units (ReLU).
  - Output: 2 units (Steering, Throttle).

### 🎮 Control Dynamics
- **Steering**: Adjusted via Exponential Moving Average (EMA) with `alpha=0.5` to prevent jitter and match expert smoothness.
- **Action Space**: Box([-1.0, 0.0], [1.0, 1.0]).
- **Observation Space**: Image (320x240 RGB) + Lidar (60 pts) + Sensors (Speed, Accel, Gyro).

---

## 3. Environment & GPU Management

### 🏗️ Software Stack
- **OS**: Windows 10/11.
- **Conda Environment**: `env_isaaclab_v6` (Python 3.10+).
- **Core Libraries**: `gymnasium`, `stable-baselines3`, `torch`, `opencv-python`.
- **Simulator**: DonkeyCar Unity Simulator (Monaco Track).

### ⚡ GPU Infrastructure
- **Acceleration**: Training uses **CUDA** (PyTorch backend).
- **Optimization**: We leverage high-speed local storage (SSD) for the Conda environment to avoid latency in neural forward passes.
- **Headless Mode**: Supported for Rank 1+ environments to save VRAM, with Rank 0 reserved for visual debugging.

---

## 4. Key Files & Locations

| File Path | Description | Roles |
| :--- | :--- | :--- |
| `run_ppo.py` | Main Training Script | PPO logic, Weight Injection, Callbacks. |
| `gym_donkeycar/wrappers.py` | Environment Wrappers | Normalization, Rewards, EMA Control, Reset logic. |
| `gym_donkeycar/envs/donkey_sim.py` | Simulator Bridge | Low-level socket comms, Watchdogs, Telemetry. |
| `test_bc_monaco.py` | Expert Reference | "Gold Standard" script for BC inference. |
| `models/` | Weight Storage | Contains `.pth` and `.h5` expert models. |

---

## 5. History of Problems & Solutions

### ❌ The Double Normalization Bug
- **Problem**: The car was "blind," doing only 10s laps or driving straight into walls.
- **Discovery**: We were dividing images by 255.0 manually, while Stable Baselines3 was *also* dividing them internally. Result: Input values near 0.0.
- **Solution**: Disabled manual normalization in `DonkeyFeaturesExtractor`.

### ❌ The Simulator Hang (Lap End)
- **Problem**: The RL agent would complete a lap and then stop forever (Hanging).
- **Discovery**: The simulator's `observe()` loop was waiting for a camera frame that Unity paused after the lap finish line.
- **Solution**: Implemented a **100ms Watchdog** in `donkey_sim.py` and updated `time_received` during lap events to wake up the loop.

### ❌ Control Mismatch (Jitter)
- **Problem**: RL car was jittery and lost speed compared to the smooth BC model.
- **Discovery**: Missing EMA smoothing on steering and missing speed initialization in the rewards.
- **Solution**: Restored EMA (alpha=0.5) in `DonkeySmoothActionWrapper`.

### ❌ Reset vs. Infinite Mode
- **Problem**: Infinite driving (no reset) caused simulator UI to block steering inputs.
- **Solution**: Restored **Forced Reset** after each lap, made robust by the new bridge watchdogs.

---

## 6. Training Best Practices
1. **Frozen Start**: Use `--freeze_steps 10000` to let the Value Network stabilize on expert actions before updating policy weights.
2. **Aggressive Crash Detect**: Monitor `stuck_count`. If speed < 0.2 for 1s, force reset.
3. **Reward Shape**: High weight on `speed` and `lap_bonus` to encourage aggressive lines.

---
> [!IMPORTANT]
> **To Sync a New Model**: Use the weight mapping dictionary in `run_ppo.py` to ensure layer names match between Keras-style BC exports and PyTorch SB3 policies.
