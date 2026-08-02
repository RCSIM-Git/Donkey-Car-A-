# 🏎️ Monaco Autonomous Racing: Executive Technical Manual (V2)

This master database documents the end-to-end development of the Monaco autonomous racing pipeline. It is designed to be a permanent reference for maintaining expert-level performance.

---

## 🗺️ Phase I: Track Mapping & Intelligence

### 1.1 SLAM Discovery
The track was mapped using a custom robotics stack centered around **GraphSLAM**.
- **Sensor**: 360-degree raw Lidar data from the Unity simulator.
- **Algorithm**: Keyframe-based pose optimization. New poses added only if the car moved >0.5m or rotated >0.2 rad.
- **Anchor Point**: `monaco_slam_anchor.npy`. A high-resolution scan at the start line used to align the coordinate system every time a new mapping session started.
- **Output**: `monaco_slam_map.npz` (Occupancy Grid).

### 1.2 The Apex Line (Path Optimization)
Mapping alone wasn't enough. We needed a professional racing line.
- **Routing**: A* algorithm calculated segments between SLAM checkpoints.
- **Costmap Inflation**: We eroded the occupancy grid by 0.2m (Inflation) to ensure the car never planned a path too close to walls.
- **Mathematic Smoothing**: **B-Splines** were applied to the jagged A* path to create a continuous, differentiable trajectory suitable for high-speed momentum.
- **Reference**: `stare/generate_optimal_path.py`.

---

## 🧠 Phase II: Behavioral Cloning (BC Expert)

### 2.1 The Architecture (Parity Model)
To ensure the transition to RL was seamless, we utilized a specific 4-layer CNN:
- **Input**: `(320, 240, 3)` JPG/RGB frames.
- **Convolutional Base**: 
  - Conv1: `(5,5), stride 2, 24 filters`
  - Conv2: `(5,5), stride 2, 32 filters`
  - Conv3: `(5,5), stride 2, 64 filters`
  - Conv4: `(3,3), stride 1, 64 filters`
- **Dense Head**: `100 -> 50 -> 2 (Steer, Throttle)`.

### 2.2 Telemetry Protocol (V70 Master Sync)
A critical discovery was that the model needed more than just vision to ignore "shadows" on the track:
- **Normalization**: Lidar divided by 50.0, Speed by 20.0, Accel by 10.0, Gyro by 5.0.
- **Persistence**: All sensor values are concatenated into a single flat vector before the MLP.

---

## ⚡ Phase III: Reinforcement Learning (PPO Synchronization)

### 3.1 Policy Initialization (Weight Injection)
Instead of starting RL from scratch (which takes millions of steps), we "Hot Start" the PPO policy by copying tensors from the BC model.
- **Layer Mapping**:
| Keras/Raw Layer | PPO Policy Layer | Status |
| :--- | :--- | :--- |
| `conv.0.weight` | `features_extractor.cnn.0.weight` | Synchronized |
| `fc.0.weight` | `mlp_extractor.policy_net.0.weight` | Synchronized |
| `fc.4.weight` | `action_net.weight` | Synchronized |

### 3.2 Reward Function Engineering
The RL agent uses a "Momentum-Preserving" reward structure:
- **Speed Bonus**: Raw $(speed / 30.0) * 0.5$.
- **Jitter Penalty**: Heavily punishes large changes in steering ($|\Delta \delta|$) to maintain BC-like smoothness.
- **Stark Penalty**: -200.0 for any `hit != none`.
- **Lap Record**: Large exponential reward for finishing a lap under 25s.

---

## 🛠️ Infrastructure & Environment

### 💻 Hardware & Stack
- **GPU**: NVIDIA (IsaacLab V6 Environment).
- **Driver**: CUDA-accelerated PPO (PyTorch).
- **Communication**: Asynchronous socket bridge with **100ms Watchdog** to prevent deadlocks at the finish line.

### 📦 Key Components Location
- **Main Brain**: `run_ppo.py`.
- **Sensors/Resets**: `gym_donkeycar/wrappers.py`.
- **Simulator Core**: `gym_donkeycar/envs/donkey_sim.py`.

---

## ⚠️ Troubleshooting & Lessons Learned (Agent History)

> [!WARNING]
> **The Blind Car Bug (Double Normalization)**:
> Previous agents fell into the trap of dividing image arrays by 255.0 manually in the wrapper. Since Stable Baselines3's `NatureCNN` does this automatically, the model received values near 0.0 (Black). 
> **Fix**: Remove manual vision division if using SB3 extractors.

> [!CAUTION]
> **Coordinate System Shift**:
> The simulator's coordinate system can shift if the "Scene Selection" screen isn't handled correctly. 
> **Fix**: Always use the `monaco_slam_anchor.npy` to verify orientation before starting a high-stakes training run.

> [!IMPORTANT]
> **The Finish Line Hang**:
> Unity "pauses" telemetry for a few milliseconds after crossing the line to calculate split times. This is enough to hang a standard Python loop.
> **Fix**: The 100ms Watchdog in `observe()` wakes the script and forces a reset.

---

## 🏁 Summary of Current Status
- **Performance**: 22.5s / lap (Consistent).
- **Loop**: Stable, infinite racing with 1-second crash recovery.
- **Sync**: 100.0% Binary Parity with Expert BC.

**Next Focus**: Increasing `ent_coef` to allow the model to explore "clipping points" even closer than the human expert.
