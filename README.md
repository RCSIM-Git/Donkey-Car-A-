# RCSIM Donkey Car - Monaco Autonomous Racing Framework

An advanced autonomous racing pipeline and interactive GUI application for the Donkey Car Unity simulator on the Monaco track. Integrates Lidar SLAM mapping, A* optimal path generation, Behavioral Cloning (BC) expert training, and PPO Reinforcement Learning.

---

## 🌟 Key Features

- **Interactive GUI Command Center** (`monaco_dashboard.py` / `run_gui.ps1`):
  - **01 Mapping**: Real-time 2D Lidar SLAM mapping, occupancy grid visualization, and PID autotuning.
  - **02 Map Lab**: A* optimal racing line generation, Voronoi cost mapping, wall dilation, and spline smoothing.
  - **03 BC Expert**: Automated data collection and Behavioral Cloning neural network training.
  - **04 RL PPO**: Proximal Policy Optimization reinforcement learning training and live evaluation.
  - **05 Vision AI**: Real-time FPV camera stream and sensor overlay.
- **English Codebase & Documentation**: All module comments, docstrings, and script outputs are fully in English.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- [Donkey Car Unity Simulator](https://github.com/tawnkramer/gym-donkeycar) (Monaco track)

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/RCSIM-Git/Donkey-Car-A-.git
cd Donkey-Car-A-
```

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Install dependencies:
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Option A: Interactive GUI Command Center (Recommended)
Run the launcher script in PowerShell:
```powershell
.\run_gui.ps1
```
Or launch directly via Python:
```bash
python monaco_dashboard.py
```

### Option B: Automated Command-Line Workflow
Execute the end-to-end pipeline script in PowerShell:
```powershell
.\run_workflow.ps1
```

Or run individual steps:
1. **SLAM Mapping**:
   ```bash
   python 01_MAPPING/run_slam_mapping.py
   ```
2. **Generate A* Racing Line**:
   ```bash
   python 02_PATH_PLANNING/generate_optimal_path.py
   ```
3. **Autonomous Driving & Data Collection**:
   ```bash
   python 02_PATH_PLANNING/drive_optimal_path_and_collect.py
   ```
4. **Train Behavioral Cloning Expert Model**:
   ```bash
   python 03_BC_EXPERT/train_bc_monaco.py
   ```
5. **Train PPO Reinforcement Learning Agent**:
   ```bash
   python 04_RL_PPO/run_ppo.py
   ```

---

## 📁 Repository Structure

```
├── monaco_dashboard.py         # Main CustomTkinter GUI Command Center
├── monaco_engines.py           # Multi-threaded process engine for GUI
├── vision_engine.py            # FPV & OpenCV visualization engine
├── monaco_config.json          # GUI configuration parameters
├── run_gui.ps1                 # One-click GUI launcher script
├── run_workflow.ps1            # End-to-end CLI workflow launcher
├── 01_MAPPING/                 # SLAM mapping & calibration scripts
├── 02_PATH_PLANNING/          # A* path generation & Pure Pursuit racer
├── 03_BC_EXPERT/               # Expert data collection & BC model training
├── 04_RL_PPO/                  # PPO reinforcement learning training
├── core_engine/                # SLAM, ICP, and planner core algorithms
├── requirements.txt            # Python dependencies list
└── README.md                   # English project documentation
```

---

## 📜 License
Developed for RCSIM Autonomous Racing Project.
