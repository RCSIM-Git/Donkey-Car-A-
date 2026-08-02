import os
import sys
import gymnasium as gym
from gymnasium.envs.registration import register

# Authors Engine: Dynamic Environment Registration
# Auto-inject backup gym_donkeycar into path if needed
backup_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'gym-donkeycar-25.10.06'))
if os.path.exists(backup_path) and backup_path not in sys.path:
    sys.path.insert(0, backup_path)

try:
    from gym_donkeycar.envs.donkey_env import (
        AvcSparkfunEnv,
        CircuitLaunchEnv,
        GeneratedRoadsEnv,
        GeneratedTrackEnv,
        MiniMonacoEnv,
        MountainTrackEnv,
        RoboRacingLeagueTrackEnv,
        ThunderhillTrackEnv,
        WarehouseEnv,
        WarrenTrackEnv,
        WaveshareEnv,
    )
except ImportError:
    print("Warning: gym_donkeycar not installed. Cannot import environment classes.")

__version__ = "1.0.0-AUTHOR-EDITION"

# Re-register environments using the new local entry points
def register_envs():
    try:
        register(id="donkey-generated-roads-v0", entry_point="gym_donkeycar.envs.donkey_env:GeneratedRoadsEnv")
        register(id="donkey-warehouse-v0", entry_point="gym_donkeycar.envs.donkey_env:WarehouseEnv")
        register(id="donkey-avc-sparkfun-v0", entry_point="gym_donkeycar.envs.donkey_env:AvcSparkfunEnv")
        register(id="donkey-generated-track-v0", entry_point="gym_donkeycar.envs.donkey_env:GeneratedTrackEnv")
        register(id="donkey-mountain-track-v0", entry_point="gym_donkeycar.envs.donkey_env:MountainTrackEnv")
        register(id="donkey-roboracingleague-track-v0", entry_point="gym_donkeycar.envs.donkey_env:RoboRacingLeagueTrackEnv")
        register(id="donkey-waveshare-v0", entry_point="gym_donkeycar.envs.donkey_env:WaveshareEnv")
        register(id="donkey-minimonaco-track-v0", entry_point="gym_donkeycar.envs.donkey_env:MiniMonacoEnv")
        register(id="donkey-warren-track-v0", entry_point="gym_donkeycar.envs.donkey_env:WarrenTrackEnv")
        register(id="donkey-thunderhill-track-v0", entry_point="gym_donkeycar.envs.donkey_env:ThunderhillTrackEnv")
        register(id="donkey-circuit-launch-track-v0", entry_point="gym_donkeycar.envs.donkey_env:CircuitLaunchEnv")
        print("Author Engine: Registered local Gymnasium environments.")
    except Exception as e:
        # If already registered, we just proceed
        pass

register_envs()

__all__ = [
    "AvcSparkfunEnv",
    "CircuitLaunchEnv",
    "GeneratedRoadsEnv",
    "GeneratedTrackEnv",
    "MiniMonacoEnv",
    "MountainTrackEnv",
    "RoboRacingLeagueTrackEnv",
    "ThunderhillTrackEnv",
    "WarehouseEnv",
    "WarrenTrackEnv",
    "WaveshareEnv",
]
