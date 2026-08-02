import time
import numpy as np
import gymnasium as gym
from gym_donkeycar.wrappers import DonkeyMultiInputWrapper, DonkeySmoothActionWrapper

def run_physics_check():
    """
    VERIFY PHYSICS (v25)
    Raw control test to check for any drift or polarity issues.
    """
    sim_path = "C:\\Users\\mbuze\\OneDrive\\Pulpit\\DonkeySimWin\\donkey_sim.exe"
    conf = {
        "exe_path": sim_path,
        "host": "127.0.0.1", "port": 9091, 
        "max_cte": 10.0, "headless": False, 
    }
    
    print("\n--- STARTING RAW PHYSICS VERIFICATION ---")
    try:
        env = gym.make("donkey-minimonaco-track-v0", conf=conf)
        # Using pure wrapper as a pass-through
        env = DonkeySmoothActionWrapper(env, throttle_min=-1.0, throttle_max=1.0, ema_alpha=1.0)
        
        obs, info = env.reset()
        
        print("\n[TEST 1] STEER 0.0 (STRAIGHT LINE) for 5 seconds...")
        for i in range(100):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.2]))
            if i % 20 == 0:
                print(f"STEP {i:3d} | CTE: {info.get('cte',0):6.2f} | Speed: {info.get('speed',0):5.2f}")
            if terminated or truncated: break
        
        print("\n[TEST 2] STEER 0.5 (RIGHT TURN) for 2 seconds...")
        for i in range(40):
            obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.1]))
            if terminated or truncated: break
        print(f"Final CTE after Right Turn: {info.get('cte',0):6.2f}")
        
        print("\n[TEST 3] STEER -0.5 (LEFT TURN) for 2 seconds...")
        for i in range(40):
            obs, reward, terminated, truncated, info = env.step(np.array([-0.5, 0.1]))
            if terminated or truncated: break
        print(f"Final CTE after Left Turn: {info.get('cte',0):6.2f}")

    except Exception as e:
        print(f"Physics check failed: {e}")
    finally:
        if 'env' in locals(): env.close()
        print("\nVerification finished.")

if __name__ == "__main__":
    run_physics_check()
