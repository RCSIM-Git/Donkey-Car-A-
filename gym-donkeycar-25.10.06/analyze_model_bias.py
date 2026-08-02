import torch
import numpy as np
from stare.train_bc import BCModel

def analyze_bias():
    weights_path = "bc_model_weights.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- ANALYZING MODEL BIAS ({weights_path}) ---")
    
    model = BCModel().to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # 1. Test on Neutral Input (Perfect Center)
    # Neutral Image: All gray
    dummy_img = torch.full((1, 3, 120, 160), 0.5).to(device)
    # Neutral Lidar: All far away (1.0)
    dummy_lidar = torch.full((1, 12), 1.0).to(device)
    # Neutral Sensors: Speed 1.0, CTE 0.0, Pivot 0.5
    dummy_sensors = torch.zeros((1, 14)).to(device)
    dummy_sensors[0, 0] = 0.5 # Speed 10m/s
    dummy_sensors[0, 3] = 0.0 # CTE 0.0
    dummy_sensors[0, 4] = 0.5 # Pivot
    
    with torch.no_grad():
        action = model(dummy_img, dummy_lidar, dummy_sensors)
        steering = action[0, 0].item()
        throttle = action[0, 1].item()
        
    print(f"\n[NEUTRAL TEST]")
    print(f"Predicted Steering: {steering:8.4f}")
    print(f"Predicted Throttle: {throttle:8.4f}")
    
    if abs(steering) > 0.05:
        print("WARNING: Model has a significant Steering Bias!")
    else:
        print("OK: Model is numerically centered on neutral inputs.")

if __name__ == "__main__":
    analyze_bias()
