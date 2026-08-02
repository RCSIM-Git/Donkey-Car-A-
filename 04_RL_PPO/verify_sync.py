import torch
import torch.nn as nn
import numpy as np

# 1. Re-define BCModel exactly as in test_bc_monaco.py
class BCModel(nn.Module):
    def __init__(self):
        super(BCModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(56070, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, img, lidar, sensors):
        x = img.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        v = self.conv(x)
        combined = torch.cat([v, lidar, sensors], dim=1)
        return self.fc(combined)

def verify():
    device = "cpu"
    bc_path = "bc_model_weights_monaco.pth"
    
    # Load BC Model
    bc_model = BCModel().to(device)
    bc_all_state = torch.load(bc_path, map_location=device)
    bc_model.load_state_dict(bc_all_state)
    bc_model.eval()
    
    # Create Dummy Data
    img_hwc = torch.zeros((1, 240, 320, 3), dtype=torch.uint8)
    # Fill with some pattern to avoid all zeros
    img_hwc[:, 100:150, 100:200, :] = 200 
    lidar = torch.ones(1, 60) * 0.5
    sensors = torch.zeros(1, 10)
    sensors[0, 0] = 0.2 # Speed
    
    # Get BC Output
    with torch.no_grad():
        bc_out = bc_model(img_hwc, lidar, sensors)
    
    print(f"BC Model Output (Mean): {bc_out.numpy()}")

    # 3. Simulate PPO Structure
    class PPOPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
                nn.Conv2d(24, 32, 5, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
            self.policy_net = nn.Sequential(
                nn.Linear(56070, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU()
            )
            self.action_net = nn.Linear(128, 2)
            
        def forward(self, image_chw, lidar, sensors):
            v = self.image_conv(image_chw.float() / 255.0)
            combined = torch.cat([v, lidar, sensors], dim=1)
            latent = self.policy_net(combined)
            return self.action_net(latent)

    ppo_policy = PPOPolicy()
    
    # Mapping logic from run_ppo.py
    mapping = {
        "conv.": "image_conv.",
        "fc.0.": "policy_net.0.",
        "fc.2.": "policy_net.2.",
        "fc.4.": "action_net.",
    }
    
    ppo_state = ppo_policy.state_dict()
    injected_count = 0
    for bc_key, bc_val in bc_all_state.items():
        for bc_prefix, ppo_prefix in mapping.items():
            if bc_key.startswith(bc_prefix):
                ppo_key = bc_key.replace(bc_prefix, ppo_prefix)
                if ppo_key in ppo_state:
                    ppo_state[ppo_key].copy_(bc_val)
                    injected_count += 1
                    break
    
    ppo_policy.load_state_dict(ppo_state)
    ppo_policy.eval()

    print(f"Injected {injected_count} tensors into PPO Policy.")
    
    # Get PPO Output
    img_chw = img_hwc.permute(0, 3, 1, 2) 
    with torch.no_grad():
        ppo_out = ppo_policy(img_chw, lidar, sensors)
        
    print(f"PPO Policy Output: {ppo_out.numpy()}")
    
    diff = torch.abs(bc_out - ppo_out).max().item()
    print(f"Max Difference: {diff:.10f}")
    
    if diff < 1e-6:
        print("SUCCESS: 100% Sync achieved!")
    else:
        print("FAILURE: Sync failed!")

if __name__ == "__main__":
    verify()
