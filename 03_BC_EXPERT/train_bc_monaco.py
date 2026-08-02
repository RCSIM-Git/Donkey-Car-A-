import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

class MonacoDataset(torch.utils.data.Dataset):
    def __init__(self, tub_path):
        print(f"Loading data from {tub_path}...")
        self.images = []
        self.lidars = []
        self.sensors = []
        self.actions = []
        
        records = [f for f in os.listdir(tub_path) if f.startswith("record_") and f.endswith(".json")]
        records.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        for r_file in tqdm(records, desc="Loading records"):
            with open(os.path.join(tub_path, r_file), 'r') as f:
                data = json.load(f)
            
            # Vision
            img = Image.open(os.path.join(tub_path, data["cam/image_array"]))
            img = img.resize((320, 240))
            self.images.append(np.array(img))
            
            # Lidar - Normalization 50.0
            lidar_raw = data.get("lidar/raw", [0.0]*180)
            lidar_60 = np.array(lidar_raw[::3], dtype=np.float32)[:60] / 50.0
            self.lidars.append(lidar_60)
            
            # Sensors (V70 Master Sync)
            speed = data.get("telemetry/speed", 0.0) / 20.0
            accel = [x / 10.0 for x in data.get("imu/accel", [0,0,0])]
            gyro = [x / 5.0 for x in data.get("imu/gyro", [0,0,0])]
            
            # GPS: Multiplier 8.0, Divisor 100.0
            gps_raw = data.get("gps/pos", [0,0,0])
            gps = [(gps_raw[0]*8.0)/100.0, (gps_raw[1]*8.0)/100.0, (gps_raw[2]*8.0)/100.0]
            
            self.sensors.append(np.array([speed] + accel + gyro + gps, dtype=np.float32))
            
            # Actions
            self.actions.append(np.array([data["user/angle"], data["user/throttle"]], dtype=np.float32))
            
        self.images = torch.from_numpy(np.array(self.images)).byte()
        self.lidars = torch.from_numpy(np.array(self.lidars)).float()
        self.sensors = torch.from_numpy(np.array(self.sensors)).float()
        self.actions = torch.from_numpy(np.array(self.actions)).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image normalization on GPU during training
        return self.images[idx], self.lidars[idx], self.sensors[idx], self.actions[idx]

class BCModel(nn.Module):
    def __init__(self):
        super(BCModel, self).__init__()
        # Vision Branch (ResNet-like small)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # Combine with Lidar and Sensors
        # Vision output: (320x240) -> (158x118) -> (77x57) -> (37x27) -> (35x25) * 64
        # Let's use a simpler flattening
        self.fc = nn.Sequential(
            nn.Linear(64 * 35 * 25 + 60 + 10, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2) # Steering, Throttle
        )

    def forward(self, img, lidar, sensors):
        x = img.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        v = self.conv(x)
        combined = torch.cat([v, lidar, sensors], dim=1)
        return self.fc(combined)

def train_monaco():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Training on {device}")
    
    import glob
    data_dir = os.path.join(PROJECT_ROOT, "data")
    tubs = glob.glob(os.path.join(data_dir, "tub_expert_monaco_*"))
    if not tubs:
        print("No data in Tub format found!")
        return
    tubs.sort(key=os.path.getmtime)
    tub_path = tubs[-1]
    
    full_dataset = MonacoDataset(tub_path)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = BCModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 30
    best_val = 1e9
    
    print(f"Start BC training (V70) on {len(full_dataset)} frames...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for img, lidar, sensor, act in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            img, lidar, sensor, act = img.to(device), lidar.to(device), sensor.to(device), act.to(device)
            
            optimizer.zero_grad()
            out = model(img, lidar, sensor)
            loss = criterion(out, act)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, lidar, sensor, act in val_loader:
                img, lidar, sensor, act = img.to(device), lidar.to(device), sensor.to(device), act.to(device)
                out = model(img, lidar, sensor)
                val_loss += criterion(out, act).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        if avg_val < best_val:
            save_path = os.path.join(PROJECT_ROOT, "GOTOWE", "03_BC_EXPERT", "bc_model_weights_monaco.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    train_monaco()
