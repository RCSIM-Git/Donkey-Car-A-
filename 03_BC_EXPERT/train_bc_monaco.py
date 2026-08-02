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
        # Vision Branch (NatureCNN + 1024 FC)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.cnn_fc = nn.Sequential(nn.Linear(59904, 1024), nn.ReLU())
        
        # Lidar Branch (60 -> 128 -> 64)
        self.lidar_fc = nn.Sequential(nn.Linear(60, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        
        # Sensors Branch (10 -> 64 -> 32)
        self.sensor_fc = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        
        # Policy Head (1024 + 64 + 32 = 1120 -> 512 -> 256 -> 2)
        self.policy_head = nn.Sequential(
            nn.Linear(1024 + 64 + 32, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, img, lidar, sensors):
        x = img.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        img_feats = self.cnn_fc(self.cnn(x))
        lidar_feats = self.lidar_fc(lidar)
        sensor_feats = self.sensor_fc(sensors)
        combined = torch.cat([img_feats, lidar_feats, sensor_feats], dim=1)
        return self.policy_head(combined)

def train_monaco():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Training on {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    import glob
    data_dir = os.path.join(PROJECT_ROOT, "data")
    tubs = glob.glob(os.path.join(data_dir, "tub_expert_monaco_*"))
    if not tubs:
        print("No data in Tub format found!")
        return
    tubs.sort(key=os.path.getmtime)
    
    # Load all available tubs using ConcatDataset
    datasets = [MonacoDataset(tp) for tp in tubs]
    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Sequential split (chronological) to prevent temporal data leakage between train/val
    total_len = len(full_dataset)
    train_size = int(0.9 * total_len)
    val_size = total_len - train_size
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_len))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = BCModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 30
    best_val = 1e9
    
    print(f"Start BC training on {len(full_dataset)} frames across {len(tubs)} tub(s)...")
    
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
            best_val = avg_val  # FIX: Update best_val to keep true best checkpoint
            save_dir = os.path.join(PROJECT_ROOT, "GOTOWE", "03_BC_EXPERT")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "bc_model_weights_monaco.pth")
            torch.save(model.state_dict(), save_path)
            
            # Save local checkpoint
            local_save = os.path.join(PROJECT_ROOT, "03_BC_EXPERT", "bc_model_weights_monaco.pth")
            torch.save(model.state_dict(), local_save)
            print(f"Saved best model (Val Loss: {best_val:.6f}) to {save_path}")

if __name__ == "__main__":
    train_monaco()
