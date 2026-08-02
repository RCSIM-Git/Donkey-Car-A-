import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

# V26.9: Monaco High-Fidelity BC Trainer (RTX 5080 Blackwell Edition)
# Support 640x480 input and expanded sensor suite

class VRAMDataset:
    def __init__(self, data_path, device="cuda", train_size=(320, 240)):
        self.device = device
        self.img_w, self.img_h = train_size # Optimized for 16GB VRAM
        self.json_files = []
        
        print(f"Scanning tubs in {data_path}...")
        # Look for latest expert tub
        tubs = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and "tub_expert" in d]
        tubs.sort(reverse=True)
        
        if not tubs:
            print("ERROR: No expert data tub found!")
            return

        target_tub = tubs[0]
        print(f"Loading data from: {target_tub}")
        
        self.json_files = [os.path.join(target_tub, f) for f in os.listdir(target_tub) if f.endswith(".json") and f.startswith("record_")]
        self.json_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
        
        self.num_samples = len(self.json_files)
        print(f"Found {self.num_samples} records.")
        
        if self.num_samples == 0: return

        # VRAM allocation (Uint8 for images to save VRAM)
        # 30k * 640 * 480 * 3 = ~27 GB. If out of VRAM, process gets killed.
        # Reduce num_samples or switch to standard DataLoader if necessary.
        print(f"Allocating VRAM for images {self.img_w}x{self.img_h}...")
        self.images = torch.zeros((self.num_samples, self.img_h, self.img_w, 3), dtype=torch.uint8, device=device)
        self.lidars = torch.zeros((self.num_samples, 60), dtype=torch.float32, device=device) # Use 60 points for precision
        self.sensors = torch.zeros((self.num_samples, 10), dtype=torch.float32, device=device) # Speed(1) + Accel(3) + Gyro(3) + GPS(3)
        self.actions = torch.zeros((self.num_samples, 2), dtype=torch.float32, device=device)

        from concurrent.futures import ThreadPoolExecutor
        print("Loading data to VRAM (Parallel Threading)...")
        with ThreadPoolExecutor(max_workers=12) as executor:
            list(tqdm(executor.map(self._load_item_to_gpu, range(self.num_samples)), total=self.num_samples))
        print(f"Dataset loaded into VRAM!")

    def _load_item_to_gpu(self, i):
        try:
            with open(self.json_files[i], "r") as f: data = json.load(f)
            # Image: Collector saves 640x480, but we load 320x240 to fit 16GB VRAM
            img_path = os.path.join(os.path.dirname(self.json_files[i]), data["cam/image_array"])
            img = Image.open(img_path).convert("RGB")
            
            # Scale to training size
            if img.size != (self.img_w, self.img_h):
                img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
            
            self.images[i].copy_(torch.from_numpy(np.array(img, dtype=np.uint8)))
            
            # Lidar: Downsampling 180 -> 60 (every 3 degrees at inc=2.0)
            lidar_raw = data.get("lidar/raw", [0.0]*180) # Default 180 at inc=2.0
            lidar_60 = np.array(lidar_raw[::3], dtype=np.float32)[:60] / 50.0 
            self.lidars[i].copy_(torch.from_numpy(lidar_60))
            
            # Sensors: Speed, Accel XYZ, Gyro XYZ, GPS Norm
            speed = data.get("telemetry/speed", 0.0) / 20.0
            accel = [x / 10.0 for x in data.get("imu/accel", [0,0,0])]
            gyro = [x / 5.0 for x in data.get("imu/gyro", [0,0,0])]
            gps = [x / 100.0 for x in data.get("gps/pos", [0,0,0])]
            
            vec = np.array([speed] + accel + gyro + gps, dtype=np.float32)
            self.sensors[i].copy_(torch.from_numpy(vec))
            
            # Action
            act = np.array([float(data.get("user/angle", 0.0)), float(data.get("user/throttle", 0.0))], dtype=np.float32)
            self.actions[i].copy_(torch.from_numpy(act))
        except:
            pass

class BCModel(nn.Module):
    def __init__(self):
        super(BCModel, self).__init__()
        # CNN for 320x240 (Optimized for Blackwell + Sync with PPO)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(), # 320x240 -> 79x59
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),# 79x59 -> 38x28
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),# 38x28 -> 36x26
            nn.Flatten()
        )
        # 36 * 26 * 64 = 59904
        self.cnn_fc = nn.Sequential(nn.Linear(59904, 1024), nn.ReLU(), nn.Dropout(0.1))
        self.lidar_fc = nn.Sequential(nn.Linear(60, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.sensor_fc = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        
        self.policy_head = nn.Sequential(
            nn.Linear(1024 + 64 + 32, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, img, lidar, sensor):
        img_feats = self.cnn_fc(self.cnn(img))
        lidar_feats = self.lidar_fc(lidar)
        sensor_feats = self.sensor_fc(sensor)
        combined = torch.cat([img_feats, lidar_feats, sensor_feats], dim=1)
        return self.policy_head(combined)

def train_monaco():
    device = torch.device("cuda")
    print("Initializing Monaco BC Training...")
    
    dataset = VRAMDataset("data", device=device)
    if dataset.num_samples == 0: return

    indices = torch.randperm(dataset.num_samples, device=device)
    train_size = int(0.9 * dataset.num_samples)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    model = BCModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    batch_size = 64 # Reduced batch size for high resolution
    epochs = 100
    
    best_val = 1e9
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        curr_train_idx = train_idx[torch.randperm(len(train_idx), device=device)]
        num_batches = (len(curr_train_idx) + batch_size - 1) // batch_size
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_indices = curr_train_idx[i*batch_size : (i+1)*batch_size]
            
            # Transfer and Normalize
            batch_imgs = dataset.images[batch_indices].float() / 255.0
            batch_imgs = batch_imgs.permute(0, 3, 1, 2) # NCHW
            
            batch_lidars = dataset.lidars[batch_indices]
            batch_sensors = dataset.sensors[batch_indices]
            batch_acts = dataset.actions[batch_indices]
            
            # Noise augmentation (Sensors)
            batch_sensors = batch_sensors + torch.randn_like(batch_sensors) * 0.01
            
            optimizer.zero_grad()
            outputs = model(batch_imgs, batch_lidars, batch_sensors)
            loss = criterion(outputs, batch_acts)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        avg_train = train_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            num_val_batches = (len(val_idx) + batch_size - 1) // batch_size
            for i in range(num_val_batches):
                b_idx = val_idx[i*batch_size : (i+1)*batch_size]
                b_imgs = dataset.images[b_idx].float() / 255.0
                b_imgs = b_imgs.permute(0, 3, 1, 2)
                outputs = model(b_imgs, dataset.lidars[b_idx], dataset.sensors[b_idx])
                val_loss += criterion(outputs, dataset.actions[b_idx]).item()
        
        avg_val = val_loss / num_val_batches
        print(f"Epoch {epoch+1} | Loss: {avg_train:.6f} | Val: {avg_val:.6f}")
        
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "bc_model_weights_monaco.pth")
            print("Saved best model.")

if __name__ == "__main__":
    train_monaco()
