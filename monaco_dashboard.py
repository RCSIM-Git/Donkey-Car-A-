import os
import time
import math
import numpy as np
import threading
import multiprocessing
import queue
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog
from monaco_engines import run_mapping_engine, PlanningEngine, run_collection_engine, run_training_engine, run_racing_engine, run_ppo_engine, run_pilot_engine

class MonacoDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

        # Window Setup
        self.title("Monaco GP - Autonomous Command Center V3.22 (Map Lab Edition)")
        self.geometry("1400x1000")
        ctk.set_appearance_mode("dark")
        
        self.entries = {}
        # V3.22: Parameter Descriptions (Polish)
        self.descriptions = {
            "max_laps": "Ile okrążeń ma przejechać bolid podczas mapowania (Zalecane: 1).",
            "gps_scale": "Przelicznik jednostek symulatora na metry. Monako wymaga 8.0.",
            "lidar_scale": "Przelicznik dystansu czujnika LIDAR. Pomaga dopasować jednostki świata Unity do skali GPS.",
            "kp": "Wzmocnienie proporcjonalne PID. Wyższe = szybsza reakcja na błąd pozycji.",
            "kd": "Wzmocnienie różniczkowe PID. Tłumi drgania (rybkowanie) bolidu.",
            "ki": "Wzmocnienie całkujące PID. Usuwa stałe przesunięcie od osi toru.",
            "lidar_range": "Maksymalny zasięg czujnika LIDAR (w metrach).",
            "max_occ_dist": "Maksymalna odległość rysowania ścian. Usuwa 'włosy' poza torem.",
            "l_free": "Siła czyszczenia pustej przestrzeni. Wyższe ujemne (np. -1.5) = szybciej usuwa szum.",
            "l_occ": "Pewność wykrycia ściany. Wyższe = ściany trudniej 'znikają' z mapy.",
            "cam_fov": "Kąt widzenia kamery. Szeroki kąt (120) pomaga w ciasnych nawrotach Monako.",
            "voxel_size": "Wielkość ziarna skanu. Mniejsze (0.1) = detale, Większe (0.3) = wydajność.",
            "map_res": "Rozmiar piksela mapy. 0.05 to precyzja 5cm na piksel.",
            "map_max_v": "Prędkość MAX na prostych podczas mapowania (Zalecane: 2.0).",
            "map_min_v": "Prędkość MIN w zakrętach podczas mapowania (Zalecane: 1.2).",
            "clean_strength": "Moc wygładzania krawędzi ścian. Usuwa 'włochaty' zarys mapy.",
            "noise_size": "Maksymalna wielkość izolowanych plam szumu do usunięcia (w pikselach).",
            "inflation": "Dodatkowy margines bezpieczeństwa przy ścianach (piksele Erozji).",
            "smoothing": "Wypukłość zakrętów trasy. Wyższe = bardziej płynna ścieżka.",
            "spline_pts": "Liczba punktów trasy. Więcej to płynniejsza jazda, ale więcej CPU.",
            "checkpoint_step": "Co ile klatek zapisu SLAM ma powstać punkt kontrolny trasy.",
            "target_speed": "Docelowa prędkość dla autopilota eksperta w fazie kolekcji.",
            "steer_gain": "Mnożnik siły skrętu. Zwiększ, jeśli bolid wypadnie z zakrętu.",
            "lookahead_max": "Jak daleko w przód bolid patrzy planując skręt.",
            "throttle_gain": "Siła przyspieszania autopilota eksperta (0.65 = 65% mocy).",
            "brake_gain": "Siła hamowania przed zakrętami (np. -0.4).",
            "max_laps_collect": "Ile okrążeń ma przejechać bolid podczas zbierania danych (Zalecane: 5-10).",
            "ai_steer_mult": "Mnoż aggressiveness zakrętów modelu AI.",
            "ppo_steps": "Liczba kroków treningu RL (PPO). Zalecane: 2 000 000.",
            "ppo_envs": "Liczba równoległych instancji symulatora dla RL.",
            "ppo_load": "Ścieżka do zapisanego modelu PPO (.zip) lub wag BC (.pth). Użyj przycisku 📁 aby wybrać plik.",
            "model_path": "Ścieżka do pliku wag modelu BC (.pth). Użyj przycisku 📁 aby wybrać plik.",
            "use_mirroring": "Podwaja zbiór danych przez lustrzane odbicie. Poprawia stabilność w zakrętach.",
            "exploration_bias": "Skłonność do skręcania w boczne odnogi (0.0 - 1.0). Pomaga odkrywać tory typu '8'.",
            "cte_kd": "Tłumienie oscylacji CTE. Wyższe = bolid łagodniej wraca na linię i mniej 'rybkuje'.",
            "steer_ema": "Wygładzanie ruchów kierownicy (0.0-1.0). 1.0 = brak wygładzania, 0.1 = bardzo leniwa kierownica."
        }
        
        # V72: Config Persistence logic
        self.config_path = os.path.join(os.path.dirname(__file__), "monaco_config.json")
        self.load_persistent_config()

        # Grid Layout
        self.grid_rowconfigure(0, weight=1); self.grid_columnconfigure(1, weight=1)

        # 1. SIDEBAR
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid_rowconfigure(5, weight=1) # Allow Tabview to expand
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.lbl_logo = ctk.CTkLabel(self.sidebar, text="DONKEY RACING COMMAND", font=("Orbitron", 20, "bold"))
        self.lbl_logo.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        # V73: Track Selection
        self.lbl_track = ctk.CTkLabel(self.sidebar, text="SELECT TRACK", font=("Orbitron", 12, "bold"), text_color="#FF9800")
        self.lbl_track.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.tracks = [
            "donkey-minimonaco-track-v0", "donkey-warehouse-v0", 
            "donkey-generated-roads-v0", "donkey-mountain-track-v0",
            "donkey-roboracingleague-track-v0", "donkey-warren-track-v0",
            "donkey-avc-sparkfun-v0", "donkey-thunderhill-track-v0",
            "donkey-circuit-launch-track-v0", "donkey-waveshare-v0",
            "donkey-generated-track-v0"
        ]
        self.opt_track = ctk.CTkComboBox(self.sidebar, values=self.tracks, width=280)
        self.opt_track.set(self.persistent_config.get("track_id", "donkey-minimonaco-track-v0"))
        self.opt_track.grid(row=2, column=0, padx=20, pady=(5, 10))

        # V74: Car customization
        lbl_car = ctk.CTkLabel(self.sidebar, text="CAR SETTINGS", font=("Orbitron", 12, "bold"), text_color="#03A9F4")
        lbl_car.grid(row=3, column=0, padx=20, pady=(10, 0))
        
        c_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        c_frame.grid(row=4, column=0, padx=20, pady=5, sticky="ew")
        
        self.ent_car_name = ctk.CTkEntry(c_frame, placeholder_text="Car Name", width=140)
        self.ent_car_name.insert(0, self.persistent_config.get("car_name", "Donkey"))
        self.ent_car_name.pack(side="left", padx=(0, 5))
        
        self.opt_car_type = ctk.CTkComboBox(c_frame, values=["f1", "donkey", "m_pitman", "tamiya"], width=130)
        self.opt_car_type.set(self.persistent_config.get("car_type", "f1"))
        self.opt_car_type.pack(side="left")

        self.tabview = ctk.CTkTabview(self.sidebar, width=300); self.tabview.grid(row=5, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_map = self.tabview.add("01 Mapping"); self.tab_plan = self.tabview.add("02 Map Lab"); self.tab_collect = self.tabview.add("03 BC Expert"); self.tab_race = self.tabview.add("04 RL PPO"); self.tab_vision = self.tabview.add("05 Vision AI")
        self.setup_tab_mapping(); self.setup_tab_planning(); self.setup_tab_bc_expert(); self.setup_tab_rl_ppo(); self.setup_tab_vision()

        # INFO PANEL
        self.frame_info = ctk.CTkFrame(self.sidebar, fg_color="#1a1a1a", corner_radius=10); self.frame_info.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")
        self.lbl_info_title = ctk.CTkLabel(self.frame_info, text="INFO BAZA WIEDZY", font=("Orbitron", 11, "bold"), text_color="#555"); self.lbl_info_title.pack(pady=(5, 0))
        self.lbl_description = ctk.CTkLabel(self.frame_info, text="Najedź myszką na parametr,\naby zobaczyć jego opis.", font=("Inter", 12), wraplength=260, text_color="#aaa"); self.lbl_description.pack(padx=10, pady=10)

        # LOG CONSOLE
        self.console = ctk.CTkTextbox(self.sidebar, width=280, height=120, font=("Consolas", 11)); self.console.grid(row=7, column=0, padx=10, pady=5)
        self.btn_stop = ctk.CTkButton(self.sidebar, text="FORCE EMERGENCY KILL", fg_color="#D32F2F", hover_color="#B71C1C", command=self.emergency_stop); self.btn_stop.grid(row=8, column=0, padx=20, pady=10)

        # 4. VIEWPORT
        self.viewport = ctk.CTkFrame(self, corner_radius=10, fg_color="#121212"); self.viewport.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.top_label = ctk.CTkLabel(self.viewport, text="[ SLAM / PLANNING ]", font=("Orbitron", 14), text_color="#555"); self.top_label.pack(pady=(20, 10))
        self.bottom_label = ctk.CTkLabel(self.viewport, text="[ FPV CAM ]", font=("Orbitron", 14), text_color="#555"); self.bottom_label.pack(side="bottom", pady=(10, 30))

        # STATE
        self.engine_process = None
        self.queue_frames = multiprocessing.Queue(maxsize=15) # Increased buffer
        self.queue_logs = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.active_engine = None
        self.last_slam_img = None
        self.last_cam_img = None
        self.viewing_cleaned = False
        
        self.update_loop()
        self.log_loop()

    def setup_tab_mapping(self):
        btn_start = ctk.CTkButton(self.tab_map, text="START MAPPING", command=lambda: self.start_process("mapping")); btn_start.pack(pady=(10, 5))
        btn_save = ctk.CTkButton(self.tab_map, text="STOP & SAVE MAP", fg_color="#2E7D32", command=lambda: self.stop_event.set()); btn_save.pack(pady=5)
        frame = ctk.CTkScrollableFrame(self.tab_map, fg_color="transparent", height=400); frame.pack(pady=10, fill="both", expand=True)
        self.sw_slam = ctk.CTkSwitch(frame, text="Enable SLAM", progress_color="#4CAF50"); self.sw_slam.select(); self.sw_slam.pack(pady=5, anchor="w")
        self.sw_autotune_map = ctk.CTkSwitch(frame, text="Autotune PID", progress_color="#2196F3"); self.sw_autotune_map.select(); self.sw_autotune_map.pack(pady=5, anchor="w")
        self.sw_smart_slam = ctk.CTkSwitch(frame, text="Smart SLAM Adapt", progress_color="#9C27B0"); self.sw_smart_slam.select(); self.sw_smart_slam.pack(pady=5, anchor="w")
        self.sw_clean_lidar = ctk.CTkSwitch(frame, text="Clean Lidar map", progress_color="#00BCD4"); self.sw_clean_lidar.select(); self.sw_clean_lidar.pack(pady=5, anchor="w")
        self.sw_global_view = ctk.CTkSwitch(frame, text="Global Map View", progress_color="#FF9800"); self.sw_global_view.pack(pady=5, anchor="w")
        self.add_param_field(frame, "Exploration Bias", "exploration_bias", "0.0", "Opt: 0.3-0.8 for Fig 8")
        self.add_param_field(frame, "Max Laps", "max_laps", "3", "Opt: 3")
        self.add_param_field(frame, "Map Max V", "map_max_v", "2.0", "Opt: 2.0")
        self.add_param_field(frame, "Map Min V", "map_min_v", "1.2", "Opt: 1.2")
        self.add_param_field(frame, "GPS Scale", "gps_scale", "8.0", "Opt: 8.0")
        self.add_param_field(frame, "Lidar Scale", "lidar_scale", "1.0", "Opt: 1.0 (Unity World Scale)")
        self.add_param_field(frame, "Lidar Range", "lidar_range", "12.0", "Opt: 12.0")
        self.add_param_field(frame, "Camera FOV", "cam_fov", "120", "Opt: 120")
        self.add_param_field(frame, "PID Kp", "kp", "1.0", "Opt: 1.0")
        self.add_param_field(frame, "PID Kd", "kd", "0.2", "Opt: 0.2")
        self.add_param_field(frame, "PID Ki", "ki", "0.001", "Opt: 0.001")
        self.add_param_field(frame, "Max Wall Dist", "max_occ_dist", "15.0", "Opt: 15.0")
        self.add_param_field(frame, "Clearing Str", "l_free", "-1.2", "Opt: -1.2")
        self.add_param_field(frame, "Wall Trust", "l_occ", "5.0", "Opt: 5.0")

    def setup_tab_planning(self):
        btn_plan = ctk.CTkButton(self.tab_plan, text="RUN OPTIMAL PLANNING", fg_color="#2E7D32", command=self.run_planning_task); btn_plan.pack(pady=10)
        frame = ctk.CTkScrollableFrame(self.tab_plan, fg_color="transparent", height=500); frame.pack(pady=10, fill="both", expand=True)
        
        lbl_lab = ctk.CTkLabel(frame, text="MAP LABORATORY (PRO)", font=("Orbitron", 13, "bold"), text_color="#00BCD4")
        lbl_lab.pack(pady=10)
        
        self.add_param_field(frame, "Clean Strength", "clean_strength", "1", "1-5 (Denoise)")
        self.add_param_field(frame, "Max Noise Size", "noise_size", "10", "Area threshold")
        
        btn_preview = ctk.CTkButton(frame, text="PREVIEW CLEANED", command=self.preview_map_lab); btn_preview.pack(pady=5)
        btn_reset = ctk.CTkButton(frame, text="RESET TO RAW", fg_color="#555", command=self.reset_map_preview); btn_reset.pack(pady=5)

        lbl_plan = ctk.CTkLabel(frame, text="TRAJECTORY PLANNING", font=("Orbitron", 13, "bold"), text_color="#FFB300")
        lbl_plan.pack(pady=(20, 10))
        
        self.add_param_field(frame, "Wall Safety", "inflation", "12", "Opt: 12 (0.6m)")
        self.add_param_field(frame, "Smoothing S", "smoothing", "5.0", "Opt: 5.0-10.0")
        self.add_param_field(frame, "Path Density", "spline_pts", "4000", "Opt: 4000")
        self.add_param_field(frame, "Search Depth", "checkpoint_step", "15", "Opt: 15-20")

    def setup_tab_pilot(self):
        btn_pilot = ctk.CTkButton(self.tab_pilot, text="START AI DRIVE", fg_color="#2E7D32", command=lambda: self.start_process("pilot")); btn_pilot.pack(pady=10)
        btn_stop_pilot = ctk.CTkButton(self.tab_pilot, text="STOP AI DRIVE", fg_color="#C62828", command=lambda: self.stop_event.set()); btn_stop_pilot.pack(pady=5)
        
        frame = ctk.CTkScrollableFrame(self.tab_pilot, fg_color="transparent", height=400); frame.pack(pady=10, fill="both", expand=True)
        lbl_ai = ctk.CTkLabel(frame, text="AI TUNING", font=("Orbitron", 13, "bold"), text_color="#4CAF50"); lbl_ai.pack(pady=5)
        self.add_file_param_field(frame, "Model File (.pth)", "model_path", "bc_model_weights_monaco.pth")
        self.add_param_field(frame, "AI Throttle Mult", "ai_throttle_mult", "1.0", "Full power: 1.0")
        self.add_param_field(frame, "AI Steer Mult", "ai_steer_mult", "1.0", "Default: 1.0")
        self.add_param_field(frame, "Lidar Range", "lidar_range", "50.0", "Opt: 50.0")
        self.add_param_field(frame, "Cam FOV", "cam_fov", "120", "Opt: 120")
        
        # New: Visualization Settings
        lbl_v = ctk.CTkLabel(frame, text="AI MONITORING", font=("Orbitron", 12, "bold"), text_color="#2196F3"); lbl_v.pack(pady=(15, 5))
        self.sw_show_cam = ctk.CTkCheckBox(frame, text="Show AI Camera", width=120); self.sw_show_cam.select(); self.sw_show_cam.pack(pady=5)
        self.sw_show_slam = ctk.CTkCheckBox(frame, text="Show AI Path Map", width=120); self.sw_show_slam.select(); self.sw_show_slam.pack(pady=5)

    def setup_tab_bc_expert(self):
        btn_bc = ctk.CTkButton(self.tab_collect, text="START COLLECTION", fg_color="#E65100", command=lambda: self.start_process("collection")); btn_bc.pack(pady=(10, 5))
        btn_stop_bc = ctk.CTkButton(self.tab_collect, text="STOP & FINISH COLLECTION", fg_color="#2E7D32", command=lambda: self.stop_event.set()); btn_stop_bc.pack(pady=5)
        frame = ctk.CTkScrollableFrame(self.tab_collect, fg_color="transparent", height=500); frame.pack(pady=10, fill="both", expand=True)
        lbl_c = ctk.CTkLabel(frame, text="AUTOPILOT TUNING", font=("Orbitron", 13, "bold"), text_color="#FF9800"); lbl_c.pack(pady=5)
        self.add_param_field(frame, "Max Laps (Collect)", "max_laps_collect", "5", "Opt: 5-10")
        self.sw_autotune_collect = ctk.CTkSwitch(frame, text="Autotune PID/Gain", progress_color="#FF9800"); self.sw_autotune_collect.deselect(); self.sw_autotune_collect.pack(pady=5, anchor="w")
        self.add_param_field(frame, "Target Speed (Max)", "target_speed", "10.0", "Straights Opt: 10-12")
        self.add_param_field(frame, "Target Speed (Min)", "target_speed_min", "2.5", "Sharp Corners Opt: 2.0-3.0")
        self.add_param_field(frame, "Curve Penalty", "curve_penalty", "15.0", "Braking force (Opt: 10-20)")
        self.add_param_field(frame, "Steer Gain", "steer_gain", "10.0", "Original Opt: 10.0")
        self.add_param_field(frame, "Lookahead Max", "lookahead_max", "4.0", "Original Opt: 4.0")
        self.add_param_field(frame, "CTE Kd (Damping)", "cte_kd", "0.5", "Opt: 0.3-1.0")
        self.add_param_field(frame, "Steer EMA (Smooth)", "steer_ema", "0.6", "Original Opt: 0.5 - 0.7")
        lbl_t = ctk.CTkLabel(frame, text="BC TRAINING", font=("Orbitron", 13, "bold"), text_color="#2196F3"); lbl_t.pack(pady=(15, 5))
        self.add_param_field(frame, "Epochs", "epochs", "30", "Opt: 30")
        self.add_param_field(frame, "Batch Size", "batch_size", "512", "Blackwell Opt: 512")
        
        # New: Vision & Sensors
        lbl_v = ctk.CTkLabel(frame, text="VISION & SENSORS", font=("Orbitron", 12, "bold"), text_color="#00BCD4"); lbl_v.pack(pady=(10, 5))
        self.add_param_field(frame, "Lidar FOV (deg)", "lidar_fov", "360", "Front-only: 180")
        self.add_param_field(frame, "Lidar Beams (cnt)", "lidar_beams", "60", "AI Features: 60-120")
        self.opt_res = ctk.CTkComboBox(frame, values=["160x120", "320x240", "640x480"])
        self.opt_res.set(self.persistent_config.get("img_res", "320x240"))
        self.opt_res.pack(pady=5)
        
        s_frame = ctk.CTkFrame(frame, fg_color="transparent"); s_frame.pack(pady=5)
        self.sw_speed = ctk.CTkCheckBox(s_frame, text="Speed", width=80)
        if self.persistent_config.get("use_speed", True): self.sw_speed.select()
        else: self.sw_speed.deselect()
        self.sw_speed.pack(side="left")
        
        self.sw_accel = ctk.CTkCheckBox(s_frame, text="Accel", width=80)
        if self.persistent_config.get("use_accel", True): self.sw_accel.select()
        else: self.sw_accel.deselect()
        self.sw_accel.pack(side="left")
        
        self.sw_gyro = ctk.CTkCheckBox(s_frame, text="Gyro", width=80)
        if self.persistent_config.get("use_gyro", True): self.sw_gyro.select()
        else: self.sw_gyro.deselect()
        self.sw_gyro.pack(side="left")
        
        self.sw_gps = ctk.CTkCheckBox(s_frame, text="GPS", width=80)
        if self.persistent_config.get("use_gps", True): self.sw_gps.select()
        else: self.sw_gps.deselect()
        self.sw_gps.pack(side="left")
        
        self.sw_mirror = ctk.CTkSwitch(frame, text="Use Mirroring x2", progress_color="#2196F3")
        if self.persistent_config.get("use_mirroring", True): self.sw_mirror.select()
        else: self.sw_mirror.deselect()
        self.sw_mirror.pack(pady=5, anchor="w")
        
        self.add_param_field(frame, "Learn Rate", "lr", "0.0001", "Opt: 0.0001")
        
        btn_save = ctk.CTkButton(frame, text="SAVE CONFIG TO JSON", fg_color="#455A64", command=self.save_persistent_config); btn_save.pack(pady=5)
        btn_train = ctk.CTkButton(frame, text="START BC TRAINING", fg_color="#1565C0", command=self.run_training_task); btn_train.pack(pady=10)
        
        lbl_r = ctk.CTkLabel(frame, text="BC MODEL TEST (AI)", font=("Orbitron", 13, "bold"), text_color="#4CAF50"); lbl_r.pack(pady=(20, 5))
        self.add_file_param_field(frame, "Model File (.pth)", "model_path", "bc_model_weights_monaco.pth")
        self.add_param_field(frame, "Speed Mult", "ai_throttle_mult", "0.8", "Opt: 0.8")
        self.add_param_field(frame, "Steer Mult", "ai_steer_mult", "1.0", "Opt: 1.0")
        btn_race = ctk.CTkButton(frame, text="START AI RACING TEST", fg_color="#2E7D32", command=lambda: self.start_process("racing")); btn_race.pack(pady=10)

    def setup_tab_rl_ppo(self):
        btn_ppo = ctk.CTkButton(self.tab_race, text="START PPO OPTIMIZATION", fg_color="#673AB7", command=lambda: self.start_process("ppo")); btn_ppo.pack(pady=10)
        frame = ctk.CTkScrollableFrame(self.tab_race, fg_color="transparent", height=500); frame.pack(pady=10, fill="both", expand=True)
        lbl_rl = ctk.CTkLabel(frame, text="PPO HYPERPARAMETERS", font=("Orbitron", 13, "bold"), text_color="#9C27B0"); lbl_rl.pack(pady=5)
        self.add_param_field(frame, "Total Steps", "ppo_steps", "2000000", "Opt: 2M")
        self.add_param_field(frame, "Parallel Envs", "ppo_envs", "1", "Opt: 1-4")
        self.add_file_param_field(frame, "Load Model/Weights (.pth/.zip)", "ppo_load", "bc_model_weights_monaco.pth")

    def setup_tab_vision(self):
        lbl_v = ctk.CTkLabel(self.tab_vision, text="AI VISION SETTINGS", font=("Orbitron", 13, "bold"), text_color="#00BCD4")
        lbl_v.pack(pady=10)
        
        self.sw_enable_detection = ctk.CTkSwitch(self.tab_vision, text="Enable Detection", progress_color="#4CAF50")
        self.sw_enable_detection.select(); self.sw_enable_detection.pack(pady=10)
        
        self.add_param_field(self.tab_vision, "Confidence Threshold", "vision_conf", "0.5", "0.1 - 1.0")
        self.add_param_field(self.tab_vision, "Detection Frequency", "vision_freq", "3", "Every N frames")
        self.add_param_field(self.tab_vision, "LKA Steer Gain", "vision_steer_gain", "0.2", "0.0 - 1.0 (Correction force)")
        
        lbl_info = ctk.CTkLabel(self.tab_vision, text="HAILO-8L PREVIEW MODE", font=("Inter", 11, "italic"), text_color="#777")
        lbl_info.pack(pady=20)

    def load_persistent_config(self):
        import json
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.persistent_config = json.load(f)
        else:
            self.persistent_config = {}

    def save_persistent_config(self):
        import json
        config = {k: v.get() for k, v in self.entries.items()}
        # V74: Sync all GUI states including Track & Car
        try:
            config["track_id"] = self.opt_track.get()
            config["car_name"] = self.ent_car_name.get()
            config["car_type"] = self.opt_car_type.get()
            config["use_speed"] = bool(self.sw_speed.get())
            config["use_accel"] = bool(self.sw_accel.get())
            config["use_gyro"] = bool(self.sw_gyro.get())
            config["use_gps"] = bool(self.sw_gps.get())
            config["use_mirroring"] = bool(self.sw_mirror.get())
            config["img_res"] = self.opt_res.get()
        except Exception as e: print(f"Config Save Warning: {e}")
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        self.log("Configuration saved to monaco_config.json")

    def add_param_field(self, parent, label_text, var_name, default_val, hint_text):
        lbl = ctk.CTkLabel(parent, text=label_text, font=("Inter", 12)); lbl.pack(anchor="w", pady=(5,0))
        inner = ctk.CTkFrame(parent, fg_color="transparent"); inner.pack(fill="x")
        
        # Use value from JSON if available
        val = self.persistent_config.get(var_name, default_val)
        entry = ctk.CTkEntry(inner, height=28, width=120); entry.insert(0, val); entry.pack(side="left")
        
        hint = ctk.CTkLabel(inner, text=f" {hint_text}", font=("Inter", 11), text_color="#777"); hint.pack(side="left", padx=5); self.entries[var_name] = entry
        entry.bind("<Enter>", lambda e, v=var_name: self.show_info(v)); entry.bind("<Leave>", lambda e: self.show_info(None))
        # V72: Auto-save on change
        entry.bind("<FocusOut>", lambda e: self.save_persistent_config())

    def add_file_param_field(self, parent, label_text, var_name, default_val, file_types=[("Weights / Models (*.pth, *.zip)", "*.pth;*.zip"), ("All Files", "*.*")]):
        lbl = ctk.CTkLabel(parent, text=label_text, font=("Inter", 12)); lbl.pack(anchor="w", pady=(5,0))
        inner = ctk.CTkFrame(parent, fg_color="transparent"); inner.pack(fill="x")
        
        val = self.persistent_config.get(var_name, default_val)
        entry = ctk.CTkEntry(inner, height=28, width=170); entry.insert(0, str(val)); entry.pack(side="left", padx=(0, 5))
        
        def browse_file():
            selected = filedialog.askopenfilename(
                initialdir=self.PROJECT_ROOT,
                title=f"Wybierz: {label_text}",
                filetypes=file_types
            )
            if selected:
                try:
                    rel_path = os.path.relpath(selected, self.PROJECT_ROOT)
                    if not rel_path.startswith(".."):
                        selected = rel_path
                except Exception:
                    pass
                entry.delete(0, 'end')
                entry.insert(0, selected)
                self.save_persistent_config()

        btn_browse = ctk.CTkButton(inner, text="📁", width=36, height=28, fg_color="#1E88E5", hover_color="#1565C0", command=browse_file)
        btn_browse.pack(side="left")
        
        self.entries[var_name] = entry
        entry.bind("<Enter>", lambda e, v=var_name: self.show_info(v)); entry.bind("<Leave>", lambda e: self.show_info(None))
        entry.bind("<FocusOut>", lambda e: self.save_persistent_config())

    def show_info(self, var_name):
        if var_name is None: self.lbl_description.configure(text="Najedź myszką na parametr,\naby zobaczyć jego opis.", text_color="#aaa")
        else: self.lbl_description.configure(text=self.descriptions.get(var_name, "Brak opisu."), text_color="#2196F3")

    def preview_map_lab(self):
        # Local Thread to clean and preview
        def run_lab():
             p_root = self.PROJECT_ROOT
             track_id = self.opt_track.get()
             track_name = track_id.split("-")[1] if "-" in track_id else track_id
             map_path = os.path.join(p_root, "data", "maps", f"{track_name}_slam_map.npz")
             if not os.path.exists(map_path): self.log(f"ERROR: Map not found at {map_path}"); return
             grid = np.load(map_path)["grid"]
             config = {k: v.get() for k, v in self.entries.items()}
             strength = int(self.entries["clean_strength"].get())
             size = int(self.entries["noise_size"].get())
             engine = PlanningEngine(p_root, config)
             cleaned = engine.apply_map_lab(grid, strength, size)
             # Convert to RGB View
             view = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB); view = cv2.flip(view, 0)
             self.last_slam_img = view; self.viewing_cleaned = True; self.log(f"Map Lab: Previewing Cleaned (Str:{strength}, Size:{size})")
        threading.Thread(target=run_lab, daemon=True).start()

    def reset_map_preview(self):
        p_root = self.PROJECT_ROOT
        track_id = self.opt_track.get()
        track_name = track_id.split("-")[1] if "-" in track_id else track_id
        map_path = os.path.join(p_root, "data", "maps", f"{track_name}_slam_map.npz")
        if os.path.exists(map_path):
            grid = np.load(map_path)["grid"]; view = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB); self.last_slam_img = cv2.flip(view, 0); self.viewing_cleaned = False; self.log(f"Map Lab: Reset to RAW view for {track_name}.")

    def start_process(self, mode):
        self.emergency_stop(); self.stop_event.clear()
        self.save_persistent_config() # V73: Ensure config is saved before engine starts
        config = {k: v.get() for k, v in self.entries.items()}
        config["track_id"] = self.opt_track.get()
        config["car_name"] = self.ent_car_name.get()
        config["car_type"] = self.opt_car_type.get()
        if mode == "mapping":
             config["autotune_pid"] = self.sw_autotune_map.get(); config["autotune_slam"] = self.sw_smart_slam.get(); config["clean_lidar"] = self.sw_clean_lidar.get(); config["use_slam"] = self.sw_slam.get(); config["global_map_view"] = self.sw_global_view.get()
        elif mode == "collection" or mode == "pilot" or mode == "racing": 
             if mode == "collection": config["autotune_pid"] = self.sw_autotune_collect.get()
             if mode == "pilot" or mode == "racing":
                 config["vision_enabled"] = self.sw_enable_detection.get()
                 config["vision_conf"] = self.entries.get("vision_conf").get()
                 config["vision_freq"] = self.entries.get("vision_freq").get()
                 
                 if mode == "pilot":
                     config["lidar_range"] = self.entries["lidar_range"].get()
                     config["cam_fov"] = self.entries["cam_fov"].get()
                     config["ai_throttle_mult"] = self.entries["ai_throttle_mult"].get()
                     config["ai_steer_mult"] = self.entries["ai_steer_mult"].get()

             config["use_mirroring"] = self.sw_mirror.get()
             config["img_res"] = self.opt_res.get()
             config["use_speed"] = self.sw_speed.get()
             config["use_accel"] = self.sw_accel.get()
             config["use_gyro"] = self.sw_gyro.get()
             config["use_gps"] = self.sw_gps.get()
        p_root = self.PROJECT_ROOT
        if mode == "mapping": self.engine_process = multiprocessing.Process(target=run_mapping_engine, args=(p_root, self.queue_frames, self.queue_logs, config, self.stop_event))
        elif mode == "collection": self.engine_process = multiprocessing.Process(target=run_collection_engine, args=(p_root, self.queue_frames, self.queue_logs, config, self.stop_event))
        elif mode == "pilot": self.engine_process = multiprocessing.Process(target=run_pilot_engine, args=(p_root, self.queue_frames, self.queue_logs, config, self.stop_event))
        elif mode == "racing": self.engine_process = multiprocessing.Process(target=run_racing_engine, args=(p_root, self.queue_frames, self.queue_logs, config))
        elif mode == "ppo": self.engine_process = multiprocessing.Process(target=run_ppo_engine, args=(p_root, self.queue_logs, config))
        self.engine_process.start(); self.active_engine = mode; self.log(f"Launching {mode} Engine...")

    def run_training_task(self):
        config = {k: v.get() for k, v in self.entries.items()}; multiprocessing.Process(target=run_training_engine, args=(self.PROJECT_ROOT, self.queue_logs, config)).start(); self.log("Step 3B: AI Training started...")
    def run_planning_task(self):
        self.last_slam_img = None # Reset preview to allow loading path from disk
        config = {k: v.get() for k, v in self.entries.items()}; threading.Thread(target=lambda: PlanningEngine(self.PROJECT_ROOT, config).run_planning(self.queue_logs), daemon=True).start(); self.log("Step 2: Optimal Line generation...")
    def emergency_stop(self):
        if self.engine_process: self.engine_process.terminate(); os.system('taskkill /f /im donkey_sim.exe'); self.engine_process = None; self.active_engine = None; self.log("EMERGENCY KILL EXECUTED.")
    def log(self, msg):
        self.console.insert("end", f"> {msg}\n"); self.console.see("end")
        if "[AUTOTUNE]" in msg:
            try:
                if "Dual-Path Mapping" in msg: 
                    # Correct parsing for V3.31 logs
                    kp = msg.split("Kp=")[1].split(",")[0]
                    kd = msg.split("Kd=")[1].strip()
                    ki = msg.split("Ki=")[1].split(",")[0]
                    self.entries["kp"].delete(0, "end"); self.entries["kp"].insert(0, kp)
                    self.entries["kd"].delete(0, "end"); self.entries["kd"].insert(0, kd)
                    self.entries["ki"].delete(0, "end"); self.entries["ki"].insert(0, ki)
                elif "Gain" in msg: 
                    gain = msg.split(":")[1].strip(); self.entries["steer_gain"].delete(0, "end"); self.entries["steer_gain"].insert(0, gain)
            except: pass
    def update_loop(self):
        try:
            while not self.queue_frames.empty():
                data = self.queue_frames.get()
                self.last_slam_img = data.get("slam")
                self.last_cam_img = data.get("cam")
                self.last_detections = data.get("detections", [])
                self.last_lines = data.get("lines", {'yellow':[], 'white':[]})
            
            if self.last_cam_img is not None:
                # V3.22: Draw Detections if available
                view = self.last_cam_img.copy()
                
                # 1. Draw Lines (Yellow & White)
                lines = getattr(self, 'last_lines', {'yellow':[], 'white':[]})
                for color_name, color_bgr in [('yellow', (0, 255, 255)), ('white', (255, 255, 255))]:
                    for line in lines.get(color_name, []):
                        cv2.line(view, (line[0], line[1]), (line[2], line[3]), color_bgr, 2)
                
                # 2. Draw Object Detections
                for det in getattr(self, 'last_detections', []):
                    box = det['box']
                    label = det['label']
                    score = det['score']
                    
                    # Draw BBox
                    cv2.rectangle(view, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    # Draw Label
                    txt = f"{label} {score:.2f}"
                    cv2.putText(view, txt, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                img = Image.fromarray(view)
                self.bottom_label.configure(image=ctk.CTkImage(light_image=img, dark_image=img, size=(800, 450)), text="")
            
            # V7.5: Auto-clear engine state if process died
            if self.engine_process and not self.engine_process.is_alive():
                self.engine_process = None; self.active_engine = None

            if self.last_slam_img is not None:
                img = Image.fromarray(self.last_slam_img)
                self.top_label.configure(image=ctk.CTkImage(light_image=img, dark_image=img, size=(600, 600)), text="")
            else:
                # V7.5: Show preview if live map is missing
                p = os.path.join(self.PROJECT_ROOT, "data", "maps", "optimal_path_preview.png")
                if os.path.exists(p):
                    img = Image.open(p)
                    self.top_label.configure(image=ctk.CTkImage(light_image=img, dark_image=img, size=(600, 600)), text="")
        except Exception as e: pass
        self.after(15, self.update_loop) # V3.24: Higher GUI Refresh rate (~60 FPS)
    def log_loop(self):
        while not self.queue_logs.empty(): self.log(self.queue_logs.get())
        self.after(100, self.log_loop)

if __name__ == "__main__":
    app = MonacoDashboard(); app.mainloop()
