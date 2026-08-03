import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO

# UTF-8 stdout reconfiguration for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Class Colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (255, 140, 0),   # person (Orange-Blue)
    1: (255, 255, 0),   # backpack (Cyan)
    2: (0, 255, 127),   # handbag (Spring Green)
    3: (0, 215, 255),   # suitcase (Gold)
    4: (0, 0, 255),     # rat (Bright Red - Alert)
    5: (147, 20, 255)   # mouse (Magenta/Purple - Alert)
}

CLASS_NAMES = {
    0: "Person",
    1: "Backpack",
    2: "Handbag",
    3: "Suitcase",
    4: "Rat",
    5: "Mouse"
}

class SecurePiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ SecurePi AI Security & Pest Detection Dashboard")
        self.root.geometry("1280x820")
        self.root.configure(bg="#1E1E2E")

        # Core State Variables
        self.model_path = self.find_best_model()
        self.model = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.annotated_frame = None
        self.source_type = "webcam"  # "webcam", "image", "video"
        self.source_val = 0

        # Class toggle states (dict of boolean vars)
        self.class_vars = {i: tk.BooleanVar(value=True) for i in range(6)}

        # Load YOLO model in thread
        self.init_model()

        # Build UI Layout
        self.setup_styles()
        self.build_ui()

    def find_best_model(self):
        detect_dir = Path("runs/detect")
        if detect_dir.exists():
            matching_pt = [
                d / "weights" / "best.pt" for d in detect_dir.iterdir()
                if d.is_dir() and (d / "weights" / "best.pt").exists()
            ]
            if matching_pt:
                latest_pt = sorted(matching_pt, key=lambda f: f.stat().st_mtime, reverse=True)[0]
                print(f"💡 GUI auto-selected latest PyTorch weights: {latest_pt}")
                return str(latest_pt)

            matching_onnx = [
                d / "weights" / "best.onnx" for d in detect_dir.iterdir()
                if d.is_dir() and (d / "weights" / "best.onnx").exists()
            ]
            if matching_onnx:
                latest_onnx = sorted(matching_onnx, key=lambda f: f.stat().st_mtime, reverse=True)[0]
                print(f"💡 GUI auto-selected latest ONNX model: {latest_onnx}")
                return str(latest_onnx)

        if Path("yolov8n.pt").exists():
            return "yolov8n.pt"
        return "yolov8n.pt"

    def init_model(self):
        try:
            print(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1E1E2E")
        style.configure("TLabel", background="#1E1E2E", foreground="#CDD6F4", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#89B4FA")
        style.configure("Alert.TLabel", font=("Segoe UI", 12, "bold"), foreground="#F38BA8")
        style.configure("TCheckbutton", background="#1E1E2E", foreground="#CDD6F4", font=("Segoe UI", 10))
        style.configure("TScale", background="#1E1E2E")

    def build_ui(self):
        # Header Banner
        header_frame = tk.Frame(self.root, bg="#181825", height=60)
        header_frame.pack(fill="x", side="top")

        title_label = tk.Label(
            header_frame,
            text="🛡️ SecurePi - Real-Time AI Detection Dashboard",
            font=("Segoe UI", 16, "bold"),
            fg="#89B4FA",
            bg="#181825",
            padx=15,
            pady=10
        )
        title_label.pack(side="left")

        self.status_banner = tk.Label(
            header_frame,
            text="STATUS: READY",
            font=("Segoe UI", 11, "bold"),
            fg="#A6E3A1",
            bg="#313244",
            padx=15,
            pady=6
        )
        self.status_banner.pack(side="right", padx=15, pady=10)

        # Main Split Container
        main_container = tk.Frame(self.root, bg="#1E1E2E")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Column: Video Viewport
        left_panel = tk.Frame(main_container, bg="#11111B", highlightbackground="#313244", highlightthickness=1)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.video_label = tk.Label(
            left_panel,
            text="🎥 Click 'Start Stream' or select a Media Source to begin",
            font=("Segoe UI", 12),
            fg="#6C7086",
            bg="#11111B"
        )
        self.video_label.pack(fill="both", expand=True)

        # Video Control Toolbar below viewport
        toolbar = tk.Frame(left_panel, bg="#181825", pady=8)
        toolbar.pack(fill="x", side="bottom")

        self.btn_start = tk.Button(
            toolbar,
            text="▶️ Start Stream",
            font=("Segoe UI", 10, "bold"),
            bg="#A6E3A1",
            fg="#11111B",
            activebackground="#94E2D5",
            command=self.start_stream,
            padx=15,
            pady=5,
            relief="flat"
        )
        self.btn_start.pack(side="left", padx=10)

        self.btn_stop = tk.Button(
            toolbar,
            text="⏹️ Stop",
            font=("Segoe UI", 10, "bold"),
            bg="#F38BA8",
            fg="#11111B",
            activebackground="#EBA0AC",
            command=self.stop_stream,
            state="disabled",
            padx=15,
            pady=5,
            relief="flat"
        )
        self.btn_stop.pack(side="left", padx=5)

        self.btn_snapshot = tk.Button(
            toolbar,
            text="📸 Take Snapshot",
            font=("Segoe UI", 10, "bold"),
            bg="#89B4FA",
            fg="#11111B",
            activebackground="#B4BEFE",
            command=self.take_snapshot,
            padx=15,
            pady=5,
            relief="flat"
        )
        self.btn_snapshot.pack(side="left", padx=5)

        # Right Column: Controls & Alert Panel
        right_panel = tk.Frame(main_container, bg="#1E1E2E", width=360)
        right_panel.pack(side="right", fill="y", padx=(5, 0))
        right_panel.pack_propagate(False)

        # Section 1: Source Selection
        src_box = tk.LabelFrame(right_panel, text=" 📹 Input Source ", bg="#181825", fg="#89B4FA", font=("Segoe UI", 11, "bold"), padx=10, pady=10)
        src_box.pack(fill="x", pady=(0, 10))

        btn_webcam = tk.Button(src_box, text="📷 Webcam", bg="#313244", fg="#CDD6F4", activebackground="#45475A", command=self.select_webcam, relief="flat", width=10)
        btn_webcam.pack(side="left", padx=4)

        btn_img = tk.Button(src_box, text="🖼️ Open Image", bg="#313244", fg="#CDD6F4", activebackground="#45475A", command=self.select_image, relief="flat", width=12)
        btn_img.pack(side="left", padx=4)

        btn_vid = tk.Button(src_box, text="🎥 Open Video", bg="#313244", fg="#CDD6F4", activebackground="#45475A", command=self.select_video, relief="flat", width=12)
        btn_vid.pack(side="left", padx=4)

        # Section 2: Confidence Threshold
        conf_box = tk.LabelFrame(right_panel, text=" 🎚️ Detection Sensitivity ", bg="#181825", fg="#89B4FA", font=("Segoe UI", 11, "bold"), padx=10, pady=10)
        conf_box.pack(fill="x", pady=(0, 10))

        self.conf_val_label = tk.Label(conf_box, text="Confidence Threshold: 25%", bg="#181825", fg="#CDD6F4", font=("Segoe UI", 10))
        self.conf_val_label.pack(anchor="w")

        self.slider_conf = tk.Scale(
            conf_box,
            from_=10,
            to=90,
            orient="horizontal",
            bg="#181825",
            fg="#89B4FA",
            troughcolor="#313244",
            highlightthickness=0,
            command=self.update_conf_label
        )
        self.slider_conf.set(25)
        self.slider_conf.pack(fill="x", pady=5)

        # Section 3: Class Filter Toggles
        cls_box = tk.LabelFrame(right_panel, text=" 🎯 Target Class Filters ", bg="#181825", fg="#89B4FA", font=("Segoe UI", 11, "bold"), padx=10, pady=10)
        cls_box.pack(fill="x", pady=(0, 10))

        grid_frame = tk.Frame(cls_box, bg="#181825")
        grid_frame.pack(fill="x")

        for idx, name in CLASS_NAMES.items():
            color_hex = f"#{CLASS_COLORS[idx][2]:02x}{CLASS_COLORS[idx][1]:02x}{CLASS_COLORS[idx][0]:02x}"
            cb = tk.Checkbutton(
                grid_frame,
                text=name,
                variable=self.class_vars[idx],
                bg="#181825",
                fg=color_hex,
                selectcolor="#313244",
                activebackground="#181825",
                activeforeground=color_hex,
                font=("Segoe UI", 10, "bold")
            )
            row, col = divmod(idx, 2)
            cb.grid(row=row, column=col, sticky="w", padx=10, pady=4)

        # Section 4: Live Detection Counts
        count_box = tk.LabelFrame(right_panel, text=" 📊 Live Object Counter ", bg="#181825", fg="#89B4FA", font=("Segoe UI", 11, "bold"), padx=10, pady=10)
        count_box.pack(fill="x", pady=(0, 10))

        self.count_labels = {}
        count_grid = tk.Frame(count_box, bg="#181825")
        count_grid.pack(fill="x")

        for idx, name in CLASS_NAMES.items():
            lbl = tk.Label(count_grid, text=f"{name}: 0", bg="#181825", fg="#CDD6F4", font=("Segoe UI", 10, "bold"))
            row, col = divmod(idx, 2)
            lbl.grid(row=row, column=col, sticky="w", padx=10, pady=2)
            self.count_labels[idx] = lbl

        # Section 5: Real-Time Event Log
        log_box = tk.LabelFrame(right_panel, text=" 🚨 Security Alert Log ", bg="#181825", fg="#F38BA8", font=("Segoe UI", 11, "bold"), padx=5, pady=5)
        log_box.pack(fill="both", expand=True)

        self.log_list = tk.Listbox(
            log_box,
            bg="#11111B",
            fg="#CDD6F4",
            selectbackground="#313244",
            highlightthickness=0,
            font=("Consolas", 9),
            bd=0
        )
        self.log_list.pack(fill="both", expand=True, side="left")

        scrollbar = tk.Scrollbar(log_box, command=self.log_list.yview, bg="#181825")
        scrollbar.pack(side="right", fill="y")
        self.log_list.config(yscrollcommand=scrollbar.set)

    def update_conf_label(self, val):
        self.conf_val_label.config(text=f"Confidence Threshold: {val}%")

    def select_webcam(self):
        self.source_type = "webcam"
        self.source_val = 0
        self.log_event("Source set to Live Webcam (Device 0)")
        if self.is_running:
            self.restart_stream()

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.source_type = "image"
            self.source_val = file_path
            self.log_event(f"Loaded image: {Path(file_path).name}")
            self.start_stream()

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.source_type = "video"
            self.source_val = file_path
            self.log_event(f"Loaded video: {Path(file_path).name}")
            self.start_stream()

    def start_stream(self):
        if self.is_running:
            return

        if self.source_type == "image":
            self.process_single_image(self.source_val)
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_banner.config(text="STATUS: STREAMING", bg="#A6E3A1", fg="#11111B")

        self.cap = cv2.VideoCapture(self.source_val)
        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop_stream(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_banner.config(text="STATUS: STOPPED", bg="#F38BA8", fg="#11111B")
        self.log_event("Video stream stopped")

    def restart_stream(self):
        self.stop_stream()
        time.sleep(0.2)
        self.start_stream()

    def process_single_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image: {img_path}")
            return

        annotated, counts = self.detect_objects(img)
        self.update_count_labels(counts)
        self.display_frame(annotated)
        self.status_banner.config(text="STATUS: IMAGE LOADED", bg="#89B4FA", fg="#11111B")

    def video_loop(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "video":
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            self.current_frame = frame
            annotated, counts = self.detect_objects(frame)
            self.annotated_frame = annotated

            self.update_count_labels(counts)
            self.display_frame(annotated)
            time.sleep(0.01)

        self.stop_stream()

    def detect_objects(self, frame):
        if self.model is None:
            return frame, {i: 0 for i in range(6)}

        conf_thresh = self.slider_conf.get() / 100.0
        results = self.model.predict(source=frame, conf=conf_thresh, verbose=False)

        annotated = frame.copy()
        counts = {i: 0 for i in range(6)}
        detected_rodents = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if not self.class_vars[cls_id].get():
                    continue  # Filtered out by checkbox

                counts[cls_id] += 1
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                color = CLASS_COLORS.get(cls_id, (0, 255, 0))
                label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.0%}"

                # Draw bounding box (thick borders)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                # Draw label background pill
                (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - lbl_h - 8), (x1 + lbl_w + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                if cls_id in (4, 5):  # Rat or Mouse
                    detected_rodents.append(CLASS_NAMES[cls_id])

        if detected_rodents:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_event(f"[{timestamp}] 🚨 RODENT DETECTED: {', '.join(detected_rodents)}")
            self.status_banner.config(text="🚨 PEST ALERT DETECTED!", bg="#F38BA8", fg="#11111B")

        return annotated, counts

    def display_frame(self, frame):
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Scale image to fit video label while preserving aspect ratio
        win_w = self.video_label.winfo_width()
        win_h = self.video_label.winfo_height()

        if win_w > 50 and win_h > 50:
            img.thumbnail((win_w, win_h), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=img_tk, text="")
        self.video_label.image = img_tk

    def update_count_labels(self, counts):
        for idx, count in counts.items():
            name = CLASS_NAMES[idx]
            self.count_labels[idx].config(text=f"{name}: {count}")

    def log_event(self, msg):
        self.log_list.insert(0, msg)
        if self.log_list.size() > 100:
            self.log_list.delete(100, tk.END)

    def take_snapshot(self):
        target_frame = self.annotated_frame if self.annotated_frame is not None else self.current_frame
        if target_frame is None:
            messagebox.showwarning("Warning", "No active frame to capture!")
            return

        out_dir = Path("runs/snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = out_dir / filename

        cv2.imwrite(str(filepath), target_frame)
        self.log_event(f"📸 Snapshot saved: {filename}")
        messagebox.showinfo("Snapshot Saved", f"Saved annotated frame to:\n{filepath.resolve()}")

def main():
    root = tk.Tk()
    app = SecurePiGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
