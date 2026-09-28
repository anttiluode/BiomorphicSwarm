"""
ACTIVE CABBAGE: THE SELF-DRIVING NEURAL NETWORK
===============================================
A fusion of 'scout_com_instanton.py' and 'cabbage3.py'.

1. THE FIELD: A Complex-Valued Neural Network (RCNet) trying to dream an image.
2. THE SCOUT: An autonomous agent that navigates the 'Frustration Field' (Error).
3. THE PHYSICS: The Scout is attracted to high-error regions (Entropy). 
   As the network learns, entropy drops, and the Scout flows elsewhere.

Usage:
- Load an image.
- Click 'Release Scout'.
- Watch the Red Dot (Attention) hunt down the imperfections in real-time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import time
import random

# ============================================================================
# 1. THE RESONANT BRAIN (The Field)
# ============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(out_features))
        self.bias_i = nn.Parameter(torch.zeros(out_features))
        # Criticality initialization
        nn.init.xavier_normal_(self.fc_r.weight, gain=0.2)
        nn.init.xavier_normal_(self.fc_i.weight, gain=0.2)

    def forward(self, z):
        # z is complex (real, imag)
        real = z[..., 0]
        imag = z[..., 1]
        out_r = self.fc_r(real) - self.fc_i(imag) + self.bias_r
        out_i = self.fc_r(imag) + self.fc_i(real) + self.bias_i
        return torch.stack([out_r, out_i], dim=-1)

class RCNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (x, y) -> Mapped to Fourier Features for coordinate embedding
        self.feature_scale = 10.0 
        self.l1 = ComplexLinear(2, 64)
        self.l2 = ComplexLinear(64, 64)
        self.l3 = ComplexLinear(64, 64)
        self.out = ComplexLinear(64, 3) # RGB Output

    def forward(self, x):
        # x is (Batch, 2) -> (x, y) normalized -1 to 1
        # Embed coordinates into complex plane
        z = torch.stack([x * np.pi * self.feature_scale, x * np.pi * self.feature_scale * 0.5], dim=-1)
        
        # Resonant Layers
        z = self.l1(z)
        z = torch.tanh(z) # Complex activation
        z = self.l2(z)
        z = torch.tanh(z)
        z = self.l3(z)
        z = torch.tanh(z)
        
        # Readout Magnitude
        z_out = self.out(z)
        # Magnitude is pixel intensity [0, 1]
        rgb = torch.sqrt(z_out[..., 0]**2 + z_out[..., 1]**2)
        return rgb

# ============================================================================
# 2. THE SCOUT (The Agent)
# ============================================================================
class Scout:
    def __init__(self):
        self.pos = np.array([0.0, 0.0]) # Normalized (x,y) from -1 to 1
        self.vel = np.array([0.0, 0.0])
        self.mass = 1.0
        self.friction = 0.92
        self.sensing_radius = 0.1 # How far it "looks" for gradients
        self.learning_rate = 0.05 # How fast it accelerates toward error

    def update(self, grad_x, grad_y):
        # Physics update based on gradient of the Error Field
        # Force = Gradient (Attracted to high error)
        force_x = grad_x
        force_y = grad_y
        
        # Acceleration
        acc_x = force_x / self.mass
        acc_y = force_y / self.mass
        
        # Velocity update
        self.vel[0] += acc_x * self.learning_rate
        self.vel[1] += acc_y * self.learning_rate
        
        # Friction (Damping)
        self.vel *= self.friction
        
        # Position update
        self.pos += self.vel
        
        # Boundary bounce
        for i in range(2):
            if self.pos[i] < -1: 
                self.pos[i] = -1; self.vel[i] *= -0.8
            if self.pos[i] > 1:  
                self.pos[i] = 1; self.vel[i] *= -0.8

# ============================================================================
# 3. THE ACTIVE CABBAGE GUI
# ============================================================================
class ActiveCabbageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Active Cabbage: The Self-Driving Neural Network")
        self.root.geometry("1100x600")
        self.root.configure(bg="#111")

        # System State
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RCNet().to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=0.005)
        self.scout = Scout()
        
        self.is_running = False
        self.target_img_np = None
        self.view_scale = 128 # The resolution of the neural view
        
        # UI Layout
        self.setup_ui()
        
    def setup_ui(self):
        # Control Panel
        frame_controls = tk.Frame(self.root, bg="#222", width=200)
        frame_controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        btn_style = {"bg": "#444", "fg": "white", "font": ("Consolas", 10), "relief": "flat"}
        
        tk.Button(frame_controls, text="Load Image", command=self.load_image, **btn_style).pack(pady=5, fill=tk.X)
        self.btn_scout = tk.Button(frame_controls, text="RELEASE SCOUT", command=self.toggle_scout, bg="#005500", fg="white", font=("Consolas", 12, "bold"))
        self.btn_scout.pack(pady=20, fill=tk.X)
        
        self.lbl_stats = tk.Label(frame_controls, text="Waiting...", bg="#222", fg="#0f0", font=("Consolas", 9), justify=tk.LEFT)
        self.lbl_stats.pack(pady=10)

        # Canvas Area
        frame_canvas = tk.Frame(self.root, bg="#000")
        frame_canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Left: Ground Truth, Right: Neural Dream + Scout Overlay
        self.lbl_view = tk.Label(frame_canvas, bg="black")
        self.lbl_view.pack(expand=True)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        
        # Load and process
        img = Image.open(path).convert('RGB')
        img.thumbnail((512, 512)) # Resize for reasonable training
        self.target_img_np = np.array(img) / 255.0
        self.h, self.w, _ = self.target_img_np.shape
        self.aspect = self.w / self.h
        
        # Create coordinate grid for the whole image (Ground Truth Reference)
        ys, xs = np.meshgrid(np.linspace(-1, 1, self.h), np.linspace(-1, 1, self.w), indexing='ij')
        self.grid_coords = np.stack([xs, ys], axis=-1).astype(np.float32)
        
        self.lbl_stats.config(text=f"Loaded: {self.w}x{self.h}")

    def toggle_scout(self):
        if not self.target_img_np is None:
            self.is_running = not self.is_running
            if self.is_running:
                self.btn_scout.config(text="RECALL SCOUT", bg="#550000")
                Thread(target=self.active_loop, daemon=True).start()
            else:
                self.btn_scout.config(text="RELEASE SCOUT", bg="#005500")

    def get_patch_loss(self, cx, cy, radius=0.15):
        """
        Samples the network and the ground truth at a specific location.
        Returns the Loss (MSE) which acts as the 'Potential Energy'.
        """
        # Create a mini-batch of coordinates around (cx, cy)
        res = 32 # Resolution of the patch
        # Map normalized coords to pixel indices
        
        # Generate local grid
        x_range = np.linspace(cx - radius, cx + radius, res)
        y_range = np.linspace(cy - radius, cy + radius, res)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        
        # Flat coords for model
        coords = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
        coords_t = torch.tensor(coords, dtype=torch.float32).to(self.device)
        
        # Get Ground Truth (nearest neighbor sampling)
        # Map -1..1 to 0..w
        pix_x = ((coords[:, 0] + 1) / 2 * (self.w - 1)).astype(int)
        pix_y = ((coords[:, 1] + 1) / 2 * (self.h - 1)).astype(int)
        
        # Clip
        pix_x = np.clip(pix_x, 0, self.w - 1)
        pix_y = np.clip(pix_y, 0, self.h - 1)
        
        target_patch = self.target_img_np[pix_y, pix_x]
        target_t = torch.tensor(target_patch, dtype=torch.float32).to(self.device)
        
        # Model Prediction
        pred_t = self.model(coords_t)
        
        # Loss
        loss = nn.MSELoss()(pred_t, target_t)
        return loss, pred_t, target_t

    def active_loop(self):
        while self.is_running:
            # 1. SENSE: Scout looks around to find the gradient of Error
            # We sample 4 points around the scout to estimate gradient
            delta = 0.05
            cx, cy = self.scout.pos
            
            # This is the "Mental Simulation" step - checking potential futures
            with torch.no_grad():
                l_center, _, _ = self.get_patch_loss(cx, cy)
                l_right, _, _  = self.get_patch_loss(cx + delta, cy)
                l_up, _, _     = self.get_patch_loss(cx, cy - delta) # y is inverted in screen coords usually
                
                # Simple finite difference gradient
                grad_x = (l_right.item() - l_center.item()) * 20.0 # Amplify signal
                grad_y = (l_up.item() - l_center.item()) * 20.0
                
            # 2. NAVIGATE: Update Scout Physics
            # We add some random noise (Brownian motion) to prevent getting stuck in local minima
            noise = np.random.randn(2) * 0.002
            self.scout.update(grad_x + noise[0], grad_y + noise[1])
            
            # 3. LEARN: Train the network AT the Scout's new location
            # The Scout's presence collapses the wave function (forces the net to decide and learn)
            self.model.train()
            self.opt.zero_grad()
            
            loss, pred, target = self.get_patch_loss(self.scout.pos[0], self.scout.pos[1])
            loss.backward()
            self.opt.step()
            
            # 4. VISUALIZE (Every few frames)
            if np.random.rand() < 0.2:
                self.update_display(pred, target, loss.item())
                
            time.sleep(0.01)

    def update_display(self, pred_t, target_t, current_loss):
        # Construct the visualization
        # We want to show the specific patch the Scout is fixing
        
        with torch.no_grad():
            patch_dream = pred_t.cpu().numpy().reshape(32, 32, 3)
            patch_real = target_t.cpu().numpy().reshape(32, 32, 3)
            
        # Resize for visibility
        viz_size = 400
        dream_big = cv2.resize(patch_dream, (viz_size, viz_size), interpolation=cv2.INTER_NEAREST)
        
        # Draw Crosshair (Scout)
        center = viz_size // 2
        cv2.circle(dream_big, (center, center), 10, (1.0, 0.0, 0.0), -1) # Red Dot
        cv2.line(dream_big, (center, center), (center + int(self.scout.vel[0]*1000), center + int(self.scout.vel[1]*1000)), (0, 1, 0), 2) # Velocity Vector
        
        # Convert to TK
        img_final = (np.clip(dream_big, 0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_final)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Stats
        stat_text = (
            f"SCOUT TELEMETRY\n"
            f"Pos: ({self.scout.pos[0]:.2f}, {self.scout.pos[1]:.2f})\n"
            f"Vel: {np.linalg.norm(self.scout.vel):.4f}\n"
            f"Local Loss: {current_loss:.5f}\n"
            f"Status: {'LEARNING' if current_loss > 0.01 else 'SURFING'}"
        )
        
        # Thread-safe update
        self.root.after(0, lambda: self._update_gui(img_tk, stat_text))

    def _update_gui(self, img_tk, text):
        self.lbl_view.config(image=img_tk)
        self.lbl_view.image = img_tk
        self.lbl_stats.config(text=text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ActiveCabbageApp(root)
    root.mainloop()