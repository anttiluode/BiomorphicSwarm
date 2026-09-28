"""
BIOMORPHIC CABBAGE: SWARM INTELLIGENCE
======================================
A simulation of the Visual Cortex (V1) learning an image.

THE CAST:
1. THE BUILDERS (Red): Magnocellular. Big, fast, fix global shapes.
2. THE DETAILERS (Blue): Parvocellular. Small, precise, fix textures.
3. THE HIVE MIND: Pheromone trails attract scouts to high-error zones.

Watch the "Baby Brain" (Chaos) evolve into "Adult Brain" (Structure).
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

# ============================================================================
# 1. THE CORTEX (Neural Field)
# ============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(out_features))
        self.bias_i = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_normal_(self.fc_r.weight, gain=0.2)
        nn.init.xavier_normal_(self.fc_i.weight, gain=0.2)

    def forward(self, z):
        real, imag = z[..., 0], z[..., 1]
        out_r = self.fc_r(real) - self.fc_i(imag) + self.bias_r
        out_i = self.fc_r(imag) + self.fc_i(real) + self.bias_i
        return torch.stack([out_r, out_i], dim=-1)

class VisualCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_scale = 10.0 
        self.l1 = ComplexLinear(2, 64)
        self.l2 = ComplexLinear(64, 64)
        self.l3 = ComplexLinear(64, 64)
        self.out = ComplexLinear(64, 3) 

    def forward(self, x):
        # Embed coordinates into complex plane
        z = torch.stack([x * np.pi * self.feature_scale, x * np.pi * self.feature_scale * 0.5], dim=-1)
        z = torch.tanh(self.l1(z))
        z = torch.tanh(self.l2(z))
        z = torch.tanh(self.l3(z))
        z_out = self.out(z)
        return torch.sqrt(z_out[..., 0]**2 + z_out[..., 1]**2)

# ============================================================================
# 2. THE SWARM (Agents)
# ============================================================================
class Scout:
    def __init__(self, type_id):
        self.pos = np.random.uniform(-1, 1, 2)
        self.vel = np.zeros(2)
        self.type = type_id # 0 = Builder (Red), 1 = Detailer (Blue)
        
        # Biomorphic Specialization
        if self.type == 0: # BUILDER (Magno)
            self.mass = 2.0
            self.learning_rate = 0.08
            self.sensing_radius = 0.2
            self.color = (0, 0, 1) # Red (BGR)
            self.max_speed = 0.05
        else: # DETAILER (Parvo)
            self.mass = 0.5
            self.learning_rate = 0.02
            self.sensing_radius = 0.05
            self.color = (1, 0, 0) # Blue (BGR)
            self.max_speed = 0.02

    def update(self, force, friction):
        acc = force / self.mass
        self.vel += acc
        self.vel *= friction
        
        # Speed Limit
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
            
        self.pos += self.vel
        
        # Bounce
        for i in range(2):
            if self.pos[i] < -1: self.pos[i] = -1; self.vel[i] *= -0.8
            if self.pos[i] > 1: self.pos[i] = 1; self.vel[i] *= -0.8

# ============================================================================
# 3. THE BIO-SIMULATION APP
# ============================================================================
class BioSwarmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biomorphic Swarm: The Living Cortex")
        self.root.geometry("1000x600")
        self.root.configure(bg="#111")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cortex = VisualCortex().to(self.device)
        self.opt = optim.Adam(self.cortex.parameters(), lr=0.005)
        
        # Spawn The Colony
        self.scouts = []
        for _ in range(15): self.scouts.append(Scout(0)) # 15 Builders
        for _ in range(35): self.scouts.append(Scout(1)) # 35 Detailers
        
        self.target_img_np = None
        self.is_running = False
        self.global_friction = 0.90 # "Plasticity" (starts low, increases)
        
        self.setup_ui()

    def setup_ui(self):
        panel = tk.Frame(self.root, bg="#222")
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Button(panel, text="Load DNA (Image)", command=self.load_image, bg="#444", fg="white").pack(pady=5, fill=tk.X)
        self.btn_run = tk.Button(panel, text="AWAKEN HIVE", command=self.toggle_hive, bg="#005500", fg="white", font=("Arial", 12, "bold"))
        self.btn_run.pack(pady=20, fill=tk.X)
        
        self.lbl_status = tk.Label(panel, text="Hive Dormant", bg="#222", fg="#0f0", font=("Consolas", 10), justify=tk.LEFT)
        self.lbl_status.pack(pady=10)
        
        self.lbl_view = tk.Label(self.root, bg="black")
        self.lbl_view.pack(side=tk.RIGHT, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        img = Image.open(path).convert('RGB')
        img.thumbnail((256, 256)) 
        self.target_img_np = np.array(img) / 255.0
        self.h, self.w, _ = self.target_img_np.shape
        self.lbl_status.config(text=f"Target Acquired: {self.w}x{self.h}")

    def toggle_hive(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_run.config(text="FREEZE HIVE", bg="#550000")
            Thread(target=self.hive_loop, daemon=True).start()
        else:
            self.btn_run.config(text="AWAKEN HIVE", bg="#005500")

    def get_gradients(self, scouts):
        # Calculate gradients for ALL scouts in a batch for speed
        grads_x = []
        grads_y = []
        losses = []
        
        # This part is a bit slow on CPU, would be instant on GPU batch
        # For demo we do simple sampling
        with torch.no_grad():
            for s in scouts:
                l_c, _, _ = self.get_patch_loss(s.pos[0], s.pos[1], 0.05)
                l_r, _, _ = self.get_patch_loss(s.pos[0] + 0.05, s.pos[1], 0.05)
                l_u, _, _ = self.get_patch_loss(s.pos[0], s.pos[1] - 0.05, 0.05)
                
                loss_val = l_c.item()
                gx = (l_r.item() - loss_val) * 30.0
                gy = (l_u.item() - loss_val) * 30.0
                
                # Pheromone attraction (pull towards center of image if lost)
                # dist = np.linalg.norm(s.pos)
                # if dist > 0.9: gx -= s.pos[0]*0.1; gy -= s.pos[1]*0.1
                
                grads_x.append(gx)
                grads_y.append(gy)
                losses.append(loss_val)
                
        return grads_x, grads_y, losses

    def get_patch_loss(self, cx, cy, radius):
        # ... (Same as previous logical mapping) ...
        # Simplified for brevity in this turn
        res = 16
        x_range = np.linspace(cx - radius, cx + radius, res)
        y_range = np.linspace(cy - radius, cy + radius, res)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        coords = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
        coords_t = torch.tensor(coords, dtype=torch.float32).to(self.device)
        
        pix_x = np.clip(((coords[:, 0] + 1) / 2 * (self.w - 1)).astype(int), 0, self.w - 1)
        pix_y = np.clip(((coords[:, 1] + 1) / 2 * (self.h - 1)).astype(int), 0, self.h - 1)
        
        target_t = torch.tensor(self.target_img_np[pix_y, pix_x], dtype=torch.float32).to(self.device)
        pred_t = self.cortex(coords_t)
        loss = nn.MSELoss()(pred_t, target_t)
        return loss, pred_t, target_t

    def hive_loop(self):
        epoch = 0
        while self.is_running:
            # 1. SENSE (Parallel)
            gx, gy, losses = self.get_gradients(self.scouts)
            
            # 2. MOVE & TRAIN (Sequential for now, could be batched)
            self.cortex.train()
            self.opt.zero_grad()
            
            total_loss = 0
            
            for i, s in enumerate(self.scouts):
                # Physics
                # Add random noise (Temperature)
                noise = np.random.randn(2) * (0.005 if s.type==0 else 0.01)
                
                # Attraction to high loss (Gradient Ascent on Error)
                s.update(np.array([gx[i], gy[i]]) + noise, self.global_friction)
                
                # Learning
                loss, _, _ = self.get_patch_loss(s.pos[0], s.pos[1], s.sensing_radius)
                loss.backward() # Accumulate gradients from all scouts
                total_loss += loss.item()

            self.opt.step()
            
            # 3. HOMEOSTASIS (Adaptive Friction)
            # As error drops, increase friction (crystallize)
            avg_loss = total_loss / len(self.scouts)
            if avg_loss < 0.005: self.global_friction = 0.85 # solidify
            elif avg_loss > 0.02: self.global_friction = 0.95 # liquefy
            
            epoch += 1
            if epoch % 5 == 0:
                self.visualize_swarm(avg_loss)
            
            # time.sleep(0.01)

    def visualize_swarm(self, loss_val):
        # We draw the "Mind's Eye"
        # Since we can't render full resolution every frame, we render a low-res background
        # and overlay the agents.
        
        viz_size = 400
        # Quick background render (center only to save time)
        with torch.no_grad():
            xs = np.linspace(-1, 1, 64)
            ys = np.linspace(-1, 1, 64)
            gx, gy = np.meshgrid(xs, ys)
            coords = np.stack([gx.flatten(), gy.flatten()], axis=-1)
            ct = torch.tensor(coords, dtype=torch.float32).to(self.device)
            bg = self.cortex(ct).cpu().numpy().reshape(64, 64, 3)
            
        bg_big = cv2.resize(bg, (viz_size, viz_size), interpolation=cv2.INTER_NEAREST)
        bg_uint = (np.clip(bg_big, 0, 1) * 255).astype(np.uint8)
        
        # Draw Scouts
        for s in self.scouts:
            cx = int((s.pos[0] + 1) / 2 * viz_size)
            cy = int((s.pos[1] + 1) / 2 * viz_size)
            # Size depends on type
            rad = 6 if s.type == 0 else 3
            cv2.circle(bg_uint, (cx, cy), rad, s.color, -1)
            
        img_pil = Image.fromarray(bg_uint)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        status = f"HIVE STATE\nScouts: {len(self.scouts)}\nGlobal Loss: {loss_val:.4f}\nPlasticity: {(1-self.global_friction)*100:.1f}%"
        self.root.after(0, lambda: self.update_gui(img_tk, status))

    def update_gui(self, img, txt):
        self.lbl_view.config(image=img)
        self.lbl_view.image = img
        self.lbl_status.config(text=txt)

if __name__ == "__main__":
    root = tk.Tk()
    app = BioSwarmApp(root)
    root.mainloop()