"""
NEURAL EXPLORER: THE INFINITE CANVAS
====================================
1. Load ANY Image.
2. Train RCNet to learn the continuous representation.
3. INTERACTIVE: Mouse Drag to Pan, Scroll to Zoom.
4. OBSERVATION: See how the neural net hallucinates details that don't exist
   in the pixel data as you zoom in 100x.
5. PERSISTENCE: Save/Load the trained brain.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
import time

# ============================================================================
# 1. THE RESONANT BRAIN
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
        real = self.fc_r(z.real) - self.fc_i(z.imag) + self.bias_r
        imag = self.fc_r(z.imag) + self.fc_i(z.real) + self.bias_i
        return torch.complex(real, imag)

class RCNet_Explorer(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        self.layer1 = ComplexLinear(1, width)
        self.layer2 = ComplexLinear(width, width)
        self.layer3 = ComplexLinear(width, width)
        self.layer4 = ComplexLinear(width, width)
        self.layer5 = ComplexLinear(width, width)
        self.out = ComplexLinear(width, 3)

    def forward(self, z):
        z = self.layer1(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        z = self.layer2(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        z = self.layer3(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        z = self.layer4(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        z = self.layer5(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        out = self.out(z)
        return torch.sigmoid(out.real)

# ============================================================================
# 2. THE GUI
# ============================================================================
class ExplorerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Explorer (Infinite Resolution)")
        self.master.geometry("1200x800")
        self.master.configure(bg="#111")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_training = False
        self.model = None
        self.target_img_np = None
        self.target_tensor = None
        
        # Camera State
        self.zoom = 1.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.drag_start = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top Bar
        toolbar = tk.Frame(self.master, bg="#333", height=50)
        toolbar.pack(fill=tk.X)
        
        btn_style = {"bg": "#444", "fg": "white", "font": ("Arial", 10), "relief": "flat"}
        
        tk.Button(toolbar, text="Load Image", command=self.load_image, **btn_style).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.btn_train = tk.Button(toolbar, text="Start Learning", command=self.toggle_train, state='disabled', **btn_style)
        self.btn_train.pack(side=tk.LEFT, padx=10)
        
        tk.Button(toolbar, text="Save Brain", command=self.save_model, **btn_style).pack(side=tk.LEFT, padx=10)
        tk.Button(toolbar, text="Load Brain", command=self.load_model, **btn_style).pack(side=tk.LEFT, padx=10)
        
        self.lbl_info = tk.Label(toolbar, text="Drag to Pan, Scroll to Zoom", bg="#333", fg="#aaa")
        self.lbl_info.pack(side=tk.RIGHT, padx=20)
        
        # Main Area
        self.canvas_frame = tk.Frame(self.master, bg="#111")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Reality vs Dream
        self.lbl_real = tk.Label(self.canvas_frame, bg="black")
        self.lbl_real.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.lbl_dream = tk.Label(self.canvas_frame, bg="black")
        self.lbl_dream.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bindings for Dream Panel
        self.lbl_dream.bind("<ButtonPress-1>", self.on_drag_start)
        self.lbl_dream.bind("<B1-Motion>", self.on_drag_move)
        # Mouse Wheel bindings
        self.lbl_dream.bind("<MouseWheel>", self.on_scroll) # Windows
        self.lbl_dream.bind("<Button-4>", self.on_scroll_linux) # Linux Up
        self.lbl_dream.bind("<Button-5>", self.on_scroll_linux) # Linux Down

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if path:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            
            self.target_img_np = img
            self.target_tensor = torch.tensor(img / 255.0, dtype=torch.float32).to(self.device)
            
            # Reset Model
            self.model = RCNet_Explorer(width=256).to(self.device)
            self.opt = optim.Adam(self.model.parameters(), lr=0.005)
            
            self.update_view()
            self.btn_train.config(state='normal', bg="#005500")

    def save_model(self):
        if self.model:
            path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
            if path:
                torch.save(self.model.state_dict(), path)
                messagebox.showinfo("Saved", "Neural Brain Saved Successfully.")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            # We need an image to explore, even if we load a brain
            if self.target_img_np is None:
                messagebox.showwarning("Warning", "Please load a reference image first (even a blank one) to initialize the canvas.")
                return

            self.model = RCNet_Explorer(width=256).to(self.device)
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            self.opt = optim.Adam(self.model.parameters(), lr=0.005)
            
            self.update_view()
            self.btn_train.config(state='normal', bg="#005500")

    def toggle_train(self):
        if self.is_training:
            self.is_training = False
            self.btn_train.config(text="Resume Learning", bg="#005500")
        else:
            self.is_training = True
            self.btn_train.config(text="Stop Learning", bg="#550000")
            Thread(target=self.train_loop, daemon=True).start()

    def train_loop(self):
        # Global Coordinates
        y, x = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))
        z = x + 1j*y
        train_coords = torch.tensor(z, dtype=torch.cfloat).unsqueeze(-1).to(self.device)
        
        while self.is_training:
            self.model.train()
            self.opt.zero_grad()
            
            pred = self.model(train_coords)
            loss = nn.MSELoss()(pred, self.target_tensor)
            
            loss.backward()
            self.opt.step()
            
            if np.random.rand() < 0.2: # Update UI occasionally
                self.master.after(0, self.update_view)
            
            time.sleep(0.01)

    def get_view_coords(self, res=512):
        span = 1.0 / self.zoom
        x0, x1 = self.center_x - span, self.center_x + span
        y0, y1 = self.center_y - span, self.center_y + span
        
        y, x = np.meshgrid(np.linspace(y0, y1, res), np.linspace(x0, x1, res))
        z = x + 1j*y
        return torch.tensor(z, dtype=torch.cfloat).unsqueeze(-1).to(self.device)

    def update_view(self):
        if self.model is None: return
        
        # 1. RCNet
        with torch.no_grad():
            self.model.eval()
            coords = self.get_view_coords(res=512)
            pred = self.model(coords)
            dream = pred.cpu().numpy()
            dream = (np.clip(dream, 0, 1) * 255).astype(np.uint8)
            img_tk = ImageTk.PhotoImage(Image.fromarray(dream))
            self.lbl_dream.config(image=img_tk); self.lbl_dream.image = img_tk
            
        # 2. Pixel Reality
        if self.target_img_np is not None:
            h, w, _ = self.target_img_np.shape
            span_pix = int((1.0 / self.zoom) * (w / 2))
            cx_pix = int((self.center_x + 1.0) / 2.0 * w)
            cy_pix = int((self.center_y + 1.0) / 2.0 * h)
            
            x0 = max(0, cx_pix - span_pix); x1 = min(w, cx_pix + span_pix)
            y0 = max(0, cy_pix - span_pix); y1 = min(h, cy_pix + span_pix)
            
            if x1 > x0 and y1 > y0:
                crop = self.target_img_np[y0:y1, x0:x1]
                crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_NEAREST)
                real_tk = ImageTk.PhotoImage(Image.fromarray(crop))
                self.lbl_real.config(image=real_tk); self.lbl_real.image = real_tk
            else:
                self.lbl_real.config(image='')

    # Controls
    def on_drag_start(self, event): self.drag_start = (event.x, event.y)
    def on_drag_move(self, event):
        if self.drag_start:
            dx = event.x - self.drag_start[0]; dy = event.y - self.drag_start[1]
            move_scale = 0.005 / self.zoom
            self.center_x -= dx * move_scale; self.center_y -= dy * move_scale
            self.drag_start = (event.x, event.y)
            self.update_view()
    def on_scroll(self, event):
        if event.delta > 0: self.zoom *= 1.1
        else: self.zoom /= 1.1
        self.zoom = max(0.1, self.zoom)
        self.update_view()
    def on_scroll_linux(self, event):
        if event.num == 4: self.zoom *= 1.1
        else: self.zoom /= 1.1
        self.zoom = max(0.1, self.zoom)
        self.update_view()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExplorerApp(root)
    root.mainloop()