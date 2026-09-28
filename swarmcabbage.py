"""
BIOMORPHIC CABBAGE: SWARM INTELLIGENCE  (fixed + instrumented, Sep 2026)
=======================================================================
A coordinate->RGB "cortex" learns an image, but only from the patches that
50 scouts choose to look at. Scouts climb the error: they go where the
cortex is still wrong.

THE CAST
  BUILDERS  (red, big)   : magnocellular. Heavy, fast, wide patches.
  DETAILERS (blue, small): parvocellular. Light, slow, narrow patches.
  CORTEX                 : the shared field. The swarm's external memory.

THREE PANES
  TARGET | HIVE MIND (what the cortex believes) | WHERE IT LOOKED (fading visits)

WHAT WAS FIXED (vs. the 2025 version)
  1. Scouts were drawn with colour (0,0,1)/(1,0,0) on a 0-255 image -> black
     blobs. Now real red / blue.
  2. The vertical error gradient pointed the wrong way (sampled ABOVE the scout,
     pushed DOWN). Now central differences on both axes, signs match the
     pixel mapping (+y = down).
  3. "Plasticity 5%" was 1-friction, which is backwards: friction 0.95 keeps
     MORE momentum. The UI now says EXPLORING / SETTLING with the real value.
  4. Tk was touched from the worker thread; ImageTk images were made there.
     Now the worker only publishes arrays; the Tk thread draws them.
  5. Pressing AWAKEN with no image crashed. There is a built-in target now.
  6. Everything is batched: one forward pass for all sensing probes, one for
     all training patches (was 150+ separate forwards per step).
  7. sqrt(|z|^2) had an infinite gradient at 0; added a tiny epsilon.
  8. The per-type learning_rate was defined but never used; it now weights
     each scout's share of the loss.

NEW
  - "Add surprise": pastes a new object into one corner of the target.
    Watch the scouts leave the solved parts and go to the new thing.
  - "Where scouts look": Swarm / Pinned / Random. Same cortex, same 50
    patches per step; only where they land differs. Random is the control
    the swarm has to beat.
  - "Use webcam": the target is your live camera, 128x128.
  - `python swarmcabbage.py --benchmark` runs the moving-vs-pinned-vs-random
    test headless and prints the numbers.
"""

import sys
import time
import argparse
from threading import Thread, Lock

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

TARGET_SIZE = 128  # the target is kept square; the cortex lives on [-1,1]^2


# ============================================================================
# 1. THE CORTEX (Neural Field)  -- unchanged architecture
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
        z = torch.stack([x * np.pi * self.feature_scale,
                         x * np.pi * self.feature_scale * 0.5], dim=-1)
        z = torch.tanh(self.l1(z))
        z = torch.tanh(self.l2(z))
        z = torch.tanh(self.l3(z))
        z_out = self.out(z)
        return torch.sqrt(z_out[..., 0] ** 2 + z_out[..., 1] ** 2 + 1e-8)


# ============================================================================
# 2. TARGETS
# ============================================================================
def make_letter_target(size=TARGET_SIZE, letter="A"):
    img = np.full((size, size, 3), 20, np.uint8)
    cv2.putText(img, letter, (int(size * 0.16), int(size * 0.84)),
                cv2.FONT_HERSHEY_SIMPLEX, size / 34.0, (240, 230, 200),
                thickness=max(2, size // 12), lineType=cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def add_surprise(target, rng=None):
    """Paste a bright new object into a random corner. Returns (new, mask)."""
    rng = rng or np.random.default_rng()
    t = (target * 255).astype(np.uint8).copy()
    s = t.shape[0]
    cx = int(s * (0.2 if rng.random() < 0.5 else 0.8))
    cy = int(s * (0.2 if rng.random() < 0.5 else 0.8))
    r = int(s * 0.13)
    colour = [(60, 220, 90), (240, 80, 60), (70, 140, 250)][rng.integers(3)]
    cv2.circle(t, (cx, cy), r, colour, -1, cv2.LINE_AA)
    cv2.circle(t, (cx, cy), r // 2, (250, 250, 250), -1, cv2.LINE_AA)
    mask = np.zeros(t.shape[:2], np.float32)
    cv2.circle(mask, (cx, cy), r, 1.0, -1)
    return t.astype(np.float32) / 255.0, mask


def to_square(img_rgb_float, size=TARGET_SIZE):
    h, w = img_rgb_float.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = img_rgb_float[y0:y0 + s, x0:x0 + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ============================================================================
# 3. THE HIVE (scouts as arrays, everything batched)
# ============================================================================
class Hive:
    N_BUILDERS = 15
    N_DETAILERS = 35

    def __init__(self, device="cpu", seed=0):
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cortex = VisualCortex().to(self.device)
        self.opt = optim.Adam(self.cortex.parameters(), lr=0.005)

        nb, nd = self.N_BUILDERS, self.N_DETAILERS
        self.n = nb + nd
        self.is_builder = np.array([True] * nb + [False] * nd)
        self.mass = np.where(self.is_builder, 2.0, 0.5)
        self.lr_weight = np.where(self.is_builder, 0.08, 0.02)
        self.lr_weight = self.lr_weight / self.lr_weight.mean()
        self.radius = np.where(self.is_builder, 0.2, 0.05)
        self.max_speed = np.where(self.is_builder, 0.05, 0.02)
        self.noise = np.where(self.is_builder, 0.005, 0.01)

        self.pos = self.rng.uniform(-1, 1, (self.n, 2))
        self.vel = np.zeros((self.n, 2))
        self.friction = 0.90
        self.mode_text = "WAKING"

        self.target = None
        self.target_t = None
        self.set_target(make_letter_target())

        self.visits = np.zeros((64, 64), np.float32)   # WHERE IT LOOKED
        self.last_scout_loss = np.zeros(self.n)
        self.steps = 0
        self.lock = Lock()

        # fixed patch templates
        self._res_train, self._res_sense = 16, 8
        self._tmpl_train = self._template(self._res_train)
        self._tmpl_sense = self._template(self._res_sense)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _template(res):
        g = np.linspace(-1, 1, res)
        gx, gy = np.meshgrid(g, g)
        return np.stack([gx.ravel(), gy.ravel()], -1)  # (res*res, 2)

    def set_target(self, img_rgb_float):
        img = to_square(np.clip(img_rgb_float, 0, 1).astype(np.float32))
        self.target = img
        self.target_t = torch.tensor(img, dtype=torch.float32, device=self.device)

    def _lookup(self, coords_np):
        """Target colour at [-1,1]^2 coords (nearest pixel, clamped at edges)."""
        s = self.target.shape[0]
        px = np.clip(((coords_np[:, 0] + 1) / 2 * (s - 1)).round().astype(int), 0, s - 1)
        py = np.clip(((coords_np[:, 1] + 1) / 2 * (s - 1)).round().astype(int), 0, s - 1)
        return self.target_t[torch.as_tensor(py, device=self.device),
                             torch.as_tensor(px, device=self.device)]

    def _patch_losses(self, centers, radii, tmpl, grad):
        """Mean squared error of the cortex in a square patch around each center.
        centers (k,2), radii (k,) -> torch (k,)"""
        k, m = len(centers), len(tmpl)
        coords = (centers[:, None, :] + radii[:, None, None] * tmpl[None]).reshape(-1, 2)
        target = self._lookup(coords)
        ct = torch.tensor(coords, dtype=torch.float32, device=self.device)
        if grad:
            pred = self.cortex(ct)
        else:
            with torch.no_grad():
                pred = self.cortex(ct)
        return ((pred - target) ** 2).mean(-1).view(k, m).mean(-1)

    # ------------------------------------------------------------------- step
    def step(self, move="swarm"):
        """move: 'swarm' (climb the error), 'pinned' (never move),
                 'random' (fresh uniform positions every step)."""
        with self.lock:
            if move == "swarm":
                self._sense_and_move()
            elif move == "random":
                self.pos = self.rng.uniform(-1, 1, (self.n, 2))
                self.vel[:] = 0

            # LEARN: one batched forward/backward over all scouts' patches
            self.cortex.train()
            self.opt.zero_grad()
            losses = self._patch_losses(self.pos, self.radius, self._tmpl_train, grad=True)
            w = torch.tensor(self.lr_weight, dtype=torch.float32, device=self.device)
            (losses * w).mean().backward()
            self.opt.step()
            self.last_scout_loss = losses.detach().cpu().numpy()
            avg = float(self.last_scout_loss.mean())

            # HOMEOSTASIS: high error -> keep momentum (explore); low -> settle
            if avg < 0.005:
                self.friction, self.mode_text = 0.85, "SETTLING"
            elif avg > 0.02:
                self.friction, self.mode_text = 0.95, "EXPLORING"

            self._deposit_visits()
            self.steps += 1
            return avg

    def _sense_and_move(self):
        e = 0.05
        offs = np.array([[e, 0], [-e, 0], [0, e], [0, -e]])
        centers = (self.pos[:, None, :] + offs[None]).reshape(-1, 2)
        radii = np.full(len(centers), 0.05)
        L = self._patch_losses(centers, radii, self._tmpl_sense, grad=False)
        L = L.cpu().numpy().reshape(self.n, 4)
        # central differences; +x = right, +y = DOWN (same as pixel rows)
        gx = (L[:, 0] - L[:, 1]) * 15.0
        gy = (L[:, 2] - L[:, 3]) * 15.0
        force = np.stack([gx, gy], -1) + self.rng.normal(size=(self.n, 2)) * self.noise[:, None]

        self.vel += force / self.mass[:, None]
        self.vel *= self.friction
        speed = np.linalg.norm(self.vel, axis=1)
        too_fast = speed > self.max_speed
        self.vel[too_fast] *= (self.max_speed[too_fast] / speed[too_fast])[:, None]
        self.pos += self.vel
        for i in range(2):  # bounce
            lo, hi = self.pos[:, i] < -1, self.pos[:, i] > 1
            self.pos[lo, i], self.pos[hi, i] = -1, 1
            self.vel[lo | hi, i] *= -0.8

    def _deposit_visits(self):
        self.visits *= 0.985
        s = self.visits.shape[0]
        yy, xx = np.mgrid[0:s, 0:s]
        for (x, y), r in zip(self.pos, self.radius):
            cx, cy = (x + 1) / 2 * (s - 1), (y + 1) / 2 * (s - 1)
            rr = max(1.0, r / 2 * (s - 1))
            self.visits += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rr * rr)) * 0.05

    # ----------------------------------------------------------------- output
    def belief(self, res=64):
        g = np.linspace(-1, 1, res)
        gx, gy = np.meshgrid(g, g)
        ct = torch.tensor(np.stack([gx.ravel(), gy.ravel()], -1),
                          dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.cortex(ct).cpu().numpy().reshape(res, res, 3)
        return np.clip(out, 0, 1)

    def full_error(self, res=64, mask=None):
        b = self.belief(res)
        t = cv2.resize(self.target, (res, res), interpolation=cv2.INTER_AREA)
        err = ((b - t) ** 2).mean(-1)
        if mask is not None:
            m = cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST) > 0.5
            return float(err.mean()), float(err[m].mean()), float(err[~m].mean())
        return float(err.mean())

    def snapshot(self):
        return {k: (v.state_dict() if hasattr(v, "state_dict") else v)
                for k, v in [("cortex", self.cortex), ("opt", self.opt)]}


# ============================================================================
# 4. HEADLESS TEST: does moving where the error is beat not moving?
# ============================================================================
def benchmark(seeds=(0, 1, 2), warm=500, after=300, modes=("swarm", "pinned", "random")):
    print("Each condition starts from a fresh cortex, learns 'A' for %d steps, then a"
          " surprise object is added and it continues %d steps.\nSame 50 patches per step"
          " in every condition; only WHERE they land differs.\n" % (warm, after))
    rows = {m: [] for m in modes}
    for mode in modes:
        for seed in seeds:
            h = Hive(seed=seed)
            for _ in range(warm):
                h.step(mode)
            learned = h.full_error()
            new_t, mask = add_surprise(h.target, np.random.default_rng(100 + seed))
            h.set_target(new_t)
            curve = []
            for t in range(after):
                h.step(mode)
                if (t + 1) % 50 == 0:
                    curve.append(h.full_error(mask=mask))
            rows[mode].append((learned, curve))
            tot, inside, _ = curve[-1]
            print(f"  {mode:7s} seed {seed}: 'A' learned to MSE {learned:.4f};"
                  f" after surprise: whole {tot:.4f}, inside surprise {inside:.4f}", flush=True)
    print("\nmean over seeds")
    print("  mode     A-learned | whole-image MSE every 50 steps after surprise"
          "            | inside the surprise")
    for mode, rs in rows.items():
        a = np.mean([r[0] for r in rs])
        c = np.array([r[1] for r in rs]).mean(0)
        print(f"  {mode:7s}  {a:.4f}  | " + " ".join(f"{v:.4f}" for v in c[:, 0])
              + " | " + " ".join(f"{v:.4f}" for v in c[:, 1]))
    print("\nNote: 'swarm' also spends 4 no-grad sensing patches per scout per step;"
          " 'pinned' and 'random' do not.")


# ============================================================================
# 5. THE GUI
# ============================================================================
def run_gui():
    import tkinter as tk
    from tkinter import filedialog
    from PIL import Image, ImageTk

    PANE = 300

    class BioSwarmApp:
        def __init__(self, root):
            self.root = root
            root.title("Biomorphic Swarm: The Living Cortex")
            root.configure(bg="#111")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.hive = Hive(device=dev, seed=int(time.time()) % 10000)
            self.running = False
            self.worker = None
            self.mode = tk.StringVar(value="swarm")
            self.webcam_on = False
            self.cap = None
            self.frame = None      # published by worker, drawn by Tk thread
            self.status = ""
            self._setup_ui()
            self._render()          # show initial state
            self.root.after(60, self._poll)
            root.protocol("WM_DELETE_WINDOW", self._close)

        # ---------------------------------------------------------------- UI
        def _setup_ui(self):
            panel = tk.Frame(self.root, bg="#222")
            panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
            btn = dict(bg="#444", fg="white", activebackground="#666")
            tk.Button(panel, text="Load Image", command=self.load_image, **btn).pack(pady=4, fill=tk.X)
            self.btn_cam = tk.Button(panel, text="Use Webcam", command=self.toggle_webcam, **btn)
            self.btn_cam.pack(pady=4, fill=tk.X)
            tk.Button(panel, text="Add Surprise", command=self.surprise, **btn).pack(pady=4, fill=tk.X)
            tk.Button(panel, text="Reset to 'A'", command=self.reset_target, **btn).pack(pady=4, fill=tk.X)
            tk.Label(panel, text="Where scouts look:", bg="#222", fg="#aaa").pack(pady=(10, 0), anchor="w")
            for val, txt in (("swarm", "Swarm (climb the error)"),
                             ("pinned", "Pinned (never move)"),
                             ("random", "Random (control)")):
                tk.Radiobutton(panel, text=txt, value=val, variable=self.mode,
                               bg="#222", fg="#ddd", selectcolor="#333",
                               activebackground="#222").pack(anchor="w")
            self.btn_run = tk.Button(panel, text="AWAKEN HIVE", command=self.toggle_hive,
                                     bg="#005500", fg="white", font=("Arial", 12, "bold"))
            self.btn_run.pack(pady=14, fill=tk.X)
            self.lbl_status = tk.Label(panel, text="Hive dormant", bg="#222", fg="#0f0",
                                       font=("Consolas", 10), justify=tk.LEFT, anchor="w")
            self.lbl_status.pack(pady=6, fill=tk.X)
            legend = ("red  = builders (wide look)\n"
                      "blue = detailers (narrow look)\n\n"
                      "Scouts climb the error:\n"
                      "they go where the hive\n"
                      "is still wrong.")
            tk.Label(panel, text=legend, bg="#222", fg="#888", font=("Consolas", 9),
                     justify=tk.LEFT).pack(pady=10, anchor="w")

            right = tk.Frame(self.root, bg="#111")
            right.pack(side=tk.RIGHT, expand=True, padx=10, pady=10)
            self.views = []
            for col, title in enumerate(("TARGET", "HIVE MIND", "WHERE IT LOOKED")):
                tk.Label(right, text=title, bg="#111", fg="#aaa",
                         font=("Consolas", 11, "bold")).grid(row=0, column=col, pady=(0, 4))
                lbl = tk.Label(right, bg="black", width=PANE, height=PANE)
                lbl.grid(row=1, column=col, padx=4)
                self.views.append(lbl)

        # ----------------------------------------------------------- actions
        def load_image(self):
            path = filedialog.askopenfilename()
            if not path:
                return
            img = cv2.imread(path)
            if img is None:
                self.lbl_status.config(text="Could not read that file")
                return
            self._stop_webcam()
            self.hive.set_target(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            if not self.running:
                self._render()

        def surprise(self):
            new, _ = add_surprise(self.hive.target)
            self.hive.set_target(new)
            if not self.running:
                self._render()

        def reset_target(self):
            self._stop_webcam()
            self.hive.set_target(make_letter_target())
            if not self.running:
                self._render()

        def toggle_webcam(self):
            if self.webcam_on:
                self._stop_webcam()
                return
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.lbl_status.config(text="No webcam found")
                self.cap = None
                return
            self.webcam_on = True
            self.btn_cam.config(text="Stop Webcam", bg="#335")

        def _stop_webcam(self):
            self.webcam_on = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.btn_cam.config(text="Use Webcam", bg="#444")

        def toggle_hive(self):
            self.running = not self.running
            if self.running:
                self.btn_run.config(text="FREEZE HIVE", bg="#550000")
                if self.worker is None or not self.worker.is_alive():
                    self.worker = Thread(target=self._loop, daemon=True)
                    self.worker.start()
            else:
                self.btn_run.config(text="AWAKEN HIVE", bg="#005500")

        # -------------------------------------------------------- the loop
        def _loop(self):
            while self.running:
                if self.webcam_on and self.cap is not None and self.hive.steps % 4 == 0:
                    ok, fr = self.cap.read()
                    if ok:
                        fr = cv2.flip(fr, 1)
                        self.hive.set_target(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
                loss = self.hive.step(self.mode.get())
                if self.hive.steps % 4 == 0:
                    self._render(loss)

        def _render(self, loss=None):
            h = self.hive
            with h.lock:
                target = h.target.copy()
                pos = h.pos.copy()
                visits = h.visits.copy()
                belief = h.belief(96)
                mode, fr = h.mode_text, h.friction
            t_img = (cv2.resize(target, (PANE, PANE), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
            b_img = (cv2.resize(belief, (PANE, PANE), interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)
            v = visits / (visits.max() + 1e-6)
            v_img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            v_img = cv2.cvtColor(cv2.resize(v_img, (PANE, PANE), interpolation=cv2.INTER_LINEAR),
                                 cv2.COLOR_BGR2RGB)
            b_img = np.ascontiguousarray(b_img)
            for (x, y), builder in zip(pos, h.is_builder):
                cx, cy = int((x + 1) / 2 * (PANE - 1)), int((y + 1) / 2 * (PANE - 1))
                rad, col = (6, (255, 70, 70)) if builder else (3, (80, 170, 255))
                cv2.circle(b_img, (cx, cy), rad + 1, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(b_img, (cx, cy), rad, col, -1, cv2.LINE_AA)
            self.frame = (t_img, b_img, v_img)
            if loss is not None:
                err = h.full_error(48)
                self.status = (f"HIVE STATE [{self.mode.get().upper()}]\nScouts: {h.n}\nStep: {h.steps}\n"
                               f"Patch loss: {loss:.4f}\nWhole-image MSE: {err:.4f}\n"
                               f"Swarm: {mode}\n  (momentum {fr:.2f})")

        def _poll(self):
            if self.frame is not None:
                imgs = [ImageTk.PhotoImage(Image.fromarray(a)) for a in self.frame]
                for lbl, im in zip(self.views, imgs):
                    lbl.config(image=im, width=PANE, height=PANE)
                    lbl.image = im
                self.frame = None
            if self.status:
                self.lbl_status.config(text=self.status)
            self.root.after(60, self._poll)

        def _close(self):
            self.running = False
            self._stop_webcam()
            self.root.destroy()

    root = tk.Tk()
    BioSwarmApp(root)
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true",
                    help="run the moving vs pinned vs random test headless")
    args = ap.parse_args()
    if args.benchmark:
        benchmark()
    else:
        run_gui()
