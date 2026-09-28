"""lineage_test.py -- the sampling strategies of the lineage, on one cortex.

cabbage3  : trains on EVERY pixel, every step (full view)
cabbage4  : ONE scout climbing the error, trains on its one 32x32 patch (r=0.15)
random1   : ONE patch of the same size, dropped at random (the control for cabbage4)
Same cortex (the swarmcabbage/cabbage4 tanh cortex), same optimiser, same step count.
"""
import numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim
from skimage import data
from swarmcabbage import VisualCortex

img = cv2.resize(data.astronaut(), (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
T = torch.tensor(img)
g = np.linspace(-1, 1, 128); gx, gy = np.meshgrid(g, g)
FULL = torch.tensor(np.stack([gx.ravel(), gy.ravel()], -1), dtype=torch.float32)

def lookup(c):
    px = np.clip(((c[:, 0] + 1) / 2 * 127).round().astype(int), 0, 127)
    py = np.clip(((c[:, 1] + 1) / 2 * 127).round().astype(int), 0, 127)
    return T[py, px]

tmpl = np.stack(np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)), -1).reshape(-1, 2)
def patch(net, cx, cy, r=0.15, grad=True):
    c = np.array([cx, cy]) + r * tmpl
    ct = torch.tensor(c, dtype=torch.float32)
    with torch.set_grad_enabled(grad):
        return ((net(ct) - lookup(c)) ** 2).mean()

def whole(net):
    with torch.no_grad():
        return float(((net(FULL).clamp(0, 1) - T.view(-1, 3)) ** 2).mean())

def run(mode, seed, steps=1500):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = VisualCortex(); opt = optim.Adam(net.parameters(), lr=0.005)
    pos, vel = np.zeros(2), np.zeros(2)
    curve = []
    for s in range(steps):
        opt.zero_grad()
        if mode == "cabbage3":
            loss = ((net(FULL) - T.view(-1, 3)) ** 2).mean()
        elif mode == "random1":
            loss = patch(net, *rng.uniform(-1, 1, 2))
        else:  # cabbage4 scout, with the y-sign fixed
            d = 0.05
            lc, lr_, ld = (patch(net, pos[0], pos[1], grad=False).item(),
                           patch(net, pos[0] + d, pos[1], grad=False).item(),
                           patch(net, pos[0], pos[1] + d, grad=False).item())
            f = np.array([lr_ - lc, ld - lc]) * 20 + rng.normal(size=2) * 0.002
            vel = (vel + f * 0.05) * 0.92; pos = pos + vel
            for i in range(2):
                if abs(pos[i]) > 1: pos[i] = np.sign(pos[i]); vel[i] *= -0.8
            loss = patch(net, *pos)
        loss.backward(); opt.step()
        if (s + 1) % 300 == 0: curve.append(whole(net))
    return curve

for mode in ["cabbage3", "cabbage4", "random1"]:
    cs = np.array([run(mode, s) for s in (0, 1)])
    print(f"{mode:9s} whole-image MSE at 300..1500 steps: " + "  ".join(f"{v:.4f}" for v in cs.mean(0)), flush=True)
