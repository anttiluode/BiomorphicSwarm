"""Renders the lineage figures from the fossil files in ../lineage (run from repo root).
Each ancestor is run as written; only the drawing is new."""
import sys, types, importlib.util, numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, label
from skimage import data
DARK = dict(facecolor="#111")

# ---------- 1. anttis-instanton: run its simulator class exactly as written ----------
src = open("lineage/1_anttis-instanton.py").read().split("# --- Visualization ---")[0]
ns = {}; exec(src, ns)
sim = ns["SubstrateParticleSimulator"](ns["GRID_SIZE"], ns["DT"], ns["DX"], ns["C_WAVE"],
                                       ns["A_COEFF_POTENTIAL"], ns["B_COEFF_POTENTIAL"])
sim.setup_simulation()
fig, axs = plt.subplots(1, 5, figsize=(20, 4.6), **DARK)
track = []
def adv(t):
    while sim.time < t:
        sim.step()
        if int(round(sim.time / sim.dt)) % 25 == 0:
            p = np.maximum(0, sim.phi); track.append((sim.time, *center_of_mass(p)[::-1]))
for ax, t in zip(axs, (5, 60, 110, 140)):
    adv(t)
    pos = np.maximum(0, sim.phi); cy, cx = center_of_mass(pos)
    nb = label(pos > 0.2 * pos.max())[1]
    ax.imshow(sim.phi, cmap="RdBu_r", origin="lower", vmin=-1.6, vmax=1.6)
    ax.plot(cx, cy, "o", ms=11, mfc="#3f3", mec="k")
    ax.set_title(f"t = {t}   blobs: {nb}", color="w"); ax.axis("off")
tr = np.array(track); a = axs[4]; a.set_facecolor("#111")
a.plot(tr[:, 0], tr[:, 1], c="#3f3"); a.set_xlabel("time", color="w"); a.set_ylabel("dot x position", color="w")
a.tick_params(colors="w"); a.set_title("the dot's x over time: the 'tunnelling' jumps", color="w")
fig.suptitle("1 · anttis-instanton.py (Jun 2025): the green dot is the field's centre of mass. The empty field is "
             "unstable, blobs appear everywhere, and the dot jumps as blobs are born and die.",
             color="w", fontsize=12, y=1.02)
plt.tight_layout(); plt.savefig("figs/lineage_1_instanton.png", dpi=72, bbox_inches="tight", **DARK); plt.close()

# ---------- 2. scout_com_instanton: its own class, stepped as written ----------
spec = importlib.util.spec_from_file_location("sci", "lineage/2_scout_com_instanton.py")
sci = importlib.util.module_from_spec(spec); spec.loader.exec_module(sci)
np.random.seed(0); h = sci.HybridFieldIntelligence()
for _ in range(1500): h.step()
S, W = np.array(h.scout_history), np.array(h.wave_com_history)
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), **DARK)
axs[0].imshow(h.attractor_field, cmap="terrain", origin="lower")
axs[0].plot(S[:, 0], S[:, 1], "-", c="#f33", lw=1.5); axs[0].plot(*S[-1], "o", ms=10, mfc="#f33", mec="w")
axs[0].set_title("the scout's landscape (fixed fractal) and its path", color="w")
axs[1].imshow(h.phi, cmap="RdBu_r", origin="lower")
axs[1].plot(W[:, 0], W[:, 1], ".", c="y", ms=2); axs[1].plot(*S[-1], "o", ms=10, mfc="#f33", mec="w")
axs[1].set_title("the wave field; yellow = its centre of mass over time", color="w")
for a in axs: a.axis("off")
fig.suptitle("2 · scout_com_instanton.py: the scout is born. It has mass, velocity, damping and noise, "
             "and climbs a landscape.", color="w", fontsize=12, y=1.03)
plt.tight_layout(); plt.savefig("figs/lineage_2_scout.png", dpi=80, bbox_inches="tight", **DARK); plt.close()

# ---------- 4. cabbage4: its network, scout and patch rule as written (incl. its y sign) ----------
class ComplexLinear(nn.Module):
    def __init__(s, i, o):
        super().__init__(); s.fc_r = nn.Linear(i, o, bias=False); s.fc_i = nn.Linear(i, o, bias=False)
        s.bias_r = nn.Parameter(torch.zeros(o)); s.bias_i = nn.Parameter(torch.zeros(o))
        nn.init.xavier_normal_(s.fc_r.weight, gain=0.2); nn.init.xavier_normal_(s.fc_i.weight, gain=0.2)
    def forward(s, z):
        r, i = z[..., 0], z[..., 1]
        return torch.stack([s.fc_r(r) - s.fc_i(i) + s.bias_r, s.fc_r(i) + s.fc_i(r) + s.bias_i], -1)
class RCNet(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = ComplexLinear(2, 64); s.l2 = ComplexLinear(64, 64)
        s.l3 = ComplexLinear(64, 64); s.out = ComplexLinear(64, 3)
    def forward(s, x):
        z = torch.stack([x * np.pi * 10, x * np.pi * 5], -1)
        z = torch.tanh(s.l1(z)); z = torch.tanh(s.l2(z)); z = torch.tanh(s.l3(z)); o = s.out(z)
        return torch.sqrt(o[..., 0] ** 2 + o[..., 1] ** 2)
torch.manual_seed(0); np.random.seed(0)
img = cv2.resize(data.astronaut(), (256, 256), interpolation=cv2.INTER_AREA) / 255.0
H = W_ = 256; net = RCNet(); opt = optim.Adam(net.parameters(), lr=0.005)
pos, vel = np.zeros(2), np.zeros(2)
def patch(cx, cy, r=0.15, res=32):
    xs, ys = np.meshgrid(np.linspace(cx - r, cx + r, res), np.linspace(cy - r, cy + r, res))
    c = np.stack([xs.ravel(), ys.ravel()], -1)
    px = np.clip(((c[:, 0] + 1) / 2 * (W_ - 1)).astype(int), 0, W_ - 1)
    py = np.clip(((c[:, 1] + 1) / 2 * (H - 1)).astype(int), 0, H - 1)
    t = torch.tensor(img[py, px], dtype=torch.float32); p = net(torch.tensor(c, dtype=torch.float32))
    return ((p - t) ** 2).mean(), p
for step in range(1500):
    with torch.no_grad():
        lc = patch(*pos)[0].item(); lr_ = patch(pos[0] + .05, pos[1])[0].item(); lu = patch(pos[0], pos[1] - .05)[0].item()
    f = np.array([(lr_ - lc) * 20, (lu - lc) * 20]) + np.random.randn(2) * .002
    vel = (vel + f * 0.05) * 0.92; pos = pos + vel
    for i in range(2):
        if pos[i] < -1: pos[i] = -1; vel[i] *= -.8
        if pos[i] > 1: pos[i] = 1; vel[i] *= -.8
    opt.zero_grad(); loss, pred = patch(*pos); loss.backward(); opt.step()
shown = pred.detach().numpy().reshape(32, 32, 3)
g = np.linspace(-1, 1, 128); gx, gy = np.meshgrid(g, g)
with torch.no_grad(): whole = net(torch.tensor(np.stack([gx.ravel(), gy.ravel()], -1), dtype=torch.float32)).numpy().reshape(128, 128, 3)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), **DARK)
axs[0].imshow(img); axs[0].set_title("target", color="w")
sh = cv2.resize(np.clip(shown, 0, 1), (256, 256), interpolation=cv2.INTER_NEAREST); axs[1].imshow(sh)
axs[1].plot(128, 128, "o", ms=12, mfc="r", mec="k"); axs[1].set_title("what cabbage4 showed you:\nthe scout's own 32×32 patch", color="w")
axs[2].imshow(np.clip(whole, 0, 1)); sx, sy = (pos + 1) / 2 * 127; axs[2].plot(sx, sy, "o", ms=10, mfc="r", mec="k")
axs[2].set_title("the whole cortex at the same moment\n(never displayed)", color="w")
for a in axs: a.axis("off")
fig.suptitle("4 · cabbage4.py after 1,500 steps: one scout keeps repainting where it stands.\n"
             "The rest of the picture was never shown, so nobody saw it failing.", color="w", fontsize=12, y=1.06)
plt.tight_layout(); plt.savefig("figs/lineage_4_cabbage4.png", dpi=80, bbox_inches="tight", **DARK); plt.close()
print("done")
