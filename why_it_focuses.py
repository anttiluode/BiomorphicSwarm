"""why_it_focuses.py -- two measurements behind the README.

1. Coarse before fine: error in the blurry (low-frequency) part of the image
   vs the detail (high-frequency) part, over training steps.
2. Rays from the centre: every first-layer unit draws one straight edge
   across the image. With zero bias that edge passes through the centre.
   We track how far the edges have moved away from the centre.
Uses the NASA astronaut photo bundled with scikit-image (public domain)."""
import numpy as np, cv2, torch
from skimage import data
from swarmcabbage import Hive

def bands(img, s=3.0):
    lo = cv2.GaussianBlur(img, (0, 0), s)
    return lo, img - lo

def edge_offsets(h):
    l1 = h.cortex.l1
    Wr, Wi = l1.fc_r.weight.detach().numpy(), l1.fc_i.weight.detach().numpy()
    k = np.pi * h.cortex.feature_scale
    # first layer pre-activations are linear in (x,y):  A.x + b
    A_re, b_re = k * Wr - 0.5 * k * Wi, l1.bias_r.detach().numpy()
    A_im, b_im = 0.5 * k * Wr + k * Wi, l1.bias_i.detach().numpy()
    A = np.vstack([A_re, A_im]); b = np.concatenate([b_re, b_im])
    return np.abs(b) / (np.linalg.norm(A, axis=1) + 1e-9)   # distance of edge line from centre

img = cv2.resize(data.astronaut(), (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
h = Hive(seed=0); h.set_target(img)
t64 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
tlo, thi = bands(t64)
snaps, log = {}, []
keep = [0, 25, 100, 400, 1500, 4000]
for step in range(4001):
    if step in keep: snaps[step] = h.belief(128)
    if step % 100 == 0 or step in (25, 50):
        b = h.belief(64); blo, bhi = bands(b)
        e_lo = ((blo - tlo) ** 2).mean() / (tlo.var() + 1e-9)
        e_hi = ((bhi - thi) ** 2).mean() / ((thi ** 2).mean() + 1e-9)
        d = edge_offsets(h)
        log.append((step, e_lo, e_hi, np.median(d), (d < 1).mean()))
        print(f"step {step:5d}  coarse err {e_lo:.3f}  detail err {e_hi:.3f}"
              f"  median edge distance from centre {np.median(d):.3f}"
              f"  edges inside picture {100*(d<1).mean():.0f}%", flush=True)
    if step < 4000: h.step("random")

# strip of snapshots
P = 160
row = [cv2.resize((img * 255).astype(np.uint8), (P, P))]
for s in keep:
    row.append(cv2.resize((snaps[s] * 255).astype(np.uint8), (P, P), interpolation=cv2.INTER_LINEAR))
gap = np.full((P, 4, 3), 17, np.uint8)
strip = np.hstack(sum([[r, gap] for r in row], [])[:-1])
lab = np.full((24, strip.shape[1], 3), 17, np.uint8)
for i, t in enumerate(["target"] + [f"step {s}" for s in keep]):
    cv2.putText(lab, t, (i * (P + 4) + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
cv2.imwrite("figs/focus_strip.png", cv2.cvtColor(np.vstack([strip, lab]), cv2.COLOR_RGB2BGR))
np.savetxt("figs/focus_log.csv", np.array(log), delimiter=",",
           header="step,coarse_err,detail_err,median_edge_dist,frac_edges_inside", comments="")
