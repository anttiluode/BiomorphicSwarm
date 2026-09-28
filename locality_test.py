"""locality_test.py -- does attention pay once writes are local?

Same Hive, same scouts, same 50 patches/step. Only the memory changes:
a 16x16 grid of Gaussian bumps with learnable colour. sigma = bump width.
Small sigma: a write changes only nearby colour (local). Large sigma: every write
moves the whole picture (global, like the neural field).
"""
import sys, numpy as np, torch, torch.nn as nn, torch.optim as optim
from swarmcabbage import Hive, add_surprise

class BumpField(nn.Module):
    def __init__(self, sigma, n=16):
        super().__init__()
        g = torch.linspace(-1, 1, n); cy, cx = torch.meshgrid(g, g, indexing="ij")
        self.c = torch.stack([cx.ravel(), cy.ravel()], -1)
        self.w = nn.Parameter(torch.full((n * n, 3), 0.3))
        self.s = sigma
    def forward(self, x):
        k = torch.exp(-torch.cdist(x, self.c) ** 2 / (2 * self.s ** 2))
        return (k @ self.w) / (k.sum(-1, keepdim=True) + 1e-6)

def run(sigma, mode, seed, warm=300, after=200):
    h = Hive(seed=seed)
    h.cortex = BumpField(sigma); h.opt = optim.Adam(h.cortex.parameters(), lr=0.02)
    for _ in range(warm): h.step(mode)
    learned = h.full_error()
    new, mask = add_surprise(h.target, np.random.default_rng(100 + seed)); h.set_target(new)
    early = None
    for t in range(after):
        h.step(mode)
        if t == 49: early = h.full_error(mask=mask)[1]
    tot, inside, _ = h.full_error(mask=mask)
    return learned, early, inside, tot

def main():
  sigmas = [float(s) for s in sys.argv[1:]] or [0.06, 0.15, 0.4, 1.0]
  print("sigma  mode    A-learned  surprise@50  surprise@200  whole@200")
  for s in sigmas:
    for mode in ("swarm", "random"):
        r = np.array([run(s, mode, seed) for seed in (0, 1)]).mean(0)
        print(f"{s:5.2f}  {mode:6s}  {r[0]:.4f}     {r[1]:.4f}       {r[2]:.4f}        {r[3]:.4f}", flush=True)

if __name__ == "__main__":
    main()
