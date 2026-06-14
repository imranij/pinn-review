"""
Reproducible Euler-Bernoulli PINN benchmark for
"A Topical Review of Machine Learning and Physics-Informed Neural Networks
 for Beam Analysis and Control".

Nondimensional EBB free vibration:
    w_xxxx + w_tt = 0,    x in [0, pi],  t in [0, 1]
Simply supported ends:  w(0,t)=w(pi,t)=0,  w_xx(0,t)=w_xx(pi,t)=0
Initial conditions:     w(x,0)=sin(x),  w_t(x,0)=0
Exact solution:         w(x,t)=sin(x) cos(t)

Hard Dirichlet BC via the trial space  w = x*(pi-x)*N(x,t).
Optimizer schedule: Adam warm-up -> L-BFGS refinement (the schedule
discussed in Section 3.5 of the paper).

Outputs the relative space-time L2 error R and wall-clock training and
inference times used to populate Tables 14-16. Run:

    python ebb_pinn.py --seed 0

Requires: torch>=2.0, numpy. CPU is sufficient; a GPU is used if available.
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn

PI = math.pi


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)


class MLP(nn.Module):
    def __init__(self, width=200, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # inputs already mapped to [-1, 1]
        xt = torch.cat([x, t], dim=1)
        n = self.net(xt)
        # undo the input scaling on x to apply the physical hard-BC factor
        xp = (x + 1.0) * 0.5 * PI          # x in [0, pi]
        return xp * (PI - xp) * n          # hard w=0 at x=0, pi


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def pde_residual(model, x, t):
    w = model(x, t)
    w_x = grad(w, x)
    w_xx = grad(w_x, x)
    w_xxx = grad(w_xx, x)
    w_xxxx = grad(w_xxx, x)
    w_t = grad(w, t)
    w_tt = grad(w_t, t)
    # chain rule for x-scaling (x_phys = (x+1)/2*pi): d/dx_phys = (2/pi) d/dx
    s = 2.0 / PI
    return (s ** 4) * w_xxxx + w_tt


def exact(xp, tp):
    return np.sin(xp) * np.cos(tp)


def sample(n, device):
    x = (torch.rand(n, 1, device=device) * 2 - 1).requires_grad_(True)
    t = (torch.rand(n, 1, device=device) * 2 - 1).requires_grad_(True)
    return x, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam_iters", type=int, default=6000)
    ap.add_argument("--lbfgs_iters", type=int, default=3000)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLP(args.width, args.depth).to(device)

    # collocation (interior) and IC points; t in [-1,1] maps to [0,1]
    Nn, Nic = 10000, 500
    xc, tc = sample(Nn, device)
    xic = (torch.rand(Nic, 1, device=device) * 2 - 1).requires_grad_(True)
    tic = (-torch.ones(Nic, 1, device=device)).requires_grad_(True)  # t=0
    xic_phys = (xic + 1) * 0.5 * PI
    w0 = torch.sin(xic_phys)

    def loss_fn():
        r = pde_residual(model, xc, tc)
        l_pde = (r ** 2).mean()
        w_ic = model(xic, tic)
        wt_ic = grad(w_ic, tic) * 2.0  # t-scaling: d/dt_phys=(2/1)d/dt
        l_ic = ((w_ic - w0) ** 2).mean() + (wt_ic ** 2).mean()
        return l_pde + 100.0 * l_ic    # IC up-weighted (see Table 16)

    # ---- Adam warm-up ----
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for it in range(args.adam_iters):
        opt.zero_grad(); L = loss_fn(); L.backward(); opt.step()
    t_adam = time.time() - t0

    # ---- L-BFGS refinement ----
    opt2 = torch.optim.LBFGS(model.parameters(), max_iter=args.lbfgs_iters,
                             tolerance_grad=1e-12, tolerance_change=1e-14,
                             history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad(); L = loss_fn(); L.backward(); return L

    t0 = time.time(); opt2.step(closure); t_lbfgs = time.time() - t0

    # ---- evaluation on a dense unseen grid ----
    nx, nt = 256, 256
    xs = np.linspace(0, PI, nx); ts = np.linspace(0, 1, nt)
    XX, TT = np.meshgrid(xs, ts)
    Xn = torch.tensor((XX.ravel() / PI * 2 - 1)[:, None], dtype=torch.float32,
                      device=device, requires_grad=False)
    Tn = torch.tensor((TT.ravel() * 2 - 1)[:, None], dtype=torch.float32,
                      device=device, requires_grad=False)
    t0 = time.time()
    with torch.no_grad():
        wp = model(Xn, Tn).cpu().numpy().ravel()
    t_infer = time.time() - t0
    we = exact(XX.ravel(), TT.ravel())
    R = np.linalg.norm(wp - we) / np.linalg.norm(we) * 100

    print(f"seed={args.seed}  arch={args.depth}x{args.width}")
    print(f"relative L2 error R = {R:.3e} %")
    print(f"train: Adam {t_adam:.1f}s + L-BFGS {t_lbfgs:.1f}s "
          f"= {t_adam + t_lbfgs:.1f}s")
    print(f"inference (256x256 grid): {t_infer*1000:.1f} ms")


if __name__ == "__main__":
    main()
