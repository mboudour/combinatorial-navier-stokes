"""Figures for the revised manuscript: fig_stability.pdf (3 panels) and fig_complex.pdf."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from cns import SC
from exp1 import annulus, torus
from sat import irregular
from study import ops, absc, scan
from shed import run
from diamond import diamonds, val
plt.rcParams.update({"font.size": 10, "font.family": "serif", "lines.linewidth": 1.2,
                     "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9})
rho = np.geomspace(1, 3000, 400)
cases = [("flat torus 5x5", torus(5), ("--", "0.55")), ("flat annulus 8x3", annulus(8, 3), (":", "0.55")),
         ("4-triangle complex (Ex. 2)", SC([(0,1,2),(0,1,3),(0,2,4),(1,3,4)]), ((0,(1,1,3,1)), "0.3")),
         ("irregular, seed 3", irregular(3), ("-", "k")), ("irregular, seed 13", irregular(13), ("-.", "k"))]
fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.4)); ax = [axs[0, 0], axs[1, 0], axs[1, 1], axs[0, 1]]
for name, K, (ls, col) in cases:
    B, G = ops(K, K.h0())
    ax[0].semilogx(rho, [absc(B, G, r) for r in rho], ls=ls, color=col, label=name)
ax[0].set_yscale("symlog", linthresh=1.0); ax[0].axhline(0, color="0.8", lw=0.6)
ax[0].set_xlabel(r"$\rho=s/\nu$"); ax[0].set_ylabel(r"spectral abscissa of $\rho B_{h_0}-G$")
ax[0].legend(frameon=False, loc="upper left"); ax[0].set_title(r"(a) transversal spectrum ($\nu=1$)", fontsize=10)
K13 = irregular(13); B13, G13 = ops(K13, K13.h0()); rz = np.linspace(150, 700, 500)
ax[3].plot(rz, [absc(B13, G13, r) for r in rz], "k-."); ax[3].axhline(0, color="0.6", lw=0.6)
ax[3].set_xlabel(r"$\rho=s/\nu$"); ax[3].set_ylabel("spectral abscissa"); ax[3].set_ylim(-0.1, 0.25)
ax[3].set_title(r"(b) irregular complex 13: window of stability", fontsize=10)
res = {m: run(3, m, 1, keep=True)[0] for m in (0.5, 2.0, 5.0)}
K = irregular(3); h = K.h0(); B, G = ops(K, h); Rc = scan(B, G)['Rc']; Rcm = scan(-B, G)['Rc']
tt = np.geomspace(1, 3e4, 500)
for m, (r, sol) in res.items():
    ax[1].semilogx(tt, [h @ sol.sol(t) / 0.01 for t in tt], label=rf"$s_0/\nu={r['s0_over_nu']:.0f}$")
    ax[2].loglog(tt, [max(np.linalg.norm(K.d1 @ sol.sol(t)), 1e-18) for t in tt], label=rf"$s_0/\nu={r['s0_over_nu']:.0f}$")
for y in (Rc, -Rcm): ax[1].axhline(y, color="0.6", ls=":", lw=0.8)
ax[1].set_xlabel("time"); ax[1].set_ylabel(r"$\langle c(t),h_0\rangle/\nu$"); ax[1].legend(frameon=False)
ax[1].set_title(r"(c) harmonic component, complex 3, $\nu=0.01$", fontsize=10)
ax[2].set_xlabel("time"); ax[2].set_ylabel(r"$\|\delta c(t)\|$"); ax[2].set_ylim(1e-12, 1e1); ax[2].legend(frameon=False)
ax[2].set_title("(d) vorticity, complex 3", fontsize=10)
fig.tight_layout(); fig.savefig("fig_stability.pdf", metadata={"CreationDate": None})
print("Rc(h0)=", Rc, " Rc(-h0)=", Rcm, {m: (r['final_over_nu'], r['final_abscissa']) for m, (r, s) in res.items()})

# Figure 2: the complex, its hole and the diamond defect of h0 on interior edges
K = irregular(3); h = K.h0(); X = K.pts
fig, ax = plt.subplots(figsize=(4.4, 4.2))
ax.add_collection(PolyCollection([X[list(t)] for t in K.T], facecolor="0.93", edgecolor="0.75", lw=0.4))
segs, vals = [], []
for p, q, o, o2 in diamonds(K):
    segs.append(X[[p, q]]); vals.append(abs(val(K, h, o, p) + val(K, h, o, q) + val(K, h, o2, p) + val(K, h, o2, q)))
lc = LineCollection(segs, array=np.array(vals), cmap="viridis", lw=2.0); ax.add_collection(lc)
cb = fig.colorbar(lc, ax=ax, shrink=0.8); cb.set_label(r"$|D^e_{f,g}(h_0)|$")
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig("fig_complex.pdf", metadata={"CreationDate": None})
print("defect range on diamonds:", min(vals), max(vals), "n diamonds", len(vals))
