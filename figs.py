import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import orth
from cns import SC
from exp1 import annulus, torus, lin_op
from sat import irregular, crit
from diamond import diamond_defect, Qform
plt.rcParams.update({"font.size": 9, "font.family": "serif", "lines.linewidth": 1.1})
OUT = {}

# ---- check 1: symmetric enstrophy form = diamond defect / 12 (entrywise) ----
rng = np.random.default_rng(0); errs = []
for seed in range(10):
    K = irregular(seed + 10)
    c = rng.normal(size=len(K.E))
    S = Qform(K, c)
    import itertools
    from diamond import diamonds, val
    pos = {t: k for k, t in enumerate(K.T)}
    pred = np.zeros_like(S)
    for p, q, o, o2 in diamonds(K):
        f, g = pos[tuple(sorted((o, p, q)))], pos[tuple(sorted((o2, p, q)))]
        e, _ = K.edge(p, q)
        sf, sg = K.d1[f, e], K.d1[g, e]
        d = (val(K, c, o, p) + val(K, c, o, q) + val(K, c, o2, p) + val(K, c, o2, q)) / 12.0
        pred[f, g] += sf * sg * d; pred[g, f] += sf * sg * d
    errs.append(np.abs(S - pred).max())
OUT["lemma_entrywise_err"] = float(max(errs))

# ---- check 2: flat complexes: stability at all Re ; irregular: finite Re_c ----
ss = np.geomspace(1, 3000, 120)
curves = {}
for name, K in [("flat torus 5x5", torus(5)), ("flat annulus 8x3", annulus(8, 3))]:
    h0 = K.H[:, 0]
    curves[name] = [np.max(np.linalg.eigvals(lin_op(K, s * h0, 1.0)).real) for s in ss]
    OUT[name] = dict(defect=float(diamond_defect(K, h0)), max_abscissa=float(max(curves[name])))
Rc_list = []
for seed in range(1, 31):
    K = irregular(seed)
    if K.betti1 != 1: continue
    h0 = K.H[:, 0]
    Rc, lam = crit(K, h0)
    Y = orth(K.d1)
    R = Y.T @ K.d1 @ np.einsum('fij,i->fj', K.W, h0).T @ Y
    Rc_list.append(dict(seed=seed, E=len(K.E), F=len(K.T), defect=float(diamond_defect(K, h0)),
                        Rc=None if Rc is None else float(Rc), hopf=None if lam is None else bool(abs(lam.imag) > 1e-6),
                        traceR=float(np.trace(R)), maxReR=float(np.linalg.eigvals(R).real.max())))
    if seed in (1, 3):
        curves[f"irregular complex #{seed}"] = [np.max(np.linalg.eigvals(lin_op(K, s * h0, 1.0)).real) for s in ss]
OUT["irregular"] = Rc_list

# ---- check 3: low-Reynolds theorem and flat-complex threshold ----
K = irregular(1); d_max = max(np.count_nonzero(K.d1[:, e]) for e in range(len(K.E)))
CK = np.sqrt(d_max / 3); mu = min(v for v in np.linalg.eigvalsh(K.d1 @ K.d1.T) if v > 1e-9)
nu = 1.0
c0 = K.P @ rng.normal(size=len(K.E)); c0 *= 0.9 * nu * np.sqrt(mu) / CK / np.linalg.norm(c0)
sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, 60), c0, rtol=1e-10, atol=1e-13, dense_output=True)
ts = np.linspace(0, 60, 200); om = np.array([np.linalg.norm(K.d1 @ sol.sol(t)) for t in ts])
gamma = mu * (nu - CK * np.linalg.norm(c0) / np.sqrt(mu))
OUT["lowRe"] = dict(CK=float(CK), mu=float(mu), max_ratio_to_bound=float(np.max(om / (om[0] * np.exp(-gamma * ts)))))

# ---- experiment: shedding of supercritical circulation ----
K = irregular(3); h0 = K.H[:, 0]; Rc, _ = crit(K, h0); Rcm, _ = crit(K, -h0)
nu = 0.01; traj = {}
for A in [0.5, 2.0, 5.0]:
    c0 = A * Rc * nu * h0 + 1e-3 * (K.P @ np.random.default_rng(0).normal(size=len(K.E)))
    sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, 3e4), c0, rtol=1e-9, atol=1e-12, method='LSODA', dense_output=True)
    tt = np.geomspace(1, 3e4, 300)
    traj[A] = (tt, [(h0 @ sol.sol(t)) / nu for t in tt], [np.linalg.norm(sol.sol(t)) for t in tt])
OUT["shedding"] = dict(Rc_plus=float(Rc), Rc_minus=float(Rcm), final={str(A): float(v[1][-1]) for A, v in traj.items()})
json.dump(OUT, open("summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != "irregular"}, indent=1))
print("irregular: n=", len(Rc_list), " Rc range", min(r['Rc'] for r in Rc_list if r['Rc']), max(r['Rc'] for r in Rc_list if r['Rc']),
      " all finite:", all(r['Rc'] for r in Rc_list), " hopf fraction", np.mean([r['hopf'] for r in Rc_list]),
      " max|traceR|", max(abs(r['traceR']) for r in Rc_list), " min maxReR", min(r['maxReR'] for r in Rc_list))

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
sty = {"flat torus 5x5": ("--", "0.5"), "flat annulus 8x3": (":", "0.5"),
       "irregular complex #1": ("-", "k"), "irregular complex #3": ("-.", "k")}
for name, v in curves.items():
    ax[0].semilogx(ss, v, ls=sty[name][0], color=sty[name][1], label=name)
ax[0].set_yscale("symlog", linthresh=1.0)
ax[0].axhline(0, color="0.8", lw=0.6)
ax[0].set_xlabel(r"$\mathrm{Re}=s/\nu$ (unit harmonic $h_0$)"); ax[0].set_ylabel("spectral abscissa")
ax[0].legend(frameon=False, fontsize=6.5); ax[0].set_title("(a) linear stability of $s h_0$", fontsize=9)
for A, (tt, a, e) in traj.items():
    ax[1].semilogx(tt, a, label=rf"$s_0/\nu={A*Rc:.0f}$")
ax[1].axhline(Rc, color="0.6", ls=":", lw=0.8); ax[1].axhline(-Rcm, color="0.6", ls=":", lw=0.8)
ax[1].set_xlabel("time"); ax[1].set_ylabel(r"$\langle c(t),h_0\rangle/\nu$")
ax[1].legend(frameon=False, fontsize=6.5); ax[1].set_title("(b) shedding of supercritical circulation", fontsize=9)
fig.tight_layout(); fig.savefig("fig_stability.pdf", metadata={"CreationDate": None})
