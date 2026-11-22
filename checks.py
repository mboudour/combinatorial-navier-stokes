"""Identity checks: diamond formula (entrywise), flat-complex spectra, low-Re bound.
Writes summary.json."""
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
# first critical values with the oriented basis are computed in study.py

# ---- check 3: low-Reynolds theorem and flat-complex threshold ----
K = irregular(1); d_max = max(np.count_nonzero(K.d1[:, e]) for e in range(len(K.E)))
CK = np.sqrt(d_max / 3); mu = min(v for v in np.linalg.eigvalsh(K.d1 @ K.d1.T) if v > 1e-9)
nu = 1.0
c0 = K.P @ rng.normal(size=len(K.E)); c0 *= 0.9 * nu * np.sqrt(mu) / CK / np.linalg.norm(c0)
sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, 60), c0, rtol=1e-10, atol=1e-13, dense_output=True)
ts = np.linspace(0, 60, 200); om = np.array([np.linalg.norm(K.d1 @ sol.sol(t)) for t in ts])
gamma = mu * (nu - CK * np.linalg.norm(c0) / np.sqrt(mu))
OUT["lowRe"] = dict(CK=float(CK), mu=float(mu), max_ratio_to_bound=float(np.max(om / (om[0] * np.exp(-gamma * ts)))))

json.dump(OUT, open("summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != "irregular"}, indent=1))
