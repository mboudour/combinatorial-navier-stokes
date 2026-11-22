"""Checks of the results added in revision: Proposition 3 bound, Theorem 4(a) with sigma_+,
Theorem 4(d) bound on Example 2, Lemma 3 with H=0, and Corollary 2 for H=0."""
import numpy as np, json
from scipy.integrate import solve_ivp
from cns import SC
from sat import irregular
from study import ops, absc, scan
R = {}
# Proposition 3 on irregular complexes with large data
rng = np.random.default_rng(11); worst = 0
for seed in (2, 5, 8):
    K = irregular(seed); h = K.h0(); nu = 0.05
    dK = max(np.count_nonzero(K.d1[:, e]) for e in range(len(K.E))); CK = np.sqrt(dK / 3)
    mu = min(v for v in np.linalg.eigvalsh(K.d1 @ K.d1.T) if v > 1e-9)
    c0 = K.P @ rng.normal(size=len(K.E)); c0 *= 3 / np.linalg.norm(c0)
    sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, 4000), c0, method='LSODA', rtol=1e-10, atol=1e-13)
    worst = max(worst, abs(h @ sol.y[:, -1] - h @ c0) / (CK * 9 / (2 * nu * np.sqrt(mu))))
R['prop3_max_ratio_to_bound'] = worst
# Theorem 4(a) with sigma_+
ok = True
for seed in range(1, 31):
    K = irregular(seed); B, G = ops(K, K.h0()); S = (B + B.T) / 2; sp = np.linalg.eigvalsh(S).max()
    mu = np.linalg.eigvalsh(G).min()
    for sg in (1, -1):
        Rc = scan(sg * B, G)['Rc']; sps = np.linalg.eigvalsh(sg * S).max()
        ok &= Rc >= mu / sps - 1e-9
R['thm4a_sigma_plus_ok'] = bool(ok)
# Theorem 4(d) on Example 2: bound -kappa vs actual
K = SC([(0, 1, 2), (0, 1, 3), (0, 2, 4), (1, 3, 4)]); h = K.h0()
B = K.d1 @ np.einsum('fij,i->jf', K.W, h); G = K.d1 @ K.d1.T; M = np.diag([7, 7, 11, 11.])
kappa = np.linalg.eigvalsh(M @ G + G @ M).min() / (2 * 11)
R['ex2_kappa'] = float(kappa)
R['ex2_max_abscissa'] = max(absc(B, G, r) for r in np.r_[-np.geomspace(1e-3, 1e6, 500), np.geomspace(1e-3, 1e6, 500)])
R['ex2_bound_holds'] = R['ex2_max_abscissa'] <= -kappa + 1e-12
# Lemma 3 and Corollary 2 with H = 0: a single triangle and a disc
K = SC([(0, 1, 2)]); L = K.P @ (-K.d1.T @ K.d1)
from scipy.linalg import null_space
Vb = null_space(K.d0.T); R['single_triangle_spec_L'] = np.linalg.eigvals(Vb.T @ L @ Vb).round(10).real.tolist()
K = SC([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)]); nu = 0.1; mu = min(v for v in np.linalg.eigvalsh(K.d1 @ K.d1.T) if v > 1e-9)
c0 = K.P @ rng.normal(size=len(K.E)) * 5
sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, 50), c0, rtol=1e-11, atol=1e-14, dense_output=True)
ts = np.linspace(0, 50, 100)
R['disc_H0_max_ratio'] = float(max(np.linalg.norm(sol.sol(t)) / (np.exp(-nu * mu * t) * np.linalg.norm(c0)) for t in ts))
print(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

# Section 4: on a complex with H^2 != 0, the form can vanish on im(delta) with a nonzero diamond defect
import itertools
from diamond import Qform, diamonds, val
K = SC([t for t in itertools.combinations(range(5), 3) if t != (2, 3, 4)])
a = np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 1.])
R2 = dict(dim_H2=int(len(K.T) - np.linalg.matrix_rank(K.d1)), div=float(np.abs(K.d0.T @ a).max()),
          form_on_im_delta=float(np.abs(K.d1.T @ Qform(K, a) @ K.d1).max()),
          defect_02_012_023=float(val(K, a, 1, 0) + val(K, a, 1, 2) + val(K, a, 3, 0) + val(K, a, 3, 2)))
print(R2)
