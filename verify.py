import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import orth, null_space
from cns import SC
from exp1 import annulus, torus, lin_op
from sat import irregular
from diamond import diamond_defect
rng = np.random.default_rng(42); R = {}
def dK(K): return max(np.count_nonzero(K.d1[:, e]) for e in range(len(K.E)))
def mu(K): return min(v for v in np.linalg.eigvalsh(K.d1 @ K.d1.T) if v > 1e-9)
# Lemma 1 bound
worst = 0
for s in range(10):
    K = irregular(s); CK = np.sqrt(dK(K)/3)
    for _ in range(50):
        a, b, w = rng.normal(size=len(K.E)), rng.normal(size=len(K.E)), rng.normal(size=len(K.T))
        worst = max(worst, abs(w @ K.cup(a, b)) / (CK * np.abs(a).max() * np.linalg.norm(w) * np.linalg.norm(b)))
R['lemma_bound_max_ratio'] = worst
# Example: regular tori flat for m=3,4,6
R['torus_flat_defects'] = {m: float(max(diamond_defect(torus(m), torus(m).H[:, j]) for j in range(torus(m).betti1))) for m in (3, 4, 6)}
# Lemma 3: spectrum of L_h on V = {0}^b1 ∪ spec(A_h)
K = irregular(3); h = 50 * K.H[:, 0]; nu = 1.0
Vb = null_space(K.d0.T); L = Vb.T @ (K.P @ (np.einsum('fij,i->jf', K.W, h) @ K.d1 - nu * K.d1.T @ K.d1)) @ Vb
Y = orth(K.d1); A = Y.T @ (K.d1 @ np.einsum('fij,i->jf', K.W, h) - nu * K.d1 @ K.d1.T) @ Y
eL = np.sort_complex(np.linalg.eigvals(L)); eA = np.sort_complex(np.r_[np.linalg.eigvals(A), np.zeros(K.betti1)])
R['lemma_spectrum_err'] = float(np.abs(eL - eA).max())
# Theorem 4(a) bound
hh = K.H[:, 0]; B = Y.T @ K.d1 @ np.einsum('fij,i->jf', K.W, hh) @ Y; G = Y.T @ K.d1 @ K.d1.T @ Y
bound = mu(K) / np.linalg.norm((B + B.T) / 2, 2)
from sat import crit
Rc, _ = crit(K, hh)
R['thm4a'] = dict(bound=float(bound), Rc=float(Rc), ok=bool(bound <= Rc + 1e-9))
# Theorem 1 exchange bound
CK = np.sqrt(dK(K)/3); m = mu(K)
c0 = K.P @ rng.normal(size=len(K.E)); c0 *= 0.8 * nu * np.sqrt(m) / CK / np.linalg.norm(c0)
sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, 400), c0, rtol=1e-11, atol=1e-14)
hinf = K.PH @ sol.y[:, -1]; gamma = nu - CK * np.linalg.norm(c0) / np.sqrt(m)
R['thm1_exchange'] = dict(actual=float(np.linalg.norm(hinf - K.PH @ c0)), bound=float(CK * np.linalg.norm(c0) * np.linalg.norm(K.d1 @ c0) / (m * gamma)))
# Theorem 3 is checked in verify_flat.py
print(R)
