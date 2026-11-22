import numpy as np
from scipy.spatial import Delaunay
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from cns import SC
from exp1 import lin_op

def irregular(seed, n=50):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1,1,(n,2)); pts = pts[np.linalg.norm(pts,axis=1)>0.35]
    return SC([tuple(int(x) for x in s) for s in Delaunay(pts).simplices if np.linalg.norm(pts[s].mean(0))>0.4])

def crit(K, h0, lo=0.0, hi=5000.0):
    f = lambda s: np.max(np.linalg.eigvals(lin_op(K, s*h0, 1.0)).real)
    ss = np.geomspace(1, hi, 300)
    vals = [f(s) for s in ss]
    k = next((i for i, v in enumerate(vals) if v > 0), None)
    if k is None: return None, None
    a, b = ss[k-1], ss[k]
    for _ in range(60):
        m = (a+b)/2; a, b = (m, b) if f(m) <= 0 else (a, m)
    ev = np.linalg.eigvals(lin_op(K, b*h0, 1.0)); lam = ev[np.argmax(ev.real)]
    return b, lam

if __name__ == "__main__":
    for seed in [1, 2, 3]:
        K = irregular(seed)
        if K.betti1 != 1: continue
        h0 = K.H[:, 0]
        for sgn in (+1, -1):
            Rc, lam = crit(K, sgn*h0)
            print(f"seed {seed} sign {sgn:+d}: Re_c={Rc:.2f}, critical eigenvalue {lam:.3f} (Hopf if imag≠0)")
        nu = 0.01
        Rc, _ = crit(K, h0)
        rng = np.random.default_rng(0)
        for A in [0.5*Rc*nu, 2*Rc*nu, 5*Rc*nu]:
            c0 = A*h0 + 1e-3*(K.P @ rng.normal(size=len(K.E)))
            sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, 3e4), c0, rtol=1e-9, atol=1e-12, method='LSODA', dense_output=False)
            aT = K.H[:,0] @ sol.y[:,-1]
            print(f"   nu={nu}: initial amplitude/nu = {A/nu:7.1f} (Re_c={Rc:.1f}) -> final amplitude/nu = {aT/nu:7.1f}, |delta c(T)|={np.linalg.norm(K.d1@sol.y[:,-1]):.1e}")
