"""Supercritical ensemble: harmonic equilibrium s0*h0 plus a coexact perturbation of norm 1e-3.
usage: python shed.py SEED MULT NPERT [RTOL] [METHOD]  -> appends to shed_results.jsonl;
each record carries run_id = seed/mult/pert/rtol/method, and shed_summary.py keeps one record per run_id."""
import sys, json, time, numpy as np
from scipy.integrate import solve_ivp
from sat import irregular
from study import ops, absc, scan

def run(seed, mult, npert, nu=0.01, T=3e4, rtol=1e-9, atol=1e-12, keep=False, method='LSODA'):
    K = irregular(seed); h = K.h0(); B, G = ops(K, h)
    Rc = scan(B, G)['Rc']; s0 = mult * Rc * nu
    PC = K.P - K.PH
    out = []
    for j in range(npert):
        g = PC @ np.random.default_rng(100 + j).normal(size=len(K.E)); p = 1e-3 * g / np.linalg.norm(g)
        c0 = s0 * h + p; t0 = time.time()
        sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, T), c0, method=method, rtol=rtol, atol=atol, dense_output=keep)
        cT = sol.y[:, -1]; aT = float(h @ cT)
        a = h @ sol.y; i5 = np.flatnonzero(np.abs(a - s0) > 0.05 * abs(s0))
        r = dict(run_id=f'{seed}/{mult}/{j}/{rtol:g}/{method}', seed=seed, mult=mult, pert=j, Rc=Rc, s0_over_nu=s0 / nu, final_over_nu=aT / nu,
                 final_abscissa=absc(np.sign(aT) * B, G, abs(aT) / nu), final_vorticity=float(np.linalg.norm(K.d1 @ cT)),
                 t_depart=float(sol.t[i5[0]]) if len(i5) else None, success=bool(sol.success), rtol=rtol, method=method, secs=time.time() - t0)
        out.append((r, sol) if keep else r)
    return out

if __name__ == "__main__":
    seed, mult, n = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    rt = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-9
    meth = sys.argv[5] if len(sys.argv) > 5 else 'LSODA'
    for r in run(seed, mult, n, rtol=rt, atol=rt * 1e-3, method=meth):
        print(json.dumps(r)); open('shed_results.jsonl', 'a').write(json.dumps(r) + '\n')
