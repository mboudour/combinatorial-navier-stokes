"""Long continuation runs on an irregular complex.
usage: python longrun.py SEED [RTOL]
Starts from the same initial condition as shed.py (perturbation 0, s0/nu = 2 Re_c(h0)), integrates
with DOP853 in segments of 1e4 up to T=1e6, and records for every segment the solver status,
message and actual end time. A run is called settled from the first segment end at which the
vorticity norm is below 1e-6. Writes long{SEED}_rtol{RTOL}.json."""
import sys, json, numpy as np
from scipy.integrate import solve_ivp
from sat import irregular
from study import ops, absc, scan
SEED = int(sys.argv[1]); RT = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-9
K = irregular(SEED); h = K.h0(); B, G = ops(K, h); Rc = scan(B, G)['Rc']; nu = 0.01
PC = K.P - K.PH; g = PC @ np.random.default_rng(100).normal(size=len(K.E))
c = 2 * Rc * nu * h + 1e-3 * g / np.linalg.norm(g)
out = []; t = 0.0; settled = None
for k in range(100):
    sol = solve_ivp(lambda t, c: K.rhs(c, nu), (t, t + 1e4), c, method='DOP853', rtol=RT, atol=RT * 1e-3)
    c = sol.y[:, -1]; t = float(sol.t[-1]); w = float(np.linalg.norm(K.d1 @ c)); a = float(h @ c)
    out.append(dict(t=t, success=bool(sol.success), message=sol.message, a_over_nu=a / nu, vort=w,
                    absc=absc(np.sign(a) * B, G, abs(a) / nu)))
    if settled is None and w < 1e-6: settled = t
    if not sol.success: break
json.dump(dict(seed=SEED, rtol=RT, method='DOP853', Rc=Rc, s0_over_nu=2 * Rc, settle_criterion='vorticity < 1e-6',
               settled_at=settled, all_success=all(o['success'] for o in out), t_end=t, traj=out),
          open(f'long{SEED}_rtol{RT:g}.json', 'w'), indent=0)
print(SEED, RT, 'settled_at', settled, 'final a/nu', out[-1]['a_over_nu'], 'vort', out[-1]['vort'], 'absc', out[-1]['absc'], 'ok', all(o['success'] for o in out), 't_end', t)
