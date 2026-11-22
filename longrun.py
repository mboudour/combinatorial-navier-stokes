"""Long run on an irregular complex (argument: seed) at s0/nu = 2 Re_c(h0): harmonic coefficient and vorticity up to T=1e6."""
import json, numpy as np
from scipy.integrate import solve_ivp
from sat import irregular
from study import ops, absc, scan
import sys; SEED = int(sys.argv[1]); K = irregular(SEED); h = K.h0(); B, G = ops(K, h); Rc = scan(B, G)['Rc']; nu = 0.01
PC = K.P - K.PH; g = PC @ np.random.default_rng(100).normal(size=len(K.E)); c = 2 * Rc * nu * h + 1e-3 * g / np.linalg.norm(g)
out = []; t = 0.0
for k in range(100):
    sol = solve_ivp(lambda t, c: K.rhs(c, nu), (t, t + 1e4), c, method='DOP853', rtol=1e-9, atol=1e-12)
    c = sol.y[:, -1]; t += 1e4
    out.append(dict(t=t, a_over_nu=float(h @ c / nu), vort=float(np.linalg.norm(K.d1 @ c)), absc=absc(np.sign(h @ c) * B, G, abs(h @ c) / nu)))
    json.dump(dict(Rc=Rc, s0_over_nu=2 * Rc, traj=out), open(f'long{SEED}.json', 'w'), indent=0)
