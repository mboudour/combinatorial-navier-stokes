import numpy as np
from scipy.linalg import null_space
from exp1 import annulus, torus, lin_op
import exp1
for name, K in [('annulus 8x3', annulus(8,3)), ('torus 5x5', torus(5))]:
    C = null_space(np.vstack([K.d0.T, K.H.T]))
    rng = np.random.default_rng(3)
    for trial in range(3):
        h0 = K.H @ rng.normal(size=K.betti1); h0 /= np.linalg.norm(h0)
        Nh = np.einsum('fij,i->jf', K.W, h0)
        Q = C.T @ K.P @ Nh @ K.d1 @ C            # transport part on coexact space
        S = (Q + Q.T) / 2
        ev = np.linalg.eigvalsh(S)
        out = []
        for s in [1, 10, 100, 1e3, 1e4]:
            out.append(round(np.max(np.linalg.eigvals(lin_op(K, s*h0, 1.0)).real), 4))
        print(name, 'sym part eig range', ev.min().round(4), ev.max().round(4), ' maxRe(L) at s=1..1e4:', out)
