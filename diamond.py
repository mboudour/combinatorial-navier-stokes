import numpy as np, itertools
from cns import SC

def diamonds(K):
    """List (p,q,o,o2) for every edge pq and pair of triangles opq, o2pq."""
    by_edge = {}
    for t in K.T:
        for e in itertools.combinations(t, 2):
            o = [v for v in t if v not in e][0]
            by_edge.setdefault(e, []).append(o)
    out = []
    for (p, q), os in by_edge.items():
        for o, o2 in itertools.combinations(os, 2):
            out.append((p, q, o, o2))
    return out

def val(K, c, a, b):
    e, s = K.edge(a, b); return s * c[e]

def diamond_defect(K, c):
    """max over diamonds of |S_f(c)+S_g(c)|, S_f(c)=c(o->p)+c(o->q)."""
    d = [abs(val(K,c,o,p)+val(K,c,o,q)+val(K,c,o2,p)+val(K,c,o2,q)) for p,q,o,o2 in diamonds(K)]
    return max(d) if d else 0.0

def Qform(K, c):
    """matrix of omega -> <omega, c ∪ delta^T omega> (symmetric part)."""
    Wc = np.einsum('fij,i->fj', K.W, c)
    M = Wc @ K.d1.T
    return (M + M.T) / 2
