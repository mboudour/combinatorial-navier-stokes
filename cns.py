"""Combinatorial Navier-Stokes (Wilson-type) on a simplicial 2-complex.
 velocity c: 1-cochain, div-free (delta* c = 0);  dc/dt = P T(c),
 <T(c), b> = <delta c, c ∪ b> - nu <delta c, delta b>,  ∪ = antisymmetrized (DEC/Whitney) cup product.
"""
import itertools, numpy as np
from scipy.linalg import null_space, orth

class SC:
    def __init__(self, tris):
        tris = [tuple(sorted(t)) for t in tris]
        self.T = sorted(set(tris))
        E = set()
        for a, b, c in self.T:
            E |= {(a, b), (a, c), (b, c)}
        self.E = sorted(E); self.V = sorted({v for e in self.E for v in e})
        self.ei = {e: k for k, e in enumerate(self.E)}; self.vi = {v: k for k, v in enumerate(self.V)}
        nV, nE, nF = len(self.V), len(self.E), len(self.T)
        self.d0 = np.zeros((nE, nV)); self.d1 = np.zeros((nF, nE))
        for k, (a, b) in enumerate(self.E):
            self.d0[k, self.vi[a]] = -1; self.d0[k, self.vi[b]] = 1
        for k, (a, b, c) in enumerate(self.T):
            self.d1[k, self.ei[(b, c)]] += 1; self.d1[k, self.ei[(a, c)]] -= 1; self.d1[k, self.ei[(a, b)]] += 1
        assert np.allclose(self.d1 @ self.d0, 0)
        # cup tensor W[f, e1, e2] : (α∪β)(f) = sum W α_e1 β_e2   (antisymmetrized, graded commutative)
        self.W = np.zeros((nF, nE, nE))
        for k, t in enumerate(self.T):
            for p in itertools.permutations(range(3)):
                sgn = np.linalg.det(np.eye(3)[list(p)])
                v = [t[i] for i in p]
                e1, s1 = self.edge(v[0], v[1]); e2, s2 = self.edge(v[1], v[2])
                self.W[k, e1, e2] += sgn * s1 * s2 / 6.0
        Z = null_space(self.d0.T)                     # div-free 1-cochains
        self.P = Z @ Z.T
        H = null_space(np.vstack([self.d0.T, self.d1]))
        self.H = H; self.PH = H @ H.T
        self.betti1 = H.shape[1]
    def h0(self, j=0):
        """Unit harmonic cochain H[:, j], oriented so that its first entry (edges in
        lexicographic order) with |value| > 1e-8 is positive."""
        h = self.H[:, j].copy()
        k = np.flatnonzero(np.abs(h) > 1e-8)[0]
        return h if h[k] > 0 else -h
    def edge(self, a, b):
        return (self.ei[(a, b)], 1) if (a, b) in self.ei else (self.ei[(b, a)], -1)
    def cup(self, a, b):
        return np.einsum('fij,i,j->f', self.W, a, b)
    def rhs(self, c, nu, f=None):
        w = self.d1 @ c                                   # vorticity (2-cochain)
        N = np.einsum('fij,f,i->j', self.W, w, c)         # b -> <w, c ∪ b>
        T = N - nu * self.d1.T @ w
        if f is not None: T = T + f
        return self.P @ T
