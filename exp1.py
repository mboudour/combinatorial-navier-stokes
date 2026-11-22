import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from cns import SC

def annulus(n=8, rings=2):
    T=[]
    for r in range(rings-1):
        for k in range(n):
            a,b,c,d=r*n+k,r*n+(k+1)%n,(r+1)*n+k,(r+1)*n+(k+1)%n
            T+=[(a,b,c),(b,c,d)]
    return SC(T)
def torus(m=5):
    idx=lambda i,j:(i%m)*m+(j%m); T=[]
    for i in range(m):
        for j in range(m):
            T+=[(idx(i,j),idx(i+1,j),idx(i+1,j+1)),(idx(i,j),idx(i,j+1),idx(i+1,j+1))]
    return SC(T)

def lin_op(K, h, nu):
    """Linearization at harmonic h restricted to coexact (div-free ⊖ harmonic) space."""
    Nh = np.einsum('fij,i->jf', K.W, h)            # omega -> N(h, omega)
    L = K.P @ (Nh @ K.d1 - nu * K.d1.T @ K.d1)
    C = null_space(np.vstack([K.d0.T, K.H.T]))     # orthonormal basis coexact
    return C.T @ L @ C

if __name__ == "__main__":
    for name, K in [('annulus 8x3', annulus(8,3)), ('torus 5x5', torus(5))]:
        print(name, 'V,E,F,b1 =', len(K.V), len(K.E), len(K.T), K.betti1)
        rng = np.random.default_rng(1)
        nu = 0.05
        for trial in range(3):
            c0 = K.P @ rng.normal(size=len(K.E)); c0 /= np.linalg.norm(c0)
            sol = solve_ivp(lambda t, c: K.rhs(c, nu), (0, 4000), c0, rtol=1e-10, atol=1e-12, method='LSODA')
            cT = sol.y[:, -1]
            a0, aT = K.H.T @ c0, K.H.T @ cT
            print(f'  trial {trial}: |delta c(T)|={np.linalg.norm(K.d1@cT):.1e}  harm0={np.round(a0,4)} harmT={np.round(aT,4)}  change={np.linalg.norm(aT-a0):.3e}')
        # stability of harmonic flows h = s*h0
        for j in range(K.betti1):
            h0 = K.H[:, j]
            ss = np.linspace(0, 200, 401)
            mx = [np.max(np.linalg.eigvals(lin_op(K, s*h0, 1.0)).real) for s in ss]
            crit = next((s for s, m in zip(ss, mx) if m > 1e-9), None)
            print(f'  harmonic mode {j}: max Re at s/nu=0: {mx[0]:.3f}; first instability at s/nu ≈ {crit}')
