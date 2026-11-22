import numpy as np
from scipy.integrate import solve_ivp
from exp1 import torus
rng=np.random.default_rng(1)
K=torus(5); dK=max(np.count_nonzero(K.d1[:,e]) for e in range(len(K.E))); CK=np.sqrt(dK/3)
m=min(v for v in np.linalg.eigvalsh(K.d1@K.d1.T) if v>1e-9); nu=0.01
# pointwise test of the differential inequality (no integration error)
worst=-np.inf
for _ in range(2000):
    v=K.P@rng.normal(size=len(K.E)); v-=K.PH@v
    v*=rng.uniform(0.01,0.99)*nu*m/CK/np.linalg.norm(K.d1@v)
    h=K.H@rng.normal(size=K.betti1)*10**rng.uniform(0,4)
    c=h+v; w=K.d1@c; dw=K.d1@K.rhs(c,nu)
    lhs=w@dw; rhs=-(nu-CK*np.linalg.norm(w)/m)*np.linalg.norm(K.d1.T@w)**2
    worst=max(worst,lhs-rhs)
print('max(lhs-rhs) pointwise:',worst)
# trajectory with moderate circulation, tight tolerance
for amp in (10,100,1000):
    v=K.P@rng.normal(size=len(K.E)); v-=K.PH@v; v*=0.5*nu*m/CK/np.linalg.norm(K.d1@v)
    c0=amp*K.H[:,0]+v; w0=np.linalg.norm(K.d1@c0); rate=m*(nu-CK*w0/m)
    T=300
    sol=solve_ivp(lambda t,c:K.rhs(c,nu),(0,T),c0,method='DOP853',rtol=1e-13,atol=1e-15,dense_output=True)
    ts=np.linspace(0,T,300); om=np.array([np.linalg.norm(K.d1@sol.sol(t)) for t in ts])
    r=om/(w0*np.exp(-rate*ts)); print(amp, 'max ratio',r.max(),'min om',om.min(),'nonmono steps',np.sum(np.diff(om)>1e-14*om[:-1].max()))
