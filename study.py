"""Revision computations: thresholds for both orientations, re-entry, mesh table,
counterexample, energy residual.  Writes study.json and table_irregular.csv."""
import json, csv, numpy as np
from scipy.linalg import orth
from scipy.stats import spearmanr
from cns import SC
from sat import irregular
from diamond import diamond_defect, diamonds

def ops(K, h):
    Y = orth(K.d1); Nh = np.einsum('fij,i->jf', K.W, h)
    return Y.T @ K.d1 @ Nh @ Y, Y.T @ K.d1 @ K.d1.T @ Y
def absc(B, G, r): return float(np.max(np.linalg.eigvals(r * B - G).real))
RHO = np.geomspace(1, 5000, 1500)
def scan(B, G):
    """first crossing (bisection), type of critical eigenvalue, number of
    unstable->stable returns on the grid, and abscissa at the end of the grid"""
    v = np.array([absc(B, G, r) for r in RHO]); pos = v >= 0
    if not pos.any(): return dict(Rc=None, osc=None, returns=0, end=float(v[-1]))
    k = int(np.argmax(pos)); a, b = (RHO[k-1] if k else 0.0), RHO[k]
    for _ in range(60):
        m = (a + b) / 2; a, b = (m, b) if absc(B, G, m) < 0 else (a, m)
    ev = np.linalg.eigvals(b * B - G); lam = ev[np.argmax(ev.real)]
    returns = int(np.sum(pos[:-1] & ~pos[1:]))
    return dict(Rc=float(b), osc=bool(abs(lam.imag) > 1e-6), freq=float(abs(lam.imag)), returns=returns, end=float(v[-1]))

def main():
  OUT = {}; rows = []
  for seed in range(1, 31):
      K = irregular(seed); h = K.h0()
      B, G = ops(K, h)
      mu = float(min(np.linalg.eigvalsh(G)))
      dK = max(np.count_nonzero(K.d1[:, e]) for e in range(len(K.E)))
      two_tri = all(np.count_nonzero(K.d1[:, e]) <= 2 for e in range(len(K.E)))
      row = dict(seed=seed, V=len(K.V), E=len(K.E), F=len(K.T), dim_im_delta=int(G.shape[0]),
                 betti1=K.betti1, mu=mu, edges_in_le2_triangles=two_tri,
                 defect=float(diamond_defect(K, h)), maxReB=float(np.linalg.eigvals(B).real.max()))
      for sgn, tag in ((1, 'p'), (-1, 'm')):
          r = scan(sgn * B, G)
          for k2, v in r.items(): row[f'{k2}_{tag}'] = v
      rows.append(row)
  with open('table_irregular.csv', 'w', newline='') as f:
      w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
  Rcs = [r[f'Rc_{t}'] for r in rows for t in 'pm']
  osc = [r[f'osc_{t}'] for r in rows for t in 'pm']
  ret = [r[f'returns_{t}'] for r in rows for t in 'pm']
  Rmin = [min(r['Rc_p'], r['Rc_m']) for r in rows]
  OUT['irregular'] = dict(n_complexes=len(rows), n_directions=len(Rcs), all_finite=all(x is not None for x in Rcs),
      Rc_min=min(Rcs), Rc_max=max(Rcs), n_osc=int(sum(osc)), n_with_return=int(sum(x > 0 for x in ret)),
      all_end_unstable=all(r[f'end_{t}'] > 0 for r in rows for t in 'pm'),
      V_range=[min(r['V'] for r in rows), max(r['V'] for r in rows)], E_range=[min(r['E'] for r in rows), max(r['E'] for r in rows)],
      F_range=[min(r['F'] for r in rows), max(r['F'] for r in rows)], all_betti1=all(r['betti1'] == 1 for r in rows),
      all_manifold_edges=all(r['edges_in_le2_triangles'] for r in rows),
      spearman_Rcmin_defect=list(map(float, spearmanr(Rmin, [r['defect'] for r in rows]))),
      spearman_Rcmin_mu=list(map(float, spearmanr(Rmin, [r['mu'] for r in rows]))),
      spearman_Rcmin_maxReB=list(map(float, spearmanr(Rmin, [r['maxReB'] for r in rows]))))

  # re-entry example: seed 13, negative orientation
  K = irregular(13); B, G = ops(K, -K.h0())
  OUT['reentry_seed13_minus'] = {str(r): absc(B, G, r) for r in (300, 450, 500, 600)}

  # counterexample (four triangles)
  K = SC([(0, 1, 2), (0, 1, 3), (0, 2, 4), (1, 3, 4)])
  h = K.h0(); hr = np.array([-3, 1, -4, 6, 4, -1, -6, 5, -5.]); hr /= np.linalg.norm(hr)
  B = K.d1 @ np.einsum('fij,i->jf', K.W, h); G = K.d1 @ K.d1.T   # im delta = C^2 here
  D = np.diag([7, 7, 11, 11.])
  rr = np.r_[-np.geomspace(1e-2, 1e6, 600), np.geomspace(1e-2, 1e6, 600)]
  OUT['counterexample'] = dict(betti1=K.betti1, dim_im_delta=int(np.linalg.matrix_rank(K.d1)), F=len(K.T),
      h_matches_reviewer=float(min(np.linalg.norm(h - hr), np.linalg.norm(h + hr))),
      defect=float(diamond_defect(K, h)), B6=(6 * np.sqrt(np.sum(np.array([-3,1,-4,6,4,-1,-6,5,-5.])**2)) * B).round(6).tolist() if False else None,
      DB_skew_err=float(np.abs(D @ B + B.T @ D).max()),
      DGGD_minors=[float(np.linalg.det((D @ G + G @ D)[:k, :k])) for k in range(1, 5)],
      spec_B=[complex(x).__repr__() for x in np.linalg.eigvals(B)],
      max_abscissa_all_rho=max(absc(B, G, r) for r in rr),
      bound=-float(min(np.linalg.eigvalsh(D @ G + G @ D))) / (2 * float(max(np.diag(D)))))
  # unnormalized B for reviewer's h
  hu = np.array([-3, 1, -4, 6, 4, -1, -6, 5, -5.])
  OUT['counterexample']['B_reviewer_h_times6'] = (6 * K.d1 @ np.einsum('fij,i->jf', K.W, hu)).round(8).tolist()
  OUT['counterexample']['G'] = G.tolist()
  OUT['counterexample']['faces'] = [list(t) for t in K.T]

  # energy identity residual: d/dt 1/2|c|^2 + nu |delta c|^2 evaluated from the vector field
  rng = np.random.default_rng(5); res = 0
  for seed in (1, 3, 13):
      K = irregular(seed)
      for _ in range(50):
          c = K.P @ rng.normal(size=len(K.E)) * 10 ** rng.uniform(-1, 2); nu = 10 ** rng.uniform(-2, 0)
          res = max(res, abs(c @ K.rhs(c, nu) + nu * np.linalg.norm(K.d1 @ c) ** 2) / (nu * np.linalg.norm(K.d1 @ c) ** 2))
  OUT['energy_identity_rel_residual'] = float(res)
  json.dump(OUT, open('study.json', 'w'), indent=1)
  print(json.dumps(OUT, indent=1)[:4000])

if __name__ == "__main__":
    main()
