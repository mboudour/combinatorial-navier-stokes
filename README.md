# combinatorial-navier-stokes

Code for the paper

> M. Boudourides, *Stability of Circulating Flows in a Combinatorial Navier–Stokes Model on Simplicial 2-Complexes* (submitted).

The model is Wilson's finite cochain model of incompressible viscous flow on a simplicial 2-complex: the velocity is a divergence-free 1-cochain, the pressure is the Hodge projection onto divergence-free cochains, the viscous term is the Hodge Laplacian, and the nonlinearity uses the antisymmetrized cup product. The code builds the complexes, the operators and the model, and reproduces every number and figure in the paper.

## Files

| File | Contents |
|---|---|
| `cns.py` | Simplicial complex class: coboundaries, cup product, Hodge projectors, harmonic basis, right-hand side of the model |
| `exp1.py` | Regular annulus and torus triangulations; linearization at a harmonic flow on the coexact subspace |
| `exp2.py` | Skew-symmetry and large-Reynolds-number checks |
| `diamond.py` | Diamonds of a complex, diamond defect, enstrophy quadratic form |
| `sat.py` | Irregular (Delaunay) complexes with a hole; critical Reynolds number by scan and bisection; shedding runs |
| `figs.py` | All numerical checks of Section 6; writes `summary.json` and `fig_stability.pdf` |
| `verify.py` | Independent checks of Lemmas 1 and 3, Example 1, Theorem 1 and Theorem 4(a) |
| `verify_flat.py` | Checks of Theorem 3 (pointwise differential inequality and trajectories on the flat torus) |

## Reproducing

```bash
pip install -r requirements.txt
python figs.py         # summary.json and fig_stability.pdf (several minutes)
python verify.py
python verify_flat.py
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

## License

MIT
