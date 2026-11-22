# combinatorial-navier-stokes

Code and results for the paper

> M. Boudourides, *Enstrophy Defects and the Stability of Circulating Flows in a Combinatorial Navier–Stokes Model* (submitted).

The model is Wilson's finite cochain model of incompressible viscous flow on a simplicial 2-complex: the velocity is a divergence-free 1-cochain, the pressure is the Hodge projection onto divergence-free cochains, the viscous term is the Hodge Laplacian, and the nonlinearity uses the antisymmetrized cup product. The code builds the complexes, the operators and the model, and reproduces every number, table and figure in the paper.

## Files

| File | Contents |
|---|---|
| `cns.py` | Simplicial complex class: coboundaries, cup product, Hodge projectors, oriented harmonic basis (`h0`), right-hand side of the model |
| `exp1.py` | Regular annulus and torus; linearization at a harmonic flow; harmonic-exchange runs (`python exp1.py`) |
| `exp2.py` | Symmetric part of the transport on the regular complexes |
| `diamond.py` | Diamonds, diamond defect, enstrophy quadratic form |
| `sat.py` | Irregular complexes (`irregular(seed)`: 50 uniform points in [-1,1]^2, points with norm <= 0.35 dropped, Delaunay, triangles with centroid norm <= 0.4 removed) |
| `checks.py` | Diamond formula (entrywise), flat-complex spectra, low-Re bound; writes `summary.json` |
| `study.py` | First critical values for both orientations, stability windows, mesh table, Example 2, energy-identity residual; writes `study.json`, `table_irregular.csv` |
| `shed.py` | Supercritical runs with coexact perturbations of norm 1e-3; appends records with a `run_id` to `shed_results.jsonl` |
| `shed_summary.py` | Summary of `shed_results.jsonl` (one record per `run_id`; reproduces the counts in Section 6) |
| `longrun.py` | Long runs from the same initial condition with DOP853 to t = 1e6, recording solver status per segment; settled = vorticity < 1e-6 (`long{seed}_rtol{rtol}.json`) |
| `figs2.py` | `fig_stability.pdf` and `fig_complex.pdf` |
| `table_tex.py` | LaTeX table of the irregular complexes |
| `verify.py`, `verify_flat.py`, `verify_rev.py` | Independent checks of the lemmas, theorems and examples |

## Reproducing

```bash
pip install -r requirements.txt
python checks.py; python study.py; python table_tex.py
rm -f shed_results.jsonl   # start from a fresh results file
for m in 0.5 2 5; do python shed.py 3 $m 10; done
for s in $(seq 1 30); do [ $s -ne 3 ] && [ $s -ne 13 ] && python shed.py $s 2 3; done
python shed.py 13 2 3 1e-9 DOP853
python longrun.py 13; python longrun.py 23; python longrun.py 9
python longrun.py 13 1e-11; python longrun.py 23 1e-11
python shed_summary.py; python figs2.py
python verify.py; python verify_flat.py; python verify_rev.py
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

## License

MIT
