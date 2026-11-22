"""LaTeX table of the thirty irregular complexes from table_irregular.csv (+ trace check)."""
import csv, numpy as np
from sat import irregular
from study import ops
rows = list(csv.DictReader(open('table_irregular.csv')))
tr = max(abs(np.trace(ops(irregular(int(r['seed'])), irregular(int(r['seed'])).h0())[0])) for r in rows)
print('max |tr B| =', tr)
def rc(r, t):
    return f"{float(r['Rc_'+t]):.1f}" + ("$^{\\ast}$" if r['osc_'+t] == 'True' else "") + ("$^{\\dagger}$" if r['returns_'+t] != '0' else "")
L = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
     r"$k$ & $V$ & $E$ & $F$ & $\mu$ & $\max|D(h_0)|$ & $\max\Rea\sigma(B_{h_0})$ & $\mathrm{Re}_c(h_0)$ & $\mathrm{Re}_c(-h_0)$\\", r"\midrule"]
for r in rows:
    L.append(f"{r['seed']} & {r['V']} & {r['E']} & {r['F']} & {float(r['mu']):.3f} & {float(r['defect']):.3f} & {float(r['maxReB']):.4f} & {rc(r,'p')} & {rc(r,'m')}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
import os; os.makedirs('paper', exist_ok=True); open('paper/table_irregular.tex', 'w').write("\n".join(L) + "\n")
