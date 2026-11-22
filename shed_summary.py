"""Summary of shed_results.jsonl as reported in Section 6. One record per run
(seed, mult, pert, rtol, method); runs at rtol 1e-9, LSODA, except complex 13 (DOP853)."""
import json
recs = {}
for l in open('shed_results.jsonl'):
    r = json.loads(l); r.setdefault('method', 'LSODA')
    recs[r.get('run_id', f"{r['seed']}/{r['mult']}/{r['pert']}/{r['rtol']:g}/{r['method']}")] = r
R = [r for r in recs.values() if r['rtol'] == 1e-9 and (r['method'] == 'LSODA') == (r['seed'] != 13)]
SETTLED = 1e-6
c3 = [r for r in R if r['seed'] == 3]; sup3 = [r for r in c3 if r['mult'] > 1]; sub3 = [r for r in c3 if r['mult'] < 1]
print('complex 3: runs', len(c3), '| subcritical final/s0 in', [round(min(r['final_over_nu'] / r['s0_over_nu'] for r in sub3), 6), round(max(r['final_over_nu'] / r['s0_over_nu'] for r in sub3), 6)])
print('  supercritical final a/nu in', [round(min(r['final_over_nu'] for r in sup3), 1), round(max(r['final_over_nu'] for r in sup3), 1)],
      '| all normally stable', all(r['final_abscissa'] < 0 for r in sup3), '| max final vorticity', max(r['final_vorticity'] for r in sup3),
      '| departure times', [round(min(r['t_depart'] for r in sup3)), round(max(r['t_depart'] for r in sup3))], '| all success', all(r['success'] for r in c3))
oth = [r for r in R if r['seed'] != 3]
ok = [r for r in oth if r['final_vorticity'] < SETTLED and r['final_abscissa'] < 0]
print('other complexes: runs', len(oth), '| settled (vorticity <', SETTLED, ') at a normally stable equilibrium by T:', len(ok),
      '| final a/nu in', [round(min(r['final_over_nu'] for r in ok)), round(max(r['final_over_nu'] for r in ok))], '| all success', all(r['success'] for r in oth))
for r in sorted(oth, key=lambda r: (r['seed'], r['pert'])):
    if r not in ok: print('  not settled by T:', r['seed'], r['pert'], round(r['final_over_nu'], 1), f"{r['final_vorticity']:.1e}", round(r['final_abscissa'], 3), r['method'])
