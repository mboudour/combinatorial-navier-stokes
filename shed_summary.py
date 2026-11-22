"""Summarize shed_results.jsonl (LSODA, rtol 1e-9 runs) for the manuscript."""
import json
R = [json.loads(l) for l in open('shed_results.jsonl')]
R9 = [r for r in R if r['rtol'] == 1e-9 and r.get('method', 'LSODA') == 'LSODA']
c3 = [r for r in R9 if r['seed'] == 3]
sup3 = [r for r in c3 if r['mult'] > 1]; sub3 = [r for r in c3 if r['mult'] < 1]
print('complex 3 subcritical: final/s0 range', min(r['final_over_nu'] / r['s0_over_nu'] for r in sub3), max(r['final_over_nu'] / r['s0_over_nu'] for r in sub3))
print('complex 3 supercritical: final range', min(r['final_over_nu'] for r in sup3), max(r['final_over_nu'] for r in sup3),
      'all stable', all(r['final_abscissa'] < 0 for r in sup3), 'max vort', max(r['final_vorticity'] for r in sup3), 'all success', all(r['success'] for r in c3),
      'depart times', min(r['t_depart'] for r in sup3), max(r['t_depart'] for r in sup3))
oth = [r for r in R9 if r['seed'] != 3 and r['seed'] != 13]
ok = [r for r in oth if r['final_abscissa'] < 0 and r['final_vorticity'] < 1e-6]
print('others (excl 13): n', len(oth), 'settled stable', len(ok), 'final range', min(r['final_over_nu'] for r in oth), max(r['final_over_nu'] for r in oth),
      'max vort', max(r['final_vorticity'] for r in oth), 'all success', all(r['success'] for r in oth))
print('not ok:', [(r['seed'], r['pert'], r['final_over_nu'], r['final_vorticity'], r['final_abscissa']) for r in oth if r not in ok])
print('seeds covered', sorted({r['seed'] for r in oth}))
