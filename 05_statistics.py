# -*- coding: utf-8 -*-
"""
05_statistics.py — kampanya sonuclarindan makale tablolari.
Kullanim (grallm_kampanya klasorunden):
  python scripts/05_statistics.py

Girdi : results/runs.csv, results/preds_*.csv
Cikti : results/table_main.csv        (kosul basina mean±SD + %95 GA)
        results/table_mcnemar.csv     (seed-eslesmis McNemar p'leri)
        results/table_bootstrap.csv   (dogruluk farki %95 bootstrap GA)
        results/perclass_<ds>.csv     (sinif bazli F1: original vs vertexcover)
        results/confusion_<ds>_<cond>.csv
"""
import os, glob, itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

RES = 'results'
COMPARISONS = [('vertexcover', 'original'), ('vertexcover', 'lead'),
               ('vertexcover', 'tfidf'), ('vertexcover', 'textrank'),
               ('vertexcover', 'random0')]

runs = pd.read_csv(os.path.join(RES, 'runs.csv'))
runs['cond_group'] = runs['condition'].replace(
    {'random0': 'random', 'random1': 'random', 'random2': 'random'})

# ---------- TABLO 1: mean ± SD + %95 GA ----------
rows = []
for (ds, cg), g in runs.groupby(['dataset', 'cond_group']):
    n = len(g)
    for m in ['test_acc', 'test_f1']:
        mean, sd = g[m].mean(), g[m].std(ddof=1)
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(dataset=ds, condition=cg, metric=m, n_runs=n,
                         mean=round(mean, 4), sd=round(sd, 4),
                         ci95_lo=round(mean - ci, 4), ci95_hi=round(mean + ci, 4)))
main = pd.DataFrame(rows)
main.to_csv(os.path.join(RES, 'table_main.csv'), index=False)
print('=== ANA TABLO (test) ===')
piv = main[main.metric == 'test_acc'].pivot(index='condition', columns='dataset', values='mean')
print(piv.to_string())

def load_preds(ds, cond, seed):
    p = os.path.join(RES, f'preds_{ds}_{cond}_s{seed}.csv')
    return pd.read_csv(p).sort_values('doc_id') if os.path.exists(p) else None

def mcnemar_p(gold, a, b):
    """Exact binomial McNemar: a-dogru/b-yanlis vs a-yanlis/b-dogru."""
    ca, cb = (a == gold), (b == gold)
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    if n01 + n10 == 0: return 1.0, n01, n10
    return stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue * 1, n01, n10

# ---------- TABLO 2: seed-eslesmis McNemar ----------
seeds = sorted(runs['seed'].unique())
mrows = []
for ds in runs['dataset'].unique():
    for c1, c2 in COMPARISONS:
        ps, n01s, n10s = [], [], []
        for s in seeds:
            p1, p2 = load_preds(ds, c1, s), load_preds(ds, c2, s)
            if p1 is None or p2 is None: continue
            gold = p1['label'].values
            pv, n01, n10 = mcnemar_p(gold, p1['pred'].values, p2['pred'].values)
            ps.append(pv); n01s.append(n01); n10s.append(n10)
        if ps:
            mrows.append(dict(dataset=ds, pair=f'{c1} vs {c2}', n_seeds=len(ps),
                              median_p=round(float(np.median(ps)), 4),
                              min_p=round(min(ps), 4), max_p=round(max(ps), 4),
                              sig_seeds_p05=sum(1 for p in ps if p < 0.05),
                              mean_n01=round(np.mean(n01s), 1),
                              mean_n10=round(np.mean(n10s), 1)))
mc = pd.DataFrame(mrows)
mc.to_csv(os.path.join(RES, 'table_mcnemar.csv'), index=False)
print('\n=== McNEMAR (seed-eslesmis, exact binomial) ===')
print(mc.to_string(index=False))

# ---------- TABLO 3: bootstrap GA (dogruluk farki) ----------
rng = np.random.default_rng(0)
brows = []
for ds in runs['dataset'].unique():
    for c1, c2 in COMPARISONS:
        diffs_all = []
        for s in seeds:
            p1, p2 = load_preds(ds, c1, s), load_preds(ds, c2, s)
            if p1 is None or p2 is None: continue
            gold = p1['label'].values
            ok1 = (p1['pred'].values == gold).astype(float)
            ok2 = (p2['pred'].values == gold).astype(float)
            n = len(gold)
            idx = rng.integers(0, n, size=(2000, n))
            diffs_all.append(ok1[idx].mean(1) - ok2[idx].mean(1))
        if diffs_all:
            d = np.concatenate(diffs_all)
            brows.append(dict(dataset=ds, pair=f'{c1} vs {c2}',
                              mean_diff=round(float(d.mean()), 4),
                              ci95_lo=round(float(np.percentile(d, 2.5)), 4),
                              ci95_hi=round(float(np.percentile(d, 97.5)), 4)))
bt = pd.DataFrame(brows)
bt.to_csv(os.path.join(RES, 'table_bootstrap.csv'), index=False)
print('\n=== BOOTSTRAP %95 GA (dogruluk farki, seed havuzlu) ===')
print(bt.to_string(index=False))

# ---------- Sinif bazli F1 + karisiklik matrisi (cogunluk oyu) ----------
def majority_preds(ds, cond):
    ps = [load_preds(ds, cond, s) for s in seeds]
    ps = [p for p in ps if p is not None]
    if not ps: return None, None
    gold = ps[0]['label'].values
    M = np.stack([p['pred'].values for p in ps])
    maj = pd.DataFrame(M).mode(axis=0).iloc[0].values
    return gold, maj

for ds in runs['dataset'].unique():
    pc = {}
    for cond in ['original', 'vertexcover']:
        gold, maj = majority_preds(ds, cond)
        if gold is None: continue
        labs = sorted(set(gold))
        pc[cond] = f1_score(gold, maj, labels=labs, average=None)
        cm = pd.DataFrame(confusion_matrix(gold, maj, labels=labs), index=labs, columns=labs)
        cm.to_csv(os.path.join(RES, f'confusion_{ds}_{cond}.csv'))
    if pc:
        out = pd.DataFrame(pc, index=labs).round(4)
        out.to_csv(os.path.join(RES, f'perclass_{ds}.csv'))
        print(f'\n=== SINIF BAZLI F1 ({ds}, cogunluk oyu) ===')
        print(out.to_string())

print('\nTum tablolar results/ klasorune yazildi.')
