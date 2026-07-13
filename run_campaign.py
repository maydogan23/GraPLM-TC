
import argparse
import os
import subprocess
import sys

import pandas as pd

SEEDS = [42, 43, 44, 45, 46]
DATASETS = ['ttc3600', 'ttc4900']
PHASE1 = ['original']
PHASE2 = ['vertexcover', 'lead', 'tfidf', 'textrank',
          'random0', 'random1', 'random2']
PHASE3 = ['vc_b10', 'vc_b20', 'vc_b30', 'vc_b40']   # maxSize sensitivity
PHASE3_DATASETS = ['ttc3600']


def done_runs():
    path = os.path.join('results', 'runs.csv')
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path)['run_id'].astype(str))
    except Exception:
        return set()


def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'Command failed with exit code {result.returncode}.')
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', type=int, choices=[1, 2, 3], default=None,
                    help='1: original, 2: summary-based, 3: maxSize; '
                         'omit to run all phases')
    ap.add_argument('--skip_variants', action='store_true',
                    help='skip variant generation if already produced')
    args = ap.parse_args()
    py = sys.executable

    if not args.skip_variants:
        for ds in DATASETS:
            vpath = os.path.join('data', f'{ds}_variants.csv')
            if os.path.exists(vpath):
                print(f'{vpath} exists; skipping (delete it to regenerate).')
            else:
                run([py, os.path.join('scripts', 'generate_variants.py'),
                     '--dataset', ds, '--data_dir', 'data'])

    conditions = (PHASE1 if args.phase == 1 else
                  PHASE2 if args.phase == 2 else
                  PHASE3 if args.phase == 3 else PHASE1 + PHASE2)
    finished = done_runs()
    todo = [(ds, c, s) for ds in DATASETS for c in conditions for s in SEEDS
            if f'{ds}_{c}_s{s}' not in finished]
    if args.phase in (None, 3):
        todo += [(ds, c, s) for ds in PHASE3_DATASETS for c in PHASE3
                 for s in SEEDS if f'{ds}_{c}_s{s}' not in finished
                 and (ds, c, s) not in todo]
    print(f'{len(todo)} runs scheduled ({len(finished)} already completed).')

    for i, (ds, cond, seed) in enumerate(todo, 1):
        print(f'--- [{i}/{len(todo)}] {ds} / {cond} / seed {seed} ---', flush=True)
        run([py, os.path.join('scripts', 'train_berturk.py'),
             '--dataset', ds, '--condition', cond, '--seed', str(seed)])
    print('Campaign complete.')


if __name__ == '__main__':
    main()
