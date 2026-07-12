# -*- coding: utf-8 -*-
"""
Kampanya orkestratörü (Windows/Linux). Bash scriptinin yerine geçer.

Kullanım (grallm_kampanya klasörünün içinden):
  python scripts/04_run_campaign.py            # her şeyi koşar
  python scripts/04_run_campaign.py --phase 1  # sadece original koşullar
  python scripts/04_run_campaign.py --phase 2  # sadece summary'ye bağımlı koşullar

Kesintiye dayanıklıdır: results/runs.csv'de tamamlanmış koşuları atlar,
kaldığı yerden devam eder.
"""
import argparse, os, subprocess, sys
import pandas as pd

SEEDS = [42, 43, 44, 45, 46]
DATASETS = ['ttc3600', 'ttc4900']
PHASE1 = ['original']
PHASE2 = ['vertexcover', 'lead', 'tfidf', 'textrank', 'random0', 'random1', 'random2']
PHASE3 = ['vc_b10', 'vc_b20', 'vc_b30', 'vc_b40']  # maxSize duyarlilik (R5.4)
PHASE3_DATASETS = ['ttc3600']  # istenirse 'ttc4900' eklenebilir

def done_runs():
    p = os.path.join('results', 'runs.csv')
    if not os.path.exists(p):
        return set()
    try:
        return set(pd.read_csv(p)['run_id'].astype(str))
    except Exception:
        return set()

def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f'HATA: komut basarisiz ({r.returncode}). Duzeltme icin ciktiyi paylasin.')
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, choices=[1, 2, 3], default=None,
                    help='1: original, 2: summary bagimli; bos: hepsi')
    ap.add_argument('--skip_variants', action='store_true',
                    help='varyant uretimini atla (zaten uretilmisse)')
    args = ap.parse_args()
    py = sys.executable

    if not args.skip_variants:
        for ds in DATASETS:
            vpath = os.path.join('data', f'{ds}_variants.csv')
            if os.path.exists(vpath):
                print(f'{vpath} mevcut, atlandi (yeniden uretmek icin dosyayi silin)')
            else:
                run([py, os.path.join('scripts', '02_generate_variants.py'),
                     '--dataset', ds, '--data_dir', 'data'])

    conditions = (PHASE1 if args.phase == 1 else
                  PHASE2 if args.phase == 2 else
                  PHASE3 if args.phase == 3 else PHASE1 + PHASE2)
    finished = done_runs()
    todo = [(ds, c, s) for ds in DATASETS for c in conditions for s in SEEDS
            if f'{ds}_{c}_s{s}' not in finished]
    if args.phase in (None, 3):  # maxSize kosullari (varsayilan: ttc3600)
        todo += [(ds, c, s) for ds in PHASE3_DATASETS for c in PHASE3
                 for s in SEEDS if f'{ds}_{c}_s{s}' not in finished
                 and (ds, c, s) not in todo]
    print(f'Toplam {len(todo)} kosu (tamamlanmis {len(finished)} kosu atlaniyor)')

    for i, (ds, cond, seed) in enumerate(todo, 1):
        print(f'--- [{i}/{len(todo)}] {ds} / {cond} / seed {seed} ---', flush=True)
        run([py, os.path.join('scripts', '03_train_berturk.py'),
             '--dataset', ds, '--condition', cond, '--seed', str(seed)])
    print('Kampanya tamam. results/runs.csv ve preds_*.csv dosyalarini paylasin.')

if __name__ == '__main__':
    main()
