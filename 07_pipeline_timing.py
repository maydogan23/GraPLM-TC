# -*- coding: utf-8 -*-
"""
07_pipeline_timing.py — ozetleme asamasinin maliyet olcumu (Comment 2.4/6.2)

Makaledeki Algorithm 1-2'nin sadik bir yeniden uygulamasini kosarak
asama basina sure olcer:
  (1) on isleme (stop-word + pseudo-lemma + cumle bolme)
  (2) graf kurma (lemma-ortusme kenarlari, Algorithm 1)
  (3) Malatya-merkezilik vertex cover secimi (Algorithm 2, butce sinirli)

NOT: text sutununda noktalama olmadigindan cumle bolme varsayilan olarak
20-token pseudo-cumledir. Cengiz Hoca'nin pipeline'i gercek cumle
bolme/lemmatizasyon kullaniyorsa, --sentences_from ile onun ara ciktisi
verilebilir; makalede olcumun "faithful re-implementation of Algorithms
1-2" oldugu acikca yazilacak.

Kullanim:
  python scripts/07_pipeline_timing.py --dataset ttc3600 --n_docs 500
  python scripts/07_pipeline_timing.py --dataset ttc4900 --n_docs 500
"""
import argparse, time, re
import numpy as np
import pandas as pd

TR_STOP = set('''ve veya ile de da ki bu su o bir iki uc icin gibi kadar
sonra once ama fakat ancak cunku ne nasil neden hangi her hic cok az daha
en mi mu mi midir ise olarak uzere göre'''.split())

def pseudo_lemma(w):
    # kaba govde: ilk 5 karakter (Turkce ekleri yaklasik keser) — sadece
    # zamanlama temsiliyeti icin; dogruluk iddiasi yok
    return w.lower()[:5]

def preprocess(text, chunk=20):
    toks = [w for w in re.findall(r'\S+', str(text))]
    sents = [toks[i:i+chunk] for i in range(0, len(toks), chunk)] or [toks]
    nodes = []
    for s in sents:
        lem = {pseudo_lemma(w) for w in s if w.lower() not in TR_STOP and len(w) > 2}
        nodes.append((s, lem))
    return nodes

def build_graph(nodes):
    n = len(nodes)
    adj = [set() for _ in range(n)]
    for i in range(n):
        li = nodes[i][1]
        for j in range(i+1, n):
            if li & nodes[j][1]:
                adj[i].add(j); adj[j].add(i)
    return adj

def malatya_vc_select(nodes, adj, budget):
    """Malatya merkezilik: dugum degeri = komsu derecelerinin toplami;
    en yuksek merkezilikli dugumu sec, grafiktan cikar, tekrarla."""
    n = len(nodes)
    alive = set(range(n))
    deg = {i: len(adj[i]) for i in alive}
    picked, total = [], 0
    while alive and total < budget:
        cent = {i: sum(deg[j] for j in adj[i] if j in alive) + deg[i] for i in alive}
        v = max(cent, key=cent.get)
        picked.append(v); total += len(nodes[v][0])
        alive.discard(v)
        for j in adj[v]:
            if j in alive: deg[j] -= 1
        deg.pop(v, None)
    picked.sort()
    return [w for i in picked for w in nodes[i][0]][:budget]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ttc3600', 'ttc4900'])
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--n_docs', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(f'{args.data_dir}/{args.dataset}_frozen.csv')
    sample = df.sample(min(args.n_docs, len(df)), random_state=args.seed)

    t_pre, t_graph, t_vc = [], [], []
    for _, r in sample.iterrows():
        budget = max(len(str(r['summary']).split()), 5)
        t0 = time.perf_counter(); nodes = preprocess(r['text']); t1 = time.perf_counter()
        adj = build_graph(nodes); t2 = time.perf_counter()
        _ = malatya_vc_select(nodes, adj, budget); t3 = time.perf_counter()
        t_pre.append(t1-t0); t_graph.append(t2-t1); t_vc.append(t3-t2)

    def s(x): return f"{np.mean(x)*1000:.2f} ± {np.std(x)*1000:.2f} ms"
    n = len(sample)
    tot = (np.sum(t_pre)+np.sum(t_graph)+np.sum(t_vc))
    print(f"[{args.dataset}] {n} belge (orneklem), belge basina ortalama:")
    print(f"  on isleme          : {s(t_pre)}")
    print(f"  graf kurma (Alg.1) : {s(t_graph)}")
    print(f"  vertex cover (Alg.2): {s(t_vc)}")
    print(f"  TOPLAM pipeline    : {s([a+b+c for a,b,c in zip(t_pre,t_graph,t_vc)])}")
    full_n = len(df)
    print(f"  {full_n} belgeye olceklenmis tahmini toplam: {tot/n*full_n:.1f} s (tek CPU cekirdegi)")

if __name__ == '__main__':
    main()
