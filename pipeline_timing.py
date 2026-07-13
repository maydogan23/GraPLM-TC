
import argparse
import re
import time

import numpy as np
import pandas as pd

TR_STOPWORDS = set('''ve veya ile de da ki bu su o bir iki uc icin gibi kadar
sonra once ama fakat ancak cunku ne nasil neden hangi her hic cok az daha
en mi mu mi midir ise olarak uzere göre'''.split())


def stem(word):
    """Coarse stem used only for timing representativeness (first 5 chars)."""
    return word.lower()[:5]


def preprocess(text, chunk=20):
    """Split text into fixed-size pseudo-sentences and extract content stems."""
    tokens = re.findall(r'\S+', str(text))
    sentences = [tokens[i:i + chunk] for i in range(0, len(tokens), chunk)] or [tokens]
    nodes = []
    for sent in sentences:
        stems = {stem(w) for w in sent
                 if w.lower() not in TR_STOPWORDS and len(w) > 2}
        nodes.append((sent, stems))
    return nodes


def build_graph(nodes):
    """Connect two nodes whose sentences share at least one content stem."""
    n = len(nodes)
    adj = [set() for _ in range(n)]
    for i in range(n):
        stems_i = nodes[i][1]
        for j in range(i + 1, n):
            if stems_i & nodes[j][1]:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def malatya_vertex_cover(nodes, adj, budget):
    """Select nodes by centrality until the token budget is reached.

    Node centrality is the sum of neighbour degrees; the highest-centrality
    node is selected and removed, and the process repeats until the selected
    sentences reach the token budget.
    """
    alive = set(range(len(nodes)))
    degree = {i: len(adj[i]) for i in alive}
    picked, total = [], 0
    while alive and total < budget:
        centrality = {i: sum(degree[j] for j in adj[i] if j in alive) + degree[i]
                      for i in alive}
        node = max(centrality, key=centrality.get)
        picked.append(node)
        total += len(nodes[node][0])
        alive.discard(node)
        for j in adj[node]:
            if j in alive:
                degree[j] -= 1
        degree.pop(node, None)
    picked.sort()
    return [w for i in picked for w in nodes[i][0]][:budget]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['ttc3600', 'ttc4900'])
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--n_docs', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(f'{args.data_dir}/{args.dataset}_frozen.csv')
    sample = df.sample(min(args.n_docs, len(df)), random_state=args.seed)

    t_pre, t_graph, t_cover = [], [], []
    for _, row in sample.iterrows():
        budget = max(len(str(row['summary']).split()), 5)
        t0 = time.perf_counter()
        nodes = preprocess(row['text'])
        t1 = time.perf_counter()
        adj = build_graph(nodes)
        t2 = time.perf_counter()
        malatya_vertex_cover(nodes, adj, budget)
        t3 = time.perf_counter()
        t_pre.append(t1 - t0)
        t_graph.append(t2 - t1)
        t_cover.append(t3 - t2)

    def fmt(x):
        return f'{np.mean(x) * 1000:.2f} +/- {np.std(x) * 1000:.2f} ms'

    n = len(sample)
    total = np.sum(t_pre) + np.sum(t_graph) + np.sum(t_cover)
    per_doc = [a + b + c for a, b, c in zip(t_pre, t_graph, t_cover)]
    print(f'[{args.dataset}] per-document means over {n} sampled documents:')
    print(f'  preprocessing           : {fmt(t_pre)}')
    print(f'  graph construction (Alg1): {fmt(t_graph)}')
    print(f'  vertex-cover (Alg2)     : {fmt(t_cover)}')
    print(f'  total pipeline          : {fmt(per_doc)}')
    print(f'  estimated total for {len(df)} documents: '
          f'{total / n * len(df):.1f} s (single CPU core)')


if __name__ == '__main__':
    main()
