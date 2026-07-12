
import argparse, os, time, json, random, math, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

COND2COL = {'original': 'text', 'vertexcover': 'summary', 'lead': 'lead',
            'tfidf': 'tfidf_sel', 'textrank': 'textrank_sel',
            'random0': 'random_r0', 'random1': 'random_r1', 'random2': 'random_r2',
            'vc_b10': 'vc_b10', 'vc_b20': 'vc_b20',
            'vc_b30': 'vc_b30', 'vc_b40': 'vc_b40'}

class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len,
                       padding='max_length', return_tensors='pt')
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        d['labels'] = self.labels[i]
        return d


def ensure_pooler_ok(model):
    """Bozuk yuklemede pooler sifir kalabiliyor; tespit et ve yeniden baslat."""
    p = model.bert.pooler.dense.weight
    if p.std().item() < 1e-6:
        print("[guard] pooler agirliklari sifir bulundu -> yeniden baslatiliyor "
              "(N(0,0.02)); cache'i temizlemeniz yine de onerilir", flush=True)
        torch.nn.init.normal_(model.bert.pooler.dense.weight, std=0.02)
        torch.nn.init.zeros_(model.bert.pooler.dense.bias)
    return model

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def evaluate(model, loader, device):
    model.eval(); preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(batch['labels'].cpu().tolist())
    return np.array(preds), np.array(gold)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ttc3600', 'ttc4900'])
    ap.add_argument('--condition', required=True, choices=list(COND2COL))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--model_name', default='dbmdz/bert-base-turkish-cased')
    ap.add_argument('--max_len', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--warmup_ratio', type=float, default=0.1)
    ap.add_argument('--attn', default='eager', choices=['eager', 'sdpa'],
                    help='attention implementasyonu (varsayilan: eager)')
    ap.add_argument('--smoke', action='store_true',
                    help='64 ornekte 30 adim overfit testi; egitim yapmaz')
    args = ap.parse_args()

    import transformers
    print(f"[env] torch={torch.__version__} transformers={transformers.__version__} "
          f"cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = f"{args.dataset}_{args.condition}_s{args.seed}"
    col = COND2COL[args.condition]

    vpath = os.path.join(args.data_dir, f"{args.dataset}_variants.csv")
    fpath = os.path.join(args.data_dir, f"{args.dataset}_frozen.csv")
    df = pd.read_csv(vpath if os.path.exists(vpath) else fpath)
    if col not in df.columns:
        raise SystemExit(f"'{col}' sutunu yok - once 02_generate_variants.py kosun.")

    labels = sorted(df['label'].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    df['y'] = df['label'].map(l2i)
    print(f"[data] siniflar={labels}", flush=True)
    parts = {s: df[df.split == s] for s in ['train', 'val', 'test']}

    tok = AutoTokenizer.from_pretrained(args.model_name)

    # --- SMOKE TEST: 64 ornegi 30 adimda ezberleyebiliyor mu? ---
    if args.smoke:
        sub = parts['train'].sample(64, random_state=0)
        ds = DS(sub[col].astype(str).values, sub['y'].values, tok, args.max_len)
        dl = DataLoader(ds, batch_size=16, shuffle=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(labels),
            attn_implementation=args.attn).to(device)
        model = ensure_pooler_ok(model)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        model.train(); step = 0
        print(f"[smoke] beklenen ilk loss ~ln({len(labels)})={math.log(len(labels)):.2f}", flush=True)
        for ep in range(20):
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch); loss = out.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[smoke] !!! adim {step}: loss NaN/Inf - kararsizlik burada", flush=True)
                    sys.exit(1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                if step % 5 == 0:
                    print(f"[smoke] adim {step:3d} loss={loss.item():.4f}", flush=True)
                step += 1
                if step >= 30: break
            if step >= 30: break
        p, g = evaluate(model, dl, device)
        acc = accuracy_score(g, p)
        print(f"[smoke] 30 adim sonrasi 64-ornek dogrulugu: {acc:.3f} "
              f"({'OK - loop calisiyor' if acc > 0.5 else 'SORUN - loop/ortam problemi'})", flush=True)
        return

    # --- NORMAL EGITIM ---
    t0 = time.time()
    loaders = {}
    for s, d in parts.items():
        ds = DS(d[col].astype(str).values, d['y'].values, tok, args.max_len)
        loaders[s] = DataLoader(ds, batch_size=args.batch_size, shuffle=(s == 'train'))
    tokenize_s = time.time() - t0

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels),
        attn_implementation=args.attn).to(device)
    model = ensure_pooler_ok(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loaders['train']) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    print(f"[cfg] total_steps={total_steps} warmup={warmup} lr={args.lr} attn={args.attn}", flush=True)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    best_val, best_state, best_epoch = -1, None, -1
    global_step = 0
    t_train = time.time()
    for ep in range(args.epochs):
        model.train(); ep_losses = []
        for batch in loaders['train']:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"!!! adim {global_step}: loss NaN/Inf. Kosu durduruldu. "
                      f"--lr 2e-5 ile tekrar deneyin.", flush=True)
                sys.exit(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ep_losses.append(loss.item())
            if global_step < 20 and global_step % 5 == 0:
                print(f"  adim {global_step:3d} loss={loss.item():.4f}", flush=True)
            global_step += 1
        vp, vg = evaluate(model, loaders['val'], device)
        vacc = accuracy_score(vg, vp)
        print(f"[{run_id}] epoch {ep} train_loss={np.mean(ep_losses):.4f} "
              f"val_acc={vacc:.4f}", flush=True)
        if vacc > best_val:
            best_val, best_epoch = vacc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_s = time.time() - t_train
    model.load_state_dict(best_state); model.to(device)

    t0 = time.time()
    tp, tg = evaluate(model, loaders['test'], device)
    infer_s = time.time() - t0
    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == 'cuda' else 0

    row = dict(run_id=run_id, dataset=args.dataset, condition=args.condition,
               seed=args.seed, best_epoch=best_epoch, val_acc=round(best_val, 4),
               test_acc=round(accuracy_score(tg, tp), 4),
               test_f1=round(f1_score(tg, tp, average='macro'), 4),
               test_prec=round(precision_score(tg, tp, average='macro'), 4),
               test_rec=round(recall_score(tg, tp, average='macro'), 4),
               tokenize_s=round(tokenize_s, 2), train_s=round(train_s, 2),
               test_inference_s=round(infer_s, 3), peak_mem_gb=round(peak_mem_gb, 2),
               max_len=args.max_len, batch=args.batch_size, lr=args.lr,
               warmup=warmup, attn=args.attn, model=args.model_name)

    os.makedirs(args.results_dir, exist_ok=True)
    runs_path = os.path.join(args.results_dir, 'runs.csv')
    pd.DataFrame([row]).to_csv(runs_path, mode='a', index=False,
                               header=not os.path.exists(runs_path))
    preds = parts['test'][['doc_id', 'label']].copy()
    preds['pred'] = [labels[i] for i in tp]
    preds.to_csv(os.path.join(args.results_dir, f'preds_{run_id}.csv'), index=False)
    print(json.dumps(row, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
