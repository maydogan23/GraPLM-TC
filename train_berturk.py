
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# Maps an experimental condition to the text column used as classifier input.
COND2COL = {
    'original': 'text', 'vertexcover': 'summary', 'lead': 'lead',
    'tfidf': 'tfidf_sel', 'textrank': 'textrank_sel',
    'random0': 'random_r0', 'random1': 'random_r1', 'random2': 'random_r2',
    'vc_b10': 'vc_b10', 'vc_b20': 'vc_b20', 'vc_b30': 'vc_b30', 'vc_b40': 'vc_b40',
}


class TextDataset(Dataset):
    """Tokenized dataset with static padding to a fixed maximum length."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len,
                             padding='max_length', return_tensors='pt')
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item['labels'] = self.labels[i]
        return item


def ensure_pooler_initialized(model):
    """Reinitialize the BERT pooler if it was loaded with all-zero weights.

    Some checkpoint downloads leave the pooler dense layer zeroed, which
    prevents the classification head from receiving a useful signal. This
    restores the standard normal initialization when that state is detected.
    """
    if model.bert.pooler.dense.weight.std().item() < 1e-6:
        torch.nn.init.normal_(model.bert.pooler.dense.weight, std=0.02)
        torch.nn.init.zeros_(model.bert.pooler.dense.bias)
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items()
                              if k != 'labels'}).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(batch['labels'].cpu().tolist())
    return np.array(preds), np.array(gold)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
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
    ap.add_argument('--attn', default='eager', choices=['eager', 'sdpa'])
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = f'{args.dataset}_{args.condition}_s{args.seed}'
    col = COND2COL[args.condition]

    vpath = os.path.join(args.data_dir, f'{args.dataset}_variants.csv')
    fpath = os.path.join(args.data_dir, f'{args.dataset}_frozen.csv')
    df = pd.read_csv(vpath if os.path.exists(vpath) else fpath)
    if col not in df.columns:
        raise SystemExit(f"Column '{col}' not found; generate variants first.")

    labels = sorted(df['label'].unique())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    df['y'] = df['label'].map(label_to_idx)
    parts = {s: df[df.split == s] for s in ['train', 'val', 'test']}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    t0 = time.time()
    loaders = {}
    for split, part in parts.items():
        ds = TextDataset(part[col].astype(str).values, part['y'].values,
                         tokenizer, args.max_len)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=(split == 'train'))
    tokenize_s = time.time() - t0

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels),
        attn_implementation=args.attn).to(device)
    model = ensure_pooler_initialized(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loaders['train']) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    best_val, best_state, best_epoch = -1.0, None, -1
    t_train = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in loaders['train']:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        val_pred, val_gold = evaluate(model, loaders['val'], device)
        val_acc = accuracy_score(val_gold, val_pred)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    train_s = time.time() - t_train

    model.load_state_dict(best_state)
    model.to(device)

    t0 = time.time()
    test_pred, test_gold = evaluate(model, loaders['test'], device)
    inference_s = time.time() - t0
    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                   if device == 'cuda' else 0.0)

    row = dict(
        run_id=run_id, dataset=args.dataset, condition=args.condition,
        seed=args.seed, best_epoch=best_epoch, val_acc=round(best_val, 4),
        test_acc=round(accuracy_score(test_gold, test_pred), 4),
        test_f1=round(f1_score(test_gold, test_pred, average='macro'), 4),
        test_prec=round(precision_score(test_gold, test_pred, average='macro'), 4),
        test_rec=round(recall_score(test_gold, test_pred, average='macro'), 4),
        tokenize_s=round(tokenize_s, 2), train_s=round(train_s, 2),
        test_inference_s=round(inference_s, 3), peak_mem_gb=round(peak_mem_gb, 2),
        max_len=args.max_len, batch=args.batch_size, lr=args.lr,
        warmup=warmup, attn=args.attn, model=args.model_name)

    os.makedirs(args.results_dir, exist_ok=True)
    runs_path = os.path.join(args.results_dir, 'runs.csv')
    pd.DataFrame([row]).to_csv(runs_path, mode='a', index=False,
                               header=not os.path.exists(runs_path))
    preds = parts['test'][['doc_id', 'label']].copy()
    preds['pred'] = [labels[i] for i in test_pred]
    preds.to_csv(os.path.join(args.results_dir, f'preds_{run_id}.csv'), index=False)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
