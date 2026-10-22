"""Step 4 - full fine-tuning of one backbone on one split with one training seed.

Same protocol for all three models: identical split, augmentation, class-weighted
cross-entropy, AdamW, batch 16, at most --epochs epochs, model selection and early
stopping on VALIDATION ROC-AUC only. The test set is scored once, with the selected
weights, after training has finished.
Writes results/finetune/<model>_split<S>_seed<R>_{history,preds}.csv
"""
import argparse, os, time, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from models import Classifier
from data import OralImages, load_split

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, choices=['vgg16', 'resnet50', 'vit_b16'])
p.add_argument('--images', required=True); p.add_argument('--splits', default='../data_splits')
p.add_argument('--split_seed', type=int, default=42); p.add_argument('--seed', type=int, default=0)
p.add_argument('--epochs', type=int, default=10); p.add_argument('--patience', type=int, default=3)
p.add_argument('--batch', type=int, default=16); p.add_argument('--accum', type=int, default=None, help='gradient accumulation steps (micro-batch = batch/accum, saves memory); default 1 for ResNet-50 (batch norm), 2 otherwise'); p.add_argument('--out', default='../results/finetune')
p.add_argument('--save_model', action='store_true')
a = p.parse_args()
if a.accum is None: a.accum = 1 if a.model == 'resnet50' else 2
LR ={'vgg16': 1e-4, 'resnet50': 1e-4, 'vit_b16': 2e-5}[a.model]     # transformer needs a smaller step
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(os.cpu_count())
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device:', DEV, flush=True)
os.makedirs(a.out, exist_ok=True); tag = f'{a.model}_split{a.split_seed}_seed{a.seed}'
S = load_split(a.splits, a.split_seed)
tr, va, te = (S[S.split == k] for k in ('train', 'val', 'test'))
g = torch.Generator().manual_seed(a.seed)
dl_tr = DataLoader(OralImages(a.images, tr, True), batch_size=a.batch, shuffle=True, generator=g, num_workers=0)
dl_va = DataLoader(OralImages(a.images, va, False), batch_size=32); dl_te = DataLoader(OralImages(a.images, te, False), batch_size=32)
m = Classifier(a.model).to(DEV)
w = torch.tensor(len(tr) / (2 * np.bincount(tr.label)), dtype=torch.float32)   # class-balanced weights
lossf = nn.CrossEntropyLoss(weight=w.to(DEV))
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)


def predict(dl):
    m.eval(); P, Y, L = [], [], 0.0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            o = m(x); L += lossf(o, y).item() * len(y); P.append(torch.softmax(o, 1)[:, 1].cpu().numpy()); Y.append(y.cpu().numpy())
    P, Y = np.concatenate(P), np.concatenate(Y); return P, Y, L / len(Y)


hist, best, best_auc, bad = [], None, -1, 0
for ep in range(1, a.epochs + 1):
    m.train(); t0 = time.time(); tl, n, corr = 0.0, 0, 0
    for x, y in dl_tr:                                  # one optimiser step per batch of 16,
        x, y = x.to(DEV), y.to(DEV); opt.zero_grad()    # computed in micro-batches to save memory
        for xm, ym in zip(x.chunk(a.accum), y.chunk(a.accum)):
            o = m(xm); l = lossf(o, ym); (l * len(ym) / len(y)).backward()
            tl += l.item() * len(ym); n += len(ym); corr += (o.argmax(1) == ym).sum().item()
        opt.step()
    pv, yv, lv = predict(dl_va); auc = roc_auc_score(yv, pv)
    hist.append({'epoch': ep, 'train_loss': tl / n, 'train_acc': corr / n, 'val_loss': lv, 'val_auc': auc,
                 'val_acc': float(((pv >= 0.5) == yv).mean()), 'epoch_time_s': time.time() - t0})
    print(json.dumps(hist[-1]), flush=True)
    if auc > best_auc + 1e-4 or (abs(auc - best_auc) <= 1e-4 and lv < min(h['val_loss'] for h in hist[:-1] + [hist[-1]])):
        best_auc, best, bad = auc, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= a.patience: break
pd.DataFrame(hist).to_csv(os.path.join(a.out, tag + '_history.csv'), index=False)
m.load_state_dict(best)
pv, yv, _ = predict(dl_va); pt, yt, _ = predict(dl_te)       # test set: scored once, here
pd.concat([pd.DataFrame({'path': va.path.values, 'split': 'val', 'label': yv, 'p_cancer': pv}),
           pd.DataFrame({'path': te.path.values, 'split': 'test', 'label': yt, 'p_cancer': pt})]).to_csv(os.path.join(a.out, tag + '_preds.csv'), index=False)
if a.save_model: torch.save({k: v.cpu() for k, v in best.items()}, os.path.join(a.out, tag + '.pt'))
print('DONE', tag, 'best val AUC %.4f' % best_auc, 'test AUC %.4f' % roc_auc_score(yt, pt))
