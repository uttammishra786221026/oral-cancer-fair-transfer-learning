"""Step 3b - linear probe (frozen backbone + new two-way head) for all models and splits.

Head = L2-regularised logistic regression with balanced class weights, trained on the
augmented training views. C is chosen on validation ROC-AUC; the test set is used once.
Writes results/linear_probe/<model>_split<S>_preds.csv
"""
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('--features', default='../results/features'); p.add_argument('--splits', default='../data_splits')
p.add_argument('--out', default='../results/linear_probe'); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
log = []
for model in ['vgg16', 'resnet50', 'vit_b16']:
    F = np.load(os.path.join(a.features, model + '.npz'), allow_pickle=True); idx = {pth: i for i, pth in enumerate(F['path'])}
    for seed in range(42, 47):
        S = pd.read_csv(os.path.join(a.splits, f'split_seed{seed}.csv'))
        I = {k: np.array([idx[x] for x in S[S.split == k].path]) for k in ('train', 'val', 'test')}
        Xtr = F['aug'][I['train']].reshape(-1, F['aug'].shape[-1]); ytr = np.repeat(F['label'][I['train']], F['aug'].shape[1])
        sc = StandardScaler().fit(Xtr)
        best = None
        for C in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]:
            clf = LogisticRegression(C=C, class_weight='balanced', max_iter=5000).fit(sc.transform(Xtr), ytr)
            auc = roc_auc_score(F['label'][I['val']], clf.predict_proba(sc.transform(F['plain'][I['val']]))[:, 1])
            if best is None or auc > best[0] + 1e-6: best = (auc, C, clf)
        auc, C, clf = best
        rows = []
        for k in ('val', 'test'):
            pr = clf.predict_proba(sc.transform(F['plain'][I[k]]))[:, 1]
            rows.append(pd.DataFrame({'path': F['path'][I[k]], 'split': k, 'label': F['label'][I[k]], 'p_cancer': pr}))
        pd.concat(rows).to_csv(os.path.join(a.out, f'{model}_split{seed}_preds.csv'), index=False)
        log.append({'model': model, 'split_seed': seed, 'C': C, 'val_auc': auc}); print(json.dumps(log[-1]), flush=True)
pd.DataFrame(log).to_csv(os.path.join(a.out, 'selected_C.csv'), index=False)
