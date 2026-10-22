"""Step 5 - turn saved predictions into the tables of the paper.

* per run: accuracy, sensitivity, specificity, precision, F1, ROC-AUC (threshold 0.5)
  and the same metrics at a threshold set on the validation set for 90 % sensitivity;
* summary: mean +/- SD over runs (5 splits for the linear probe, 3 seeds for fine-tuning);
* on the reference split (seed 42, training seed 0): bootstrap 95 % CIs, pairwise exact
  McNemar and DeLong tests.
Writes results/*.csv
"""
import glob, os, itertools
import numpy as np, pandas as pd
from metrics import screening_metrics, bootstrap_ci, mcnemar, delong, threshold_for_sensitivity

R = '../results'; NAMES = {'vgg16': 'VGG16', 'resnet50': 'ResNet-50', 'vit_b16': 'ViT-B/16'}
KEYS = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc', 'balanced_accuracy']


def load(protocol):
    out = []
    for f in sorted(glob.glob(f'{R}/{protocol}/*_preds.csv')):
        b = os.path.basename(f).replace('_preds.csv', ''); parts = b.split('_split')
        model = parts[0]; rest = parts[1].split('_seed'); split = int(rest[0]); seed = int(rest[1]) if len(rest) > 1 else 0
        out.append((model, split, seed, pd.read_csv(f)))
    return out


rows, per_img = [], {}
for protocol in ('linear_probe', 'finetune'):
    for model, split, seed, d in load(protocol):
        v, t = d[d.split == 'val'], d[d.split == 'test']
        thr90 = threshold_for_sensitivity(v.label, v.p_cancer, 0.90)
        for thr_name, thr in (('0.5', 0.5), ('val_sens90', thr90)):
            m = screening_metrics(t.label, t.p_cancer, thr)
            rows.append({'protocol': protocol, 'model': NAMES[model], 'split_seed': split, 'train_seed': seed, 'threshold': thr_name, 'thr_value': thr, **m})
        per_img[(protocol, model, split, seed)] = t.sort_values('path')
runs = pd.DataFrame(rows); runs.to_csv(f'{R}/all_runs_test_metrics.csv', index=False)
summ = runs.groupby(['protocol', 'threshold', 'model'])[KEYS].agg(['mean', 'std', 'count']).round(4)
summ.columns = ['_'.join(c) for c in summ.columns]; summ.reset_index().to_csv(f'{R}/summary_mean_sd.csv', index=False)
print(summ.reset_index()[['protocol', 'threshold', 'model', 'accuracy_mean', 'accuracy_std', 'sensitivity_mean', 'specificity_mean', 'f1_mean', 'auc_mean', 'auc_std', 'auc_count']].to_string(index=False))

ci_rows, test_rows = [], []
for protocol in ('linear_probe', 'finetune'):
    ref = {m: per_img.get((protocol, m, 42, 0)) for m in NAMES}
    ref = {m: t for m, t in ref.items() if t is not None}
    for m, t in ref.items():
        point = screening_metrics(t.label, t.p_cancer); ci = bootstrap_ci(t.label.values, t.p_cancer.values)
        ci_rows.append({'protocol': protocol, 'model': NAMES[m], **{k: point[k] for k in ci}, **{k + '_lo': ci[k][0] for k in ci}, **{k + '_hi': ci[k][1] for k in ci}})
    for a, b in itertools.combinations(ref, 2):
        ta, tb = ref[a], ref[b]; assert (ta.path.values == tb.path.values).all()
        bb, cc, pm = mcnemar(ta.label.values, ta.p_cancer.values, tb.p_cancer.values)
        aa, ab, pd_ = delong(ta.label.values, ta.p_cancer.values, tb.p_cancer.values)
        test_rows.append({'protocol': protocol, 'model_A': NAMES[a], 'model_B': NAMES[b], 'A_right_B_wrong': bb, 'A_wrong_B_right': cc,
                          'mcnemar_p': pm, 'auc_A': aa, 'auc_B': ab, 'delong_p': pd_})
pd.DataFrame(ci_rows).to_csv(f'{R}/reference_split_bootstrap_ci.csv', index=False)
pd.DataFrame(test_rows).to_csv(f'{R}/reference_split_pairwise_tests.csv', index=False)
print(pd.DataFrame(test_rows).round(4).to_string(index=False))
