"""Step 7 - paper figures from the saved CSV files (300 dpi PNG, IEEE column width)."""
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

R, F = '../results', '../figures'; os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], 'font.size': 8,
                     'axes.linewidth': 0.8, 'savefig.dpi': 300, 'legend.frameon': False})
NAMES = {'vgg16': 'VGG16', 'resnet50': 'ResNet-50', 'vit_b16': 'ViT-B/16'}; COL = {'vgg16': '#1f77b4', 'resnet50': '#d62728', 'vit_b16': '#2ca02c'}

# Fig. 2 ROC curves on the reference test split, both protocols
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.85), sharey=True, gridspec_kw={'wspace': 0.12})
for a_, prot, title in ((ax[0], 'linear_probe', '(a) Frozen backbone'), (ax[1], 'finetune', '(b) Full fine-tuning')):
    for m in NAMES:
        f = f'{R}/{prot}/{m}_split42_preds.csv' if prot == 'linear_probe' else f'{R}/{prot}/{m}_split42_seed0_preds.csv'
        if not os.path.exists(f): continue
        d = pd.read_csv(f); t = d[d.split == 'test']; fpr, tpr, _ = roc_curve(t.label, t.p_cancer)
        from sklearn.metrics import auc as A
        a_.plot(fpr, tpr, color=COL[m], lw=1.1, label=f'{NAMES[m]} ({A(fpr, tpr):.3f})')
    a_.plot([0, 1], [0, 1], ':', color='gray', lw=0.7); a_.set_title(title, fontsize=8); a_.set_xlabel('False positive rate')
    a_.legend(fontsize=6, loc='lower right'); a_.set_xlim(0, 1); a_.set_ylim(0, 1.01); a_.set_xticks([0, 0.5, 1]); a_.set_xticklabels(['0', '0.5', '1'])
ax[0].set_ylabel('Sensitivity'); fig.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.88, wspace=0.18); fig.savefig(f'{F}/fig_roc.png'); plt.close(fig)

# Fig. 3 training curves (fine-tuning, reference split, seed 0)
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.7))
for m in NAMES:
    f = f'{R}/finetune/{m}_split42_seed0_history.csv'
    if not os.path.exists(f): continue
    h = pd.read_csv(f)
    ax[0].plot(h.epoch, h.train_loss, '--', color=COL[m], lw=0.9); ax[0].plot(h.epoch, h.val_loss, '-', color=COL[m], lw=1.1, label=NAMES[m])
    ax[1].plot(h.epoch, h.val_auc, '-o', color=COL[m], lw=1.0, ms=2, label=NAMES[m])
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss (dashed: train)'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Validation ROC-AUC')
ax[0].legend(fontsize=6); fig.tight_layout(pad=0.3); fig.savefig(f'{F}/fig_training.png'); plt.close(fig)

# Fig. 4 mean +/- SD of AUC, sensitivity, specificity over runs
runs = pd.read_csv(f'{R}/all_runs_test_metrics.csv'); runs = runs[runs.threshold == '0.5']
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
for a_, prot, title in ((ax[0], 'linear_probe', '(a) Frozen, 5 splits'), (ax[1], 'finetune', '(b) Fine-tuned, 3 seeds')):
    d = runs[runs.protocol == prot]; x = np.arange(3); w = 0.26
    for j, (k, lab) in enumerate((('auc', 'AUC'), ('sensitivity', 'Sens.'), ('specificity', 'Spec.'))):
        g = d.groupby('model')[k].agg(['mean', 'std']).reindex(list(NAMES.values()))
        a_.bar(x + (j - 1) * w, g['mean'], w, yerr=g['std'], capsize=1.5, label=lab, color=['#4c72b0', '#dd8452', '#55a868'][j], error_kw={'lw': 0.6})
    a_.set_xticks(x); a_.set_xticklabels(list(NAMES.values()), fontsize=6.5); a_.set_title(title, fontsize=8); a_.set_ylim(0.4, 1.05)
ax[0].set_ylabel('Test score'); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=6.5, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9)); fig.savefig(f'{F}/fig_bars.png'); plt.close(fig)

# Fig. 5 confusion matrices, fine-tuned, reference split
fig, ax = plt.subplots(1, 3, figsize=(3.5, 1.45))
for a_, m in zip(ax, NAMES):
    f = f'{R}/finetune/{m}_split42_seed0_preds.csv'
    if not os.path.exists(f): continue
    t = pd.read_csv(f); t = t[t.split == 'test']; cm = confusion_matrix(t.label, (t.p_cancer >= 0.5).astype(int), labels=[0, 1])
    a_.imshow(cm, cmap='Blues'); a_.set_title(NAMES[m], fontsize=7)
    for i in range(2):
        for j in range(2): a_.text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    a_.set_xticks([0, 1]); a_.set_xticklabels(['N', 'C'], fontsize=6); a_.set_yticks([0, 1]); a_.set_yticklabels(['N', 'C'], fontsize=6)
    a_.set_xlabel('Predicted', fontsize=6)
ax[0].set_ylabel('True', fontsize=6); fig.tight_layout(pad=0.5); fig.savefig(f'{F}/fig_confusion.png', bbox_inches='tight', pad_inches=0.03); plt.close(fig)
print('figures written')
