"""Fig. 1 - overview of the leakage-safe comparison protocol (drawn with matplotlib)."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], 'font.size': 7})
fig, ax = plt.subplots(figsize=(3.5, 2.55)); ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis('off')
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.15', fc=c, ec='black', lw=0.6)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=6.3)
def arr(x1, y1, x2, y2): ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=0.7))
box(0.1, 6.1, 3.1, 1.1, '405 clinical photographs\n266 normal, 139 cancer', '#e8f0fb')
box(3.6, 6.1, 3.0, 1.1, 'dHash (256 bit)\nnear-duplicate groups', '#e8f0fb')
box(7.0, 6.1, 2.9, 1.1, 'stratified group split\n70/15/15, 5 seeds', '#e8f0fb')
arr(3.2, 6.65, 3.6, 6.65); arr(6.6, 6.65, 7.0, 6.65)
box(0.1, 4.2, 9.8, 1.2, 'identical for all models: same splits and augmentation (crop, flip, ±10°,\nbrightness/contrast), class-weighted loss, own input scaling applied once,\nmodel selection on validation ROC-AUC only', '#fff4e0')
arr(8.45, 6.1, 8.45, 5.4)
for i, (n, s) in enumerate([('VGG16', '134 M'), ('ResNet-50', '23.5 M'), ('ViT-B/16', '85.8 M')]):
    box(0.1 + i * 3.3, 2.55, 3.1, 1.1, f'{n} (ImageNet)\n{s} parameters', '#eaf6ea'); arr(1.65 + i * 3.3, 4.2, 1.65 + i * 3.3, 3.65)
box(0.1, 0.95, 4.75, 1.05, 'Mode A: frozen backbone\n+ logistic head (5 splits)', '#f6eaf6')
box(5.15, 0.95, 4.75, 1.05, 'Mode B: full fine-tuning\nAdamW, early stop (3 seeds)', '#f6eaf6')
arr(4.0, 2.55, 2.5, 2.0); arr(6.0, 2.55, 7.5, 2.0); ax.text(5.0, 2.25, 'each model, both modes', ha='center', fontsize=5.8, style='italic')
ax.text(5.0, 0.35, 'test set scored once: Sens, Spec, F1, ROC-AUC, bootstrap CI, McNemar, DeLong, Grad-CAM', ha='center', fontsize=6.3, style='italic')
arr(2.5, 0.95, 4.0, 0.55); arr(7.5, 0.95, 6.0, 0.55)
fig.tight_layout(pad=0.1); fig.savefig('../figures/fig_pipeline.png', dpi=300); print('ok')
