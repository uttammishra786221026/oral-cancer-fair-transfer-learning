"""Step 6 - visual explanations on test images of the reference split.

Grad-CAM (last convolutional block) for VGG16 and ResNet-50, attention rollout for
ViT-B/16. Uses the fine-tuned weights saved by finetune.py --save_model.
Writes figures/fig_explain.png and results/explain_cases.csv
"""
import argparse, os
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from models import Classifier
from data import EVAL_TF

p = argparse.ArgumentParser(); p.add_argument('--images', required=True); p.add_argument('--ft', default='../results/finetune')
p.add_argument('--n', type=int, default=4); a = p.parse_args()


def gradcam(m, x, layer):
    acts, grads = {}, {}
    def fwd(mod, i, o):
        acts['a'] = o; o.register_hook(lambda g: grads.__setitem__('g', g))
    for mod in m.modules():                     # in-place ReLUs would overwrite the hooked activations
        if isinstance(mod, torch.nn.ReLU): mod.inplace = False
    h1 = layer.register_forward_hook(fwd)
    m.zero_grad(); out = m(x); out[0, 1].backward(); h1.remove()
    w = grads['g'].mean((2, 3), keepdim=True); cam = Fn.relu((w * acts['a']).sum(1, keepdim=True))
    cam = Fn.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0].detach().numpy()
    return cam / (cam.max() + 1e-8), torch.softmax(out, 1)[0, 1].item()


def rollout(m, x):
    A = []
    hooks = [blk.attn.register_forward_hook(lambda mod, i, o: None) for blk in m.backbone.blocks]
    for h in hooks: h.remove()
    def attn_fwd(mod, inp, out):
        xin = inp[0]; B, N, C = xin.shape
        qkv = mod.qkv(xin).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]; att = (q @ k.transpose(-2, -1)) * mod.scale; A.append(att.softmax(-1).mean(1)[0].detach())
    hooks = [blk.attn.register_forward_hook(attn_fwd) for blk in m.backbone.blocks]
    with torch.no_grad(): out = m(x)
    for h in hooks: h.remove()
    R = torch.eye(A[0].shape[-1])
    for att in A:
        att = 0.5 * att + 0.5 * torch.eye(att.shape[-1]); att = att / att.sum(-1, keepdim=True); R = att @ R
    cam = R[0, 1:].reshape(14, 14).numpy(); cam = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), torch.softmax(out, 1)[0, 1].item()


models = {}
for name in ['vgg16', 'resnet50', 'vit_b16']:
    m = Classifier(name); m.load_state_dict(torch.load(f'{a.ft}/{name}_split42_seed0.pt', map_location='cpu')); models[name] = m.eval()
P = pd.read_csv(f'{a.ft}/resnet50_split42_seed0_preds.csv'); T = P[P.split == 'test']
pick = pd.concat([T[T.label == 1].sample(a.n // 2, random_state=1), T[T.label == 0].sample(a.n - a.n // 2, random_state=1)])
fig, ax = plt.subplots(len(pick), 4, figsize=(9.2, 2.35 * len(pick))); rows = []
for r, (_, row) in enumerate(pick.iterrows()):
    im = Image.open(os.path.join(a.images, row.path)).convert('RGB'); x = EVAL_TF(im).unsqueeze(0); show = np.asarray(im.resize((224, 224)))
    ax[r, 0].imshow(show); ax[r, 0].set_title(('Cancer' if row.label else 'Normal') + ' (ground truth)', fontsize=8)
    for c, name in enumerate(['vgg16', 'resnet50', 'vit_b16'], 1):
        m = models[name]
        if name == 'vgg16': cam, pc = gradcam(m, x.clone().requires_grad_(True), m.backbone.features[28])
        elif name == 'resnet50': cam, pc = gradcam(m, x.clone().requires_grad_(True), m.backbone.layer4)
        else: cam, pc = rollout(m, x)
        ax[r, c].imshow(show); ax[r, c].imshow(cam, cmap='jet', alpha=0.42)
        ax[r, c].set_title({'vgg16': 'VGG16', 'resnet50': 'ResNet-50', 'vit_b16': 'ViT-B/16'}[name] + f'  p(cancer)={pc:.2f}', fontsize=8)
        rows.append({'path': row.path, 'label': row.label, 'model': name, 'p_cancer': pc})
    for c in range(4): ax[r, c].axis('off')
fig.tight_layout(); os.makedirs('../figures', exist_ok=True); fig.savefig('../figures/fig_explain.png', dpi=300); pd.DataFrame(rows).to_csv('../results/explain_cases.csv', index=False)
print('saved')
