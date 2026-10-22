"""Step 3a - frozen-backbone features for the linear-probe protocol.

For every image: one plain (resized) view and --views augmented views, all passed
once through the frozen ImageNet backbone. The features are reused by every split.
Writes results/features/<model>.npz
"""
import argparse, os, time
import numpy as np, pandas as pd, torch
from PIL import Image
from models import Classifier
from data import TRAIN_TF, EVAL_TF

p = argparse.ArgumentParser()
p.add_argument('--model', required=True); p.add_argument('--images', required=True)
p.add_argument('--table', default='../data_splits/images.csv'); p.add_argument('--views', type=int, default=5)
p.add_argument('--out', default='../results/features'); a = p.parse_args()
if os.path.exists(os.path.join(a.out, a.model + '.npz')): print('already done:', a.model); raise SystemExit
torch.manual_seed(0); torch.set_num_threads(os.cpu_count()); os.makedirs(a.out, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T = pd.read_csv(a.table); m = Classifier(a.model).to(DEV).eval()
plain, aug = [], []; t0 = time.time()
with torch.no_grad():
    for k, r in T.iterrows():
        im = Image.open(os.path.join(a.images, r.path)).convert('RGB')
        x = torch.stack([EVAL_TF(im)] + [TRAIN_TF(im) for _ in range(a.views)])
        f = m.features(x.to(DEV)).cpu().numpy().astype(np.float32); plain.append(f[0]); aug.append(f[1:])
        if k % 50 == 0: print(a.model, k, '%.1f s' % (time.time() - t0), flush=True)
np.savez_compressed(os.path.join(a.out, a.model + '.npz'), path=T.path.values, label=T.label.values, plain=np.array(plain), aug=np.array(aug))
print('DONE', a.model, '%.1f s' % (time.time() - t0))
