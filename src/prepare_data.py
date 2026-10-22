"""Step 1 - build a leakage-safe binary dataset (oral cancer vs normal).

* reads the two original (non-augmented) class folders of the public
  Shaz-5/oral-cancer-classification repository (MIT licence);
* finds near-duplicate photographs with a 256-bit difference hash (dHash)
  and joins them into groups, so copies of one photograph never fall into
  different partitions;
* draws five stratified, group-aware 70/15/15 splits (seeds 42-46).

Output: data_splits/images.csv (one row per image) and
        data_splits/split_seed<S>.csv (image -> train/val/test).
"""
import argparse, glob, os
import numpy as np, pandas as pd
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--src', required=True, help='.../Content_3class_new')
p.add_argument('--out', default='data_splits')
p.add_argument('--hash_threshold', type=int, default=20, help='max differing bits (of 256) to call two images duplicates')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
CLASSES = {'normal': 0, 'oral_cancer': 1}


def dhash(path, n=16):
    g = np.asarray(Image.open(path).convert('L').resize((n + 1, n), Image.LANCZOS), dtype=float)
    return (g[:, 1:] > g[:, :-1]).flatten()


rows = []
for c, y in CLASSES.items():
    for f in sorted(glob.glob(os.path.join(a.src, c, '*'))):
        im = Image.open(f)
        rows.append({'path': os.path.relpath(f, a.src), 'label': y, 'class': c, 'width': im.size[0], 'height': im.size[1]})
df = pd.DataFrame(rows)
H = np.array([dhash(os.path.join(a.src, p_)) for p_ in df.path])
D = (H[:, None, :] != H[None, :, :]).sum(-1)

# union-find over pairs closer than the threshold
parent = list(range(len(df)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
pairs = 0; cross = 0
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if D[i, j] <= a.hash_threshold:
            pairs += 1; cross += int(df.label[i] != df.label[j]); parent[find(i)] = find(j)
df['group'] = [find(i) for i in range(len(df))]
df['group'] = df.group.astype('category').cat.codes
assert cross == 0, 'a duplicate pair has two different labels'
df.to_csv(os.path.join(a.out, 'images.csv'), index=False)
gsz = df.groupby('group').size()
print(f'{len(df)} images ({(df.label == 0).sum()} normal, {(df.label == 1).sum()} cancer); '
      f'{pairs} near-duplicate pairs, {len(gsz)} groups, {int((gsz > 1).sum())} groups with >1 image, '
      f'{int(gsz[gsz > 1].sum())} images in such groups; cross-class duplicates: {cross}')


def split(seed, fr=(0.70, 0.15, 0.15)):
    """stratified group split: groups are shuffled per class and filled into
    train/val/test until each partition holds its share of that class."""
    rng = np.random.default_rng(seed); out = {}
    for y in (0, 1):
        g = df[df.label == y].groupby('group').size()
        order = rng.permutation(g.index.values); n = g.sum()
        target = np.array(fr) * n; fill = np.zeros(3)
        for gid in order:
            k = int(np.argmax(target - fill))           # partition with the largest remaining deficit
            fill[k] += g[gid]; out[gid] = ['train', 'val', 'test'][k]
    s = df[['path', 'label', 'group']].copy(); s['split'] = s.group.map(out)
    return s


for seed in range(42, 47):
    s = split(seed); s.to_csv(os.path.join(a.out, f'split_seed{seed}.csv'), index=False)
    t = pd.crosstab(s.split, s.label)
    assert s.groupby('group').split.nunique().max() == 1
    print(f'seed {seed}:', {k: (int(t.loc[k, 0]), int(t.loc[k, 1])) for k in ('train', 'val', 'test')})
