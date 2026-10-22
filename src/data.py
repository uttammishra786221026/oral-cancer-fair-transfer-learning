"""Image loading and the single augmentation policy shared by all models."""
import os, numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

SIZE = 224
# same policy for every model: flips, small rotation, mild brightness/contrast.
# Hue and saturation are NOT changed because mucosal colour carries diagnostic information.
TRAIN_TF = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.80, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
                      T.RandomRotation(10), T.ColorJitter(brightness=0.15, contrast=0.15), T.ToTensor()])
EVAL_TF = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])   # values in [0,1]; each model normalises internally


class OralImages(Dataset):
    def __init__(self, root, table, train):
        self.root, self.t, self.tf = root, table.reset_index(drop=True), TRAIN_TF if train else EVAL_TF
    def __len__(self):
        return len(self.t)
    def __getitem__(self, i):
        r = self.t.iloc[i]; im = Image.open(os.path.join(self.root, r.path)).convert('RGB')
        return self.tf(im), int(r.label)


def load_split(split_dir, seed):
    return pd.read_csv(os.path.join(split_dir, f'split_seed{seed}.csv'))
