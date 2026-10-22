"""Step 2 - the three ImageNet-pretrained backbones used in the paper.

All three take a 224x224 RGB image and end in a new two-way linear layer.
Weights are loaded from local files (see README "Pretrained weights"):
  VGG16       Keras ImageNet weights (vgg16_weights_tf_dim_ordering_tf_kernels.h5),
              converted here to PyTorch; Caffe-style input (BGR, mean subtraction)
  ResNet-50   timm ImageNet-1k weights (resnet50_ram-a26f946b.pth); ImageNet mean/std
  ViT-B/16    timm/Google ImageNet-21k -> 1k weights (jx_vit_base_p16_224-80ecf9dd.pth),
              the same checkpoint as google/vit-base-patch16-224; mean = std = 0.5
"""
import os
import numpy as np, torch, torch.nn as nn, timm

WEIGHTS = os.environ.get('OC_WEIGHTS', 'weights')
FEAT_DIM = {'vgg16': 4096, 'resnet50': 2048, 'vit_b16': 768}


class CaffeNorm(nn.Module):
    """input in [0,1] RGB -> Keras 'caffe' preprocessing (BGR, 0-255, mean removed)"""
    def __init__(self):
        super().__init__(); self.register_buffer('m', torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1))
    def forward(self, x):
        return x[:, [2, 1, 0]] * 255.0 - self.m


class Norm(nn.Module):
    def __init__(self, mean, std):
        super().__init__(); self.register_buffer('m', torch.tensor(mean).view(1, 3, 1, 1)); self.register_buffer('s', torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, x):
        return (x - self.m) / self.s


class VGG16Keras(nn.Module):
    """torch re-implementation of keras.applications.VGG16 (include_top up to fc2)."""
    def __init__(self):
        super().__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        L, c = [], 3
        for v in cfg:
            if v == 'M': L.append(nn.MaxPool2d(2, 2))
            else: L += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]; c = v
        self.features = nn.Sequential(*L)
        self.fc = nn.Sequential(nn.Linear(25088, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 4096), nn.ReLU(inplace=True))

    def forward_features(self, x):
        x = self.features(x)                          # N,512,7,7
        x = x.permute(0, 2, 3, 1).flatten(1)          # Keras Flatten order (H, W, C)
        return self.fc(x)


def load_vgg16_keras(path):
    import h5py
    m = VGG16Keras(); f = h5py.File(path, 'r')
    convs = [l for l in m.features if isinstance(l, nn.Conv2d)]
    names = [f'block{b}_conv{i}' for b, n in zip(range(1, 6), (2, 2, 3, 3, 3)) for i in range(1, n + 1)]
    def g(name):
        grp = f[name] if name in f else f['model_weights'][name]
        ks = sorted(grp.keys()); grp = grp[ks[0]] if len(ks) == 1 and isinstance(grp[ks[0]], h5py.Group) else grp
        w = [k for k in grp.keys() if '_W' in k or 'kernel' in k][0]; b = [k for k in grp.keys() if '_b' in k or 'bias' in k][0]
        return np.array(grp[w]), np.array(grp[b])
    for conv, n in zip(convs, names):
        w, b = g(n); conv.weight.data = torch.from_numpy(w.transpose(3, 2, 0, 1).copy()); conv.bias.data = torch.from_numpy(b)
    for lin, n in zip([m.fc[0], m.fc[2]], ['fc1', 'fc2']):
        w, b = g(n); lin.weight.data = torch.from_numpy(w.T.copy()); lin.bias.data = torch.from_numpy(b)
    return m


class Classifier(nn.Module):
    def __init__(self, name, n_classes=2):
        super().__init__(); self.name = name
        if name == 'vgg16':
            self.norm = CaffeNorm(); self.backbone = load_vgg16_keras(os.path.join(WEIGHTS, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
        elif name == 'resnet50':
            self.norm = Norm([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)
            sd = torch.load(os.path.join(WEIGHTS, 'resnet50_ram-a26f946b.pth'), map_location='cpu')
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False); assert not missing, missing
        elif name == 'vit_b16':
            self.norm = Norm([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
            sd = torch.load(os.path.join(WEIGHTS, 'jx_vit_base_p16_224-80ecf9dd.pth'), map_location='cpu')
            sd = {k: v for k, v in sd.items() if not k.startswith('head')}
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False); assert not missing, missing
        else:
            raise ValueError(name)
        self.head = nn.Linear(FEAT_DIM[name], n_classes)

    def features(self, x):
        x = self.norm(x)
        return self.backbone.forward_features(x) if self.name == 'vgg16' else self.backbone(x)

    def forward(self, x):
        return self.head(self.features(x))
