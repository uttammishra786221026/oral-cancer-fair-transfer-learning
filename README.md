# Oral cancer screening from clinical photographs: a fair comparison of VGG16, ResNet-50 and ViT-B/16

This repository contains everything needed to reproduce the experiments of the paper

> **Fair Comparison of VGG16, ResNet-50 and Vision Transformer for Oral Cancer Screening from Clinical Photographs under a Leakage-Safe Protocol**
> Uttam Mishra. Submitted to the 16th IEEE International Conference on Communication Systems and Network Technologies (CSNT 2027).

It holds the code, the exact data splits, every per-image prediction, the result tables (CSV) and the figures used in the paper.

---

## 1. What the study does, in short

Three ImageNet-pretrained networks are compared on one task: deciding whether a clinical photograph of the mouth shows **oral cancer** or **normal mucosa**.

| Backbone | Family | Parameters (without the head) |
|---|---|---|
| VGG16 | plain deep CNN | 134.3 M |
| ResNet-50 | residual CNN | 23.5 M |
| ViT-B/16 | vision transformer | 85.8 M |

The point of the study is **fairness**. All three models get the same images, the same splits, the same augmentation, the same class weighting and the same rule for choosing the final model (validation data only). The test images are used **once**, at the very end.

Two training modes are run for every model:

* **Mode A – frozen backbone (linear probe):** the pretrained network is not changed; only a new two-class head is trained. Repeated on 5 different splits.
* **Mode B – full fine-tuning:** all weights are trained. Repeated with 3 training seeds on the reference split (seed 42).

## 2. Headline results

| Mode | Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1 (%) | ROC-AUC |
|---|---|---|---|---|---|---|
| A: frozen (5 splits) | VGG16 | 80.9 ± 6.8 | 75.9 ± 6.2 | 83.5 ± 8.0 | 73.4 ± 8.6 | 0.883 ± 0.053 |
| A: frozen (5 splits) | ResNet-50 | 91.1 ± 3.1 | 82.7 ± 5.5 | 95.5 ± 3.7 | 86.4 ± 4.6 | 0.978 ± 0.016 |
| A: frozen (5 splits) | ViT-B/16 | 92.4 ± 2.5 | 92.4 ± 7.2 | 92.5 ± 2.5 | 89.2 ± 3.8 | 0.983 ± 0.010 |
| B: fine-tuned (3 seeds) | VGG16 | 80.0 ± 10.4 | 96.7 ± 5.8 | 71.7 ± 17.7 | 77.2 ± 8.3 | 0.980 ± 0.005 |
| B: fine-tuned (3 seeds) | ResNet-50 | 96.7 ± 0.0 | 98.3 ± 2.9 | 95.8 ± 1.4 | 95.2 ± 0.1 | 0.993 ± 0.005 |
| B: fine-tuned (3 seeds) | ViT-B/16 | 98.9 ± 1.0 | 100.0 ± 0.0 | 98.3 ± 1.4 | 98.4 ± 1.4 | 1.000 ± 0.000 |

Mean ± standard deviation on the test partitions, threshold 0.5.

Full tables: [`results/summary_mean_sd.csv`](results/summary_mean_sd.csv), [`results/reference_split_bootstrap_ci.csv`](results/reference_split_bootstrap_ci.csv), [`results/reference_split_pairwise_tests.csv`](results/reference_split_pairwise_tests.csv), [`results/all_runs_test_metrics.csv`](results/all_runs_test_metrics.csv).

## 3. Step-by-step: how the work was done

### Step 1 – Data source
* Public GitHub repository **Shaz-5/oral-cancer-classification** (MIT licence): <https://github.com/Shaz-5/oral-cancer-classification>. Its authors state that the photographs were provided by a hospital.
* Only the two original (non-augmented) folders were used: `Content_3class_new/normal` (266 images) and `Content_3class_new/oral_cancer` (139 images), **405 images** in total.
* The `pre_cancer` folder and all `augmented_*` folders were **not** used. Augmented copies of the same photographs would leak into the test set.

### Step 2 – Removing near-duplicate leakage (`src/prepare_data.py`)
* Every image is turned into a 256-bit *difference hash* (a compact fingerprint of its brightness pattern).
* Two images whose fingerprints differ in at most 20 of the 256 bits are treated as copies of the same photograph.
* Copies are joined into groups. Result: **25 near-duplicate pairs**, giving 24 groups that hold 49 images. No pair mixed the two classes.
* Every group is kept inside one partition, so a photograph and its copy can never be split between training and testing.

### Step 3 – Splitting (`src/prepare_data.py`)
* Stratified, group-aware 70 % / 15 % / 15 % split into train / validation / test, drawn separately for each class.
* Five splits with seeds 42–46 are saved in [`data_splits/`](data_splits). Seed 42 is the *reference split*: 284 train, 61 validation and 60 test images.

### Step 4 – Pretrained weights (`src/models.py`)
| Model | File | Source |
|---|---|---|
| VGG16 | `vgg16_weights_tf_dim_ordering_tf_kernels.h5` | official Keras release, <https://github.com/fchollet/deep-learning-models/releases/tag/v0.1> (converted to PyTorch by `load_vgg16_keras`) |
| ResNet-50 | `resnet50_ram-a26f946b.pth` | timm release, <https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-weights> |
| ViT-B/16 | `jx_vit_base_p16_224-80ecf9dd.pth` | timm release of the original Google ImageNet-21k→1k ViT (same weights as `google/vit-base-patch16-224`), <https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-vitjx> |

Each model applies the input scaling that its own pretraining used. The scaling is applied **once**, inside the model, so there is no double rescaling. The VGG16 conversion was checked: on sample images the converted network and ResNet-50 gave the same ImageNet top-1 class.

### Step 5 – Mode A, frozen backbone (`src/extract_features.py`, `src/linear_probe.py`)
* Each image passes once in plain form and 5 times with random augmentation through the frozen backbone. The features are stored.
* A logistic-regression head (balanced class weights) is trained on the augmented training features. Its regularisation strength is chosen from 8 values using **validation ROC-AUC only**.
* Repeated for all 5 splits.

### Step 6 – Mode B, full fine-tuning (`src/finetune.py`)
* AdamW, weight decay 1e-4, batch size 16, at most 10 epochs, early stopping after 3 epochs without a better validation ROC-AUC.
* Learning rate 1e-4 for VGG16 and ResNet-50, and 2e-5 for ViT-B/16.
* Class-weighted cross-entropy loss.
* Same augmentation for all models: random resized crop (80–100 %), horizontal flip, ±10° rotation, ±15 % brightness and contrast. Hue and saturation are not changed, because mucosal colour matters clinically.
* Each batch of 16 is processed as 2 micro-batches of 8 (gradient accumulation) to fit in 8 GB of RAM. VGG16 and ViT-B/16 have no batch normalisation, so this gives the same update as one batch of 16. ResNet-50 was run with whole batches of 16.
* The best validation epoch is kept, and the test set is scored once.

### Step 7 – Evaluation (`src/metrics.py`, `src/evaluate.py`)
* Cancer = positive class. Reported: accuracy, sensitivity, specificity, precision, F1, ROC-AUC and balanced accuracy.
* Two thresholds: the fixed value 0.5, and a *screening* threshold chosen on validation data to reach 90 % sensitivity.
* On the reference split: 95 % bootstrap confidence intervals (2000 resamples), the exact McNemar test (accuracy) and the DeLong test (ROC-AUC) for every pair of models.

### Step 8 – Figures and explanations (`src/make_figures.py`, `src/explain.py`)
* ROC curves, training curves, mean ± SD bars and confusion matrices.
* Grad-CAM for the CNNs and attention rollout for ViT-B/16, drawn on test images.

## 4. How to reproduce

```bash
git clone https://github.com/uttammishra786221026/oral-cancer-fair-transfer-learning.git
cd oral-cancer-fair-transfer-learning
pip install -r requirements.txt

# data (MIT licence) - only the two original folders are used
git clone https://github.com/Shaz-5/oral-cancer-classification.git

# pretrained weights (put the three files in ./weights)
mkdir weights && cd weights
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
wget https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth
wget https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
cd ..

bash run_all.sh oral-cancer-classification/Content_3class_new weights
```

A CPU is enough. On a 2-core machine, Mode A takes about 30 minutes and Mode B about 4 hours. A GPU is much faster.

## 5. Repository layout

```
src/prepare_data.py      step 2-3  duplicate grouping and splits
src/models.py            step 4    the three backbones (+ Keras->PyTorch VGG16 converter)
src/data.py              shared augmentation and image loading
src/extract_features.py  step 5    frozen features
src/linear_probe.py      step 5    Mode A heads
src/finetune.py          step 6    Mode B training
src/metrics.py           step 7    metrics, bootstrap, McNemar, DeLong
src/evaluate.py          step 7    tables
src/make_figures.py      step 8    figures
src/explain.py           step 8    Grad-CAM / attention rollout
data_splits/             images.csv (hash groups) + split_seed42..46.csv
results/                 per-image predictions, training histories, all metric tables
figures/                 all figures of the paper (300 dpi)
logs/                    console output of every run
```

## 6. Limitations

* One small, single-source dataset (405 images, no patient IDs, likely one clinic). No external test set.
* The labels come from the source repository; histopathology confirmation is not documented.
* Results are for research only. This is **not** a medical device and must not be used for diagnosis.

## 7. Licence

The code is released under the MIT licence (see `LICENSE`). The images belong to the original dataset authors (MIT licence). They are not redistributed here; download them from the source repository.
