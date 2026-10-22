#!/usr/bin/env bash
# One command for a fresh GPU machine (Colab / Kaggle / lab server), run from this folder:
#   bash GPU_RUN.sh
set -e
pip install -q -r requirements.txt
[ -d oral-cancer-classification ] || git clone --depth 1 https://github.com/Shaz-5/oral-cancer-classification.git
mkdir -p weights && cd weights
[ -f vgg16_weights_tf_dim_ordering_tf_kernels.h5 ] || wget -q https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
[ -f resnet50_ram-a26f946b.pth ] || wget -q https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth
[ -f jx_vit_base_p16_224-80ecf9dd.pth ] || wget -q https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
cd ..
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
bash run_all.sh "$PWD/oral-cancer-classification/Content_3class_new" "$PWD/weights"
