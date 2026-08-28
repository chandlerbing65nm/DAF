
# Towards Continual Test-Time Adaptation of Vision-Language Models in Open-Vocabulary Semantic Segmentation


## Requirements 
- [Python 3.10.13](https://www.python.org/)
- [PyTorch 2.1.2](https://pytorch.org/)
- [MMSegmentation 1.2.2](https://github.com/open-mmlab/mmsegmentation)


## Getting Started
### Step 1: Requirements
To run DAF, please install the following packages, and conda environment:

```bash
conda create -n daf python==3.10.13
conda activate daf
pip install "numpy<2"
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/rocm5.6
pip install -U openmim
mim install -r requirements.txt
```

> **Note:** All experiments were conducted on a single AMD Instinct MI200 GPU with ROCm 5.6.

---
### Step 2: Prepare Datasets

We evaluate DAF on PASCAL VOC 2012, using two class configurations:

- **PASCAL VOC 20** – The 20 foreground categories (background excluded).
- **PASCAL VOC 21** – The 20 foreground categories plus a challenging background label.

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. Please note that we only use the validation split of each dataset.

Additionally, inspired by [ImageNet-C](https://github.com/hendrycks/robustness), we generate 15 corruption types (e.g., noise, blur, weather, compression) *on-the-fly* at test time, allowing us to effectively evaluate each adaptation method's robustness to diverse distribution shifts. 

---
### Step 3: Perform Adaptation

DAF supports two adaptation methods:

- **TENT** – Entropy minimization over visual encoder LayerNorm parameters.
- **MLMP** – Multi-level multi-prompt optimization with entropy, diversity, and cross-modal anchor consistency losses.

The following example runs MLMP adaptation on PASCAL VOC 20:

```python
python main.py \
    --adapt \
    --method mlmp \
    --loss_ent True --lamb_ent 1.0 \
    --loss_div True --lamb_div 2.0 \
    --loss_cmac True --lamb_cmac 0.5 \
    --module_safs True --alpha_safs 0.5 \
    --prompt_dir prompts.yaml \
    --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 \
    --alpha_cls 1.0 \
    --ovss_type naclip \
    --ovss_backbone ViT-B/32 \
    --token_merge False --merge_type algm \
    --algm_layers 1 7 --algm_threshold 0.8 --algm_window_size 2 2 \
    --save_dir .save/PascalVOC20Dataset/dafm/ \
    --data_dir /path/to/VOCdevkit/VOC2012/ \
    --dataset PascalVOC20Dataset \
    --workers 4 \
    --init_resize 224 224 \
    --patch_size 224 224 \
    --patch_stride 112 \
    --corruptions_list gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression \
    --lr 1e-3 \
    --optimizer sgd  \
    --steps 1 \
    --batch-size 8 \
    --trials 1 \
    --seed "$SEED" \
    --plot_loss \
    --class_extensions \
    --reset_mode continual \
```

To use TENT instead, change `--method mlmp` to `--method tent` and remove the MLMP-specific arguments (`--vision_outputs`, `--alpha_cls`, `--loss_feat_cons`, `--feat_cons_type`). To evaluate on PASCAL VOC 21, change `--dataset PascalVOC20Dataset` to `--dataset PascalVOC21Dataset` and update `--save_dir` accordingly.

---
## License

This source code is released under the MIT license, which can be found [here](LICENCE). This project integrates elements from the following repositories; we gratefully acknowledge the authors for making their work open-source:
- [MLMP](https://github.com/dosowiechi/MLMP) (MIT licensed)
- [TENT](https://github.com/DequanWang/tent) (MIT licensed)
