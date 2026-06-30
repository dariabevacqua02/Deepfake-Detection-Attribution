# 🕵️ Deepfake Detection & Attribution

A deep learning pipeline that not only detects whether a face image is **real or AI-generated**, but also **attributes** a fake image to the specific generative model that produced it (Midjourney, DALL·E 3, Stable Diffusion, etc.). Six CNN/Transformer architectures are benchmarked under an identical transfer-learning pipeline, and an **open-set / out-of-distribution (OOD) rejection mechanism** is added on top via a model-agnostic dynamic confidence threshold.

---

## Overview

The task is framed as an **11-class closed-set classification problem**: 1 "Real" class (genuine human faces) vs. 10 "Fake" classes, one per generative model. On top of the standard closed-set classifier, a **dynamic confidence threshold** turns the system into an **open-set detector**: predictions the model isn't confident about are flagged as `UNKNOWN` instead of being forced into one of the 11 known classes — crucial for a domain like deepfake detection, where new generators appear constantly and a deployed model will inevitably encounter generators it has never seen during training.

## Key Features

- **Multi-architecture benchmark**: ResNet18, DenseNet121, VGG16-BN, InceptionV3, MobileNetV3-Large, and ViT-B/16, all pretrained on ImageNet.
- **Two-phase transfer learning**: backbone frozen for the first epochs (head-only warm-up), then fully unfrozen with a 10× lower learning rate — stabilizes training and avoids catastrophic forgetting of pretrained features.
- **High-throughput data pipeline**: images are pre-resized and cached to disk once (multi-threaded), so training reads small, uniform JPEGs instead of doing expensive on-the-fly resizing every epoch.
- **Aggressive augmentation**: random crop, flips, rotation, color jitter, random grayscale, Gaussian blur, and random erasing — designed to discourage the network from latching onto trivial, generator-specific artifacts (compression, color profile, resolution) instead of genuine generative fingerprints.
- **Dynamic confidence threshold (open-set detection)**: a per-model, data-driven threshold rejects low-confidence predictions as `UNKNOWN`, without retraining or architectural changes — see the dedicated section below.
- **Automated model comparison**: ranked metrics tables, grouped bar charts, per-class F1 heatmaps, and an auto-generated textual summary of the best/worst performer.

## Dataset

- **Real**: 1,000 images sampled from [CelebA-HQ](https://www.kaggle.com/datasets/matteospata/celeba-hq) (256×256 high-quality face crops).
- **Fake**: 1,000 images per generator from a closed-set deepfake attribution dataset, covering 10 modern image generators:

  `Adobe Firefly · DALL·E 3 · Flux.1 · Flux.1.1 Pro · Freepik · Leonardo AI · Midjourney · Stable Diffusion 3.5 · Stable Diffusion XL · Starry AI`

All 11 classes are perfectly balanced (1,000 images each) and split **70% / 15% / 15%** into train / validation / test (7,700 / 1,650 / 1,650 images). Images are pre-processed once into a 256×256 JPEG cache to remove I/O and resizing overhead from the training loop.

> ⚠️ The dataset is **not included** in this repository. Source it from CelebA-HQ and the relevant Kaggle deepfake-attribution dataset, then update `CELEBA_HQ_DIR` and `FAKE_DIR` in the notebook.

## Methodology

1. **Dataset preparation** — gathers real and fake images, performs a stratified 70/15/15 split per class, and copies files into `train/val/test` folders.
2. **Caching** — every image is resized to 256×256 and re-saved as JPEG (quality 95) using a thread pool, so all downstream epochs read small, pre-decoded files.
3. **EDA** — class balance, image dimension distributions, and per-channel RGB statistics across real vs. fake classes.
4. **Data loading & augmentation** — standard 224×224 pipeline for CNNs (`RandomCrop`, flips, rotation, color jitter, random erasing) and a dedicated 299×299 pipeline for InceptionV3, which has stricter input-size requirements.
5. **Two-phase training** — phase 1 trains only the new classification head (backbone frozen); phase 2 unfreezes the full network at `LR/10`. Both phases use AdamW, cosine annealing, mixed-precision (AMP), gradient clipping, and label smoothing.
6. **Closed-set evaluation** — accuracy, weighted precision/recall/F1, confusion matrices, and per-class classification reports on the 11-class test set.
7. **Open-set detection** — a dynamic confidence threshold (see below) is calibrated on the validation set and applied to the test set, splitting predictions into `KNOWN` (confidently classified into one of the 11 classes) and `UNKNOWN` (rejected).
8. **Model comparison** — all six models are ranked by F1-score, with grouped bar charts, a metrics heatmap, and an automatically generated interpretive summary.

## Why a Dynamic Confidence Threshold?

A classifier trained on a fixed, closed set of classes has no built-in notion of "I've never seen anything like this." When a new generator appears — one that wasn't in the training data — a standard softmax classifier will still confidently assign it to one of the known classes, because softmax outputs always sum to 1 regardless of how unfamiliar the input is. This project addresses that gap with a deliberately simple, cheap-to-compute rejection rule:

```
threshold = mean(max_softmax_prob) − k · std(max_softmax_prob)     (k = 0.5, computed on the validation set)
```

A prediction is accepted only if its maximum softmax probability is above this threshold; otherwise it is labeled `UNKNOWN`.

**Why relative to the mean and std, instead of a fixed value like 0.9?** Different architectures are calibrated very differently — in this project alone, ViT's average confidence (0.893, σ = 0.076) is both higher *and* tighter than MobileNetV3's (0.831, σ = 0.164). A single fixed cutoff would be too strict for some models and too lax for others. By anchoring the threshold to each model's *own* confidence distribution, the rule self-calibrates per architecture: confident, well-clustered models naturally get a tighter, higher threshold, while more uncertain or noisier models get a comparatively lower one — without any manual tuning per model.

**Why this is a practical alternative to Bayesian Neural Networks.** Proper uncertainty quantification — Bayesian Neural Networks, MC-Dropout, or Deep Ensembles — estimates a full predictive distribution by sampling multiple stochastic forward passes (often 20–100 per prediction) or by training and maintaining several independent models. That multiplies both training and inference cost by the number of samples/members, typically requires specialized variational layers or ensemble infrastructure, and complicates an already resource-constrained training run (this project trains six full networks on two T4 GPUs as it is). The dynamic threshold approach, by contrast:

- Reuses the **existing, already-trained softmax output** — no architectural changes, no retraining, no extra parameters.
- Needs only **one calibration pass** over the validation set to compute mean and std, and a single comparison at inference time — the same cost as a normal forward pass.
- Is **fully model-agnostic**: it works identically for a 12M-parameter MobileNet or an 86M-parameter ViT.

The trade-off is honest: max-softmax confidence is a weaker, less theoretically grounded uncertainty signal than a true Bayesian posterior, and it is known to sometimes be overconfident on adversarial or pathological out-of-distribution inputs. But for a practical open-set deepfake detector — where the goal is a fast, deployable, per-model-calibrated safety net rather than a formally calibrated probability — it captures most of the practical benefit of Bayesian uncertainty estimation at a tiny fraction of the computational cost.

## Results (Closed-Set, Test Set)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **ViT-B/16** | **98.55%** | **98.58%** | **98.55%** | **98.55%** |
| VGG16-BN | 98.48% | 98.50% | 98.48% | 98.48% |
| DenseNet121 | 97.76% | 97.81% | 97.76% | 97.76% |
| InceptionV3 | 97.70% | 97.80% | 97.70% | 97.71% |
| ResNet18 | 97.33% | 97.36% | 97.33% | 97.34% |
| MobileNetV3-Large | 96.79% | 96.84% | 96.79% | 96.79% |

ViT-B/16 achieves the best overall performance, closely followed by VGG16-BN. The lighter MobileNetV3 trades some accuracy for efficiency, as expected.

### Open-Set Detection (k = 0.5)

| Model | Threshold | UNKNOWN (test) | Accuracy (closed-set) | Accuracy on accepted (KNOWN) |
|---|---|---|---|---|
| ViT-B/16 | 0.8547 | 188 (11.4%) | 98.55% | 99.93% |
| InceptionV3 | 0.8456 | 173 (10.5%) | 97.70% | 99.53% |
| DenseNet121 | 0.8289 | 331 (20.1%) | 97.76% | 99.85% |
| VGG16-BN | 0.8164 | 310 (18.8%) | 98.48% | 100.00% |
| ResNet18 | 0.7845 | 376 (22.8%) | 97.33% | 99.84% |
| MobileNetV3-Large | 0.7493 | 384 (23.3%) | 96.79% | 99.68% |

Rejecting low-confidence predictions consistently pushes accuracy on the accepted subset above **99.5%** for every model, confirming that the rejected images are concentrated where the model is genuinely most error-prone — exactly the behavior an OOD rejection mechanism should exhibit.

## Project Structure

```
.
├── Deepfake_Detection_Attribution.ipynb   # main notebook (full pipeline)
├── README.md
└── outputs/                                # generated on run: EDA plots, curves, confusion matrices, threshold analysis
```

## Getting Started

### Requirements

- Python 3.10+
- A CUDA-capable GPU (developed and benchmarked on 2× Tesla T4, 16 GB VRAM each)
- PyTorch 2.x

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow
```

### Usage

1. Download CelebA-HQ and the deepfake-attribution dataset (10 generator classes).
2. Update `CELEBA_HQ_DIR` and `FAKE_DIR` in the notebook's setup cell to point to your local copies.
3. Run the notebook top to bottom: `jupyter notebook Deepfake_Detection_Attribution.ipynb`.
4. To reuse trained weights without retraining, point the evaluation cells to the `.pth` checkpoints saved under `MODEL_DIR`.

## Tech Stack

`PyTorch` · `torchvision` · `scikit-learn` · `pandas` / `NumPy` · `Matplotlib` / `Seaborn` · `ThreadPoolExecutor` (parallel pre-caching)

## Acknowledgments

- Real face images: [CelebA-HQ](https://www.kaggle.com/datasets/matteospata/celeba-hq).
- Fake images spanning 10 generative models from a closed-set deepfake attribution dataset on Kaggle.
- Pretrained backbones from `torchvision.models` (ImageNet weights).

## License

This project is released under the MIT License. The underlying datasets retain their own licenses — please refer to their original sources for terms of use.

## Disclaimer

This project is for **research and educational purposes only**. Deepfake detection models can fail silently on generators not seen during training; the `UNKNOWN` mechanism mitigates but does not eliminate this risk. The model should not be relied upon as the sole basis for high-stakes authenticity decisions.
