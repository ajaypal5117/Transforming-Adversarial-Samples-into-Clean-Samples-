# Adversarial Denoising with Feature-Guided Loss

This repository implements a deep learning pipeline for **denoising adversarial examples** using feature-aligned loss functions. The denoiser is trained to align intermediate representations between clean and adversarial inputs, extracted from a **frozen pretrained classifier** (MNIST or CIFAR-10). Additionally, it includes an adversarial sample detector based on the **dual-manifold approach**, leveraging Monte Carlo dropout uncertainties and deep feature representations.

---

## 🔧 Environment Setup

### 1. Python Version

* Requires Python **3.12** or higher.

### 2. Install Dependencies

```bash
# Create and activate a conda virtual environment
conda create -n keermada python=3.12
conda activate keermada

# Install requirements
pip install -r requirements.txt
```

---

## 📂 Repository Structure

### Purifier

```
.
├── adv_data/                       # Adversarial datasets (.pt)
│   ├── cifar10_fgsm_adv.pt
│   ├── mnist_pgd_adv.pt
│   └── ...
├── checkpoints/                    # Saved model checkpoints
│   ├── cifar10_FGSM_latest_model.pth
│   └── cifar10_FGSM_best_model.pth
├── classifier.py                   # CNN architectures & feature extractor
├── data_loader.py                  # Loads clean + adversarial datasets
├── logs/                           # Training & test logs
│   ├── cifar10_FGSM_train.out
│   └── cifar10_PGD_test.out
├── main.py                         # Entry point: train/test pipeline
├── model.py                        # Builds Net: classifier + denoiser + loss
├── saved_models/                   # Pretrained classifier weights
│   ├── mnist_classifier.pth
│   └── cifar10_classifier.pth
├── requirements.txt                # Python dependencies
└── train.py                        # Training, validation, testing logic
```

### Detector

* **detector\_training.py**

  * Loads models/data, generates noisy samples, extracts features & uncertainties
  * Computes dual-manifold scores and trains a logistic regression detector

* **util.py**

  * Data utilities: `get_data`, `get_noisy_samples`
  * Model loader: `get_model`
  * Feature extraction: `extract_features`, `get_deep_features`
  * Uncertainty estimation: `get_mc_uncertainties`
  * Scoring: `mahalanobis_distance`, `get_dual_manifold_scores`
  * Detection training: `train_lr`, `compute_roc`

---

## 📑 Dataset Format

* Each adversarial dataset must follow:

```
adv_data/{dataset}_{attack}_adv.pt
```

Where:

* `{dataset}`: `cifar10` or `mnist`
* `{attack}`: `fgsm`, `pgd`, `bim-a`, `bim-b`

Each `.pt` file contains:

```python
{
  "adv_train": {
    "clean":  Tensor[N, C, H, W],
    "adv":    Tensor[N, C, H, W],
    "labels": Tensor[N]
  },
  "adv_test": {
    "clean":  Tensor[N, C, H, W],
    "adv":    Tensor[N, C, H, W],
    "labels": Tensor[N]
  }
}
```

* **Pretrained classifier weights** must be placed under `saved_models/`:

```
saved_models/{dataset}_classifier.pth
```

Examples:

```
mnist_classifier.pth
cifar10_classifier.pth
```

These are auto-loaded by `model.py`.

---

## 🚀 Usage

### 1. Train Classifiers

```bash
python classifier_training.py --dataset cifar10
```

Models are saved in `saved_models/`.

### 2. Generate Adversarial Data

* **MNIST**: Download from [Google Drive](https://drive.google.com/drive/folders/1MOpH3OUdAZF1BHFwu4TTWQrNQx3s0yO4?usp=sharing), place in `adv_data/`.
* **CIFAR-10**: Generate using:

```bash
python adv_generation.py --dataset cifar10 --attack FGSM --model_path saved_models/cifar10_classifier.pth
```

### 3. Run Detector

Modify `train_detector.sh` for dataset/paths, then run:

```bash
bash train_detector.sh
```

### 4. Run Purifier

```bash
# For CIFAR-10
your-terminal$ ./run_denoiser_cifar10.sh

# For MNIST
your-terminal$ ./run_denoiser_mnist.sh
```

### 5. Run Integrated Detector–Purifier Model

```bash
bash run_integrated_model.sh
```

---

## 📊 Outputs

### Detector

After running detector, the following files are saved under `<save_dir>`:

```
saved_results/
├── mc_uncerts_normal_<dataset>_<attack>.npy
├── mc_uncerts_noisy_<dataset>_<attack>.npy
├── mc_uncerts_adv_<dataset>_<attack>.npy
├── deep_features_train_<dataset>_<attack>.npy
├── deep_features_test_normal_<dataset>_<attack>.npy
├── deep_features_test_noisy_<dataset>_<attack>.npy
├── deep_features_test_adv_<dataset>_<attack>.npy

lr_model_<dataset>_<attack>.pkl       # Trained logistic regression detector
manifolds_<dataset>_<attack>.pkl     # Clean & adversarial class manifolds
roc_curve_<attack>.png               # ROC curve plot
```

### Purifier

```
logs/
├── {dataset}_{attack}_train.out     # Training & validation metrics
├── {dataset}_{attack}_test.out      # Clean, adversarial, denoised accuracy

checkpoints/
├── {dataset}_{attack}_latest_model.pth   # Last epoch
├── {dataset}_{attack}_best_model.pth     # Best adversarial accuracy

final_test_{dataset}_{attack}.npz    # NumPy archive of final predictions
```

---

## 📝 Notes

* Supported datasets: **MNIST**, **CIFAR-10**
* Supported attacks: **FGSM**, **PGD**, **BIM-a**, **BIM-b**

---
