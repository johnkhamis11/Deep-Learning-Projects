# CIFAR-10 Image Classification with a CNN and Data Augmentation

Classifying 32×32 colour images into **10 classes** with convolutional neural networks. The project compares a simple baseline CNN against an improved CNN that uses data augmentation, batch normalization and dropout.

📓 Notebook: `CIFAR10_CNN_Data_Augmentation.ipynb`

---

## Dataset

- **Source:** CIFAR-10 via `keras.datasets.cifar10`, downloaded automatically.
- **Size:** 50,000 training and 10,000 test images (32×32×3).
- **Classes (balanced):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- **Split:** 45,000 train / 5,000 validation (stratified) / 10,000 test.

## Workflow

1. Load the data and inspect shapes, pixel range and class balance.
2. Visualize sample images and the class distribution.
3. Normalize pixels to the 0–1 range.
4. Train a **baseline CNN**: 2 × (Conv2D + MaxPool) → `Dense(64)` → softmax, about 168K parameters.
5. Train an **improved CNN**:
   - Augmentation layers: `RandomFlip('horizontal')` and `RandomRotation(0.05)`
   - 3 × (Conv2D + BatchNormalization + MaxPool + Dropout 0.25), with 32 → 64 → 128 filters
   - `Dense(128) → Dropout(0.5) → Dense(10, softmax)`, about 358K parameters
6. Compare training and validation curves for both models.
7. Evaluate the improved CNN with a per-class classification report, confusion matrix and a gallery of misclassified images.

Both models use Adam, sparse categorical cross-entropy and early stopping on validation loss.

## Results (test set, 10,000 images)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Baseline CNN | 0.671 | 0.679 | 0.671 | 0.670 |
| **Improved CNN** | **0.763** | **0.768** | **0.763** | **0.763** |

**Takeaway:** the improved architecture adds about **9 percentage points** of accuracy. The baseline's training accuracy keeps climbing while its validation accuracy stalls, a sign of overfitting that augmentation and regularization reduce.

**Per-class behaviour (improved CNN):**

- Easiest classes: **automobile** (F1 0.89), **ship** (0.87) and **truck** (0.86)
- Hardest classes: **cat** (F1 0.59), then **bird**, **dog** and **deer** (about 0.67 to 0.68)

Cats and dogs are the most commonly confused animals, as expected at 32×32 resolution.

## Saved artifacts

| File | Description |
|---|---|
| `cifar10_cnn.keras` | Trained improved CNN |
| `class_names.pkl` | List of class names in label order |

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
jupyter notebook CIFAR10_CNN_Data_Augmentation.ipynb
```

Run all cells in order. The dataset (about 170 MB) is downloaded on the first run. A GPU is recommended but not required. The random seed is fixed (`SEED = 42`).

## Limitations and ideas

- 76% is a solid result for a small from-scratch CNN, but state-of-the-art models exceed 95%.
- Possible next steps: stronger augmentation (`RandomCrop`, `RandomContrast`, Mixup/CutMix), deeper VGG-style or ResNet blocks, a learning-rate schedule, and transfer learning on upscaled images.
