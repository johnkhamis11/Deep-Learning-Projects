# Cats vs Dogs Image Classification with a CNN

A binary image classifier that tells **cats from dogs**. It compares a simple baseline CNN against an improved CNN that uses data augmentation, batch normalization and dropout.

📓 Notebook: `Cats_vs_Dogs_CNN.ipynb`

---

## Dataset

- **Source:** Microsoft's *Kaggle Cats and Dogs* dataset (`PetImages`), downloaded inside the notebook with `curl` and `unzip`.
- **Full dataset:** about 12,500 cat images and 12,500 dog images.
- **Subset used:** the first **1,500 images per class** (3,000 in total), resized to **128×128 RGB**. Corrupted images are skipped automatically.
- **Split (stratified):** 70% train / 15% validation / 15% test (2,100 / 450 / 450).

## Workflow

1. Download and extract the dataset.
2. Load a balanced subset, resize it and build the label array.
3. Visualize sample images and the class distribution.
4. Split the data into train, validation and test sets.
5. Train a **baseline CNN**: `Rescaling → 2 × (Conv2D + MaxPool) → Dense(64) → Dense(1, sigmoid)`, about 3.7M parameters.
6. Train an **improved CNN**:
   - Augmentation layers: `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`
   - 3 × (Conv2D + BatchNormalization + MaxPool + Dropout 0.25)
   - `Dense(128) → Dropout(0.5) → Dense(1, sigmoid)`, about 4.3M parameters
7. Evaluate with a classification report, ROC-AUC, confusion matrix, ROC curves and a gallery of misclassified images.

Both models use Adam, binary cross-entropy and early stopping on validation loss.

## Results (test set, 450 images)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | 0.676 | 0.685 | 0.649 | 0.667 | 0.736 |
| **Improved CNN** | **0.724** | **0.759** | **0.658** | **0.705** | **0.786** |

**Takeaway:** the baseline reaches about 92% training accuracy but only about 67% validation accuracy, a clear case of overfitting. Augmentation, batch normalization and dropout in the improved model close part of that gap and add roughly 5 points of accuracy.

## Saved artifact

- `cats_dogs_cnn.keras`: the trained improved CNN.

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow
jupyter notebook Cats_vs_Dogs_CNN.ipynb
```

The first cells download about 800 MB, so they need an internet connection. They use shell commands (`!curl`, `!unzip`), which work in Jupyter, Colab or Kaggle. On Windows, download and extract the dataset manually into `data/PetImages`. A GPU is recommended but not required.

## Limitations and ideas

- Only 3,000 of the roughly 25,000 images are used, which caps accuracy. Using the full dataset would likely help a lot.
- **Transfer learning** (for example MobileNetV2, EfficientNet or ResNet pretrained on ImageNet) would typically push accuracy above 95%.
- Other ideas: a deeper network, learning-rate scheduling, and Grad-CAM to see what the model looks at.
