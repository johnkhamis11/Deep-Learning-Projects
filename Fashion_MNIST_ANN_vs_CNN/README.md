# Fashion-MNIST Clothing Classification: ANN vs CNN

Classifying 28×28 grayscale images of clothing items into **10 categories**, and comparing a fully-connected network (ANN) against a convolutional neural network (CNN) built with Keras.

📓 Notebook: `Fashion_MNIST_ANN_vs_CNN.ipynb`

---

## Dataset

- **Source:** Fashion-MNIST via `keras.datasets.fashion_mnist`, downloaded automatically.
- **Size:** 60,000 training and 10,000 test images (28×28, grayscale).
- **Classes (perfectly balanced, 6,000 each in training):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
- **Split:** 54,000 train / 6,000 validation (stratified) / 10,000 test.

## Workflow

1. Load the data and check shapes, pixel range (0–255) and class balance.
2. Visualize sample images and the class distribution.
3. Normalize pixels to 0–1 and reshape to `(28, 28, 1)`.
4. Train an **ANN**: `Flatten → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(10, softmax)`, about 235K parameters.
5. Train a **CNN**: `2 × (Conv2D + MaxPool) → Flatten → Dense(128) → Dropout(0.3) → Dense(10, softmax)`, about 225K parameters.
6. Evaluate the CNN with a per-class report, confusion matrix and a gallery of misclassified images.
7. Compare both models on the test set.

Both models use Adam, sparse categorical cross-entropy and early stopping on validation loss (patience 5).

## Results (test set, 10,000 images)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| ANN | 0.885 | 0.885 | 0.885 | 0.885 |
| **CNN** | **0.908** | **0.909** | **0.908** | **0.908** |

**Takeaway:** the CNN beats the ANN by about **2.3 points** with a slightly *smaller* parameter count (225K vs 235K). Convolutions exploit the spatial structure of images, so they use their parameters more efficiently than dense layers.

**Per-class behaviour (CNN):**

- Easiest: **Trouser, Sandal and Bag** (F1 about 0.98), then **Ankle boot** (0.97) and **Sneaker** (0.96)
- Hardest: **Shirt** (F1 0.73), which is often confused with T-shirt/top, Pullover and Coat, followed by Coat (0.85), T-shirt/top (0.86) and Pullover (0.86)

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
jupyter notebook Fashion_MNIST_ANN_vs_CNN.ipynb
```

Run all cells in order. The dataset is small (about 30 MB) and downloads on the first run. A GPU is not required. The random seed is fixed (`SEED = 42`).

## Limitations and ideas

- The notebook does not save the trained models.
- Fashion-MNIST is relatively easy, and strong CNNs exceed 93–94%. Possible next steps: batch normalization, a deeper network, data augmentation, and a learning-rate schedule.
- The Shirt class is the main weak spot, so it would be the first place to look when improving the model.
