# Reuters Newswire Topic Classification: ANN vs LSTM vs BiLSTM

Multi-class text classification of Reuters newswire articles into **46 topics**, comparing an embedding-based feed-forward network (ANN) with an LSTM and a Bidirectional LSTM in Keras.

📓 Notebook: `Reuters_News_Classification_RNN.ipynb`

---

## Dataset

- **Source:** Reuters newswire topics via `keras.datasets.reuters`, downloaded automatically.
- **Size:** 8,982 training and 2,246 test articles, in **46 topic classes**.
- **Class imbalance:** very strong. The two largest topics (class 3 with 3,159 articles and class 4 with 1,949) make up over half of the training set, while the smallest classes have only 10–20 articles.
- **Preprocessing:** articles are already integer-encoded. Only the **10,000 most frequent words** are kept, and each article is padded or truncated to **200 tokens**. About 21% of articles are longer than 200 tokens.
- **Split:** 8,083 train / 899 validation (stratified) / 2,246 test.

## Workflow

1. Load the data, decode one article back into text and inspect the topic counts.
2. Visualize topic distribution and article length.
3. Pad sequences to a fixed length (`maxlen = 200`) and create the validation split.
4. Train three models, all starting with `Embedding(10000, 64)` and ending with a `Dense(46, softmax)` output:
   - **ANN:** `GlobalAveragePooling1D → Dense(64) → Dropout(0.3)`, about 647K parameters
   - **LSTM:** `LSTM(64) → Dropout(0.3)`, about 676K parameters
   - **BiLSTM:** `Bidirectional(LSTM(64)) → Dropout(0.3)`, about 712K parameters
5. Compare validation loss and accuracy across epochs.
6. Evaluate on the test set with a confusion matrix and a per-class accuracy check.

All models use Adam, sparse categorical cross-entropy and early stopping on validation loss (patience 3).

## Results (test set, 2,246 articles)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| **ANN** | **0.772** | **0.383** | **0.317** | **0.328** | **0.747** |
| LSTM | 0.619 | 0.085 | 0.104 | 0.090 | 0.584 |
| BiLSTM | 0.617 | 0.065 | 0.084 | 0.068 | 0.573 |

**Takeaway:** the simple ANN (average of word embeddings) clearly beats both recurrent models on every metric. This is a useful result in itself: for topic classification, which depends mostly on *which words appear* rather than their order, and with only about 9K training articles, a simple bag-of-embeddings model is a strong baseline that recurrent models struggle to match.

**Class imbalance matters:** the large gap between accuracy (0.77) and macro F1 (0.33) shows that even the best model does well on frequent topics but poorly on rare ones. The LSTM and BiLSTM effectively ignore many small classes, and the per-class check shows several topics with 0% accuracy. Accuracy alone overstates how well the models do.

## Saved artifacts

| File | Description |
|---|---|
| `reuters_bilstm.keras` | Trained BiLSTM model |
| `word_index.pkl` | Reuters word-to-index dictionary |

> ⚠️ **Note:** the notebook saves the **BiLSTM**, which is the *weakest* model in the comparison. The ANN would be the better one to export. Change the last cell to `ann_model.save("reuters_ann.keras")` if you want to deploy the best model.

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
jupyter notebook Reuters_News_Classification_RNN.ipynb
```

Run all cells in order. The dataset is small and downloads on the first run. The random seed is fixed (`SEED = 42`).

## Limitations and ideas

- **Imbalance handling:** try `class_weight`, oversampling of rare topics, or grouping very small classes.
- **Better recurrent models:** the LSTM and BiLSTM stop early with high validation loss. More regularization, pretrained embeddings (GloVe) or a longer `maxlen` could change the picture.
- **Strong baselines to add:** TF-IDF with Logistic Regression or Linear SVM, which often do very well on this dataset.
- A fine-tuned Transformer would likely do best, although the dataset is small.
