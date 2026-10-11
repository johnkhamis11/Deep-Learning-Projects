# IMDB Movie Review Sentiment Analysis: SimpleRNN vs LSTM vs GRU

Binary sentiment classification (**positive / negative**) of movie reviews, comparing three recurrent architectures built with Keras: SimpleRNN, LSTM and GRU.

📓 Notebook: `IMDB_Sentiment_RNN_LSTM_GRU.ipynb`

---

## Dataset

- **Source:** IMDB movie reviews via `keras.datasets.imdb`, downloaded automatically.
- **Size:** 25,000 training and 25,000 test reviews, perfectly balanced between positive and negative.
- **Preprocessing:** reviews are already integer-encoded. Only the **10,000 most frequent words** are kept, and every review is padded or truncated to **200 tokens**. About **42.7%** of reviews are longer than 200 tokens, so they are truncated.
- **Split:** 22,500 train / 2,500 validation (stratified) / 25,000 test.

## Workflow

1. Load the data and inspect shapes, sample sequences, and decode one review back into text.
2. Analyze review length and sentiment distribution, with a histogram and boxplot.
3. Pad sequences to a fixed length (`maxlen = 200`) and create the validation split.
4. Train three models with the same structure, `Embedding(10000, 32) → recurrent layer (32 units) → Dense(1, sigmoid)`:
   - **SimpleRNN**
   - **LSTM** (with Dropout 0.3)
   - **GRU** (with Dropout 0.3)
5. Compare validation loss and accuracy across epochs.
6. Evaluate on the test set: classification report, ROC-AUC, confusion matrix and ROC curves.
7. Predict the sentiment of a custom review typed by the user.

All models use Adam, binary cross-entropy and early stopping on validation loss (patience 3).

## Results (test set, 25,000 reviews)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **LSTM** | **0.869** | 0.859 | 0.884 | **0.871** | **0.944** |
| GRU | 0.851 | 0.832 | 0.880 | 0.855 | 0.928 |
| SimpleRNN | 0.819 | 0.835 | 0.796 | 0.815 | 0.892 |

**Takeaway:**

- **LSTM** performs best on all headline metrics.
- **GRU** is close and has fewer parameters, so it is a good lighter alternative.
- **SimpleRNN** is clearly the weakest. It struggles with long sequences because of vanishing gradients, and its validation loss starts rising after only a couple of epochs.

## Try your own review

The notebook includes a snippet that encodes a custom sentence with the IMDB word index and returns a probability:

```python
review = "this movie was great and the acting was amazing"
# → probability close to 1 means Positive
```

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
jupyter notebook IMDB_Sentiment_RNN_LSTM_GRU.ipynb
```

Run all cells in order. The dataset is downloaded on the first run. The random seed is fixed (`SEED = 42`).

## Limitations and ideas

- Reviews are truncated to 200 tokens and the vocabulary is limited to 10K words, so some information is lost.
- Possible next steps: Bidirectional LSTM/GRU, stacked recurrent layers, pretrained embeddings (GloVe / Word2Vec), and fine-tuning a Transformer such as BERT, which typically reaches 90%+ accuracy.
- Saving the trained model and building a small demo app are natural extensions, since the notebook does not export the model.
