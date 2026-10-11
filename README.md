# Deep Learning Projects

A collection of 9 hands-on projects covering **tabular data, images, text and time series**. Each project follows the same workflow (EDA, visualization, preprocessing, modelling, evaluation) and compares several models, usually a classical or baseline model against a neural network, so the results show *when* deep learning actually helps.

Everything is built with **Python, TensorFlow/Keras and scikit-learn**.

---

## Projects at a glance

| # | Project | Domain | Task | Models compared | Best result |
|---|---|---|---|---|---|
| 1 | [Adult Income](Adult_Income_ML_vs_Neural_Network/) | Tabular | Binary classification | Logistic Regression, Random Forest, XGBoost, NN | **XGBoost**: ROC-AUC 0.927 |
| 2 | [California Housing](California_Housing_ML_vs_ANN/) | Tabular | Regression | Linear Regression, Random Forest, ANN | **Random Forest**: R² 0.783, RMSE 0.451 |
| 3 | [Customer Churn](Customer_Churn_ANN/) | Tabular | Binary classification (imbalanced) | 8 classical models + ANN | **XGBoost**: ROC-AUC 0.841 (ANN 0.833) |
| 4 | [Fashion-MNIST](Fashion_MNIST_ANN_vs_CNN/) | Images | 10-class classification | ANN vs CNN | **CNN**: 90.8% accuracy |
| 5 | [CIFAR-10](CIFAR10_CNN_Data_Augmentation/) | Images | 10-class classification | Baseline CNN vs improved CNN (augmentation) | **Improved CNN**: 76.3% accuracy |
| 6 | [Cats vs Dogs](Cats_vs_Dogs_CNN/) | Images | Binary classification | Baseline CNN vs improved CNN (augmentation) | **Improved CNN**: 72.4% accuracy, ROC-AUC 0.786 |
| 7 | [IMDB Sentiment](IMDB_Sentiment_RNN_LSTM_GRU/) | Text | Binary classification | SimpleRNN, LSTM, GRU | **LSTM**: 86.9% accuracy, ROC-AUC 0.944 |
| 8 | [Reuters News](Reuters_News_Classification_RNN/) | Text | 46-class classification | ANN, LSTM, BiLSTM | **ANN**: 77.2% accuracy |
| 9 | [Jena Temperature](Jena_Temperature_Forecasting_RNN/) | Time series | Regression (24 h ahead) | Naive baseline, SimpleRNN, LSTM, GRU | **GRU**: MAE 2.26 °C, R² 0.861 |

Each folder contains its own notebook and a detailed `README.md` with the dataset, workflow, full results table and limitations.

---

## Key takeaways across projects

**1. On tabular data, gradient boosting and forests beat neural networks.**
XGBoost won on Adult Income and Customer Churn, and Random Forest won on California Housing. In the churn project, all strong models landed within about 0.02 AUC of each other, so the extra complexity of a neural network bought very little.

**2. On images, CNNs clearly beat dense networks, and regularization pays off.**
On Fashion-MNIST the CNN beat the ANN by about 2.3 points with *fewer* parameters. On CIFAR-10 and Cats vs Dogs, adding augmentation, batch normalization and dropout improved accuracy by about 9 and 5 points over the baseline CNNs, and reduced overfitting.

**3. On text, the simplest model that fits the task can win.**
For IMDB sentiment, where word order matters, LSTM was best and SimpleRNN was weakest. For Reuters topic classification, which depends mostly on *which* words appear, a simple embedding-average ANN beat both LSTM and BiLSTM.

**4. Always compare against a strong baseline.**
In temperature forecasting, a naive "same as now" predictor reached R² 0.826, and the best neural model reached only 0.861. Without that baseline, the models would have looked more impressive than they are.

**5. Accuracy can be misleading.**
Reuters has 77% accuracy but only 0.33 macro F1 because of class imbalance, and the churn and income projects use ROC-AUC, recall and F1 for the same reason.

---

## Good practices used throughout

- **Reproducibility:** fixed random seed (`SEED = 42`) in every notebook.
- **No data leakage:** scalers and encoders are fitted on training data only. The time-series project splits chronologically and normalizes with training-period statistics.
- **Proper splits:** stratified train / validation / test splits wherever the target is categorical.
- **Early stopping** on validation loss (or AUC) to avoid overfitting.
- **Imbalance handling:** class weights, stratification and threshold tuning (on validation data, not test data).
- **Error analysis:** confusion matrices, per-class reports and galleries of misclassified examples.

---

## Repository structure

```
.
├── README.md
├── Adult_Income_ML_vs_Neural_Network/
│   ├── Adult_Income_ML_vs_Neural_Network.ipynb
│   └── README.md
├── California_Housing_ML_vs_ANN/
├── Customer_Churn_ANN/
├── Fashion_MNIST_ANN_vs_CNN/
├── CIFAR10_CNN_Data_Augmentation/
├── Cats_vs_Dogs_CNN/
├── IMDB_Sentiment_RNN_LSTM_GRU/
├── Reuters_News_Classification_RNN/
└── Jena_Temperature_Forecasting_RNN/
```

---

## Getting started

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install the main dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib \
            xgboost lightgbm pillow jupyter

# 3. Open any project's notebook
jupyter notebook
```

Then run the cells of a notebook from top to bottom.

**Data sources:** most datasets download automatically (Keras datasets, scikit-learn, OpenML). Two need attention:

- **Customer Churn:** download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it next to the notebook.
- **Cats vs Dogs:** the notebook downloads about 800 MB using shell commands (`!curl`, `!unzip`), so it works best in Jupyter, Colab or Kaggle.

**Hardware:** all projects run on a CPU. A GPU speeds up the image projects (CIFAR-10, Cats vs Dogs) noticeably.

---

## Tech stack

| Area | Libraries |
|---|---|
| Deep learning | TensorFlow / Keras |
| Classical ML | scikit-learn, XGBoost, LightGBM |
| Data & visualization | pandas, NumPy, Matplotlib, Seaborn |
| Utilities | joblib, Pillow |

---

## Notes

- The results in each README come from the saved notebook runs. They may vary slightly with different library versions or hardware.
- These are learning and portfolio projects. In particular, the Adult Income dataset contains sensitive attributes and should not be used for real decisions about people.
