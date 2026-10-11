# Telco Customer Churn Prediction: ANN vs Classical ML

Predicting which telecom customers are likely to **churn** (leave the company). A Keras neural network is benchmarked against eight tuned classical machine-learning models on an imbalanced dataset, with class weighting and decision-threshold tuning.

📓 Notebook: `Customer_Churn_ANN.ipynb`

---

## Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) on Kaggle (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). **Download it and place it next to the notebook.**
- **Size:** 7,043 customers × 21 columns.
- **Target:** `Churn` (Yes / No), imbalanced at about **73.5% / 26.5%**.
- **Features:** demographics, tenure, services (phone, internet, streaming, security, ...), contract type, payment method, monthly and total charges.
- **Data quality:** 11 rows have a blank `TotalCharges`. They are brand-new customers with `tenure = 0`, and they were dropped, leaving 7,032 rows.

## Key EDA insights

| Factor | Churn rate |
|---|---|
| Month-to-month contract | **42.7%** |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Fiber optic internet | **41.9%** |
| DSL internet | 19.0% |
| No internet service | 7.4% |
| Electronic check payment | **45.3%** |
| Other payment methods | 15–19% |

Short-contract, fiber-optic and electronic-check customers churn the most.

## Workflow

1. **EDA and visualization:** distributions by churn, boxplots, categorical count plots, churn rate per category, correlation heatmap.
2. **Preprocessing:**
   - Drop `customerID`, convert `TotalCharges` to numeric, map the target to 0/1
   - Stratified split: 4,781 train / 844 validation / 1,407 test
   - `ColumnTransformer` with `StandardScaler` (numeric) and `OneHotEncoder` (categorical), fitted on train only, giving **40 features**
3. **Classical baselines:** `GridSearchCV` (5-fold, scored by ROC-AUC) with imbalance handling for Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM and SVC.
4. **ANN (Keras):** 3 hidden layers (64 → 32 → 16), each with He initialization, BatchNormalization and Dropout 0.3, about 5.7K parameters. Training uses:
   - Class weights, so missing a churner costs more
   - `EarlyStopping` on validation AUC
   - `ReduceLROnPlateau`
   - Metrics: accuracy, AUC and recall
5. **Threshold tuning:** the best F1 threshold is picked on the **validation** set (0.56), not on the test set.
6. **Comparison:** all models side by side on the test set.

## Results (test set, 1,407 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.733 | 0.499 | **0.824** | **0.622** | **0.841** |
| Gradient Boosting | 0.793 | 0.638 | 0.513 | 0.569 | 0.838 |
| LightGBM | 0.744 | 0.512 | 0.783 | 0.619 | 0.836 |
| Logistic Regression | 0.724 | 0.488 | 0.794 | 0.605 | 0.835 |
| Random Forest | 0.718 | 0.481 | 0.781 | 0.595 | 0.834 |
| **ANN (Keras)** | 0.721 | 0.485 | 0.802 | 0.605 | 0.833 |
| Decision Tree | 0.733 | 0.499 | 0.794 | 0.613 | 0.828 |
| SVC | 0.729 | 0.494 | 0.799 | 0.611 | 0.822 |
| KNN | 0.772 | 0.574 | 0.551 | 0.562 | 0.815 |

**ANN with the tuned threshold (0.56):** accuracy 0.75, churn precision 0.52, churn recall 0.74, churn F1 0.61. This gives a better balance between catching churners and false alarms than the default 0.5 threshold.

**Takeaway:** all strong models land within about 0.02 AUC of each other. XGBoost is marginally best, and the ANN is competitive with the classical models. On a small tabular dataset like this, a neural network offers little advantage over gradient boosting.

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm tensorflow joblib
jupyter notebook Customer_Churn_ANN.ipynb
```

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle into the notebook folder.
2. Run all cells in order. The random seed is fixed (`SEED = 42`).

## Limitations and ideas

- Accuracy is a poor metric here, so the project focuses on **ROC-AUC, recall and F1**.
- Precision for churners is only about 0.5, so retention campaigns would contact many customers who would have stayed. The right threshold depends on the cost of a missed churner versus a wasted offer.
- Possible next steps: SHAP explanations, probability calibration, a cost-based threshold, and an ensemble of XGBoost and the ANN.
- The notebook does not save the trained model. Add `model.save(...)` and `joblib.dump(preprocessor, ...)` if you want to deploy it.
