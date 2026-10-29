# 🍷 Wine Quality Classification Pipeline

A supervised machine learning model engineered with Python and Scikit-Learn to classify red wine quality based on chemical traits.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com
   cd wine-quality-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute the classification pipeline:
   ```bash
   python src/train.py
   ```

## 📊 Evaluation Metrics (Test Set Results)

| Evaluation Metric | Score Baseline | Strategic Operational Interpretation |
| :--- | :--- | :--- |
| **5-Fold CV Accuracy** | 81.42% | Stable performance across data folds; low variance validation. |
| **Test Set Accuracy** | 82.19% | Strong generalized accuracy on fresh, unseen chemical test frames. |
| **ROC-AUC Score** | 0.8934 | Excellent model discrimination capacity between classes. |

### Top 3 Predictive Drivers
1. **Alcohol Content** (Primary structural feature)
2. **Acid Ratio** (Engineered structural balance feature)
3. **Sulphates** (Preservative chemical marker)
