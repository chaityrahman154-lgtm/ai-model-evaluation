# 🤖 Software Quality Classifier — Multi-Model ML Evaluation

A machine learning project that answers a practical software engineering question: **can we predict whether a software project is high, medium, or low quality based on its measurable code metrics?** Built as a course project for **CSE — Artificial Intelligence** at BRAC University, this experiment trains and rigorously compares three classifiers on a real software quality dataset, then visualizes which model earns the right to make that prediction.

## ✨ Project Overview

Software teams generate mountains of code metrics — test coverage, complexity scores, commit frequency, bug rates — but turning those numbers into a quality verdict is hard. This project frames that as a **multi-class classification problem**: given a set of software attributes, classify a project's quality as `High`, `Medium`, or `Low`.

Rather than settling for one model, this project runs a **head-to-head evaluation** of three fundamentally different ML approaches and lets the metrics decide the winner. The full pipeline covers data cleaning, feature engineering, model training, and a suite of evaluation visualizations that go well beyond a single accuracy score.

## 🧪 Models Compared

| Model | Approach | Why Included |
|---|---|---|
| **Logistic Regression** | Linear probabilistic classifier | Strong, interpretable baseline |
| **Naive Bayes** | Probabilistic with feature independence assumption | Efficient; good with structured feature sets |
| **Neural Network** | 3-layer feedforward (64→32→3 neurons, ReLU + Softmax) | Captures non-linear relationships between metrics |

A **K-Means clustering** pass is also run unsupervised to explore whether natural quality groupings exist in the data without labels.

## 📊 Evaluation Framework

This is where the project goes beyond a typical homework submission. Each model is evaluated across multiple lenses:

- **Accuracy & Weighted F1-Score** — Side-by-side bar chart comparing all three models
- **Confusion Matrices** — Per-model breakdown revealing exactly which quality classes get misclassified and why
- **ROC Curves with AUC Scores** — One-vs-rest multiclass ROC analysis showing the true discriminative power of each model
- **Class Distribution Plot** — Dataset balance check before training to flag potential bias

## 🛠️ Technical Pipeline
**Stack:** Python · scikit-learn · TensorFlow/Keras · pandas · NumPy · matplotlib · seaborn

## 🚀 Getting Started

```bash
git clone https://github.com/chaityrahman154-lgtm/ai-model-evaluation.git
cd ai-model-evaluation
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
python model_evaluation.py
```

The script outputs all plots sequentially: class distribution → accuracy/F1 comparison → confusion matrices → ROC curves.

## 💡 Key Takeaways

- Accuracy alone is misleading when class distributions are uneven — F1-score and ROC-AUC tell a more honest story
- The Neural Network captures complex feature interactions that linear models miss, but at the cost of interpretability
- Confusion matrix analysis reveals *which* quality tier is hardest to classify — a finding that matters for real deployment decisions

## License
MIT License — see `LICENSE` for details.
