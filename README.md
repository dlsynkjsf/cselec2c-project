# Prediction of Employee Attrition using Random Forest and Extreme Gradient Boosting on the Employee Attrition Classification Dataset

**Group 6 - 3CSD, Machine Learning Final Project**

- _BIHASA, Christian Louie_
- _CARIGMA, Johanna Louisse_
- _DALISAY, Nikolas Josef_
- _EVANGELISTA, Arthur Justin Rosmar_
- _PINEZA, Marian Therese_

---

## 1. Overview

This project develops an end-to-end machine learning pipeline for predicting voluntary employee attrition. The system explicitly accounts for the asymmetric business cost of classification errors: a missed leaver (false negative) is treated as more expensive than an unnecessary retention intervention (false positive). The pipeline combines cost-sensitive predictive modelling with model-agnostic interpretability methods, and is designed to support human-resources decision-making at both the population and the individual level.

### 1.1 Objective

To develop and evaluate classifiers that identify employees at risk of attrition while minimising a weighted misclassification cost, and to deliver per-employee explanations using SHAP (feature attribution) and DiCE (counterfactual recommendations).

### 1.2 Dataset

The dataset used is the [Employee Attrition Classification Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset) published on Kaggle by Stealth Technologies. It consists of employee records labelled with the attrition outcome (`Stayed` or `Left`), covering demographic, role, compensation, satisfaction, and work-environment attributes (23 predictors after removal of the `Employee ID` identifier). The data is provided in two stratified files:

- `data/train.csv` — training observations
- `data/test.csv` — held-out test observations

---

## 2. Methodology

The project is structured as a **comparative study of three model families**: baseline classifiers (Logistic Regression and CART decision tree), a deep-learning approach (Deep Neural Network), and state-of-the-art ensemble methods (Random Forest and XGBoost). The hypothesis under evaluation is that, while the deep neural network may achieve the strongest headline metrics, the SOTA ensemble family offers the more defensible operational choice once the requirements of human-resources decision-making are taken into account: per-employee classifications must be auditable, and the recommended interventions must be traceable to specific feature evidence rather than to an opaque output probability. The interpretability layer (SHAP and DiCE) is therefore applied to the SOTA ensemble.

### 2.1 Preprocessing

Features are partitioned into three groups, each with an appropriate transformation:

| Group | Transformer | Examples |
|-------|-------------|----------|
| Numerical | `StandardScaler` | `Age`, `Monthly Income`, `Years at Company` |
| Ordinal | `OrdinalEncoder` | `Work-Life Balance`, `Job Satisfaction`, `Education Level` |
| Nominal | `OneHotEncoder` | `Job Role`, `Marital Status`, `Overtime` |

The transformations are bundled into a single `ColumnTransformer` (the `preprocessor`), which is fit on the training data only and applied to the validation and test sets. The training data is further partitioned into an 80/20 train/validation split with stratification on the target variable.

### 2.2 Cost-Sensitive Framework

Misclassification outcomes are formalised in the standard 2×2 cost matrix, with **unequal weights** that explicitly penalise false negatives more heavily than false positives:

| | Predicted `Stayed` | Predicted `Left` |
|---|---|---|
| **Actually `Stayed`** | True Negative — cost 0 | False Positive — cost 1 |
| **Actually `Left`** | False Negative — cost 2 | True Positive — cost 0 |

The 2:1 ratio reflects the operational reality that the two error types are not symmetric. A false negative (failing to flag an employee who is, in fact, about to leave) is an irrecoverable error: once the resignation occurs, the company absorbs the full downstream cost of recruitment, onboarding, training, and lost productivity. A false positive (flagging an employee who would have stayed) is a recoverable error, addressed at most by a precautionary check-in or a retention conversation. The framework therefore prefers to err on the side of over-flagging: it is acceptable to classify a willing-to-stay employee as flight-risk, but it is not acceptable to classify a flight-risk employee as retaining.

The cost ratio is operationalised through:

- `class_weight={0: 1, 1: 2}` for the scikit-learn estimators and the Keras DNN
- `scale_pos_weight=2` for XGBoost

### 2.3 Models

Six classifiers are trained and evaluated, organised into three families to enable a like-for-like comparison:

| Family | Models |
|--------|--------|
| Baseline | Logistic Regression, Decision Tree (CART) |
| Deep Learning | Deep Neural Network (TensorFlow) |
| State-of-the-Art Ensembles | Random Forest, XGBoost |

A soft-voting ensemble of the two SOTA models (Random Forest + XGBoost) is also evaluated and serves as the configuration to which the explainability layer is applied.

### 2.4 Three-Stage Comparison

To isolate the contribution of each cost-handling technique, three configurations are compared on identical data:

1. **Baseline** — default estimators, default 0.5 decision threshold.
2. **Threshold-Only Optimization** — the baseline estimators with a per-model decision threshold tuned on the validation set to minimise the weighted cost.
3. **Cost-Aware + Hyperparameter-Tuned** — estimators retrained with class-weighting and hyperparameters selected through `RandomizedSearchCV`, with the decision threshold subsequently tuned on the validation set.

All thresholds are selected on validation data and reported on the held-out test set to prevent leakage.

### 2.5 Explainability

A single output probability is insufficient for an operational human-resources setting: a deep neural network cannot, on its own, justify why a specific employee was flagged, nor can it indicate which interventions would change that classification. Two complementary explanation methods are therefore layered on top of the SOTA ensemble -- providing the **diagnosis** for each individual prediction and the **prescription** that follows from it:

- **SHAP (TreeExplainer)** — supplies the diagnosis. It produces a global feature-importance bar plot, a global beeswarm plot showing the directional effect of feature values, and a local waterfall plot that decomposes the prediction for the highest-risk employee in the test sample into individual feature contributions.
- **DiCE (genetic method)** — supplies the prescription. It generates counterfactual explanations for the three highest-risk employees, identifying minimal feature changes that would flip the prediction from `Left` to `Stayed`. The search is restricted to **actionable variables** (compensation, overtime, work-life balance, recognition, promotions, distance from home, remote work, leadership opportunities, innovation opportunities) so that every recommendation corresponds to an intervention a human-resources team can plausibly implement.

This pairing addresses the central limitation of the deep-learning baseline. The neural network may achieve marginally higher headline metrics, but a human reviewer cannot (and arguably should not) act on a single number from a black-box model when the decision concerns an individual employee's career.

---

## 3. Setup and Reproducibility

### 3.1 Environment

- **Python**: 3.11.x — the reference environment used by the team. Other 3.11 patch versions are expected to behave identically.
- **Random seed**: `SEED = 42` is applied to Python's `random`, NumPy, TensorFlow, and all scikit-learn estimators that accept a `random_state` argument. Repeated executions in the same environment produce identical results.

### 3.2 Installation

From the project root, open a terminal and create a Python 3.11 virtual environment, then install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependency versions are pinned in `requirements.txt`. The notebook depends on `scikit-learn==1.8.0`, `xgboost==3.2.0`, `tensorflow==2.21.0`, `shap==0.51.0`, and `dice-ml==0.12`; using different major versions of these packages is likely to break either the SHAP or DiCE integration.

### 3.3 Hardware Notes

TensorFlow ≥ 2.11 does not provide GPU support on native Windows; the deep neural network therefore trains on CPU. The dataset and model sizes used in this project are small enough that CPU training completes within a few minutes, and no GPU is required to reproduce the reported results.

---

## 4. How to Run

### 4.1 Launching the Notebook

From the project root, with the virtual environment activated:

```bash
jupyter notebook
```

Open `3CSD_Group 6_Implementation.ipynb` and execute the cells in order (`Cell → Run All`). The notebook is organised into eleven steps:

| Step | Description |
|------|-------------|
| 1 | Data loading, preprocessing, and train/validation split |
| 2 | Baseline training of the six classifiers |
| 3 | Baseline benchmark on the test set at the default 0.5 threshold |
| 4 | Cost matrix definition (`COST_FN = 2`, `COST_FP = 1`) |
| 5 | Per-model decision threshold optimisation |
| 6 | Cost-aware setup (`COST_WEIGHTS = {0: 1, 1: 2}`) |
| 7 | Cost-aware retraining with hyperparameter search and threshold tuning |
| 8 | Three-stage comparison plot and business-impact summary |
| 9 | SHAP explainability (global bar, beeswarm, local waterfall) |
| 10 | DiCE counterfactual explanations for the three highest-risk employees |
| 11 | Persistence of trained artifacts and per-employee risk scores |

### 4.2 Expected Runtime

End-to-end execution completes in approximately five to ten minutes on a recent CPU. The hyperparameter search in Step 7 is the dominant cost.

### 4.3 Outputs

Step 11 writes the following to the `artifacts/` directory:

- `preprocessor.joblib` — the fitted `ColumnTransformer`
- `best_rf.joblib`, `best_xgb.joblib` — the Stage B base learners
- `tuned_ensemble.joblib` — the final soft-voting ensemble
- `threshold.txt` — the decision threshold selected on validation
- `risk_scores.csv` — per-employee test-set predictions including `P_Left`, the predicted label, and a categorical risk band (Low / Medium / High)

---

## 5. Results

The HP-Tuned Voting Ensemble is identified as the recommended deployment configuration, producing the lowest total weighted cost on the held-out test set among the configurations evaluated. Detailed per-model metrics (accuracy, precision, recall, F1, and selected threshold) and the three-stage comparison plot are produced in Step 8 of the notebook. The corresponding business-impact summary at the end of Step 8 reports the resulting confusion matrix in operational terms (correctly retained employees, unnecessary bonuses, unanticipated leavers, correctly flagged leavers) and the associated total weighted cost.
