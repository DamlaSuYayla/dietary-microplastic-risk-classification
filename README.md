# Dietary Microplastic Exposure Risk Classification Across 109 Countries Using Ensemble Machine Learning and SHAP-Based Explainable AI 

An end-to-end machine learning and Explainable AI (XAI) pipeline designed to classify human dietary microplastic exposure risks globally using longitudinal data (1990-2018).

---

## 📌 Project Overview
Dietary microplastic contamination is an escalating global health threat. This project introduces a scalable machine learning architecture that transitions from **unsupervised target discovery** to **supervised ensemble classification**, complemented by **SHAP (SHapley Additive exPlanations)** to ensure model transparency for public health policymaking.

### Key Features:
* **Multi-Country Scope:** Analyzes 723 observations from 19 food categories across 109 countries.
* **Hybrid Architecture:** Uses K-Means clustering to algorithmically derive objective risk tiers (Low, Medium, High).
* **Class Imbalance Mitigation:** Utilizes SMOTE to handle class distribution issues prior to model training.
* **Explainable AI (XAI):** Leverages global SHAP analysis to unpack "black-box" model mechanics and reveal key exposure drivers.

---

## 🏗️ Pipeline Architecture
The workflow consists of the following phases:
1. **Data Preprocessing & Normalization:** Standard scaling and missing value validation.
2. **Dimensionality Reduction:** Principal Component Analysis (PCA) to address multicollinearity.
3. **Target Derivation:** Unsupervised K-Means clustering ($k=3$, optimized via Elbow and Silhouette methods).
4. **Resampling:** Over-sampling minority risk classes using SMOTE.
5. **Supervised Learning:** Benchmarking Random Forest and XGBoost against a Multinomial Logistic Regression baseline.
6. **Validation:** Stratified 5-fold cross-validation and chronological temporal validation (Train $\le$ 2010 $\rightarrow$ Test $>$ 2010).

---

## 📊 Experimental Results

### Performance Benchmarking
Our ensemble models significantly outperformed the traditional statistical baseline:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression** | 0.8690 | 0.8843 | 0.8690 | 0.8713 | 0.9695 |
| **Random Forest (Best)** | **0.9862** | **0.9863** | **0.9862** | **0.9862** | **0.9996** |
| **XGBoost** | 0.9724 | 0.9732 | 0.9724 | 0.9726 | 0.9993 |

### Generalizability & Robustness
* **5-Fold Cross Validation:** Random Forest achieved a mean accuracy of `0.9862`, proving internal consistency.
* **Temporal Validation (Data Drift Test):** Evaluated on unseen post-2010 data, the Random Forest model maintained a strong chronological generalizability with an accuracy of **0.9495**.

---

## 🔍 Explainable AI (XAI) Insights
Using SHAP analysis, we visualized the global feature importance:
* **Primary Drivers:** Baseline contamination concentration (`total_ug_per_kg`) emerged as the single most influential feature across models.
* **Dietary Drivers:** Higher consumption of dairy products (`total_milk`) and engineered aggregate variables (`total_intake`) were the dominant predictors shifting predictions toward High-Risk classifications.

*(Tip: Place your SHAP summary plot or correlation heatmap images here using `![SHAP Plot](reports/figures/shap_importance.png)`)*

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Environment:** Google Colab / Jupyter Notebook
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `matplotlib`, `seaborn`

---

## 🚀 How to Run
   ```bash
https://colab.research.google.com/drive/1Txz5hrVtjI8PxL8U6-cnC8wcPZKVv9dr?usp=sharing
