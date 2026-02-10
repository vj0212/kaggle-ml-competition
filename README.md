# kaggle-ml-competition
# AnaVerse 2.0_N – Sensor-Based Anomaly Detection (Top 10% Solution)

## Project Overview
This repository contains my solution to **AnaVerse 2.0_N**, a machine learning competition focused on **anomaly detection using sensor-generated tabular data**.  
The objective was to identify anomalous patterns from multivariate sensor readings by building robust and well-evaluated predictive models.

I ranked **111 out of 1090 participants**, placing the solution within the **top ~10%** of submissions.

---

## Problem Statement
The task involved predicting anomalies based on sensor measurements captured over time. The dataset consisted of structured tabular data with numerical features representing sensor readings, requiring careful preprocessing, feature engineering, and model selection to handle noise, imbalance, and non-linear patterns.

---

## Evaluation Criteria
Submissions were evaluated based on:
- **Data Exploration and Preprocessing (20%)**
  - Handling missing values and outliers
  - Feature engineering and correlation analysis
- **Modeling (60%)**
  - Use of classical and advanced machine learning models
  - Justification of model choice and tuning strategy
- **Model Evaluation (20%)**
  - Accuracy, Precision, Recall, and F1-score (per class)
  - Robustness and generalization of the model

Final evaluation emphasized **F1-score and accuracy across classes**.

---

## Approach and Methodology

### 1. Data Exploration and Preprocessing
**What I did**
- Performed exploratory data analysis to understand feature distributions and class imbalance
- Identified missing values and potential outliers
- Applied data cleaning and preprocessing techniques to ensure model-ready inputs

**Why**
- Sensor data is often noisy and incomplete
- Poor preprocessing directly degrades anomaly detection performance

**How**
- Used pandas and NumPy for inspection and cleaning
- Applied scaling and transformations where required
- Prepared features in alignment with the expected `sample_submission` format

---

### 2. Feature Engineering
**What I did**
- Created derived features to better capture sensor behavior
- Analyzed correlations to reduce redundancy
- Selected features that improved class separability

**Why**
- Raw sensor readings alone are often insufficient for anomaly detection
- Feature engineering improves signal extraction from noisy data

**How**
- Statistical transformations and aggregations
- Correlation analysis to guide feature selection
- Iterative experimentation based on validation performance

---

### 3. Modeling Strategy
**What I did**
- Implemented and compared multiple machine learning models:
  - Classical models: Logistic Regression, KNN, Decision Trees
  - Advanced models: Random Forest, Gradient Boosting (e.g. XGBoost)
- Performed hyperparameter tuning and model comparison

**Why**
- Classical models provide strong baselines and interpretability
- Tree-based ensemble models handle non-linearity and feature interactions effectively in tabular data

**How**
- Used scikit-learn and XGBoost for model development
- Applied cross-validation and iterative tuning
- Selected the final model based on F1-score and overall robustness

---

### 4. Model Evaluation
**What I did**
- Evaluated models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Validated predictions against competition-specific metrics

**Why**
- Anomaly detection problems require balanced performance across classes
- F1-score is critical when dealing with class imbalance

**How**
- Used validation splits and metric-based comparison
- Ensured predictions matched the required submission schema

---

## Results
- Achieved a **Top ~10% leaderboard position (111 / 1090 participants)**
- **Private score:** 0.800660338
- **Public score:** 0.796404275
- Built a stable and generalizable anomaly detection pipeline with consistent performance across validation and leaderboard evaluations


---

## Tools and Technologies
- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- Jupyter Notebook

---

## Repository Structure
- `notebooks/` – Jupyter notebook containing EDA, feature engineering, and model experimentation
- `LICENSE` – MIT License

---

## Reproducibility
The project separates exploratory analysis from reusable logic to support reproducibility and clarity. While raw competition data is excluded, the workflow and methodology are fully documented and transferable to similar anomaly detection problems.

---

## Author
Vikram Kumar  
Data Analyst | Machine Learning | Data Science  

LinkedIn: https://www.linkedin.com/in/vikram-kumar-955157218/  
GitHub: https://github.com/vj0212

