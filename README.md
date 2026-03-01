# 🔐 Smart Insider Threat Detection & Employee Risk Profiling Framework

A hybrid machine learning framework for detecting insider threats using behavioral analytics, psychometric profiling, anomaly detection, clustering, and supervised classification.

---

## 📌 Overview

Insider threats remain one of the most difficult cybersecurity risks due to privileged access and subtle behavioral deviations. 

This project proposes a **multi-stage hybrid ML framework** integrating:

- Psychometric (OCEAN) trait analysis
- SQL-based data integration
- Domain-driven feature engineering
- Isolation Forest (Anomaly Detection)
- K-Means (Behavioral Clustering)
- Random Forest (Supervised Threat Classification)

The framework transforms raw organizational logs into interpretable, risk-stratified intelligence.

---

## 🧠 Key Features

✔ Hybrid anomaly + clustering + classification pipeline  
✔ Psychometric-behavioral feature fusion  
✔ Composite risk scoring system  
✔ Semi-supervised label generation  
✔ SMOTE-based imbalance handling  
✔ Risk stratification into 4 threat levels  

---

## 🗂 Dataset & Data Pipeline

This project uses the **CERT Insider Threat Dataset (Version R6.2)** as the primary data source.

### 📌 Raw Datasets Used
The following datasets were integrated:

- **Logon Dataset** – User authentication records  
- **Email Dataset** – Internal communication activity logs  
- **Device Dataset** – File access and endpoint interaction records  
- **User Metadata Dataset** – Organizational roles, hierarchy, and user attributes  

---

### 🔄 Data Integration & Transformation

All raw datasets were integrated using **SQL-based relational joins** (via `user_id`) to create a unified behavioral database.  

After preprocessing and feature engineering, an aggregated dataset was generated:

📂 `engineered_behavioral_dataset.csv`  
Contains engineered behavioral and psychometric features used for anomaly detection.

---

### 🤖 Semi-Supervised Label Generation

K-Means clustering was applied on the engineered dataset to generate behavioral risk groups.  

This produced:

📂 `cluster_labeled_dataset.csv`  
Contains cluster-derived threat labels used for supervised learning.

---

### 🎯 Final Model Training

The labeled dataset was then used to train a Random Forest classifier for multi-class insider threat prediction.

## 🏗 Methodology Pipeline

1. SQL-based data integration  
2. Data preprocessing & cleaning  
3. Domain-driven feature engineering  
4. Isolation Forest anomaly detection  
5. K-Means behavioral clustering  
6. Random Forest threat classification  
7. Risk visualization & interpretation  

---

## 📊 Model Performance

### Isolation Forest (Anomaly Detection)
- Accuracy: **98%**
- Precision: 0.877
- Recall: 0.70
- F1 Score: 0.78

### Random Forest (Threat Classification)
- Accuracy: **94.9%**
- Precision: 0.951
- Recall: 0.949
- F1 Score: 0.950

---

## 🏷 Threat Categories

- 🟢 Benign / Stable
- 🟡 Negligent / Overworked
- 🟠 Opportunistic / Suspicious
- 🔴 Malicious Insider

---

## 🧩 Engineered Risk Features

- Neuroticism After-Hours Risk
- Openness Activity Risk
- Composite Risk Score
- Risk Ranking
- After-hours Activity Ratio
- Device Usage Diversity

---

## 📈 Visual Insights

- Login time distribution analysis
- Role-based after-hours activity
- OCEAN personality distribution
- PC usage anomaly distribution
- Feature importance analysis
- Confusion matrices

---

## 🚀 Why This Project Matters

Traditional rule-based systems struggle to detect subtle insider threats.  

This framework:
- Combines behavioral + psychological analytics
- Reduces false positives
- Provides interpretable risk intelligence
- Enables proactive enterprise security

---

## 🛠 Tech Stack

- Python
- SQL
- Pandas
- Scikit-learn
- SMOTE (Imbalanced-Learn)
- Matplotlib / Seaborn

---

## 📚 Research Contribution

This project integrates anomaly detection, clustering, and classification into a unified, interpretable framework — addressing key limitations in existing insider threat detection systems.

---

## 📄 Research Paper

Full IEEE-style paper available in this repository.

---
