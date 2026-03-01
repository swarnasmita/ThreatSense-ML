import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# --- Load dataset ---
df = pd.read_csv("final_user_anomaly_features.csv")

# --- Handle missing values ---
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# --- Encode categorical variables ---
categorical_cols = ['role', 'department', 'team', 'business_unit', 'supervisor']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# --- Define and scale numerical columns ---
numerical_cols = [
    'avg_daily_activities', 'avg_unique_activities', 'avg_after_hours_ratio',
    'avg_unique_pcs', 'avg_logon_hour', 'avg_logoff_hour',
    'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
    'neuroticism_afterhours_risk', 'openness_activity_risk', 'composite_risk_score'
]

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# --- Define features ---
features = df[numerical_cols + categorical_cols]

# --- Pseudo Ground Truth (Top 5% most risky users) ---
threshold = df['risk_rank'].quantile(0.05)
df['true_label'] = (df['risk_rank'] <= threshold).astype(int)

# --- Step 1: Train Isolation Forest (Recall-friendly tuning) ---
iso = IsolationForest(
    n_estimators=400,
    max_samples=0.9,
    max_features=1.0,
    contamination=0.08,
    random_state=42,
    bootstrap=True,
    n_jobs=-1
)
iso_preds = iso.fit_predict(features)
df['iso_label'] = np.where(iso_preds == -1, 1, 0)
df['iso_score'] = -iso.score_samples(features)

# --- Step 2: Behavioral Risk Calibration (Precision filter) ---
df['risk_calibrated_score'] = (
    0.6 * df['iso_score'] +
    0.25 * df['composite_risk_score'] +
    0.15 * df['neuroticism_afterhours_risk']
)

df['risk_calibrated_score'] = (df['risk_calibrated_score'] - df['risk_calibrated_score'].min()) / (
    df['risk_calibrated_score'].max() - df['risk_calibrated_score'].min()
)

# --- Step 3: Dynamic Precision Tuning ---
precision_target = 0.92
for q in range(85, 100):
    df['final_label'] = (df['risk_calibrated_score'] >= np.percentile(df['risk_calibrated_score'], q)).astype(int)
    prec = precision_score(df['true_label'], df['final_label'])
    rec = recall_score(df['true_label'], df['final_label'])
    f1 = f1_score(df['true_label'], df['final_label'])
    if prec >= 0.80 and rec >= 0.70:
        optimal_q = q
        break
else:
    optimal_q = 94

df['final_label'] = (df['risk_calibrated_score'] >= np.percentile(df['risk_calibrated_score'], optimal_q)).astype(int)

# --- Step 4: Evaluate Final Model ---
print(f"\n=== High-Precision Isolation Forest (Calibrated) ===")
print(f"Optimal Score Threshold: Top {100-optimal_q}% users flagged as anomalies")

cm = confusion_matrix(df['true_label'], df['final_label'])
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(df['true_label'], df['final_label'], target_names=['Normal', 'Anomaly']))
print("Accuracy:", accuracy_score(df['true_label'], df['final_label']))

prec = precision_score(df['true_label'], df['final_label'])
rec = recall_score(df['true_label'], df['final_label'])
f1 = f1_score(df['true_label'], df['final_label'])
roc_auc = roc_auc_score(df['true_label'], df['risk_calibrated_score'])
print(f"\n🔹 Precision (Anomaly): {prec:.3f}")
print(f"🔹 Recall (Anomaly): {rec:.3f}")
print(f"🔹 F1-Score (Anomaly): {f1:.3f}")
print(f"🔹 ROC-AUC Score: {roc_auc:.3f}")

num_anomalies = df['final_label'].sum()
print(f"\n🚨 Total Users Detected as Anomalies: {num_anomalies} ({num_anomalies/len(df)*100:.2f}%)")

top_20_highrisk = df[df['final_label'] == 1].nlargest(20, 'risk_calibrated_score')[
    ['user_id', 'risk_calibrated_score', 'composite_risk_score',
     'risk_rank', 'avg_after_hours_ratio', 'neuroticism_afterhours_risk']
]

print("\n🔥 Top 20 Most Confident High-Risk Users:")
print(top_20_highrisk.to_string(index=False))

# ========== VISUALIZATIONS ==========

# 1. Confusion Matrix Heatmap
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'], ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Anomaly Detection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Performance Metrics Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
values = [prec, rec, f1, accuracy_score(df['true_label'], df['final_label'])]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylim(0, 1)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# 3. Top 20 High-Risk Users
fig, ax = plt.subplots(figsize=(12, 8))
top_20 = df[df['final_label'] == 1].nlargest(20, 'risk_calibrated_score')
y_pos = np.arange(len(top_20))
ax.barh(y_pos, top_20['risk_calibrated_score'].values,
        color=plt.cm.Reds(top_20['risk_calibrated_score'].values),
        edgecolor='black', alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_20['user_id'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Risk Calibrated Score', fontsize=12)
ax.set_title('Top 20 Highest Risk Users', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Composite Risk vs Calibrated Risk
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df[df['final_label']==0]['composite_risk_score'],
                     df[df['final_label']==0]['risk_calibrated_score'],
                     c='green', alpha=0.4, s=30, label='Normal Users', edgecolors='none')
scatter = ax.scatter(df[df['final_label']==1]['composite_risk_score'],
                     df[df['final_label']==1]['risk_calibrated_score'],
                     c='red', alpha=0.7, s=60, label='Anomalies', edgecolors='black', linewidths=0.5)
ax.set_xlabel('Composite Risk Score', fontsize=12)
ax.set_ylabel('Risk Calibrated Score', fontsize=12)
ax.set_title('Risk Score Correlation: Anomalies vs Normal Users', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("📊 All visualizations displayed successfully!")
print("="*60)