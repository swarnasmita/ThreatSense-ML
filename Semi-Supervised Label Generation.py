# ============================================================
# 🚨 Unsupervised Threat Classification (Based on Anomaly + Behavior)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

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

# --- Define features for clustering ---
behavioral_features = [
    'avg_daily_activities', 'avg_unique_activities', 'avg_after_hours_ratio',
    'avg_unique_pcs', 'avg_logon_hour', 'avg_logoff_hour',
    'Openness', 'Conscientiousness', 'Extraversion',
    'Agreeableness', 'Neuroticism', 'neuroticism_afterhours_risk',
    'openness_activity_risk', 'composite_risk_score'
]

# --- Normalize data for K-Means ---
scaler = MinMaxScaler()
X = scaler.fit_transform(df[behavioral_features])

# ============================================================
# 🧠 Step 1: Behavioral Clustering
# ============================================================

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['behavior_cluster'] = kmeans.fit_predict(X)

# ============================================================
# 🔍 Step 2: Load or Merge with Isolation Forest Results
# ============================================================

if 'final_label' not in df.columns:
    print("⚠️ Warning: 'final_label' not found. Assigning pseudo anomalies (top 5% composite risk).")
    threshold = df['composite_risk_score'].quantile(0.95)
    df['final_label'] = (df['composite_risk_score'] >= threshold).astype(int)

# ============================================================
# 🧩 Step 3: Enhanced Threat Inference Logic (Balanced Distribution)
# ============================================================

# Compute composite threat indicator
df['combined_threat_score'] = (
        0.6 * df['composite_risk_score'] +
        0.25 * df['neuroticism_afterhours_risk'] +
        0.15 * df['openness_activity_risk']
)

# Normalize it
df['combined_threat_score'] = (
                                      df['combined_threat_score'] - df['combined_threat_score'].min()
                              ) / (df['combined_threat_score'].max() - df['combined_threat_score'].min())

# Define dynamic thresholds based on percentile
p_critical = df['combined_threat_score'].quantile(0.97)
p_high = df['combined_threat_score'].quantile(0.90)
p_medium = df['combined_threat_score'].quantile(0.70)


def infer_threat(row):
    # Critical (Malicious Insider): top 3% threat score OR anomaly with risky cluster
    if row['combined_threat_score'] >= p_critical or (row['final_label'] == 1 and row['behavior_cluster'] in [2, 3]):
        return "Malicious Insider"
    # High Risk: high risk score or anomaly in moderately risky cluster
    elif row['combined_threat_score'] >= p_high or (row['final_label'] == 1 and row['behavior_cluster'] == 1):
        return "Opportunistic / Suspicious"
    # Medium Risk: elevated behavioral risk (overworked / stressed)
    elif row['combined_threat_score'] >= p_medium or row['behavior_cluster'] == 1:
        return "Negligent / Overworked"
    # Low Risk: normal behavior, low composite risk
    else:
        return "Benign / Stable"


df['threat_label'] = df.apply(infer_threat, axis=1)

# ============================================================
# 📊 Step 4: K-MEANS CLUSTERING RESULTS & EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("🧠 K-MEANS CLUSTERING RESULTS")
print("=" * 80)

# Cluster distribution
print("\n📊 Cluster Distribution:")
cluster_dist = df['behavior_cluster'].value_counts().sort_index()
for cluster_id, count in cluster_dist.items():
    pct = count / len(df) * 100
    print(f"  Cluster {cluster_id}: {count:>6} users ({pct:>5.2f}%)")

# Cluster characteristics (centroids)
print("\n📈 Cluster Centroids (Key Features):")
cluster_profiles = df.groupby('behavior_cluster')[behavioral_features].mean()
print(cluster_profiles.round(3).to_string())

# Threat label distribution per cluster
print("\n🏷️ Threat Label Distribution by Cluster:")
threat_by_cluster = pd.crosstab(df['behavior_cluster'], df['threat_label'], normalize='index') * 100
print(threat_by_cluster.round(2).to_string())

# Overall threat distribution
print("\n🎯 Overall Threat Label Distribution:")
threat_dist = df['threat_label'].value_counts()
for label, count in threat_dist.items():
    pct = count / len(df) * 100
    print(f"  {label:<30} {count:>6} ({pct:>5.2f}%)")

# ============================================================
# 📈 Step 5: Summary Statistics per Threat Type
# ============================================================

print("\n" + "=" * 80)
print("📈 THREAT TYPE SUMMARY STATISTICS")
print("=" * 80)

summary = df.groupby('threat_label')[['avg_after_hours_ratio', 'avg_unique_pcs', 'composite_risk_score']].mean()
summary['count'] = df['threat_label'].value_counts()
summary['percentage'] = (summary['count'] / len(df) * 100).round(2)
print(summary.to_string())

# ============================================================
# 🧾 Step 6: Threat Classification Mapping Table
# ============================================================

print("\n" + "=" * 80)
print("🧾 THREAT CLASSIFICATION CATEGORIES")
print("=" * 80)

mapping_table = pd.DataFrame({
    "Threat Type": ["🟢 Benign / Stable", "🟡 Negligent / Overworked",
                    "🟠 Opportunistic / Suspicious", "🔴 Malicious Insider"],
    "Behavioral Traits": [
        "Low after-hours, high conscientiousness, normal activity",
        "High after-hours, fatigue, low consistency, stress risk",
        "Moderate risk-taking, high openness, curiosity-driven behavior",
        "High device usage, late logins, high neuroticism & risk score"
    ],
    "Risk Level": ["Low", "Medium", "High", "Critical"]
})
print(mapping_table.to_string(index=False))

# ============================================================
# 📊 Step 7: Visualizations
# ============================================================

# Threat distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='threat_label',
              order=['Benign / Stable', 'Negligent / Overworked',
                     'Opportunistic / Suspicious', 'Malicious Insider'],
              palette=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
plt.title("🧠 Threat Type Distribution (K-Means Based)", fontsize=14, weight='bold')
plt.xlabel("Count")
plt.ylabel("Threat Type")
plt.tight_layout()
plt.show()

