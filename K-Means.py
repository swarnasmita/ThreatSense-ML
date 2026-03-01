# ============================================================
# 📘 Behavioral Cluster Radar Chart (Aligned with Risk Table)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- Load the dataset ---
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

# --- Select behavioral + psychometric features ---
behavioral_features = [
    'avg_daily_activities', 'avg_unique_activities', 'avg_after_hours_ratio',
    'avg_unique_pcs', 'avg_logon_hour', 'avg_logoff_hour',
    'Openness', 'Conscientiousness', 'Extraversion',
    'Agreeableness', 'Neuroticism', 'neuroticism_afterhours_risk',
    'openness_activity_risk', 'composite_risk_score'
]

# --- Normalize for clustering ---
scaler = MinMaxScaler()
X = scaler.fit_transform(df[behavioral_features])

# --- K-Means clustering ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['behavior_cluster'] = kmeans.fit_predict(X)

# ============================================================
# 📈 Radar Chart Preparation
# ============================================================

# Selected features for interpretability
key_features = [
    'avg_after_hours_ratio',
    'avg_unique_pcs',
    'Conscientiousness',
    'openness_activity_risk',
    'neuroticism_afterhours_risk',
    'composite_risk_score'
]

# Compute mean per cluster
radar_data = df.groupby('behavior_cluster')[key_features].mean()

# Standardize + Normalize
scaler = StandardScaler()
radar_scaled = pd.DataFrame(scaler.fit_transform(radar_data), columns=key_features, index=radar_data.index)
radar_normalized = (radar_scaled - radar_scaled.min()) / (radar_scaled.max() - radar_scaled.min())

# ============================================================
# 🎯 Map clusters to behavioral interpretations
# ============================================================

cluster_labels = {
    0: "Cluster 0 Low Risk — Stable & Disciplined",
    1: "Cluster 1 Medium Risk — Overworked / Stressed",
    2: "Cluster 2 High Risk — Creative but Inconsistent",
    3: "Cluster 3 Critical Risk — Potential Insider Threat"
}

cluster_colors = {
    0: "#2ecc71",  # Green (Low Risk)
    1: "#e67e22",
    2: "#f1c40f",
    3: "#e74c3c"  # Red (Critical)
}

# ============================================================
# 🧭 Radar Chart Plot
# ============================================================

categories = key_features
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Axis Labels
plt.xticks(angles[:-1], categories, color='black', size=11, weight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.ylim(0, 1)

# Plot all clusters
for idx, (cluster, row) in enumerate(radar_normalized.iterrows()):
    values = row.tolist()
    values += values[:1]

    label = cluster_labels[cluster]
    color = cluster_colors[cluster]

    ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=color, label=label)
    ax.fill(angles, values, alpha=0.25, color=color)

# Title and legend
plt.title("Behavioral Cluster Profiles & Risk Classification",
          size=15, weight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)
plt.tight_layout()
plt.show()

# ============================================================
# 📊 Print Summary Table
# ============================================================

summary_table = pd.DataFrame({
    "Cluster": [0, 1, 2, 3],
    "Behavioral Interpretation": [
        "Low after-hours, high conscientiousness",
        "High after-hours + high neuroticism",
        "High openness + moderate risk",
        "High device usage + composite risk"
    ],
    "Risk Level": [
        "🟢 Stable & Disciplined",
        "🟠 Overworked / Stressed",
        "🟡 Creative but Inconsistent",
        "🔴 Potential Insider-Risk / Multitasking"
    ]
})
print("\n=== Behavioral Cluster Summary ===")
print(summary_table.to_string(index=False))
