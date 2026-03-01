# ============================================================
# Supervised Threat Classification with Multiple Models
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("⚡ ENHANCED SUPERVISED THREAT CLASSIFICATION")
print("=" * 80)

# ============================================================
# 📥 STEP 1: Load and Merge Datasets
# ============================================================

print("\n📥 Step 1: Loading datasets...")
base = pd.read_csv("final_user_anomaly_features.csv")
classified = pd.read_csv("final_threat_classification_results.csv")

# Merge on user_id
df = pd.merge(base, classified[['user_id', 'threat_label']], on='user_id', how='left')
df['threat_label'].fillna('Benign / Stable', inplace=True)

print(f"✅ Loaded {len(df):,} users")
print("\n📊 Threat Label Distribution:")
for label, count in df['threat_label'].value_counts().items():
    pct = count / len(df) * 100
    print(f"  {label:<30} {count:>6} ({pct:>5.2f}%)")

# Encode threat labels
le = LabelEncoder()
df['threat_label_encoded'] = le.fit_transform(df['threat_label'])
class_names = le.classes_

# ============================================================
# 🔧 STEP 2: Feature Engineering
# ============================================================

print("\n🔧 Step 2: Feature engineering...")

# Base features
base_features = [
    'avg_daily_activities', 'avg_after_hours_ratio',
    'avg_logon_hour', 'avg_logoff_hour', 'avg_unique_pcs',
    'neuroticism_afterhours_risk', 'openness_activity_risk',
    'composite_risk_score'
]

# Add personality traits if available
personality_features = ['Openness', 'Conscientiousness', 'Extraversion',
                        'Agreeableness', 'Neuroticism']
for feat in personality_features:
    if feat in df.columns:
        base_features.append(feat)

# Add engineered features
if 'avg_logon_hour' in df.columns and 'avg_logoff_hour' in df.columns:
    df['work_duration'] = df['avg_logoff_hour'] - df['avg_logon_hour']
    base_features.append('work_duration')

if 'avg_after_hours_ratio' in df.columns and 'avg_daily_activities' in df.columns:
    df['after_hours_intensity'] = df['avg_after_hours_ratio'] * df['avg_daily_activities']
    base_features.append('after_hours_intensity')

# Ensure all features exist
features = [f for f in base_features if f in df.columns]

print(f"✅ Using {len(features)} features for modeling")

X = df[features].fillna(0)
y = df['threat_label_encoded']

# ============================================================
# ⚖️ STEP 3: Data Preparation with SMOTE
# ============================================================

print("\n⚖️ Step 3: Preparing data with class balancing...")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n📊 Data Split:")
print(f"  Training: {len(X_train):,} samples")
print(f"  Testing:  {len(X_test):,} samples")

print("\n  Original training distribution:")
train_dist = pd.Series(y_train).value_counts().sort_index()
for threat_class, count in train_dist.items():
    print(f"    {class_names[threat_class]:<30} {count:>5}")

# Apply SMOTE
print("\n🔄 Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\n  Balanced training: {len(X_train_balanced):,} samples")
balanced_dist = pd.Series(y_train_balanced).value_counts().sort_index()
for threat_class, count in balanced_dist.items():
    print(f"    {class_names[threat_class]:<30} {count:>5}")

# ============================================================
# 🤖 STEP 4: Train Multiple Models
# ============================================================

print("\n" + "=" * 80)
print("🤖 TRAINING MULTIPLE MODELS")
print("=" * 80)

# Train Random Forest
print("\n🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)
print("✅ Random Forest trained successfully")

# ============================================================
# 📊 STEP 5: Comprehensive Model Evaluation
# ============================================================

print("\n" + "=" * 80)
print("📊 RANDOM FOREST MODEL EVALUATION")
print("=" * 80)

# Predictions
y_pred_train = rf_model.predict(X_train_balanced)
y_pred_test = rf_model.predict(X_test)

# Overall metrics
train_acc = accuracy_score(y_train_balanced, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
balanced_acc = balanced_accuracy_score(y_test, y_pred_test)

train_f1 = f1_score(y_train_balanced, y_pred_train, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, average='weighted')

test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)

print(f"\n📊 Overall Performance:")
print(f"  Training Accuracy:     {train_acc:.4f}")
print(f"  Test Accuracy:         {test_acc:.4f}")
print(f"  Balanced Accuracy:     {balanced_acc:.4f}")
print(f"  Test Precision:        {test_precision:.4f}")
print(f"  Test Recall:           {test_recall:.4f}")
print(f"  Test F1 Score:         {test_f1:.4f}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train_balanced, y_train_balanced,
                             cv=cv, scoring='f1_weighted')
print(f"  CV F1 Score (5-fold):  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_test, labels=range(len(class_names)), zero_division=0
)

print(f"\n📋 Per-Class Metrics (Test Set):")
print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 80)

for i, class_name in enumerate(class_names):
    print(f"{class_name:<30} {precision[i]:<12.3f} {recall[i]:<12.3f} "
          f"{f1[i]:<12.3f} {int(support[i]):<10}")

# Detailed classification report
print(f"\n📄 Detailed Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Random Forest', fontsize=14, weight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


# ============================================================
# 🎯 STEP 6: Feature Importance Analysis
# ============================================================

print("\n" + "=" * 80)
print("🎯 FEATURE IMPORTANCE ANALYSIS (Random Forest)")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Top 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))



# ============================================================
# 📊 BAR PLOTS: Accuracy, Precision, Recall, F1-Score
# ============================================================

plt.figure(figsize=(8, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [test_acc, test_precision, test_recall, test_f1]
colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

bars = plt.bar(metrics, values, color=colors, alpha=0.8)
plt.title('Model Performance Metrics (Random Forest)', fontsize=14, fontweight='bold')
plt.ylim(0, 1.1)
plt.ylabel('Score')

# Add value labels on top of each bar
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{bar.get_height():.3f}", ha='center', fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 🎯 FEATURE IMPORTANCE (Base Features Only)
# ============================================================

print("\n" + "=" * 80)
print("🎯 FEATURE IMPORTANCE (Base Behavioral Features)")
print("=" * 80)

# Define original base features (before any were added to the list)
original_base_features = [
    'avg_daily_activities', 'avg_after_hours_ratio',
    'avg_logon_hour', 'avg_logoff_hour', 'avg_unique_pcs',
    'neuroticism_afterhours_risk', 'openness_activity_risk',
    'composite_risk_score'
]

# Filter feature importance for only original base features that are in the model
base_feature_importance = feature_importance[
    feature_importance['Feature'].isin(original_base_features)
].copy()

# Sort by importance descending
base_feature_importance = base_feature_importance.sort_values(by='Importance', ascending=False)

print("\n📊 Base Feature Importances:")
if len(base_feature_importance) > 0:
    print(base_feature_importance.to_string(index=False))
else:
    print("⚠️  No base features found in the model.")

# --- Plot Base Feature Importance ---
if len(base_feature_importance) > 0:
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=base_feature_importance,
        x='Importance',
        y='Feature',
        palette='viridis',
        alpha=0.85
    )

    plt.title('Base Behavioral Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Behavioral Features', fontsize=12)
    plt.grid(axis='x', alpha=0.3)

    # Add numeric labels
    for index, value in enumerate(base_feature_importance['Importance']):
        plt.text(value + 0.002, index, f"{value:.3f}", va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("⚠️  Cannot plot - no base features available.")


# ============================================================
# 💾 STEP 7: Save Results
# ============================================================

print("\n💾 Saving results...")

# Save performance metrics
performance_metrics = {
    'Metric': ['Training Accuracy', 'Test Accuracy', 'Balanced Accuracy',
               'Test Precision', 'Test Recall', 'Test F1 Score',
               'CV F1 Mean', 'CV F1 Std'],
    'Score': [train_acc, test_acc, balanced_acc, test_precision,
              test_recall, test_f1, cv_scores.mean(), cv_scores.std()]
}
pd.DataFrame(performance_metrics).to_csv("rf_performance_metrics.csv", index=False)
print("✅ Saved performance metrics to 'rf_performance_metrics.csv'")

# Save per-class metrics
per_class_metrics = pd.DataFrame({
    'Class': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})
per_class_metrics.to_csv("rf_per_class_metrics.csv", index=False)
print("✅ Saved per-class metrics to 'rf_per_class_metrics.csv'")

# Save feature importance
feature_importance.to_csv("feature_importance_rf.csv", index=False)
print("✅ Saved feature importance to 'feature_importance_rf.csv'")

# Save predictions
df['predicted_threat'] = le.inverse_transform(rf_model.predict(X_scaled))
df['prediction_correct'] = df['threat_label'] == df['predicted_threat']

output_cols = ['user_id', 'threat_label', 'predicted_threat', 'prediction_correct']
df[output_cols].to_csv("threat_predictions.csv", index=False)
print("✅ Saved predictions to 'threat_predictions.csv'")

print("\n" + "=" * 80)
print("✅ RANDOM FOREST THREAT CLASSIFICATION COMPLETE")
print("=" * 80)
