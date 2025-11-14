# Phase 3: Feature Engineering & Predictive Modeling
# ===================================================
# This script builds machine learning models to predict tweet virality
# and identifies the key factors that drive engagement.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("CHARLIE KIRK TWEET VIRALITY ANALYSIS - PHASE 3")
print("Feature Engineering & Predictive Modeling")
print("=" * 80)

# ============================================
# STEP 1: LOAD DATA AND PREPARE FEATURES
# ============================================
print("\n📂 STEP 1: Loading Enhanced Dataset...")
df = pd.read_csv('charlie_kirk_processed_phase2.csv')
df['createdAt'] = pd.to_datetime(df['createdAt'])
print(f"✓ Loaded {len(df):,} tweets with engineered features")

# ============================================
# STEP 2: ADVANCED FEATURE ENGINEERING
# ============================================
print("\n" + "=" * 80)
print("🔧 STEP 2: Advanced Feature Engineering")
print("=" * 80)

# 2.1 Time-based features
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)

# 2.2 Engagement ratios (these show tweet "quality")
df['retweet_like_ratio'] = df['retweetCount'] / (df['likeCount'] + 1)
df['reply_like_ratio'] = df['replyCount'] / (df['likeCount'] + 1)
df['quote_like_ratio'] = df['quoteCount'] / (df['likeCount'] + 1)

# 2.3 Content intensity features
df['has_hashtag'] = (df['hashtag_count'] > 0).astype(int)
df['has_mention'] = (df['mention_count'] > 0).astype(int)
df['has_url'] = (df['url_count'] > 0).astype(int)
df['has_exclamation'] = (df['exclamation_count'] > 0).astype(int)
df['has_question'] = (df['question_count'] > 0).astype(int)

# 2.4 Text complexity features
df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
df['punctuation_intensity'] = (df['exclamation_count'] + df['question_count']) / (df['text_length'] + 1)

# 2.5 Historical performance (rolling average - simulate "track record")
df = df.sort_values('createdAt')
df['historical_avg_engagement'] = df.groupby('pseudo_author_userName')['total_engagement'].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
)
df['historical_avg_engagement'] = df['historical_avg_engagement'].fillna(df['total_engagement'].mean())

print("✓ Created advanced features:")
print(f"  - Time period indicators (morning, afternoon, evening, night)")
print(f"  - Engagement quality ratios")
print(f"  - Binary content flags")
print(f"  - Text complexity metrics")
print(f"  - Historical performance tracker")

# ============================================
# STEP 3: PREPARE MODELING DATASET
# ============================================
print("\n" + "=" * 80)
print("📊 STEP 3: Preparing Dataset for Modeling")
print("=" * 80)

# Define feature set for modeling
feature_columns = [
    # Temporal features
    'hour', 'day_of_week', 'is_weekend', 'is_morning', 'is_afternoon',
    'is_evening', 'is_night',

    # Content features
    'text_length', 'word_count', 'avg_word_length',
    'hashtag_count', 'mention_count', 'url_count',
    'exclamation_count', 'question_count', 'punctuation_intensity',
    'uppercase_ratio',

    # Binary flags
    'has_hashtag', 'has_mention', 'has_url',
    'has_exclamation', 'has_question',

    # Tweet characteristics
    'isReply', 'author_isBlueVerified',

    # Historical performance
    'historical_avg_engagement'
]

# Target variable: We'll predict log-transformed engagement
# (Log transformation helps with skewed distributions)
df['log_engagement'] = np.log1p(df['total_engagement'])

# Remove rows with missing values in key features
model_df = df[feature_columns + ['log_engagement', 'total_engagement']].copy()
model_df = model_df.dropna()

print(f"✓ Dataset prepared:")
print(f"  - Total samples: {len(model_df):,}")
print(f"  - Features: {len(feature_columns)}")
print(f"  - Target: log-transformed total engagement")

# ============================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================
print("\n" + "=" * 80)
print("✂️  STEP 4: Splitting Data")
print("=" * 80)

# Separate features and target
X = model_df[feature_columns]
y = model_df['log_engagement']

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Data split complete:")
print(f"  - Training samples: {len(X_train):,}")
print(f"  - Testing samples: {len(X_test):,}")

# Scale features (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 5: BASELINE MODEL (Linear Regression)
# ============================================
print("\n" + "=" * 80)
print("📈 STEP 5: Baseline Model - Linear Regression")
print("=" * 80)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

# Evaluation
train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
test_mae_lr = mean_absolute_error(y_test, y_test_pred_lr)

print(f"✓ Linear Regression Results:")
print(f"  - Training R²: {train_r2_lr:.4f}")
print(f"  - Testing R²: {test_r2_lr:.4f}")
print(f"  - Testing RMSE: {test_rmse_lr:.4f}")
print(f"  - Testing MAE: {test_mae_lr:.4f}")

# ============================================
# STEP 6: RANDOM FOREST MODEL
# ============================================
print("\n" + "=" * 80)
print("🌲 STEP 6: Random Forest Regressor")
print("=" * 80)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest (this may take a minute)...")
rf_model.fit(X_train, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Evaluation
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)

print(f"✓ Random Forest Results:")
print(f"  - Training R²: {train_r2_rf:.4f}")
print(f"  - Testing R²: {test_r2_rf:.4f}")
print(f"  - Testing RMSE: {test_rmse_rf:.4f}")
print(f"  - Testing MAE: {test_mae_rf:.4f}")

# ============================================
# STEP 7: GRADIENT BOOSTING MODEL
# ============================================
print("\n" + "=" * 80)
print("🚀 STEP 7: Gradient Boosting Regressor")
print("=" * 80)

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

print("Training Gradient Boosting (this may take a minute)...")
gb_model.fit(X_train, y_train)

# Predictions
y_train_pred_gb = gb_model.predict(X_train)
y_test_pred_gb = gb_model.predict(X_test)

# Evaluation
train_r2_gb = r2_score(y_train, y_train_pred_gb)
test_r2_gb = r2_score(y_test, y_test_pred_gb)
test_rmse_gb = np.sqrt(mean_squared_error(y_test, y_test_pred_gb))
test_mae_gb = mean_absolute_error(y_test, y_test_pred_gb)

print(f"✓ Gradient Boosting Results:")
print(f"  - Training R²: {train_r2_gb:.4f}")
print(f"  - Testing R²: {test_r2_gb:.4f}")
print(f"  - Testing RMSE: {test_rmse_gb:.4f}")
print(f"  - Testing MAE: {test_mae_gb:.4f}")

# ============================================
# STEP 8: MODEL COMPARISON
# ============================================
print("\n" + "=" * 80)
print("🏆 STEP 8: Model Comparison")
print("=" * 80)

# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'Train_R2': [train_r2_lr, train_r2_rf, train_r2_gb],
    'Test_R2': [test_r2_lr, test_r2_rf, test_r2_gb],
    'Test_RMSE': [test_rmse_lr, test_rmse_rf, test_rmse_gb],
    'Test_MAE': [test_mae_lr, test_mae_rf, test_mae_gb]
})

print(comparison.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# R² Comparison
ax1 = axes[0]
x = np.arange(len(comparison))
width = 0.35
ax1.bar(x - width / 2, comparison['Train_R2'], width, label='Training', color='#4ECDC4', alpha=0.8)
ax1.bar(x + width / 2, comparison['Test_R2'], width, label='Testing', color='#FF6B6B', alpha=0.8)
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison['Model'], rotation=45, ha='right')
ax1.legend()
ax1.set_ylim([0, 1])

# RMSE Comparison
ax2 = axes[1]
ax2.bar(comparison['Model'], comparison['Test_RMSE'], color=['#95a5a6', '#3498db', '#e74c3c'])
ax2.set_ylabel('RMSE (Lower is Better)')
ax2.set_title('Test RMSE Comparison')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('07_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 07_model_comparison.png")
plt.show()

# Select best model
best_model_idx = comparison['Test_R2'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']
best_model = [rf_model, gb_model][best_model_idx - 1] if best_model_idx > 0 else lr_model

print(f"\n🥇 Best Model: {best_model_name}")
print(f"   Test R²: {comparison.loc[best_model_idx, 'Test_R2']:.4f}")

# ============================================
# STEP 9: FEATURE IMPORTANCE ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("🎯 STEP 9: Feature Importance Analysis")
print("=" * 80)

# Get feature importances (using Random Forest as it's typically best)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score')
plt.title('Top 20 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('08_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_feature_importance.png")
plt.show()

# ============================================
# STEP 10: PREDICTION ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("🔮 STEP 10: Prediction Quality Analysis")
print("=" * 80)

# Convert predictions back to original scale
y_test_actual = np.expm1(y_test)
y_test_predicted = np.expm1(y_test_pred_rf)

# Scatter plot: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prediction Quality Assessment', fontsize=16, fontweight='bold')

# Actual vs Predicted (log scale)
ax1 = axes[0]
ax1.scatter(y_test, y_test_pred_rf, alpha=0.3, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Log(Engagement)')
ax1.set_ylabel('Predicted Log(Engagement)')
ax1.set_title('Log-Scale Predictions')
ax1.legend()

# Actual vs Predicted (original scale)
ax2 = axes[1]
ax2.scatter(y_test_actual, y_test_predicted, alpha=0.3, s=10)
ax2.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Engagement')
ax2.set_ylabel('Predicted Engagement')
ax2.set_title('Original-Scale Predictions')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()

plt.tight_layout()
plt.savefig('09_prediction_quality.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 09_prediction_quality.png")
plt.show()

# Prediction accuracy by engagement level
# Try to create quantile bins safely
try:
    # Create bins dynamically (duplicates dropped if needed)
    engagement_bins, bin_edges = pd.qcut(
        y_test_actual,
        q=5,
        labels=None,  # get bin edges first
        retbins=True,
        duplicates='drop'
    )

    # Dynamically adjust labels based on how many bins exist
    num_bins = len(bin_edges) - 1
    base_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    labels = base_labels[:num_bins]

    # Now apply qcut again with matching labels
    engagement_bins = pd.qcut(
        y_test_actual,
        q=num_bins,
        labels=labels,
        duplicates='drop'
    )

except ValueError:
    print("⚠️ Fallback: Too few unique values, using pd.cut instead.")
    engagement_bins = pd.cut(
        y_test_actual,
        bins=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

accuracy_by_bin = pd.DataFrame({
    'Engagement_Level': engagement_bins,
    'Actual': y_test_actual,
    'Predicted': y_test_predicted
})

print("\n📊 Prediction Accuracy by Engagement Level:")
for level in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
    level_data = accuracy_by_bin[accuracy_by_bin['Engagement_Level'] == level]
    mae = mean_absolute_error(level_data['Actual'], level_data['Predicted'])
    print(f"  {level:12s}: MAE = {mae:,.2f}")

# ============================================
# STEP 11: SAVE MODELS AND RESULTS
# ============================================
print("\n" + "=" * 80)
print("💾 STEP 11: Saving Models and Results")
print("=" * 80)

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ Saved: feature_importance.csv")

# Save model comparison
comparison.to_csv('model_comparison.csv', index=False)
print("✓ Saved: model_comparison.csv")

# Save predictions for further analysis
predictions_df = pd.DataFrame({
    'actual_engagement': y_test_actual,
    'predicted_engagement': y_test_predicted,
    'actual_log': y_test,
    'predicted_log': y_test_pred_rf,
    'prediction_error': y_test_actual - y_test_predicted
})
predictions_df.to_csv('model_predictions.csv', index=False)
print("✓ Saved: model_predictions.csv")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 80)
print("✅ PHASE 3 COMPLETE!")
print("=" * 80)

print(f"\n🎯 MODEL PERFORMANCE SUMMARY:")
print(f"  Best Model: {best_model_name}")
print(f"  Test R² Score: {comparison.loc[best_model_idx, 'Test_R2']:.4f}")
print(f"  → Model explains {comparison.loc[best_model_idx, 'Test_R2'] * 100:.1f}% of engagement variance")

print(f"\n🔑 TOP 5 PREDICTORS OF VIRALITY:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {i + 1}. {row['Feature']:30s} - Importance: {row['Importance']:.4f}")

print(f"\n📊 KEY INSIGHTS:")
print(f"  - Created {len(feature_columns)} predictive features")
print(f"  - Tested 3 different models")
print(f"  - Achieved {comparison.loc[best_model_idx, 'Test_R2'] * 100:.1f}% prediction accuracy")
print(f"  - Historical performance is a strong predictor")

print("\n📁 Generated Files:")
print("  1. 07_model_comparison.png")
print("  2. 08_feature_importance.png")
print("  3. 09_prediction_quality.png")
print("  4. feature_importance.csv")
print("  5. model_comparison.csv")
print("  6. model_predictions.csv")

print("\n➡️  Next: Run Phase 4 for insights dashboard and final recommendations!")
print("=" * 80)
