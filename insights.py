# Phase 4: Insights Dashboard & Final Recommendations - FULLY FIXED VERSION
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys

warnings.filterwarnings('ignore')

print("=" * 80)
print("CHARLIE KIRK TWEET VIRALITY ANALYSIS - PHASE 4")
print("Insights Dashboard & Actionable Recommendations")
print("=" * 80)

# ============================================
# STEP 1: DEBUG - Show current directory and files
# ============================================
print("\n🔍 DEBUGGING INFORMATION:")
print("-" * 80)
print(f"Current working directory: {os.getcwd()}")
print(f"\nFiles in current directory:")

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if csv_files:
    for f in csv_files:
        file_size = os.path.getsize(f) / 1024
        print(f"  ✓ {f} ({file_size:.1f} KB)")
else:
    print("  ❌ No CSV files found in current directory")

print("-" * 80)

# ============================================
# STEP 2: Check for required files with flexible naming
# ============================================
print("\n📂 Checking for required files...")

def find_file(possible_names):
    for name in possible_names:
        if os.path.exists(name):
            return name
    return None

processed_data_file = find_file([
    'charlie_kirk_processed_phase2.csv',
    'charlie_kirk_phase2.csv',
    'processed_phase2.csv',
    'phase2.csv'
])

feature_importance_file = find_file([
    'feature_importance.csv',
    'features.csv'
])

model_comparison_file = find_file([
    'model_comparison.csv',
    'models.csv'
])

predictions_file = find_file([
    'model_predictions.csv',
    'predictions.csv'
])

print("\nFile Search Results:")
print("-" * 80)
print(f"Processed Data: {processed_data_file if processed_data_file else '❌ NOT FOUND'}")
print(f"Feature Importance: {feature_importance_file if feature_importance_file else '❌ NOT FOUND'}")
print(f"Model Comparison: {model_comparison_file if model_comparison_file else '❌ NOT FOUND'}")
print(f"Predictions: {predictions_file if predictions_file else '❌ NOT FOUND'}")
print("-" * 80)

# ============================================
# STEP 3: Decide which version to run
# ============================================
can_run_full = all([
    processed_data_file,
    feature_importance_file,
    model_comparison_file,
    predictions_file
])

if not can_run_full:
    print("\n⚠️  MISSING FILES DETECTED")
    print("=" * 80)
    missing = []
    if not processed_data_file:
        missing.append("Processed data (from Phase 2)")
    if not feature_importance_file:
        missing.append("Feature importance (from Phase 3)")
    if not model_comparison_file:
        missing.append("Model comparison (from Phase 3)")
    if not predictions_file:
        missing.append("Predictions (from Phase 3)")

    print("\nMissing files:")
    for m in missing:
        print(f"  ❌ {m}")

    print("\n🔧 TROUBLESHOOTING:")
    print("-" * 80)
    print("1. Make sure Phase 3 completed successfully")
    print("2. Check if files are in a different folder (like .venv/)")
    print("3. Move these files into the same folder as this script")

    if not processed_data_file:
        print("\n❌ Cannot proceed without Phase 2 data.")
        sys.exit(0)

    # --- Simplified Dashboard ---
    print("\n📊 Running SIMPLIFIED version (Phase 2 data only)")
    df = pd.read_csv(processed_data_file)
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    viral_threshold = df['total_engagement'].quantile(0.90)
    df['is_viral'] = df['total_engagement'] >= viral_threshold

    print(f"\n✓ Loaded {len(df):,} tweets")
    print(f"✓ Date range: {df['createdAt'].min().date()} to {df['createdAt'].max().date()}")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    fig.suptitle('Charlie Kirk Tweet Analysis - Insights Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['total_engagement'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(viral_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Viral: {viral_threshold:,.0f}')
    ax1.set_xlabel('Total Engagement')
    ax1.set_ylabel('Number of Tweets')
    ax1.set_title('Engagement Distribution', fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, df['total_engagement'].quantile(0.95))

    # Hours
    ax2 = fig.add_subplot(gs[0, 1])
    hourly_engagement = df.groupby('hour')['total_engagement'].mean()
    colors = ['#FF6B6B' if x == hourly_engagement.idxmax() else '#4ECDC4'
              for x in hourly_engagement.index]
    ax2.bar(hourly_engagement.index, hourly_engagement.values, color=colors)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Avg Engagement')
    ax2.set_title('Optimal Posting Hours', fontweight='bold')

    # Day of Week
    ax3 = fig.add_subplot(gs[0, 2])
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_engagement = df.groupby('day_name')['total_engagement'].mean().reindex(day_order)
    colors_day = ['#FF6B6B' if day in ['Saturday', 'Sunday'] else '#4ECDC4' for day in day_order]
    ax3.bar(range(7), daily_engagement.values, color=colors_day)
    ax3.set_xticks(range(7))
    ax3.set_xticklabels([d[:3] for d in day_order])
    ax3.set_ylabel('Avg Engagement')
    ax3.set_title('Day Performance', fontweight='bold')

    plt.tight_layout()
    plt.savefig('10_simplified_dashboard.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 10_simplified_dashboard.png")
    plt.show()
    sys.exit(0)

# ============================================
# FULL VERSION
# ============================================
print("\n✅ All files found! Loading data...")

df = pd.read_csv(processed_data_file)
df['createdAt'] = pd.to_datetime(df['createdAt'])
feature_importance = pd.read_csv(feature_importance_file)
model_comparison = pd.read_csv(model_comparison_file)
predictions = pd.read_csv(predictions_file)

print(f"✓ Loaded {len(df):,} tweets")
print(f"✓ Loaded {len(feature_importance)} features")
print(f"✓ Loaded {len(model_comparison)} model comparisons")
print(f"✓ Loaded {len(predictions):,} predictions")

# ============================================
# EXECUTIVE DASHBOARD
# ============================================
print("\n" + "=" * 80)
print("📊 Creating Executive Summary Dashboard")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

fig.suptitle('Charlie Kirk Tweet Virality Analysis - Executive Dashboard',
             fontsize=20, fontweight='bold', y=0.98)

# Header Metrics
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
metrics_text = f"""
📈 DATASET OVERVIEW                           🎯 MODEL PERFORMANCE                        🏆 TOP INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Tweets: {len(df):,}                    Best Model: {model_comparison.loc[model_comparison['Test_R2'].idxmax(), 'Model']}        Viral Threshold: {df['total_engagement'].quantile(0.9):,.0f}
Date Range: {df['createdAt'].min().strftime('%Y-%m-%d')} to {df['createdAt'].max().strftime('%Y-%m-%d')}           R² Score: {model_comparison['Test_R2'].max():.3f}                          Avg Engagement: {df['total_engagement'].mean():,.0f}
Daily Tweets: {len(df) / ((df['createdAt'].max() - df['createdAt'].min()).days + 1):.1f}                   Accuracy: {model_comparison['Test_R2'].max() * 100:.1f}%                             Peak Hour: {df.groupby('hour')['total_engagement'].mean().idxmax():.0f}:00
"""
ax1.text(0.05, 0.5, metrics_text, fontsize=11, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Engagement Distribution
ax2 = fig.add_subplot(gs[1, 0])
viral_threshold = df['total_engagement'].quantile(0.9)
ax2.hist(df['total_engagement'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(viral_threshold, color='red', linestyle='--', linewidth=2,
            label=f'Viral: {viral_threshold:,.0f}')
ax2.set_xlabel('Total Engagement')
ax2.set_ylabel('Number of Tweets')
ax2.set_title('Engagement Distribution', fontweight='bold')
ax2.legend()
ax2.set_xlim(0, df['total_engagement'].quantile(0.95))

# Optimal Posting Hours
ax3 = fig.add_subplot(gs[1, 1])
hourly_engagement = df.groupby('hour')['total_engagement'].mean()
colors = ['#FF6B6B' if x == hourly_engagement.idxmax() else '#4ECDC4'
          for x in hourly_engagement.index]
ax3.bar(hourly_engagement.index, hourly_engagement.values, color=colors)
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Avg Engagement')
ax3.set_title('Optimal Posting Hours', fontweight='bold')

# Day of Week
ax4 = fig.add_subplot(gs[1, 2])
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_engagement = df.groupby('day_name')['total_engagement'].mean().reindex(day_order)
colors_day = ['#FF6B6B' if day in ['Saturday', 'Sunday'] else '#4ECDC4' for day in day_order]
ax4.bar(range(7), daily_engagement.values, color=colors_day)
ax4.set_xticks(range(7))
ax4.set_xticklabels([d[:3] for d in day_order])
ax4.set_ylabel('Avg Engagement')
ax4.set_title('Day Performance', fontweight='bold')

# Top Features
ax5 = fig.add_subplot(gs[2, :])
top_10 = feature_importance.head(10)
ax5.barh(range(len(top_10)), top_10['Importance'], color='coral')
ax5.set_yticks(range(len(top_10)))
ax5.set_yticklabels(top_10['Feature'])
ax5.set_xlabel('Importance Score')
ax5.set_title('Top 10 Predictors', fontweight='bold')
ax5.invert_yaxis()

# --- ✅ FIXED SECTION: Content Impact ---
ax6 = fig.add_subplot(gs[3, 0])
possible_features = {
    'has_hashtag': ['has_hashtag', 'contains_hashtag', 'hashtags_present'],
    'has_mention': ['has_mention', 'contains_mention', 'mentions_present'],
    'has_url': ['has_url', 'contains_url', 'urls_present'],
    'has_exclamation': ['has_exclamation', 'contains_exclamation', 'exclamation_present']
}

content_features = []
for default_name, variants in possible_features.items():
    for v in variants:
        if v in df.columns:
            content_features.append(v)
            break

if not content_features:
    print("⚠️ No content-related columns found. Skipping Content Impact section.")
    ax6.text(0.5, 0.5, "No Content Features Found", ha='center', va='center', fontsize=12)
    ax6.axis('off')
else:
    baseline = df['total_engagement'].mean()
    impacts = [df[df[feat] == True]['total_engagement'].mean() / baseline for feat in content_features]
    labels = [feat.replace('has_', '').replace('contains_', '').capitalize() for feat in content_features]
    colors_content = ['green' if x > 1 else 'red' for x in impacts]
    ax6.barh(labels, impacts, color=colors_content)
    ax6.axvline(1, color='black', linestyle='--')
    ax6.set_xlabel('Multiplier')
    ax6.set_title('Content Impact', fontweight='bold')

# Tweet Type
ax7 = fig.add_subplot(gs[3, 1])
if 'isReply' in df.columns:
    original_eng = df[~df['isReply']]['total_engagement'].mean()
    reply_eng = df[df['isReply']]['total_engagement'].mean()
    ax7.bar(['Original', 'Reply'], [original_eng, reply_eng], color=['#FF6B6B', '#4ECDC4'])
    ax7.set_ylabel('Avg Engagement')
    ax7.set_title('Tweet Type', fontweight='bold')
else:
    ax7.text(0.5, 0.5, "No isReply column found", ha='center', va='center', fontsize=12)
    ax7.axis('off')

# Model Performance
ax8 = fig.add_subplot(gs[3, 2])
ax8.bar(range(len(model_comparison)), model_comparison['Test_R2'],
        color=['#95a5a6', '#3498db', '#e74c3c'])
ax8.set_xticks(range(len(model_comparison)))
ax8.set_xticklabels(model_comparison['Model'], fontsize=10)
ax8.set_ylabel('R² Score')
ax8.set_title('Model Performance', fontweight='bold')
ax8.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('10_executive_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 10_executive_dashboard.png")
plt.show()

print("\n" + "=" * 80)
print("✅ PHASE 4 COMPLETE!")
print("=" * 80)
print(f"\n🎯 Best Model: {model_comparison.loc[model_comparison['Test_R2'].idxmax(), 'Model']}")
print(f"📊 R² Score: {model_comparison['Test_R2'].max():.3f}")
print(f"🏆 Top Predictor: {feature_importance.iloc[0]['Feature']}")
print(f"⏰ Best Time: {int(hourly_engagement.idxmax())}:00")
print("\n✓ Dashboard saved successfully!")
print("=" * 80)
