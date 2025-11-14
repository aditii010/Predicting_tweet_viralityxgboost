# Phase 2: Exploratory Data Analysis & Visualization
# ===================================================
# This script performs comprehensive EDA to uncover patterns in Charlie Kirk's tweets
# and understand what drives engagement.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Enhanced plotting settings
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("CHARLIE KIRK TWEET VIRALITY ANALYSIS - PHASE 2")
print("Exploratory Data Analysis & Visualization")
print("=" * 80)

# ============================================
# LOAD PROCESSED DATA FROM PHASE 1
# ============================================
print("\n📂 Loading processed data...")
df = pd.read_csv('charlie_kirk_processed_phase1.csv')
df['createdAt'] = pd.to_datetime(df['createdAt'])
print(f"✓ Loaded {len(df):,} tweets")

# ============================================
# SECTION 1: ENGAGEMENT DISTRIBUTION ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("📊 SECTION 1: Understanding Engagement Distribution")
print("=" * 80)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Engagement Metrics Distribution', fontsize=16, fontweight='bold')

# 1.1 Total Engagement Distribution
ax1 = axes[0, 0]
ax1.hist(df['total_engagement'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Total Engagement')
ax1.set_ylabel('Number of Tweets')
ax1.set_title('Distribution of Total Engagement')
ax1.axvline(df['total_engagement'].median(), color='red', linestyle='--',
            label=f'Median: {df["total_engagement"].median():,.0f}')
ax1.axvline(df['total_engagement'].mean(), color='green', linestyle='--',
            label=f'Mean: {df["total_engagement"].mean():,.0f}')
ax1.legend()

# 1.2 Log-scale Distribution (better for skewed data)
ax2 = axes[0, 1]
log_engagement = np.log1p(df['total_engagement'])
ax2.hist(log_engagement, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Log(Total Engagement + 1)')
ax2.set_ylabel('Number of Tweets')
ax2.set_title('Log-Transformed Engagement Distribution')

# 1.3 Likes vs Retweets
ax3 = axes[1, 0]
ax3.scatter(df['likeCount'], df['retweetCount'], alpha=0.3, s=10)
ax3.set_xlabel('Likes')
ax3.set_ylabel('Retweets')
ax3.set_title('Likes vs Retweets Relationship')
ax3.set_xscale('log')
ax3.set_yscale('log')

# 1.4 Engagement Type Breakdown
ax4 = axes[1, 1]
engagement_types = ['likeCount', 'retweetCount', 'replyCount', 'quoteCount']
avg_engagement = [df[col].mean() for col in engagement_types]
ax4.bar(engagement_types, avg_engagement, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax4.set_ylabel('Average Count')
ax4.set_title('Average Engagement by Type')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('01_engagement_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_engagement_distribution.png")
plt.show()

# Define viral threshold (top 10%)
viral_threshold = df['total_engagement'].quantile(0.90)
df['is_viral'] = df['total_engagement'] >= viral_threshold

print(f"\n📈 Engagement Insights:")
print(f"  - Viral threshold (top 10%): {viral_threshold:,.0f} total engagement")
print(f"  - Viral tweets: {df['is_viral'].sum():,} ({df['is_viral'].sum()/len(df)*100:.1f}%)")
print(f"  - Average engagement (viral): {df[df['is_viral']]['total_engagement'].mean():,.0f}")
print(f"  - Average engagement (non-viral): {df[~df['is_viral']]['total_engagement'].mean():,.0f}")

# ============================================
# SECTION 2: TEMPORAL PATTERNS
# ============================================
print("\n" + "=" * 80)
print("📅 SECTION 2: When Does Virality Happen?")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Temporal Engagement Patterns', fontsize=16, fontweight='bold')

# 2.1 Engagement by Hour of Day
ax1 = axes[0, 0]
hourly_engagement = df.groupby('hour').agg({
    'total_engagement': 'mean',
    'pseudo_id': 'count'
}).reset_index()
hourly_engagement.columns = ['hour', 'avg_engagement', 'tweet_count']

ax1_twin = ax1.twinx()
ax1.bar(hourly_engagement['hour'], hourly_engagement['avg_engagement'],
        alpha=0.7, color='steelblue', label='Avg Engagement')
ax1_twin.plot(hourly_engagement['hour'], hourly_engagement['tweet_count'],
              color='red', marker='o', linewidth=2, label='Tweet Count')

ax1.set_xlabel('Hour of Day (24-hour format)')
ax1.set_ylabel('Average Engagement', color='steelblue')
ax1_twin.set_ylabel('Number of Tweets', color='red')
ax1.set_title('Engagement & Activity by Hour of Day')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# 2.2 Engagement by Day of Week
ax2 = axes[0, 1]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_engagement = df.groupby('day_name')['total_engagement'].mean().reindex(day_order)
colors = ['#FF6B6B' if day in ['Saturday', 'Sunday'] else '#4ECDC4' for day in day_order]
ax2.bar(range(7), daily_engagement.values, color=colors)
ax2.set_xticks(range(7))
ax2.set_xticklabels([d[:3] for d in day_order], rotation=0)
ax2.set_ylabel('Average Engagement')
ax2.set_title('Engagement by Day of Week')

# 2.3 Tweet Volume Over Time
ax3 = axes[1, 0]
df['year_month'] = df['createdAt'].dt.to_period('M')
monthly_tweets = df.groupby('year_month').size()
monthly_tweets.plot(ax=ax3, color='purple', linewidth=2)
ax3.set_xlabel('Time')
ax3.set_ylabel('Number of Tweets')
ax3.set_title('Tweet Volume Over Time')
ax3.tick_params(axis='x', rotation=45)

# 2.4 Engagement Over Time
ax4 = axes[1, 1]
monthly_engagement = df.groupby('year_month')['total_engagement'].mean()
monthly_engagement.plot(ax=ax4, color='green', linewidth=2, marker='o')
ax4.set_xlabel('Time')
ax4.set_ylabel('Average Engagement')
ax4.set_title('Average Engagement Trend Over Time')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('02_temporal_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_temporal_patterns.png")
plt.show()

# Best times to tweet
best_hour = hourly_engagement.loc[hourly_engagement['avg_engagement'].idxmax(), 'hour']
best_day = daily_engagement.idxmax()

print(f"\n⏰ Best Timing Insights:")
print(f"  - Best hour to tweet: {int(best_hour)}:00 (avg engagement: {hourly_engagement[hourly_engagement['hour']==best_hour]['avg_engagement'].values[0]:,.0f})")
print(f"  - Best day to tweet: {best_day} (avg engagement: {daily_engagement[best_day]:,.0f})")
print(f"  - Most active hour: {int(hourly_engagement.loc[hourly_engagement['tweet_count'].idxmax(), 'hour'])}:00")

# ============================================
# SECTION 3: CONTENT ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("📝 SECTION 3: Content Characteristics Analysis")
print("=" * 80)

# 3.1 Extract basic text features
df['text_length'] = df['text'].astype(str).str.len()
df['word_count'] = df['text'].astype(str).str.split().str.len()
df['hashtag_count'] = df['text'].astype(str).str.count('#')
df['mention_count'] = df['text'].astype(str).str.count('@')
df['url_count'] = df['text'].astype(str).str.count('http')
df['exclamation_count'] = df['text'].astype(str).str.count('!')
df['question_count'] = df['text'].astype(str).str.count('\?')
df['uppercase_ratio'] = df['text'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Content Features vs Engagement', fontsize=16, fontweight='bold')

# 3.1 Text Length
ax1 = axes[0, 0]
ax1.scatter(df['text_length'], df['total_engagement'], alpha=0.2, s=10)
ax1.set_xlabel('Text Length (characters)')
ax1.set_ylabel('Total Engagement')
ax1.set_title('Text Length vs Engagement')
ax1.set_yscale('log')

# 3.2 Word Count
ax2 = axes[0, 1]
bins = [0, 10, 20, 30, 40, 50, 100, 500]
df['word_bin'] = pd.cut(df['word_count'], bins=bins)
word_engagement = df.groupby('word_bin', observed=True)['total_engagement'].mean()
word_engagement.plot(kind='bar', ax=ax2, color='teal')
ax2.set_xlabel('Word Count Range')
ax2.set_ylabel('Average Engagement')
ax2.set_title('Word Count vs Engagement')
ax2.tick_params(axis='x', rotation=45)

# 3.3 Hashtags
ax3 = axes[0, 2]
hashtag_engagement = df.groupby('hashtag_count')['total_engagement'].mean().head(10)
ax3.bar(hashtag_engagement.index, hashtag_engagement.values, color='orange')
ax3.set_xlabel('Number of Hashtags')
ax3.set_ylabel('Average Engagement')
ax3.set_title('Hashtags vs Engagement')

# 3.4 Mentions
ax4 = axes[1, 0]
mention_engagement = df.groupby('mention_count')['total_engagement'].mean().head(10)
ax4.bar(mention_engagement.index, mention_engagement.values, color='purple')
ax4.set_xlabel('Number of Mentions')
ax4.set_ylabel('Average Engagement')
ax4.set_title('Mentions vs Engagement')

# 3.5 URLs
ax5 = axes[1, 1]
url_engagement = df.groupby('url_count')['total_engagement'].mean().head(5)
ax5.bar(url_engagement.index, url_engagement.values, color='green')
ax5.set_xlabel('Number of URLs')
ax5.set_ylabel('Average Engagement')
ax5.set_title('URLs vs Engagement')

# 3.6 Punctuation Impact
ax6 = axes[1, 2]
has_exclamation = df[df['exclamation_count'] > 0]['total_engagement'].mean()
no_exclamation = df[df['exclamation_count'] == 0]['total_engagement'].mean()
has_question = df[df['question_count'] > 0]['total_engagement'].mean()
no_question = df[df['question_count'] == 0]['total_engagement'].mean()

categories = ['Has !\nmarks', 'No !\nmarks', 'Has ?\nmarks', 'No ?\nmarks']
values = [has_exclamation, no_exclamation, has_question, no_question]
colors_punct = ['#FF6B6B', '#FFB6B6', '#4ECDC4', '#A0E4E4']
ax6.bar(categories, values, color=colors_punct)
ax6.set_ylabel('Average Engagement')
ax6.set_title('Punctuation Impact on Engagement')

plt.tight_layout()
plt.savefig('03_content_features.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_content_features.png")
plt.show()

print(f"\n📊 Content Insights:")
print(f"  - Average text length: {df['text_length'].mean():.0f} characters")
print(f"  - Average word count: {df['word_count'].mean():.1f} words")
print(f"  - Tweets with hashtags: {(df['hashtag_count'] > 0).sum():,} ({(df['hashtag_count'] > 0).sum()/len(df)*100:.1f}%)")
print(f"  - Tweets with mentions: {(df['mention_count'] > 0).sum():,} ({(df['mention_count'] > 0).sum()/len(df)*100:.1f}%)")
print(f"  - Tweets with URLs: {(df['url_count'] > 0).sum():,} ({(df['url_count'] > 0).sum()/len(df)*100:.1f}%)")

# ============================================
# SECTION 4: TWEET TYPE ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("🔄 SECTION 4: Tweet Type Performance")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Tweet Type Analysis', fontsize=16, fontweight='bold')

# 4.1 Original vs Reply Performance
ax1 = axes[0]
tweet_type_engagement = df.groupby('isReply')['total_engagement'].mean()
labels = ['Original Tweet', 'Reply']
colors = ['#FF6B6B', '#4ECDC4']
ax1.bar(labels, tweet_type_engagement.values, color=colors)
ax1.set_ylabel('Average Engagement')
ax1.set_title('Original Tweets vs Replies')

# Add value labels on bars
for i, v in enumerate(tweet_type_engagement.values):
    ax1.text(i, v + 50, f'{v:,.0f}', ha='center', fontweight='bold')

# 4.2 Blue Verified Impact
ax2 = axes[1]
verified_engagement = df.groupby('author_isBlueVerified')['total_engagement'].mean()
labels_verified = ['Not Verified', 'Blue Verified']
colors_verified = ['#95a5a6', '#1DA1F2']
ax2.bar(labels_verified, verified_engagement.values, color=colors_verified)
ax2.set_ylabel('Average Engagement')
ax2.set_title('Verification Status Impact')

# Add value labels
for i, v in enumerate(verified_engagement.values):
    ax2.text(i, v + 50, f'{v:,.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('04_tweet_types.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_tweet_types.png")
plt.show()

print(f"\n🎭 Tweet Type Insights:")
print(f"  - Original tweets avg engagement: {df[~df['isReply']]['total_engagement'].mean():,.0f}")
print(f"  - Replies avg engagement: {df[df['isReply']]['total_engagement'].mean():,.0f}")
print(f"  - Engagement boost for originals: {(df[~df['isReply']]['total_engagement'].mean() / df[df['isReply']]['total_engagement'].mean() - 1) * 100:.1f}%")

# ============================================
# SECTION 5: CORRELATION ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("🔗 SECTION 5: Feature Correlation Analysis")
print("=" * 80)

# Select features for correlation
correlation_features = [
    'total_engagement', 'likeCount', 'retweetCount', 'replyCount',
    'text_length', 'word_count', 'hashtag_count', 'mention_count',
    'url_count', 'exclamation_count', 'question_count', 'hour', 'day_of_week'
]

# Create correlation matrix
correlation_matrix = df[correlation_features].corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('05_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_correlation_matrix.png")
plt.show()

# Find strongest correlations with total_engagement
engagement_correlations = correlation_matrix['total_engagement'].sort_values(ascending=False)
print("\n🎯 Top Correlations with Total Engagement:")
print(engagement_correlations.head(8))

# ============================================
# SECTION 6: VIRAL vs NON-VIRAL COMPARISON
# ============================================
print("\n" + "=" * 80)
print("🚀 SECTION 6: Viral vs Non-Viral Tweet Characteristics")
print("=" * 80)

# Compare features
comparison_features = ['text_length', 'word_count', 'hashtag_count', 'mention_count',
                       'url_count', 'exclamation_count', 'question_count']

viral_stats = df[df['is_viral']][comparison_features].mean()
non_viral_stats = df[~df['is_viral']][comparison_features].mean()

comparison_df = pd.DataFrame({
    'Viral': viral_stats,
    'Non-Viral': non_viral_stats,
    'Difference (%)': ((viral_stats - non_viral_stats) / non_viral_stats * 100)
})

print("\n📊 Feature Comparison:")
print(comparison_df.round(2))

# Visualize comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_features))
width = 0.35

bars1 = ax.bar(x - width/2, viral_stats.values, width, label='Viral', color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x + width/2, non_viral_stats.values, width, label='Non-Viral', color='#4ECDC4', alpha=0.8)

ax.set_xlabel('Features')
ax.set_ylabel('Average Count')
ax.set_title('Viral vs Non-Viral Tweet Characteristics', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f.replace('_', ' ').title() for f in comparison_features], rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('06_viral_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_viral_comparison.png")
plt.show()

# ============================================
# SAVE ENHANCED DATASET
# ============================================
print("\n" + "=" * 80)
print("💾 Saving Enhanced Dataset")
print("=" * 80)

# Save with all new features
df.to_csv('charlie_kirk_processed_phase2.csv', index=False)
print("✓ Saved: charlie_kirk_processed_phase2.csv")

print("\n" + "=" * 80)
print("✅ PHASE 2 COMPLETE!")
print("=" * 80)
print("\n📈 Summary of Findings:")
print(f"  - Analyzed {len(df):,} tweets")
print(f"  - Created 6 visualization sets")
print(f"  - Extracted {len(comparison_features)} content features")
print(f"  - Identified viral threshold: {viral_threshold:,.0f} engagement")
print(f"\n🎯 Key Insights for Modeling:")
print(f"  - Best posting time: {int(best_hour)}:00, {best_day}")
print(f"  - Original tweets outperform replies by {(df[~df['isReply']]['total_engagement'].mean() / df[df['isReply']]['total_engagement'].mean() - 1) * 100:.1f}%")
print(f"  - Strongest engagement predictor: {engagement_correlations.index[1]} (r={engagement_correlations.values[1]:.3f})")
print("\n📊 Generated Visualizations:")
print("  1. 01_engagement_distribution.png")
print("  2. 02_temporal_patterns.png")
print("  3. 03_content_features.png")
print("  4. 04_tweet_types.png")
print("  5. 05_correlation_matrix.png")
print("  6. 06_viral_comparison.png")
print("\n➡️  Next: Run Phase 3 for feature engineering and model building!")
print("=" * 80)
