
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visual style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("CHARLIE KIRK TWEET VIRALITY ANALYSIS - PHASE 1")
print("=" * 70)

# ============================================
# STEP 1: LOAD THE DATA
# ============================================
print("\n📂 STEP 1: Loading the dataset...")

# Load the CSV file
# Replace 'for_export_charlie_kirk.csv' with your actual file path
df = pd.read_csv('/Users/aditisikarwar/Downloads/for_export_charlie_kirk.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  - Total rows: {len(df):,}")
print(f"  - Total columns: {len(df.columns)}")

# ============================================
# STEP 2: BASIC DATA INSPECTION
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 2: Basic Data Inspection")
print("=" * 70)

# Display column names and types
print("\nColumn Information:")
print("-" * 70)
for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
    non_null = df[col].notna().sum()
    null_pct = (df[col].isna().sum() / len(df)) * 100
    print(f"{i:2d}. {col:30s} | Type: {str(dtype):10s} | Non-null: {non_null:7,} ({100-null_pct:.1f}%)")

# Display first few rows
print("\n" + "-" * 70)
print("First 5 rows of the dataset:")
print("-" * 70)
print(df.head())

# ============================================
# STEP 3: DATA QUALITY CHECK
# ============================================
print("\n" + "=" * 70)
print("🔍 STEP 3: Data Quality Assessment")
print("=" * 70)

# Check for missing values
print("\nMissing Values Analysis:")
print("-" * 70)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
else:
    print("✓ No missing values found!")

# Check for duplicates
print(f"\nDuplicate Rows: {df.duplicated().sum():,}")

# ============================================
# STEP 4: ENGAGEMENT METRICS ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("📈 STEP 4: Engagement Metrics Overview")
print("=" * 70)

# Key engagement columns
engagement_cols = ['likeCount', 'retweetCount', 'replyCount', 'quoteCount', 'viewCount', 'bookmarkCount']

# Summary statistics for engagement metrics
print("\nEngagement Statistics:")
print("-" * 70)
print(df[engagement_cols].describe().round(2))

# Calculate total engagement (our main target variable)
df['total_engagement'] = df['likeCount'] + df['retweetCount'] + df['replyCount'] + df['quoteCount']

print("\n" + "-" * 70)
print("Total Engagement Statistics:")
print("-" * 70)
print(f"Mean: {df['total_engagement'].mean():,.2f}")
print(f"Median: {df['total_engagement'].median():,.2f}")
print(f"Max: {df['total_engagement'].max():,.0f}")
print(f"Standard Deviation: {df['total_engagement'].std():,.2f}")

# ============================================
# STEP 5: TEMPORAL ANALYSIS SETUP
# ============================================
print("\n" + "=" * 70)
print("📅 STEP 5: Temporal Data Analysis")
print("=" * 70)

# Convert createdAt to datetime
df['createdAt'] = pd.to_datetime(df['createdAt'])

# Extract temporal features
df['year'] = df['createdAt'].dt.year
df['month'] = df['createdAt'].dt.month
df['day'] = df['createdAt'].dt.day
df['hour'] = df['createdAt'].dt.hour
df['day_of_week'] = df['createdAt'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_name'] = df['createdAt'].dt.day_name()

# Date range
print(f"\nDate Range:")
print(f"  Earliest tweet: {df['createdAt'].min()}")
print(f"  Latest tweet: {df['createdAt'].max()}")
print(f"  Time span: {(df['createdAt'].max() - df['createdAt'].min()).days} days")

# Tweets by year
print("\nTweets by Year:")
print(df['year'].value_counts().sort_index())

# ============================================
# STEP 6: CONTENT ANALYSIS BASICS
# ============================================
print("\n" + "=" * 70)
print("📝 STEP 6: Content Analysis Basics")
print("=" * 70)

# Language distribution
print("\nLanguage Distribution:")
print(df['lang'].value_counts().head(10))

# Tweet type distribution (original vs reply)
print(f"\nTweet Types:")
print(f"  Original tweets: {(~df['isReply']).sum():,} ({(~df['isReply']).sum()/len(df)*100:.1f}%)")
print(f"  Replies: {df['isReply'].sum():,} ({df['isReply'].sum()/len(df)*100:.1f}%)")

# Blue verification status
print(f"\nBlue Verification:")
print(df['author_isBlueVerified'].value_counts())

# ============================================
# STEP 7: IDENTIFY TOP PERFORMING TWEETS
# ============================================
print("\n" + "=" * 70)
print("🏆 STEP 7: Top Performing Tweets")
print("=" * 70)

# Top 5 tweets by total engagement
top_tweets = df.nlargest(5, 'total_engagement')[['text', 'total_engagement', 'likeCount', 'retweetCount', 'createdAt']]

print("\nTop 5 Most Engaging Tweets:")
print("-" * 70)
for i, row in top_tweets.iterrows():
    print(f"\n{row['createdAt'].strftime('%Y-%m-%d')} | Total Engagement: {row['total_engagement']:,.0f}")
    print(f"  Likes: {row['likeCount']:,.0f} | RTs: {row['retweetCount']:,.0f}")
    print(f"  Text: {row['text'][:100]}...")

# ============================================
# STEP 8: SAVE PROCESSED DATA
# ============================================
print("\n" + "=" * 70)
print("💾 STEP 8: Saving Processed Data")
print("=" * 70)

# Save the processed dataframe for next phase
df.to_csv('charlie_kirk_processed_phase1.csv', index=False)
print("✓ Processed data saved as 'charlie_kirk_processed_phase1.csv'")

print("\n" + "=" * 70)
print("✅ PHASE 1 COMPLETE!")
print("=" * 70)
print("\nNext Steps:")
print("  → Run Phase 2 for detailed exploratory data analysis")
print("  → We'll create visualizations and identify patterns")
print("=" * 70)
