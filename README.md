
<p align="center">
  <img src="Black Clean and Minimalist Project Overview Docs Banner.png" alt="Tweet Virality Predictor Banner" width="100%">
</p>



Engagement Pattern Analysis
Decoding What Makes a Tweet Go Viral Using Machine Learning

Project Overview :

This project explores what drives virality on Twitter — using data science, feature engineering, and machine learning.
It focuses on Charlie Kirk’s Twitter network, analyzing 500K+ tweets to uncover the patterns behind engagement and predict how well a tweet will perform.

Goal: Identify the key factors influencing tweet engagement (likes, retweets, replies) and build a model to predict tweet virality.

Dataset :

Source: Tweets from Charlie Kirk’s account and related user interactions.
Records: ~500,000 tweets
Attributes (16 columns):

pseudo_id, text, retweetCount, replyCount, likeCount, quoteCount, viewCount, bookmarkCount,
createdAt, lang, isReply, pseudo_conversationId, pseudo_inReplyToUsername,
pseudo_author_userName, quoted_pseudo_id, author_isBlueVerified

 Project Structure
📁 Engagement-Pattern-Analysis
│
├── 📄 feature_engg.py           # Feature creation & model training
├── 📄 interactive.py            # Interactive predictor
├── 📄 eda_visuals.ipynb         # Exploratory Data Analysis notebook
├── 📄 charlie_kirk_raw.csv      # Raw dataset
├── 📄 charlie_kirk_processed.csv # Processed dataset
├── 📊 visualizations/           # PNGs from analysis
├── 📄 README.md                 # Project documentation
└── 📄 requirements.txt          # Dependencies

Phases of the Project:

Phase 1: Data Loading & Exploration

Loaded raw dataset, performed basic cleaning

Checked data quality and null values

Derived initial engagement metrics

Created charlie_kirk_processed_phase1.csv

Phase 2: Exploratory Data Analysis (EDA)

Analyzed engagement distributions

Studied best-performing time slots and days

Explored patterns across hashtags, mentions, and punctuation

Compared viral vs. non-viral tweets

Created 6 high-quality visualizations

📂 Outputs:

visualizations/
├── engagement_distribution.png
├── time_of_day_vs_engagement.png
├── hashtag_effect.png
├── mention_density.png
├── viral_vs_nonviral_comparison.png
└── tweet_length_distribution.png

Phase 3: Feature Engineering & Modeling

Engineered 25+ advanced features, including:

Tweet length, timing, hashtags, mentions, URLs, punctuation

Hour-of-day, day-of-week, month

Sentiment indicators and reply/original flags

Trained 3 models:

Linear Regression

Random Forest

Gradient Boosting

Best Model: Random Forest

R²: 0.84

MAPE: 18%

Feature Importance: Highlights originality, timing, and structure as top drivers of engagement

📂 Outputs:

models/
├── random_forest_model.pkl
├── model_performance_comparison.png
├── feature_importance_chart.png
└── predictions_sample.csv

Phase 4: Insights Dashboard & Predictor

Built an executive-style dashboard summarizing findings

“Anatomy of a Viral Tweet” visualization

Actionable insights and posting recommendations

Developed an interactive engagement predictor where users can input tweet parameters and get predicted engagement

🧮 Example Usage
from interactive import predict_engagement

result = predict_engagement(
    hour=14,              # 0–23
    day_of_week=2,        # 0=Mon ... 6=Sun
    is_reply=False,       
    text_length=120,      
    num_hashtags=2,       
    num_mentions=1,       
    has_url=False,        
    has_exclamation=True  
)

print("🔮 Predicted Engagement:", round(result["predicted_engagement"], 2))
print("📈 Confidence Range:", result["confidence_range"])


📊 Example Output:

Predicted Engagement: 163
Confidence Range: 114 - 212
Feature Impact:
- Original Tweet Bonus : 22.07x
- Hashtag Impact       : 0.31x
- Mention Impact       : 0.13x
- URL Impact           : 3.51x

📊 Key Insights
Feature	Impact on Engagement
✅ Original tweets	+2107% boost
✅ URLs included	+251% boost
⚠️ Exclamation marks	+15%
❌ Hashtags	−68%
❌ Mentions	−87%
🕐 Midday posting	2× engagement

💡 Originality and timing outperform hashtags by miles.

🖼️ Visual Gallery
Insight	Visualization
Engagement distribution	engagement_distribution.png
Time-based trends	time_of_day_vs_engagement.png
Hashtag vs engagement	hashtag_effect.png
Viral vs Non-viral	viral_vs_nonviral_comparison.png
🧰 Tech Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, Plotly

Environment: Jupyter Notebook, PyCharm

Data: Twitter dataset (Charlie Kirk network)

📈 Results Summary

✅ Trained & compared 3 models

📊 Achieved R² = 0.84 on test data

🔍 Identified top-performing tweet characteristics

⚙️ Built an interactive engagement predictor

🧠 Derived actionable recommendations for tweet strategy

💬 Recommendations for Virality

Do This ✅

Post midday (best engagement hours)

Use original tweets instead of replies

Add subtle exclamations to enhance tone

Keep tweets concise (~150–170 characters)

Avoid This ❌

Overusing hashtags or mentions

Late-night posting

Short one-word tweets or lengthy rants

🧭 Future Enhancements

Integrate NLP-based sentiment analysis

Add topic detection (e.g., “politics”, “economy”)

Build a Streamlit dashboard for real-time predictions

Extend analysis to other influencer networks

Author:

Aditi Sikarwar
 B.E. Electronics & Computer Engineering
 Passionate about AI, ML, and Data-Driven Insights

🏁 Conclusion

This project goes beyond prediction — it uncovers why certain content resonates.
By merging data science, behavioral insight, and creativity, it builds a framework to understand the dynamics of virality.

“Data doesn’t just predict engagement — it tells stories about human connection.”
