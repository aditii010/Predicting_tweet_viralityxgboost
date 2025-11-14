from interactive import predict_engagement

# Example: customize your own tweet scenario
result = predict_engagement(
    hour=18,              # 6 PM
    day_of_week=4,        # Friday
    is_reply=False,        # Original tweet
    text_length=140,       # Length of tweet
    num_hashtags=2,        # Number of hashtags
    num_mentions=1,        # Number of mentions
    has_url=False,         # Contains URL?
    has_exclamation=True   # Contains exclamation mark?
)

# Print results
print("\n===============================")
print("🔮 TWEET ENGAGEMENT PREDICTION")
print("===============================")
print(f"Predicted Engagement: {result['predicted_engagement']:,}")
print(f"Confidence Range: {result['confidence_range'][0]:,} - {result['confidence_range'][1]:,}")
print("\nImpact Breakdown:")
for factor, impact in result["impact_breakdown"].items():
    print(f"  • {factor:25}: {impact}")
