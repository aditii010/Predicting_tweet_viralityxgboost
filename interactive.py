import pandas as pd
import numpy as np

print("=" * 80)
print("🔮 CHARLIE KIRK TWEET ENGAGEMENT PREDICTOR")
print("=" * 80)
print("\nThis tool predicts expected engagement based on tweet characteristics")
print("Using insights from the Random Forest model analysis")
print("=" * 80)

# Load the data
df = pd.read_csv('charlie_kirk_processed_phase2.csv')
feature_importance = pd.read_csv('feature_importance.csv')

# Compute base engagement and multipliers
base_engagement = df['total_engagement'].mean()
hour_impacts = df.groupby('hour')['total_engagement'].mean() / base_engagement
day_impacts = df.groupby('day_of_week')['total_engagement'].mean() / base_engagement

content_impacts = {
    'original_vs_reply': df[~df['isReply']]['total_engagement'].mean() / df[df['isReply']]['total_engagement'].mean(),
    'has_hashtag': df[df['hashtag_count'] > 0]['total_engagement'].mean() / df[df['hashtag_count'] == 0]['total_engagement'].mean(),
    'has_mention': df[df['mention_count'] > 0]['total_engagement'].mean() / df[df['mention_count'] == 0]['total_engagement'].mean(),
    'has_url': df[df['url_count'] > 0]['total_engagement'].mean() / df[df['url_count'] == 0]['total_engagement'].mean(),
    'has_exclamation': df[df['exclamation_count'] > 0]['total_engagement'].mean() / df[df['exclamation_count'] == 0]['total_engagement'].mean(),
}


def predict_engagement(
    hour=12,
    day_of_week=2,
    is_reply=False,
    text_length=100,
    num_hashtags=1,
    num_mentions=0,
    has_url=False,
    has_exclamation=False
):
    """
    Predict expected tweet engagement based on given parameters.
    Returns a dictionary with predicted engagement, confidence range, and breakdown.
    """
    prediction = base_engagement
    breakdown = {'Base Engagement': round(base_engagement, 2)}

    # Hour multiplier
    hour_mult = hour_impacts.get(hour, 1.0)
    prediction *= hour_mult
    breakdown['Hour Impact'] = f"{hour_mult:.2f}x"

    # Day multiplier
    day_mult = day_impacts.get(day_of_week, 1.0)
    prediction *= day_mult
    breakdown['Day Impact'] = f"{day_mult:.2f}x"

    # Original vs reply
    if not is_reply:
        prediction *= content_impacts['original_vs_reply']
        breakdown['Original Tweet Bonus'] = f"{content_impacts['original_vs_reply']:.2f}x"
    else:
        breakdown['Reply Penalty'] = "1.00x (no bonus)"

    # Hashtag multiplier
    if num_hashtags > 0:
        hashtag_mult = content_impacts['has_hashtag']
        prediction *= hashtag_mult
        breakdown['Hashtag Impact'] = f"{hashtag_mult:.2f}x"

    # Mention multiplier
    if num_mentions > 0:
        mention_mult = content_impacts['has_mention']
        prediction *= mention_mult
        breakdown['Mention Impact'] = f"{mention_mult:.2f}x"

    # URL multiplier
    if has_url:
        url_mult = content_impacts['has_url']
        prediction *= url_mult
        breakdown['URL Impact'] = f"{url_mult:.2f}x"

    # Exclamation multiplier
    if has_exclamation:
        exc_mult = content_impacts['has_exclamation']
        prediction *= exc_mult
        breakdown['Exclamation Impact'] = f"{exc_mult:.2f}x"

    # Text length adjustment
    optimal_length = df['text_length'].median()
    length_diff = abs(text_length - optimal_length) / optimal_length
    length_mult = 1 - (length_diff * 0.1)
    length_mult = max(0.8, min(1.2, length_mult))
    prediction *= length_mult
    breakdown['Length Adjustment'] = f"{length_mult:.2f}x"

    return {
        'predicted_engagement': round(prediction),
        'confidence_range': (round(prediction * 0.7), round(prediction * 1.3)),
        'impact_breakdown': breakdown
    }


# Only run this section if executed directly (not when imported)
if __name__ == "__main__":
    print("\n✅ Predictor module loaded successfully!\n")

    sample = predict_engagement(
        hour=14,
        day_of_week=3,
        is_reply=False,
        text_length=120,
        num_hashtags=2,
        num_mentions=1,
        has_url=False,
        has_exclamation=True
    )

    print("🔮 Predicted Engagement:", sample['predicted_engagement'])
    print("📊 Confidence Range:", sample['confidence_range'])
    print("🧩 Breakdown:")
    for key, val in sample['impact_breakdown'].items():
        print(f"  • {key:25}: {val}")

    print("\nModule ready to be imported in test.py ✅")
