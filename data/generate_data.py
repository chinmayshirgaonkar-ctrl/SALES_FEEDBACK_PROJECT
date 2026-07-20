import pandas as pd
import numpy as np
from textblob import TextBlob
import random
from datetime import datetime, timedelta
import os

print("Generating big data (100,000 rows)... This may take a minute.")

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
NUM_RECORDS = 5000   
REGIONS = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
CATEGORIES = ['Electronics', 'Clothing', 'Home & Garden', 'Software', 'Toys']

# Sample feedback texts mapped to general ratings
feedback_samples = {
    'positive': [
        "Absolutely love this product!", "Great quality, will buy again.", 
        "Fast shipping and excellent customer service.", "Exceeded my expectations.",
        "Best purchase I've made all year.", "Highly recommend this to everyone."
    ],
    'neutral': [
        "It's okay, nothing special.", "Average quality for the price.", 
        "Arrived on time, works as expected.", "Does the job, but could be better.",
        "Not bad, but I've seen better."
    ],
    'negative': [
        "Terrible experience, broke after one use.", "Customer support was useless.", 
        "Waste of money, do not buy.", "Item arrived damaged.",
        "Complete garbage, requesting a refund."
    ]
}

# Generate Data
dates = [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(int(NUM_RECORDS))]
regions = np.random.choice(REGIONS, NUM_RECORDS)
categories = np.random.choice(CATEGORIES, NUM_RECORDS)
sales_amounts = np.round(np.random.uniform(10.0, 5000.0, NUM_RECORDS), 2)

# Generate feedback based on a weighted distribution
sentiments_keys = np.random.choice(['positive', 'neutral', 'negative'], NUM_RECORDS, p=[0.6, 0.25, 0.15])
feedbacks = [random.choice(feedback_samples[key]) for key in sentiments_keys]

# Assign numerical ratings based on sentiment categories
ratings = []
for key in sentiments_keys:
    if key == 'positive':
        ratings.append(random.choice([4, 5]))
    elif key == 'neutral':
        ratings.append(3)
    else:
        ratings.append(random.choice([1, 2]))

df = pd.DataFrame({
    'Date': dates,
    'Region': regions,
    'Category': categories,
    'Sales_Amount': sales_amounts,
    'Rating': ratings,
    'Feedback': feedbacks
})

print("Calculating TextBlob Sentiment... (This is processed upfront for dashboard speed)")

# Calculate polarity and assign label
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

sentiment_results = df['Feedback'].apply(get_sentiment)
df['Sentiment_Label'] = [res[0] for res in sentiment_results]
df['Sentiment_Polarity'] = [res[1] for res in sentiment_results]

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), 'sales_feedback_data.csv')
df.to_csv(output_path, index=False)

print(f"Success! {NUM_RECORDS} rows generated and saved to {output_path}")
