# ---------------------------------------------------------------
# TechX Sentiment Analysis
# Description:
#   I implement an ensemble sentiment analyzer using three models:
#   1. TextBlob (Lexicon-based)
#   2. VADER (Lexicon/Rule-based)
#   3. RoBERTa (Transformer-based, specialized for tweets/social media)
#
#   Instead of calculating a simple average,
#   I implement Majority Voting.
#   
#   Logic:
#   1. Each model gives a vote: "Positive", "Negative", or "Neutral".
#   2. The final sentiment is the one with the most votes.
#   3. If there is a tie , we default to the  most advanced model's decision.
# ---------------------------------------------------------------

# Import Packages
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk
from collections import Counter

# Ensure VADER lexicon exists
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# Initialize Models
print("Loading models, please wait...")

vader = SentimentIntensityAnalyzer()

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
bert_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------

# Converts a numerical score to a label.
def get_sentiment_label(score, threshold=0.1):
    if score > threshold:
        return "Positive"
    elif score < -threshold:
        return "Negative"
    else:
        return "Neutral"

# Get score from each model
def get_textblob_data(text):
    score = TextBlob(text).sentiment.polarity
    return score, get_sentiment_label(score)

def get_vader_data(text):
    score = vader.polarity_scores(text)["compound"]
    # VADER often over-scores polite words, so we can stick to standard logic
    return score, get_sentiment_label(score)

def get_bert_data(text):
    result = bert_pipeline(text)[0]
    score = result["score"]
    label_raw = result["label"]

    # Normalize labels and scores
    if label_raw == "neutral":
        final_score = 0.0
        final_label = "Neutral"
    elif label_raw == "positive":
        final_score = score
        final_label = "Positive"
    elif label_raw == "negative":
        final_score = -score
        final_label = "Negative"
    else:
        final_score = 0.0
        final_label = "Neutral"
        
    return final_score, final_label


# Ensemble: Majority Voting
def analyze_sentiment_voting(text):
    # 1. Get data from all models
    s_blob, l_blob = get_textblob_data(text)
    s_vader, l_vader = get_vader_data(text)
    s_bert, l_bert = get_bert_data(text)

    # 2. Collect Votes
    votes = [l_blob, l_vader, l_bert]
    vote_counts = Counter(votes)
    
    # 3. Determine Winner
    winner, count = vote_counts.most_common(1)[0]
    
    # Tie-breaking Logic 
    # If we have a 3-way tie (1 Pos, 1 Neg, 1 Neu), count is 1.
    # In this case, we trust BERT (RoBERTa) as the tie-breaker because it's the smartest model.
    if count == 1:
        final_decision = l_bert + " (BERT Decision)"
    else:
        final_decision = winner

    return {
        "TextBlob": (s_blob, l_blob),
        "VADER": (s_vader, l_vader),
        "BERT": (s_bert, l_bert),
        "Votes": dict(vote_counts),
        "Final Decision": final_decision
    }

# ------------------------------------------------
# Main Execution
# ------------------------------------------------

text = input("Enter a sentence: ")

results = analyze_sentiment_voting(text)

print("\n=== Ensemble Sentiment Analysis (Majority Voting) ===")

print(f"TextBlob : {results['TextBlob'][0]:.4f} [{results['TextBlob'][1]}]")
print(f"VADER    : {results['VADER'][0]:.4f} [{results['VADER'][1]}]")
print(f"BERT     : {results['BERT'][0]:.4f} [{results['BERT'][1]}]")

print(f"Vote Tally: {results['Votes']}")
print(f"Final Sentiment: {results['Final Decision']}")