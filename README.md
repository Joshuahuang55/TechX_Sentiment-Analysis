# TechX Sentiment Analysis Assignment

This is my submission for the TechX "AI Basics" project.

I implemented a sentiment analysis script that uses **Ensemble Majority Voting** to improve accuracy. instead of relying on a single model, it combines predictions from three sources:

1.  **TextBlob** (Basic Lexicon)
2.  **VADER** (Rule-based)
3.  **RoBERTa** (Transformer model by Hugging Face)

## Why use Majority Voting?
During testing, I found that rule-based models (like VADER) often misclassify neutral commands (e.g., *"Please attend the meeting"*) as **Positive** because of polite words like "Please".

By using a majority vote with RoBERTa (which supports neutral labels), the system correctly identifies these as **Neutral**.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install textblob nltk transformers torch
    ```

2.  **Run the script:**
    ```bash
    python3 TechX_Sentiment_Analysis.py
    ```

---
**Author:** ChingHeng Huang
