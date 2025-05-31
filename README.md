# Persian Emotion Classification

A machine learning project for classifying emotions in Persian tweets using two approaches: Multinomial Naive Bayes and Class-Adaptive Regularization (CAR).

## Overview

This project implements and compares two models for emotion classification in Persian tweets:

- Multinomial Naive Bayes (MNB)
- Class-Adaptive Regularization (CAR)

The models classify tweets into six emotions: anger, fear, joy, sadness, disgust, and surprise.

## Dataset

The dataset used in this project is available on Kaggle:
[Persian Twitter Dataset for Sentiment Analysis](https://www.kaggle.com/datasets/mohammadalimkh/persian-twitter-dataset-sentiment-analysis)

## Performance

| Model | Accuracy | Macro Avg F1 |
| ----- | -------- | ------------ |
| MNB   | 81%      | 0.81         |
| CAR   | 90%      | 0.90         |

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- hazm (Persian NLP)
- tqdm
- imbalanced-learn

## Project Structure

```
.
├── dataset/
│   └── raw/           # Raw tweet data for each emotion
├── models/            # Saved model files
├── vocab/            # Vocabulary files
└── src/
    └── main.ipynb    # Main implementation notebook
```

## Usage

1. Install dependencies
2. Place your dataset in the `dataset/raw/` directory
3. Run the notebook `src/main.ipynb`

## Features

- Persian text preprocessing using Hazm
- TF-IDF vectorization
- Class imbalance handling
- Model evaluation and comparison

