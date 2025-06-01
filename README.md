# Tweet Sentiment Analysis

This project performs **sentiment classification on tweets**, identifying whether a tweet expresses a **positive** or **negative** sentiment. It leverages a fine-tuned transformer-based model hosted on Hugging Face, and demonstrates solid performance on real-world test data.

## Source of the dataset

https://www.kaggle.com/datasets/kazanova/sentiment140

## Model

The core of this project is a pre-trained and fine-tuned DistilBERT model available on Hugging Face:

**[Model on Hugging Face](https://huggingface.co/Charles1954/Tweet-sentiment-analysis)**

- Architecture: `distilbert-base-uncased`
- Task: Binary sentiment classification (positive / negative)

## Evaluation

The model was evaluated on a test set of **31,999 tweets**. The following metrics summarize its performance:

### Classification Report

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.81      | 0.82   | 0.82     | 16035   |
| Positive  | 0.82      | 0.81   | 0.82     | 15964   |
| **Accuracy** | -       | -      | **0.82** | 31999   |

> Macro avg F1-score: **0.82**

### Confusion Matrix

- **True Negative (TN)**: 13,201
- **False Positive (FP)**: 2,834
- **False Negative (FN)**: 3,014
- **True Positive (TP)**: 12,950

The confusion matrix shows balanced performance with similar error rates across both classes.

## Model Defects and Improvement Plan

- **Model Defects**: The model is not perfect and may have some issues with class imbalance.
- **Defects**: 
  - This model may not be suitable for Judging neutral statements, as it is trained on a binary sentiment classification task (dataset does not include neutral statements).
  - The model may not be suitable for long texts, as it is limited to a maximum input length of 128 tokens.
- **Improvement Plan**: To improve the model's performance, we can try the following:
  - Collect more data, including neutral statements.
  - Use a different pre-trained model with a larger dataset, such as RoBERTa.
  - Use a different fine-tuning strategy, such as transfer learning or distillation.

## Usage

To load and use the model for inference:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="Charles1954/Tweet-sentiment-analysis")
classifier("I like pizza!")  # → [{'label': 'positive', 'score': ...}]
