# Tweet Sentiment Analysis

This project performs **sentiment classification on tweets**, identifying whether a tweet expresses a **positive** or **negative** sentiment. It leverages a fine-tuned transformer-based model hosted on Hugging Face, and demonstrates solid performance on real-world test data.

## The complete path
```
data/                                            
├── sentiment140_split.json                      # cleaned dataset
└── training.1600000.processed.noemoticon.csv    # dataset

models/                                         
└── bert/                                        # model folder
    ├── config.json
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt

report/                                          
├── test/                                       
│   ├── classification_report.txt                # classification report
│   ├── confusion_matrix.png                     # picture of confusion matrix
│   └── predictions.csv                          # predictions
└── val/                                         
    ├── classification_report.txt                # classification report
    ├── confusion_matrix.png                     # picture of confusion matrix
    └── predictions.csv                          # predictions

src/
├── data_acquisition/
│   └── download_dataset.py                      # The script for downloading data
├── model_download/
│   └── download_model.py                        # The script for downloading model
├── preprocessing/
│   └── json_transform.py                        # The script for preprocessing dataset
├── training/
│   ├── evaluate_model.py                        # The script for evaluating model using the trained model
│   ├── run_training.py                          # The script for runing train_model script
│   └── train_model.py                           # The script for training model
└── infer.py                                     # The script as an example about how to use this model
```

## Source of the dataset

https://www.kaggle.com/datasets/kazanova/sentiment140

## Model

The core of this project is a pre-trained and fine-tuned DistilBERT model available on Hugging Face:

**[Model on Hugging Face](https://huggingface.co/Charles1954/Tweet-sentiment-analysis/tree/main)**

- Architecture: `distilbert-base-uncased`
- Task: Binary sentiment classification (positive / negative)

## Evaluation

The model was evaluated on a test set of **95,999 tweets**. The following metrics summarize its performance:

### Classification Report

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.82      | 0.80   | 0.81     | 47,835  |
| Positive  | 0.80      | 0.82   | 0.81     | 48,164  |
| **Accuracy** | –         | –      | **0.81** | 95,999  |

- **Macro avg F1-score:** 0.81  
- **Weighted avg F1-score:** 0.81

This indicates that the model performs consistently across both classes, achieving strong balance between precision and recall.

### Confusion Matrix

The confusion matrix shows that the model maintains relatively balanced error rates between classes.

![confusion_matrix](https://github.com/user-attachments/assets/9fb1cd6a-285c-4168-ba3f-61f06a34ecfd)

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
