import os
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from load_data_to_pickle import load_data_to_pickle
from compute_metrics import compute_metrics
from split_train_test import split_train_test
import argparse


MODEL_NAME = "full_transcript_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tune_thresholds(probs, true_labels, thresholds=np.linspace(0.1, 0.9, 9)):
    n_classes = true_labels.shape[1]
    best_thresholds = []

    for i in range(n_classes):
        best_f1 = 0
        best_thresh = 0.5  # default
        for thresh in thresholds:
            pred = (probs[:, i] > thresh).astype(int)
            f1 = f1_score(true_labels[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        best_thresholds.append(best_thresh)

    return np.array(best_thresholds)

def train_model(is_transcript_model=True, save_model_name = MODEL_NAME):
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = load_data_to_pickle(tokenizer, is_transcript_model)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['topics'])
    num_labels = labels.shape[1]
    train_dataset, val_dataset, _ = split_train_test(df, labels, tokenizer, is_transcript_model)
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
).to(DEVICE)
    
    training_args = TrainingArguments(
    output_dir=f'./{save_model_name}',
    run_name=f'{save_model_name}',
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=1000,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    learning_rate=4.376302413966977e-05,
    weight_decay=0.2857263130755122,
    logging_dir='./logs',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained(f"saved_models/{save_model_name}")
    tokenizer.save_pretrained(f"saved_models/{save_model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_full_transcript_model", type=bool, default=True)
    parser.add_argument("--save_model_name", type=str, default=MODEL_NAME)
    args = parser.parse_args()
    train_model(args.is_full_transcript_model, args.save_model_name)
