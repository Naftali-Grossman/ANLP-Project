import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score
from load_data_to_pickle import load_data_to_pickle
import matplotlib.pyplot as plt
from constants import MAX_TOKENS_LENGTH, TED_TOPICS
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import Dataset
import os
from ted_dataset import TEDDataset
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
import argparse

from get_files import get_dataset

os.environ["WANDB_DISABLED"] = "true"





def plot_f1_and_count_single_axis(df_part, title):
    x = np.arange(len(df_part))
    width = 0.4
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width/2, df_part['f1'], width, label='F1 Score', color='tab:blue')
    ax.bar(x + width/2, df_part['count'], width, label='Normalized Train Count', color='tab:orange')
    ax.set_ylabel("F1 Score / Normalized Count")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df_part['label'], rotation=90)
    ax.set_xlabel("Labels")
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_f1_and_count_both_axes(val_true_labels, pred_labels, train_dataset, mlb, df, title):
    
    f1_per_class = f1_score(val_true_labels, pred_labels, average=None)

    label_counts = sum(train_dataset.labels)  # shape: (num_labels,)
    labels = mlb.fit_transform(df['topics'])
    num_labels = labels.shape[1]
    label_names = mlb.classes_
    counts_normalized = label_counts / len(train_dataset)

    df_metrics = pd.DataFrame({
        'label': label_names,
        'f1': f1_per_class,
        'count': counts_normalized
    })

    # Sort by F1 for splitting
    df_sorted = df_metrics.sort_values(by='f1', ascending=False).reset_index(drop=True)

    # Split into top and bottom halves
    half = len(df_sorted) // 2
    df_top = df_sorted.iloc[:half]
    df_bottom = df_sorted.iloc[half:]

    # Plotting F1 score and count for top and bottom halves
    plot_f1_and_count_single_axis(df_top, "Top Half of Labels by F1 Score")
    plot_f1_and_count_single_axis(df_bottom, "Bottom Half of Labels by F1 Score")
    

    

def test_model_from_scratch(from_scratch=False, is_full_transcript_model=True, manual_model = False, model_name = ""):
    #Loading model and tokenizer
    if not manual_model:
        model_name = "Naftali1996/Full-Transcript" if is_full_transcript_model else "Naftali1996/Titles"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = load_data_to_pickle(tokenizer, is_full_transcript_model)
    mlb = MultiLabelBinarizer()

    
    if is_full_transcript_model:
    # Loading datasets, 70% train, 15% val, 15% test
        train_dataset = get_dataset("train_dataset.pkl")
        val_dataset = get_dataset("val_dataset.pkl")
        test_dataset = get_dataset("test_dataset.pkl")
    else:
        train_dataset = get_dataset("train_title_dataset.pkl")
        val_dataset = get_dataset("val_title_dataset.pkl")
        test_dataset = get_dataset("test_title_dataset.pkl")

    #Loading datasets
    args = TrainingArguments(
    output_dir="./temp",
    per_device_eval_batch_size=1  # Reduce memory usage
)
    trainer =Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    if from_scratch:
        test_outputs = trainer.predict(test_dataset)
        val_outputs = trainer.predict(val_dataset)
    else:
        if is_full_transcript_model:
            val_outputs = get_dataset("val_outputs_transcript.pkl")
            test_outputs = get_dataset("test_outputs_transcript.pkl")
        else:
            val_outputs = get_dataset("val_outputs_title.pkl")
            test_outputs = get_dataset("test_outputs_title.pkl")


    val_logits = val_outputs.predictions
    val_true_labels = val_outputs.label_ids

    # Apply sigmoid if logits are raw
    val_probs = 1 / (1 + np.exp(-val_logits))
    test_probs = 1 / (1 + np.exp(-test_outputs.predictions))

    # Apply per-label threshold
    best_thresholds = tune_thresholds(val_probs, val_true_labels)
    pred_labels = (val_probs > best_thresholds).astype(int)

    title = "Full Transcript Model" if is_full_transcript_model else "Full Title Model"

    compute_metrics_2(test_outputs.predictions, test_dataset.labels, best_thresholds,plot=True, title=title)

    if is_full_transcript_model:
        plot_f1_and_count_both_axes(val_true_labels, pred_labels, train_dataset, mlb, df, "Top Half of Labels by F1 Score")
        plot_reliability(test_probs, best_thresholds, test_dataset)    
        plot_metric_vs_duration(test_dataset,test_probs, best_thresholds, "F1 Score")
        plot_metrics_by_percent(test_dataset, tokenizer, trainer, best_thresholds)
        generalized_domains(tokenizer, trainer, best_thresholds)

def compute_metrics_2(logits, labels, best_thresholds,plot=False, title=None):
    probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()  # apply sigmoid to get probabilities
    preds = (probs > best_thresholds).astype(int)

    # Micro
    precision_micro = precision_score(labels, preds, average='micro')
    recall_micro = recall_score(labels, preds, average='micro')
    f1_micro = f1_score(labels, preds, average='micro')

    # Macro
    precision_macro = precision_score(labels, preds, average='macro')
    recall_macro = recall_score(labels, preds, average='macro')
    f1_macro = f1_score(labels, preds, average='macro')

    # Global Top-2 Accuracy
    n_samples, n_labels = labels.shape
    top2_preds = np.argsort(probs, axis=1)[:, -2:]
    total_correct = 0
    for i in range(n_samples):
        true_indices = set(np.where(labels[i] == 1)[0])
        predicted_indices = top2_preds[i]
        total_correct += sum([1 for idx in predicted_indices if idx in true_indices])

    top2_precision = total_correct / (2 * n_samples)

    # Collect metrics
    metrics = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'top2_precision': top2_precision
    }

    # === Plot as table ===
    if plot:
        df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        fig, ax = plt.subplots(figsize=(8, len(df)*0.5 + 1))
        ax.axis('off')

        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        plt.title(f"Metrics for {title}")
        plt.show()

    return metrics
def plot_metric_and_count_single_axis(df_part, metric_col, metric_name, title):
    x = np.arange(len(df_part))
    width = 0.4

    fig, ax = plt.subplots(figsize=(16, 6))

    # counts_scaled = df_part['count'] / df_part['count'].max()

    ax.bar(x - width/2, df_part[metric_col], width, label=metric_name, color='tab:blue')
    ax.bar(x + width/2, df_part['count'], width, label='Normalized Train Count', color='tab:orange')

    ax.set_ylabel(f"{metric_name} / Normalized Count")
    ax.set_ylim(0, 1.05)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df_part['label'], rotation=90)
    ax.set_xlabel("Labels")

    ax.legend()
    fig.tight_layout()
    plt.show()

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

def plot_reliability(test_probs, best_thresholds, test_dataset, n_bins=10):
    preds = (test_probs > best_thresholds).astype(int)

    confidences = test_probs.flatten()
    predictions = preds.flatten()
    truth = test_dataset.labels.flatten()

    # Only consider predicted positives for calibration
    predicted_positives = predictions == 1
    confidences = confidences[predicted_positives]
    correct = (predictions == truth)[predicted_positives]
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1

    ece = 0.0
    total = len(confidences)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        idx = bin_indices == i
        if np.any(idx):
            bin_acc = correct[idx].float().mean()
            bin_conf = confidences[idx].mean()
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(idx.sum())

            bin_size = idx.sum()
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(bin_confs, bin_accs, marker='o', label='Reliability')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Percision')
    plt.title(f'Reliability Diagram - ECE={ece:.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_sample_metrics(true_labels, pred_labels):
    f1_list = []

    for true, pred in zip(true_labels, pred_labels):
        f1 = f1_score(true, pred, zero_division=0)
        f1_list.append(f1)
    return f1_list

def plot_metric_vs_duration(test_dataset,test_probs, best_thresholds, metric_name):
    pred_labels = (test_probs > best_thresholds).astype(int)
    true_labels = test_dataset.labels
    metric_values = []

    for true, pred in zip(true_labels, pred_labels):
        f1 = f1_score(true, pred, zero_division=0)
        metric_values.append(f1)
    
    lengths = [len(t.split()) for t in test_dataset.transcripts]

    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, metric_values, alpha=0.6, label=metric_name, color='tab:blue')
    plt.title(f'{metric_name} vs. Transcript Length')
    plt.xlabel('Transcript Length (words)')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    plt.show()



class TruncatedTestDataset(Dataset):
    def __init__(self, df, tokenizer, labels, percent):
        self.texts = df['transcript'].apply(
            lambda x: ' '.join(x.split()[:max(1, int(len(x.split()) * percent / 100))])
        ).tolist()
        self.encodings = tokenizer(self.texts, truncation=True, padding=True, max_length=MAX_TOKENS_LENGTH, return_tensors='pt')
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def plot_metrics_by_percent(test_dataset, tokenizer, trainer, best_thresholds):
    # metrics_df = pd.DataFrame(metrics_by_percent)

    percentages = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    metrics_by_percent = []
    test_df = pd.DataFrame({'transcript': test_dataset.transcripts})

    for p in percentages:
        print(f"Evaluating {p}% of transcript...")

        truncated_dataset = TruncatedTestDataset(test_df, tokenizer, test_dataset.labels, p)
        outputs = trainer.predict(truncated_dataset)
        metrics = compute_metrics_2(outputs.predictions, truncated_dataset.labels,best_thresholds)
        metrics['percent'] = p
        metrics_by_percent.append(metrics)
    metrics_df = pd.DataFrame(metrics_by_percent)
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['percent'], metrics_df['f1_macro'], label='F1 Macro', marker='o')
    plt.plot(metrics_df['percent'], metrics_df['precision_macro'], label='Precision Macro', marker='x')
    plt.plot(metrics_df['percent'], metrics_df['recall_macro'], label='Recall Macro', marker='s')
    plt.plot(metrics_df['percent'], metrics_df['top2_precision'], label='Top2 Precision', marker='D', color='purple')

    plt.title("Model Performance vs Transcript Percentage")
    plt.xlabel("Percentage of Transcript Used")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()





def generate_data(df, text_name, tags_name, ted_topics, test_size):
  mlb = MultiLabelBinarizer(classes=ted_topics)
  mlb.fit([])  # fit to initialize with classes

  labels = mlb.transform(df[tags_name])
  num_labels = labels.shape[1]
  print(num_labels, len(ted_topics))

  X = np.array(df[text_name].tolist()).reshape(-1, 1)
  y = labels

  # Shuffle with a fixed random state for reproducibility
  X, y = shuffle(X, y, random_state=42)

  # First split: train (70%) vs temp (30%)
  X, y, _, _ = iterative_train_test_split(X, y, test_size=1-test_size)

  # Flatten text arrays
  X = X.ravel().tolist()

  return X, y, labels

def generalized_domains(tokenizer= None, trainer= None, best_thresholds= None):
    mn_ds_df = get_dataset("mn_ds_df.pkl")
    mn_ds_X, mn_ds_y, mn_ds_labels = generate_data(mn_ds_df, "content", "new_tags", TED_TOPICS, 0.8)
    mn_ds_dataset = TEDDataset(mn_ds_X, mn_ds_y, tokenizer)
    mn_ds_test_outputs = trainer.predict(mn_ds_dataset)
    compute_metrics_2(mn_ds_test_outputs.predictions, mn_ds_test_outputs.label_ids, best_thresholds, plot=True, title="News Articles")


    podcasts_episodess_df = get_dataset('podcasts_episodess_df.pkl')
    podcasts_episodess_X, podcasts_episodess_y, podcasts_episodess_labels = generate_data(podcasts_episodess_df, "summary", "new_tags", TED_TOPICS, 0.8)
    podcasts_episodess_dataset = TEDDataset(podcasts_episodess_X, podcasts_episodess_y, tokenizer)
    podcasts_episodess_test_outputs = trainer.predict(podcasts_episodess_dataset)
    compute_metrics_2(podcasts_episodess_test_outputs.predictions, podcasts_episodess_test_outputs.label_ids, best_thresholds, plot=True, title="Podcasts")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_scratch", type=bool, default=False)
    parser.add_argument("--is_full_transcript_model", type=bool, default=True)
    args = parser.parse_args()
    test_model_from_scratch(args.from_scratch, args.is_full_transcript_model)