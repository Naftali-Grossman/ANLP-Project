from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import numpy as np

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()  # apply sigmoid to get probabilities
    preds = (probs > 0.25).astype(int)  # convert to binary predictions
    labels = labels.astype(int)  # make sure labels are integers

    # Micro
    precision_micro = precision_score(labels, preds, average='micro')
    recall_micro = recall_score(labels, preds, average='micro')
    f1_micro = f1_score(labels, preds, average='micro')

    # Macro
    precision_macro = precision_score(labels, preds, average='macro')
    recall_macro = recall_score(labels, preds, average='macro')
    f1_macro = f1_score(labels, preds, average='macro')


    # === Global Top-2 Accuracy ===
    n_samples, n_labels = labels.shape
    top2_preds = np.argsort(probs, axis=1)[:, -2:]  # shape (n_samples, 2)

    # Flatten top-2 predictions and create a matching array of labels
    total_correct = 0
    for i in range(n_samples):
        true_indices = set(np.where(labels[i] == 1)[0])
        predicted_indices = top2_preds[i]
        total_correct += sum([1 for idx in predicted_indices if idx in true_indices])

    top2_precision = total_correct / (2 * n_samples)


    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'top2_precision': top2_precision
    }