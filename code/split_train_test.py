from sklearn.utils import shuffle
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from ted_dataset import TEDDataset

def split_train_test(df, labels, tokenizer, is_transcript_model=True):
    column_name = 'transcript' if is_transcript_model else 'title'
    X = np.array(df[column_name].tolist()).reshape(-1, 1)  # needs 2D shape
    y = labels

    # Shuffle with a fixed random state for reproducibility
    X, y = shuffle(X, y, random_state=42)

    # First split: train (70%) vs temp (30%)
    X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.3)

    # Second split: eval (15%) vs test (15%) from temp (30%)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

    # Flatten text arrays
    train_texts = X_train.ravel().tolist()
    val_texts = X_val.ravel().tolist()
    test_texts = X_test.ravel().tolist()

    # Now build datasets
    train_dataset = TEDDataset(train_texts, y_train, tokenizer)
    val_dataset = TEDDataset(val_texts, y_val, tokenizer)
    test_dataset = TEDDataset(test_texts, y_test, tokenizer)


    return train_dataset, val_dataset, test_dataset
