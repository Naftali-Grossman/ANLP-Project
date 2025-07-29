import pandas as pd
from constants import MINIMUM_TOPIC_APPEARANCE, MAX_TOKENS_LENGTH
from clean_transcript import clean_transcript
import ast
from collections import Counter
from get_files import get_dataset

def load_data_to_pickle(tokenizer, transcript_model=True):
    needed_column = "transcript" if transcript_model else "title"
    df = get_dataset("ted_talks_en.csv")
    df[needed_column] = df[needed_column].apply(clean_transcript)

    df = df.dropna(subset=[needed_column, "topics"])

    # Convert the 'topics' column from string representation of list to actual list
    df['topics'] = df['topics'].apply(ast.literal_eval) 

    # remove low frequency topics
    tags_list = [tag for sublist in df['topics'] for tag in sublist]
    tags_counter = Counter(tags_list)
    low_freq_tags = {tag for tag, count in tags_counter.items() if count < MINIMUM_TOPIC_APPEARANCE}
    # tag not start with "TED"
    low_freq_tags = {tag for tag in low_freq_tags if not tag.startswith("TED")}
    df['topics'] = df['topics'].apply(lambda tags: [tag for tag in tags if tag not in low_freq_tags and not tag.startswith("TED")])
    df = df[df["topics"].map(len) > 0]

    # Check token count and filter out transcripts that exceed MAX_TOKENS_LENGTH
    def check_token_count(transcript):
        tokens = tokenizer(transcript, return_tensors="pt", truncation=False)
        return len(tokens["input_ids"][0])
    
    df["token_count"] = df[needed_column].apply(check_token_count)
    if transcript_model:
        df = df[df["token_count"] < MAX_TOKENS_LENGTH]
    return df
