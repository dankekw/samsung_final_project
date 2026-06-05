import os
import pandas as pd
from datasets import Dataset, load_dataset

def load_summarization_dataset(dataset_hf_link, train_size, test_size):
    fin_data = load_dataset(dataset_hf_link)
    fin_data_pd = fin_data['train'].to_pandas()

    train_data = fin_data_pd[['Article', 'Summary']].iloc[:train_size]
    test_data = fin_data_pd[['Article', 'Summary']].iloc[train_size:train_size+test_size]

    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    return train_dataset, train_dataset

def load_sentiment_dataset(dataset_hf_link):
    fin_semantic_ds = load_dataset("nickmuchi/financial-classification")
    
    return fin_semantic_ds["train"], fin_semantic_ds["test"]