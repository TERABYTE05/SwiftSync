import os
import re
from datasets import load_dataset, Audio, DatasetDict, concatenate_datasets, Features, Value
import config

def load_and_prepare_data():
    # It will load the Common Voice dataset from TSV files, cleans it and prepares it for training
    # This will return a dictionary containing the 'train' and 'test' splits cleaned

    column_names = [
        "client_id", "path", "sentence_id", "sentence", "sentence_domain",
        "up_votes", "down_votes", "age", "gender", "accents", "variant",
        "locale", "segment"
    ]
    feature_types = Features({col: Value("string") for col in column_names})

    # load dataset from .tsv files
    try:
        train_df = load_dataset(
            "csv",
            data_files = [os.path.join(config.root_path, "train.tsv")],
            delimiter = "\t",
            features = feature_types,
        )["train"]
        
        validated_df = load_dataset(
            "csv",
            data_files = [os.path.join(config.root_path, "validated.tsv")],
            delimiter = "\t",
            features = feature_types,
        )["train"]

        test_df = load_dataset(
            "csv",
            data_files = [os.path.join(config.root_path, "test.tsv")],
            delimiter = "\t",
            features = feature_types,
        )["train"]
        
    except FileNotFoundError as e:
        print("Could not find dataset files")
        print(e)
        exit()

    # Combine train and validated splits
    train_dataset = concatenate_datasets([train_df, validated_df])
    common_voice = DatasetDict({"train": train_dataset, "test": test_df})  

    # Remove unwanted columns
    remove_columns = ["client_id", "sentence_id", "sentence_domain", "up_votes", "down_votes", 
                      "age", "gender", "accents", "variant", "locale", "segment"] 
    common_voice = common_voice.remove_columns(remove_columns)

    return common_voice

if __name__ == '__main__':
    print("Loading and preparing data...")
    prepared_data = load_and_prepare_data()
    print("Sample from training set:")
    print(prepared_data["train"][0])