import os
import re
from datasets import load_dataset, Audio, DatasetDict, concatenate_datasets, Features, Value
from . import config

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
    
    # Map audio path and clean data
    def map_audio_path(batch):
        batch["audio"] = os.path.join(config.clips_path, batch["path"])
        return batch

    common_voice = common_voice.map(map_audio_path, num_proc=1)

    # Clean the transcription text
    chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\‘\”\\']"
    def remove_special_characters(batch):
        if batch["sentence"]:
            batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
        else:
            batch["sentence"] = ""
        return batch

    common_voice = common_voice.map(remove_special_characters, num_proc=1)
    
    # Filter out empty sentences
    common_voice = common_voice.filter(lambda example: example['sentence'] is not None and len(example['sentence']) > 0)

    # Load audio and resample
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16_000))

    return common_voice

if __name__ == '__main__':
    print("Loading and preparing data...")
    prepared_data = load_and_prepare_data()
    print("Sample from training set:")
    print(prepared_data["train"][0])