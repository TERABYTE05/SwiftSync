import re
from datasets import load_dataset, DatasetDict

def load_and_prepare_datasets():
    # Load the dataset from HF and preprocess dataset
    print('Loading dataset...')
    dataset = DatasetDict()
    dataset['train'] = load_dataset('mozilla-foundation/common_voice_17_0', 'fr', split='train', token = True, trust_remote_code = True)
    dataset['test'] = load_dataset('mozilla-foundation/common_voice_17_0', 'fr', split='test', token = True, trust_remote_code = True)
    print('Loaded data successfully...')

    # Selecting only subset of data
    print('Selecting subset of data for training and testing...')
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(100000)) 
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(10000))
    
    # Remove unnecessary columns
    print('Removing unnecessary columns...')
    cols_to_remove = ['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant']
    dataset = dataset.remove_columns(cols_to_remove)

    # Remove unnecessary characters 
    def remove_special_char(batch):
        chars_to_remove = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
        if batch['sentence']:
            batch['sentence'] = re.sub(chars_to_remove, '', batch['sentence'].lower())
        return batch
    
    print('Removing special characters...')
    dataset = dataset.map(remove_special_char)

    return dataset

if __name__ == "__main__":
    dataset = load_and_prepare_datasets()
    print(dataset)
