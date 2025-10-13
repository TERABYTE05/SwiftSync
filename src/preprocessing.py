import re
from datasets import load_dataset, DatasetDict

def load_and_prepare_datasets():
    # Load the dataset from HF and preprocess dataset
    print('Loading dataset')
    dataset = DatasetDict()
    dataset['train'] = load_dataset('mozilla-foundation/common_voice_17_0', 'fr', split='train', token = True, trust_remote_code = True)
    dataset['test'] = load_dataset('mozilla-foundation/common_voice_17_0', 'fr', split='test', token = True, trust_remote_code = True)
    
    # Remove unnecessary columns
    print('removing unnecessary columns')
    cols_to_remove = ['client_id', 'sentence_id', 'sentence_domain', 'up_votes', 'down_votes', 'age', 'gender', 'accents', 'variant', 'locale', 'segment']
    dataset['train'] = dataset['train'].remove_columns(cols_to_remove)
    dataset['test'] = dataset['test'].remove_columns(cols_to_remove)

    # Remove unnecessary characters 
    def remove_special_char(batch):
        chars_to_remove = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
        batch['sentence'] = re.sub(chars_to_remove, '', batch['sentence'])
        return batch
    print('removing special characters')
    dataset['train'] = dataset['train'].map(remove_special_char)
    dataset['test'] = dataset['test'].map(remove_special_char)

    return dataset

if __name__ == "__main__":
    dataset = load_and_prepare_datasets()
    print(dataset)
