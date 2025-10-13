import json
from . import preprocessing, config
from datasets import Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

def main():
    dataset = preprocessing.load_and_prepare_datasets()

    def extract_all_chars(batch):
        all_text = ' '.join(batch['sentence'])
        vocab = list(set(all_text))
        return {'vocab' : [vocab], 'all_text' : [all_text]}

    # Create a vocabulary from dataset
    print('Preparing vocabulary list')
    vocab_train = dataset['train'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['train'].column_names, num_proc=4)
    vocab_test = dataset['test'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['test'].column_names, num_proc=4)

    # Create a vocab dictionary
    vocab_list = list(set(vocab_train['vocab'][0]) | set(vocab_test['vocab'][0]))
    vocab_dict = {char: i for i, char in enumerate(sorted(vocab_list))}
    if ' ' in vocab_dict:
        vocab_dict['|'] = vocab_dict[' ']
        del vocab_dict[' ']
    else:
        vocab_dict['|'] = len(vocab_dict)

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    # Dump vocab into a json file
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)
    print('Prepared vocabulary list')

    print('Loading Tokenizer and Feature Extractor')
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('./', unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
    tokenizer.push_to_hub('jager26/wav2vec2-xls-r-300m-common-voice-fr-ft')

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Resampling audio to 16kHz
    print('Resampling audio to 16kHz')
    dataset['train'] = dataset['train'].cast_column('audio', Audio(sampling_rate = 16000))
    dataset['test'] = dataset['test'].cast_column('audio', Audio(sampling_rate = 16000))

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        batch['input_length'] = len(batch['input_values'])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    dataset['train'] = dataset['train'].map(prepare_dataset, remove_columns=dataset['train'].column_names, num_proc=4)
    dataset['test'] = dataset['test'].map(prepare_dataset, remove_columns=dataset['test'].column_names, num_proc=4)
    print('Resampling done')

    # Filter all sequences that are longer than 5 seconds
    print('Filter sequences that are longer than 5 seconds')
    max_input_length = 5.0
    dataset['train'] = dataset['train'].filter(lambda x: x < max_input_length * processor.feature_extractor.sampling_rate, input_columns=['input_length'])
    print('Done filtering sequencies that are longer than 5 seconds')

if __name__ == '__main__':
    main()