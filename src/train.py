import torch
import numpy as np
import evaluate
from dataclasses import dataclass
from datasets import Audio
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from . import config
from .preprocessing import load_and_prepare_datasets
from .processor import save_processor

def main():
    # Load and Prepare Data
    print("Loading and preparing data...")
    dataset = load_and_prepare_datasets()
    train_data = dataset['train']
    test_data = dataset['test']
    print("Data loaded successfully...")

    # Create the Processor
    print("Creating model processor...")
    processor = save_processor(dataset)
    print("Processor created successfully...")

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

    dataset = dataset.map(prepare_dataset, num_proc=4)
    print('Resampling done')

    # Filter all sequences that are longer than 5 seconds
    print('Filter sequences that are longer than 5 seconds')
    max_input_length = 5.0
    dataset['train'] = dataset['train'].filter(lambda x: x < max_input_length * processor.feature_extractor.sampling_rate, input_columns=['input_length'])
    print('Done filtering sequencies that are longer than 5 seconds')

    # Define the Data Collator
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            batch = self.processor.pad(input_features, padding = self.padding, return_tensors = "pt")
            labels_batch = self.processor.pad(labels = label_features, padding = self.padding, return_tensors = "pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Define Evaluation Metric
    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Load the Model 
    model = Wav2Vec2ForCTC.from_pretrained(
        config.model, 
        attention_dropout = 0.0,
        hidden_dropout = 0.0,
        feat_proj_dropout = 0.0,
        mask_time_prob = 0.05,
        layerdrop = 0.0,
        ctc_loss_reduction = "mean", 
        pad_token_id = processor.tokenizer.pad_token_id,
        vocab_size = len(processor.tokenizer),
    )

    model.freeze_feature_encoder()

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir  =config.repo_name,
        group_by_length = True,
        per_device_train_batch_size = 16, 
        gradient_accumulation_steps = 2,
        evaluation_strategy = "steps",
        num_train_epochs = 3, 
        gradient_checkpointing = True,
        fp16 = True, 
        save_steps = 500,
        eval_steps = 500,
        logging_steps = 500,
        learning_rate = 3e-4,
        warmup_steps = 500,
        save_total_limit = 2,
        push_to_hub = True,
    )

    # Initialize the Trainer 
    trainer = Trainer(
        model = model,
        data_collator = data_collator,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = train_data,
        eval_dataset = test_data,
        tokenizer = processor.feature_extractor,
    )

    # Start Training
    print("\nFine tuning the model...")
    trainer.train()
    print("Fine tuning completed sucessfully...")

if __name__ == "__main__":
    main()