import torch
import numpy as np
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from . import config
from .preprocessing import load_and_prepare_datasets
from .processor import create_and_save_processor 

def main():
    # Load and Prepare Data
    print("Loading and preparing data...")
    dataset = load_and_prepare_datasets()
    print("Data loading and preprocessing completed...")

    # Create the Processor
    print("\nCreating model processor...")
    processor = create_and_save_processor(dataset)
    print("Processor created successfully...")

    # Dataset Preparation for the Model 
    print("\nPreparing dataset for the model...")
    def prepare_dataset(batch):
        audio = batch["audio"]
        # Process the audio waveform into 'input_values' for Trainer
        batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
        batch['input_length'] = len(batch['input_values'])
        batch["labels"] = processor(text = batch["sentence"]).input_ids
        return batch
    prepared_dataset = dataset.map(prepare_dataset, num_proc = 1)
    
    print('Filtering sequences longer than 5 seconds...')
    max_input_length = 5.0 * 16000
    prepared_dataset['train'] = prepared_dataset['train'].filter(
        lambda x: x < max_input_length, input_columns=['input_length']
    )
    print('Filtering completed successfully...')

    # Define data collator, metrics, model, and trainer
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
    data_collator = DataCollatorCTCWithPadding(processor = processor, padding = True)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens = False)
        wer = wer_metric.compute(predictions = pred_str, references = label_str)
        return {"wer": wer}
        
    model = Wav2Vec2ForCTC.from_pretrained(
        config.model,
        ctc_loss_reduction = "mean",
        pad_token_id = processor.tokenizer.pad_token_id,
        vocab_size = len(processor.tokenizer)
    )
    model.freeze_feature_encoder()
    
    training_args = TrainingArguments(
        output_dir = config.repo_name,
        group_by_length = True,
        per_device_train_batch_size = config.train_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        evaluation_strategy = "steps",
        num_train_epochs = config.num_train_epochs,
        fp16 = config.USE_FP16,
        save_steps =  config.save_steps,
        eval_steps = config.save_steps,
        logging_steps = config.save_steps,
        learning_rate = config.learning_rate,
        warmup_steps = config.warmup_steps,
        save_total_limit = 1,
        push_to_hub = True,
    )
    
    trainer = Trainer(
        model = model,
        data_collator = data_collator,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = prepared_dataset["train"],
        eval_dataset = prepared_dataset["test"],
        tokenizer = processor.feature_extractor,
    )
    
    print("\nFine tuning the model...")
    trainer.train()
    print("Fine tuning completed successfully...")

if __name__ == "__main__":
    main()