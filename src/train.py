import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from .processor import processor
from . import config, preprocessing

@dataclass
class DataCollatorCTCWithPadding:
    # Data collator that will dynamically pad the inputs received.
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

model = Wav2Vec2ForCTC.from_pretrained(
    'facebook/wav2vec2-xls-r-300m', 
    attention_dropout = 0.0,
    hidden_dropout = 0.0,
    feat_proj_dropout = 0.0,
    mask_time_prob = 0.05,
    layerdrop = 0.0,
    ctc_loss_reduction = "mean", 
    pad_token_id = processor.tokenizer.pad_token_id,
    vocab_size = len(processor.tokenizer),
)
model.freeze_feature_extractor()

training_args = TrainingArguments(
  output_dir = config.repo_name,
  group_by_length = True,
  per_device_train_batch_size = 16,
  gradient_accumulation_steps = 2,
  evaluation_strategy = "steps",
  num_train_epochs = 30,
  gradient_checkpointing = True,
  use_fp16 = True,
  save_steps = 400,
  eval_steps = 400,
  logging_steps = 400,
  learning_rate = 3e-4,
  warmup_steps = 500,
  save_total_limit = 2,
  push_to_hub = True,
)