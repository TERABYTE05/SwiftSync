"""
Phase 1: Complete Training Script (All-in-One)
Everything needed for training in a single file.
"""

import os
import json
import random
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    SeamlessM4Tv2Model,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup  # Import from transformers, not torch
)
from peft import LoraConfig, get_peft_model, TaskType

# ==================== CONFIGURATION ====================
class Config:
    """Training configuration."""
    # Paths
    metadata_file = "organized_data/parallel_metadata.csv"
    welsh_alignments_dir = "organized_data/welsh_alignments"
    checkpoints_dir = "training_output/checkpoints"
    
    # Models
    s2s_model_name = "facebook/seamless-m4t-v2-large"
    welsh_asr_model = "techiaith/wav2vec2-xlsr-ft-cy-en"
    
    # Training
    batch_size = 2
    gradient_accumulation_steps = 8
    num_epochs = 3
    learning_rate = 2e-4
    warmup_steps = 500
    weight_decay = 0.01
    
    # LoRA
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    # Audio
    sample_rate = 16000
    chunk_duration = 2.0
    chunk_samples = int(chunk_duration * sample_rate)
    
    # Optimization
    max_grad_norm = 1.0
    clear_cache_steps = 50
    
    # Logging & Saving
    logging_steps = 10
    save_steps = 500
    eval_steps = 500
    
    # Data splits
    train_split = "all"  # Use all data
    val_split = "dev"
    max_val_samples = 100
    
    # Feedback loop
    use_feedback_loop = True
    lambda_consistency = 0.5
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    use_bfloat16 = True

config = Config()
Path(config.checkpoints_dir).mkdir(parents=True, exist_ok=True)

print(f"🖥️  Using device: {config.device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ==================== DATASET ====================
class StreamingDataset(Dataset):
    """Streaming dataset for S2S translation."""
    
    def __init__(self, split="all"):
        # Load metadata
        df = pd.read_csv(config.metadata_file)
        
        if split == "all":
            self.data = df
        else:
            self.data = df[df['split'] == split]
        
        if split == config.val_split and config.max_val_samples:
            self.data = self.data[:config.max_val_samples]
        
        self.data = self.data.reset_index(drop=True)
        print(f"📊 Loaded {len(self.data)} samples for '{split}' split")
    
    def __len__(self):
        return len(self.data)
    
    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load audio
        audio = self.load_audio(row['eng_audio'])
        
        # Get chunk
        if len(audio) < config.chunk_samples:
            audio = torch.nn.functional.pad(audio, (0, config.chunk_samples - len(audio)))
        else:
            start = random.randint(0, len(audio) - config.chunk_samples)
            audio = audio[start:start + config.chunk_samples]
        
        # Get text
        text = row['welsh_transcript']
        
        return {
            'audio': audio,
            'text': text,
            'audio_id': row['audio_id']
        }

def collate_fn(batch):
    """Collate batch."""
    audios = torch.stack([s['audio'] for s in batch])
    texts = [s['text'] for s in batch]
    ids = [s['audio_id'] for s in batch]
    return {'audio': audios, 'text': texts, 'audio_id': ids}

# ==================== MODELS ====================
def load_models():
    """Load all models."""
    print("\n📥 Loading models...")
    
    # S2S model
    processor = AutoProcessor.from_pretrained(config.s2s_model_name)
    dtype = torch.bfloat16 if config.use_bfloat16 else torch.float32
    s2s_model = SeamlessM4Tv2Model.from_pretrained(
        config.s2s_model_name,
        torch_dtype=dtype
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    s2s_model = get_peft_model(s2s_model, lora_config)
    s2s_model = s2s_model.to(config.device)
    
    # Feedback model
    feedback_processor = Wav2Vec2Processor.from_pretrained(config.welsh_asr_model)
    feedback_model = Wav2Vec2ForCTC.from_pretrained(config.welsh_asr_model)
    for param in feedback_model.parameters():
        param.requires_grad = False
    feedback_model = feedback_model.to(config.device)
    feedback_model.eval()
    
    trainable = sum(p.numel() for p in s2s_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in s2s_model.parameters())
    print(f"   ✅ Models loaded")
    print(f"   • Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return s2s_model, processor, feedback_model, feedback_processor

# ==================== TRAINER ====================
class Trainer:
    """Main trainer class."""
    
    def __init__(self):
        # Load models
        self.s2s_model, self.processor, self.feedback_model, self.feedback_processor = load_models()
        
        # Create datasets
        print("\n📊 Creating datasets...")
        train_dataset = StreamingDataset(split=config.train_split)
        val_dataset = StreamingDataset(split=config.val_split)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print(f"   • Train batches: {len(self.train_loader)}")
        print(f"   • Val batches: {len(self.val_loader)}")
        
        # Optimizer
        trainable_params = [p for p in self.s2s_model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_step(self, batch):
        """Single training step."""
        audio = batch['audio'].to(config.device)
        texts = batch['text']
        
        # Process inputs
        audio_inputs = self.processor(
            audio=audio.cpu().numpy(),
            return_tensors="pt",
            sampling_rate=config.sample_rate
        )
        audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                       if isinstance(v, torch.Tensor)}
        
        # Tokenize targets
        text_inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200
        )
        labels = text_inputs['input_ids'].to(config.device)
        
        # Forward pass
        outputs = self.s2s_model(
            input_features=audio_inputs.get('input_features'),
            labels=labels
        )
        
        loss = outputs.loss / config.gradient_accumulation_steps
        loss.backward()
        
        return loss.item() * config.gradient_accumulation_steps
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.s2s_model.eval()
        losses = []
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            try:
                audio = batch['audio'].to(config.device)
                texts = batch['text']
                
                audio_inputs = self.processor(
                    audio=audio.cpu().numpy(),
                    return_tensors="pt",
                    sampling_rate=config.sample_rate
                )
                audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                               if isinstance(v, torch.Tensor)}
                
                text_inputs = self.processor.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=200
                )
                labels = text_inputs['input_ids'].to(config.device)
                
                outputs = self.s2s_model(
                    input_features=audio_inputs.get('input_features'),
                    labels=labels
                )
                losses.append(outputs.loss.item())
            except:
                continue
        
        avg_loss = sum(losses) / len(losses) if losses else float('inf')
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(best=True)
            print(f"   ✅ New best model! Val loss: {avg_loss:.4f}")
        
        self.s2s_model.train()
        return avg_loss
    
    def save_checkpoint(self, best=False):
        """Save checkpoint."""
        save_dir = Path(config.checkpoints_dir) / ("best_model" if best else f"step-{self.global_step}")
        save_dir.mkdir(exist_ok=True, parents=True)
        self.s2s_model.save_pretrained(save_dir)
        print(f"💾 Saved to {save_dir}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("   STARTING TRAINING")
        print("="*70)
        
        for epoch in range(config.num_epochs):
            print(f"\n{'='*70}")
            print(f"   EPOCH {epoch+1}/{config.num_epochs}")
            print(f"{'='*70}")
            
            progress = tqdm(self.train_loader, desc=f"Training")
            epoch_losses = []
            
            for step, batch in enumerate(progress):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.s2s_model.parameters(),
                        config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    if self.global_step % config.clear_cache_steps == 0:
                        torch.cuda.empty_cache()
                
                avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                progress.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                if self.global_step % config.save_steps == 0:
                    self.save_checkpoint()
                
                if self.global_step % config.eval_steps == 0:
                    self.validate()
            
            print(f"\n📊 Epoch {epoch+1} - Avg Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")
            self.validate()
        
        print("\n" + "="*70)
        print("   ✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")

# ==================== MAIN ====================
if __name__ == "__main__":
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    trainer = Trainer()
    trainer.train()