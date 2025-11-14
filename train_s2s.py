"""
COMPLETE FIXED VERSION - Addresses ALL diagnostic issues
1. Fixed tuple unpacking in generation
2. Added missing config attributes
3. Proper decoding logic
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from transformers import get_linear_schedule_with_warmup
import logging
import warnings
import pandas as pd
import torchaudio
import random
from pathlib import Path

# Optional metrics
try:
    import jiwer
    from sacrebleu import corpus_bleu
    METRICS_AVAILABLE = True
except:
    METRICS_AVAILABLE = False
    print("⚠️  Install metrics: pip install jiwer sacrebleu")

warnings.filterwarnings("ignore")

# ==================== COMPLETE FIXED CONFIG ====================
class Config:
    # Paths
    metadata_file = "organized_data/parallel_metadata.csv"
    checkpoints_dir = "training_output_final/checkpoints"
    logs_dir = "training_output_final/logs"
    
    # Model
    s2s_model_name = "facebook/seamless-m4t-v2-large"
    
    # Training - OPTIMIZED
    batch_size = 2
    gradient_accumulation_steps = 8
    num_epochs = 10  # FIXED: Increased from 1
    learning_rate = 5e-5  # FIXED: Reduced for stability
    warmup_steps = 100  # FIXED: Reduced for small dataset
    weight_decay = 0.01
    
    # LoRA
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    # Audio
    sample_rate = 16000
    chunk_duration = 2.0
    hop_duration = 0.5
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)
    
    # Optimization
    max_grad_norm = 1.0
    clear_cache_steps = 50
    
    # Logging - OPTIMIZED
    logging_steps = 10
    save_steps = 100  # FIXED: Reduced for frequent saves
    eval_steps = 100  # FIXED: Reduced for frequent validation
    log_sample_predictions = 5
    
    # Data - FIXED: Added missing attributes
    train_split = "train"
    val_split = "dev"
    max_train_samples = None  # FIXED: Added
    max_val_samples = 50  # FIXED: Reduced
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    use_bfloat16 = False  # FIXED: Disabled for RTX 5050 compatibility
    target_lang = "cym"

config = Config()

for dir_path in [config.checkpoints_dir, config.logs_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# ==================== LOGGING ====================
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.logs_dir) / f"train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("STREAMING S2S TRAINING - COMPLETE FIX")
    logger.info("="*70)
    logger.info(f"Training for {config.num_epochs} epochs")
    logger.info(f"Learning rate: {config.learning_rate}")
    return logger

logger = setup_logging()

# ==================== MODELS ====================
from transformers import SeamlessM4Tv2Model, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

def load_models():
    logger.info("\nLoading models...")
    
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
    
    trainable = sum(p.numel() for p in s2s_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in s2s_model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return {"s2s_model": s2s_model, "processor": processor}

# ==================== DATASET ====================
class StreamingDataset(Dataset):
    def __init__(self, split="train"):
        df = pd.read_csv(config.metadata_file)
        
        if split == "all":
            self.data = df
        else:
            self.data = df[df['split'] == split]
        
        # Apply max samples limits
        if split == config.train_split and config.max_train_samples:
            self.data = self.data[:config.max_train_samples]
        elif split == config.val_split and config.max_val_samples:
            self.data = self.data[:config.max_val_samples]
        
        self.data = self.data.reset_index(drop=True)
        
        # Generate chunks
        self.chunk_indices = []
        for idx, row in self.data.iterrows():
            try:
                waveform, sr = torchaudio.load(row['eng_audio'])
                audio_len = waveform.shape[-1]
                if sr != config.sample_rate:
                    audio_len = int(audio_len * config.sample_rate / sr)
                
                num_chunks = max(1, (audio_len - config.chunk_samples) // config.hop_samples + 1)
                
                for chunk_idx in range(num_chunks):
                    self.chunk_indices.append({
                        'data_idx': idx,
                        'chunk_idx': chunk_idx,
                        'total_chunks': num_chunks,
                    })
            except Exception as e:
                logger.warning(f"Skip {row['eng_audio']}: {e}")
        
        logger.info(f"  {len(self.data)} files → {len(self.chunk_indices)} chunks ({split})")
    
    def __len__(self):
        return len(self.chunk_indices)
    
    def load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
                waveform = resampler(waveform)
            return waveform.squeeze()
        except:
            return torch.zeros(config.chunk_samples)
    
    def __getitem__(self, idx):
        chunk_info = self.chunk_indices[idx]
        row = self.data.iloc[chunk_info['data_idx']]
        
        audio = self.load_audio(row['eng_audio'])
        
        start_sample = chunk_info['chunk_idx'] * config.hop_samples
        end_sample = start_sample + config.chunk_samples
        
        if end_sample > len(audio):
            chunk = torch.nn.functional.pad(audio[start_sample:], (0, end_sample - len(audio)))
        else:
            chunk = audio[start_sample:end_sample]
        
        # Text chunking
        words = row['welsh_transcript'].split()
        total_chunks = chunk_info['total_chunks']
        chunk_idx = chunk_info['chunk_idx']
        
        if total_chunks == 1:
            chunk_text = row['welsh_transcript']
        else:
            words_per_chunk = max(1, len(words) // total_chunks)
            start_word = chunk_idx * words_per_chunk
            end_word = min(start_word + words_per_chunk, len(words))
            chunk_text = ' '.join(words[start_word:end_word])
        
        return {
            'audio': chunk,
            'text': chunk_text if chunk_text else row['welsh_transcript'],
            'audio_id': f"{row['audio_id']}_c{chunk_idx}",
        }

def collate_fn(batch):
    return {
        'audio': torch.stack([s['audio'] for s in batch]),
        'text': [s['text'] for s in batch],
        'audio_id': [s['audio_id'] for s in batch],
    }

# ==================== FIXED GENERATION ====================
def generate_translations(model, processor, batch):
    """
    COMPLETE FIX: Properly handle tuple outputs from PEFT models.
    The issue: PEFT returns (text_tokens, speech_units) tuple
    """
    try:
        audio = batch["audio"].to(config.device)
        batch_size = len(batch["audio"])
        
        # Process audio
        audio_inputs = processor(
            audio=audio.cpu().numpy(),
            return_tensors="pt",
            sampling_rate=config.sample_rate,
        )
        audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                       if isinstance(v, torch.Tensor)}
        
        # Generate (TEXT ONLY for validation - faster and no tuple issues)
        with torch.no_grad():
            outputs = model.generate(
                **audio_inputs,
                tgt_lang=config.target_lang,
                generate_speech=False,  # CRITICAL: Text-only for validation
                max_new_tokens=50,
                num_beams=3,
                do_sample=False,
            )
        
        # CRITICAL FIX: Proper tuple handling
        logger.debug(f"Output type: {type(outputs)}")
        
        if isinstance(outputs, tuple):
            # PEFT/SeamlessM4T returns (text_sequences, audio_codes) tuple
            logger.debug(f"Tuple length: {len(outputs)}")
            logger.debug(f"Element 0 type: {type(outputs[0])}")
            logger.debug(f"Element 1 type: {type(outputs[1])}")
            
            # Text sequences are in first element
            generated_ids = outputs[0]
            
            # Verify it's a tensor with integers
            if isinstance(generated_ids, torch.Tensor):
                if generated_ids.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    # If floats, it's probably logits - need to argmax
                    logger.warning("Got floats instead of token IDs, applying argmax")
                    generated_ids = torch.argmax(generated_ids, dim=-1)
            else:
                logger.error(f"Unexpected generated_ids type: {type(generated_ids)}")
                return [""] * batch_size
                
        elif hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences
        else:
            generated_ids = outputs
        
        # Ensure it's a 2D tensor [batch_size, seq_len]
        if generated_ids.dim() == 3:
            # If 3D [batch, beam, seq], take first beam
            generated_ids = generated_ids[:, 0, :]
        
        logger.debug(f"Generated IDs shape: {generated_ids.shape}")
        logger.debug(f"Generated IDs dtype: {generated_ids.dtype}")
        logger.debug(f"Sample IDs: {generated_ids[0][:10].tolist()}")
        
        # Decode
        try:
            translations = processor.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"batch_decode failed: {e}, trying tokenizer")
            try:
                translations = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            except Exception as e2:
                logger.error(f"tokenizer.batch_decode also failed: {e2}")
                # Last resort: manual decode
                translations = []
                for ids in generated_ids:
                    try:
                        # Ensure it's a list of ints
                        ids_list = ids.cpu().tolist()
                        text = processor.tokenizer.decode(ids_list, skip_special_tokens=True)
                        translations.append(text.strip())
                    except Exception as e3:
                        logger.error(f"Manual decode failed: {e3}")
                        translations.append("")
        
        # Clean up
        translations = [t.strip() for t in translations]
        
        # Log first translation for debugging
        if translations:
            logger.debug(f"First translation: '{translations[0]}'")
        
        return translations
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return [""] * len(batch["audio"])

# ==================== METRICS ====================
def calculate_wer(refs, hyps):
    if not METRICS_AVAILABLE:
        return 1.0
    try:
        valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
        if not valid:
            return 1.0
        r, h = zip(*valid)
        return jiwer.wer(list(r), list(h))
    except:
        return 1.0

def calculate_bleu(refs, hyps):
    if not METRICS_AVAILABLE:
        return 0.0
    try:
        valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
        if not valid:
            return 0.0
        r, h = zip(*valid)
        refs_list = [[ref] for ref in r]
        bleu = corpus_bleu(list(h), refs_list)
        return bleu.score
    except:
        return 0.0

# ==================== TRAINER ====================
class Trainer:
    def __init__(self):
        models = load_models()
        self.model = models["s2s_model"]
        self.processor = models["processor"]
        
        logger.info("\nCreating datasets...")
        train_dataset = StreamingDataset(split=config.train_split)
        val_dataset = StreamingDataset(split=config.val_split)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        logger.info(f"  Train: {len(self.train_loader)} batches")
        logger.info(f"  Val: {len(self.val_loader)} batches")
        logger.info(f"  Total steps: {len(self.train_loader) * config.num_epochs}")
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, config.warmup_steps, total_steps
        )
        
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_bleu = 0.0
        self.best_wer = float("inf")
    
    def train_step(self, batch):
        audio = batch["audio"].to(config.device)
        texts = batch["text"]
        
        try:
            audio_inputs = self.processor(
                audio=audio.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=config.sample_rate,
            )
            audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                           if isinstance(v, torch.Tensor)}
            
            text_inputs = self.processor.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=100
            )
            labels = text_inputs["input_ids"].to(config.device)
            
            outputs = self.model(**audio_inputs, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            
            return loss.item() * config.gradient_accumulation_steps
        except Exception as e:
            logger.error(f"Train error: {e}")
            return 0.0
    
    @torch.no_grad()
    def validate(self):
        logger.info("\n" + "="*70)
        logger.info("VALIDATION")
        logger.info("="*70)
        self.model.eval()
        
        losses = []
        all_refs = []
        all_hyps = []
        samples = []
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Val", leave=False)):
            try:
                audio = batch["audio"].to(config.device)
                texts = batch["text"]
                
                # Loss
                audio_inputs = self.processor(
                    audio=audio.cpu().numpy(),
                    return_tensors="pt",
                    sampling_rate=config.sample_rate,
                )
                audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                               if isinstance(v, torch.Tensor)}
                
                text_inputs = self.processor.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=100
                )
                labels = text_inputs["input_ids"].to(config.device)
                
                outputs = self.model(**audio_inputs, labels=labels)
                losses.append(outputs.loss.item())
                
                # Generate
                translations = generate_translations(self.model, self.processor, batch)
                all_refs.extend(texts)
                all_hyps.extend(translations)
                
                # Collect samples
                if batch_idx < 3:
                    for i in range(min(2, len(translations))):
                        if i < len(batch['audio_id']):
                            samples.append({
                                'id': batch['audio_id'][i],
                                'ref': texts[i],
                                'hyp': translations[i],
                            })
                
            except Exception as e:
                logger.warning(f"Val batch error: {e}")
        
        if not losses:
            self.model.train()
            return float("inf"), 0.0, 1.0
        
        avg_loss = sum(losses) / len(losses)
        bleu = calculate_bleu(all_refs, all_hyps)
        wer = calculate_wer(all_refs, all_hyps)
        empty_count = sum(1 for h in all_hyps if not h.strip())
        
        logger.info(f"\nStep {self.global_step}:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  BLEU: {bleu:.2f}")
        logger.info(f"  WER: {wer:.4f}")
        logger.info(f"  Empty: {empty_count}/{len(all_hyps)}")
        
        if samples:
            logger.info("\nSamples:")
            for s in samples[:5]:
                logger.info(f"\n  {s['id']}")
                logger.info(f"  REF: {s['ref']}")
                logger.info(f"  HYP: {s['hyp']}")
        
        # Save best
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("loss")
            logger.info("  ✨ Best loss!")
        
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self.save_checkpoint("bleu")
            logger.info("  ✨ Best BLEU!")
        
        if wer < self.best_wer:
            self.best_wer = wer
            self.save_checkpoint("wer")
            logger.info("  ✨ Best WER!")
        
        self.model.train()
        return avg_loss, bleu, wer
    
    def save_checkpoint(self, metric):
        save_dir = Path(config.checkpoints_dir) / f"best_{metric}"
        save_dir.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(save_dir)
    
    def train(self):
        logger.info("\n" + "="*70)
        logger.info("TRAINING START")
        logger.info("="*70)
        
        for epoch in range(config.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
            
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            losses = []
            
            for step, batch in enumerate(progress):
                loss = self.train_step(batch)
                losses.append(loss)
                
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    if self.global_step % config.clear_cache_steps == 0:
                        torch.cuda.empty_cache()
                
                avg = sum(losses[-10:]) / min(10, len(losses))
                progress.set_postfix({'loss': f'{avg:.4f}'})
                
                if self.global_step > 0 and self.global_step % config.eval_steps == 0:
                    self.validate()
            
            logger.info(f"Epoch {epoch+1} complete - Avg Loss: {sum(losses)/len(losses):.4f}")
            self.validate()
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Best Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best BLEU: {self.best_bleu:.2f}")
        logger.info(f"Best WER: {self.best_wer:.4f}")

# ==================== MAIN ====================
if __name__ == "__main__":
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    try:
        trainer = Trainer()
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    except Exception as e:
        logger.error(f"\nFailed: {e}", exc_info=True)
        raise