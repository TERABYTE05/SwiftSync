"""
CORRECTED: Streaming Speech-to-Speech Translation Training
Uses proper workflow: Speech-to-Text fine-tuning approach
Based on Facebook's official fine-tuning methodology
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from transformers import get_linear_schedule_with_warmup
import logging
import warnings
import jiwer
from sacrebleu import corpus_bleu
from pathlib import Path
import torchaudio
import pandas as pd
import random

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
class Config:
    metadata_file = "organized_data/parallel_metadata.csv"
    checkpoints_dir = "training_output_streaming/checkpoints"
    logs_dir = "training_output_streaming/logs"
    
    # Use Speech-to-Text model instead of full S2S for fine-tuning
    s2st_model_name = "facebook/seamless-m4t-v2-large"
    
    batch_size = 4  # Increased for better training
    gradient_accumulation_steps = 4
    num_epochs = 3
    learning_rate = 5e-5  # Lower LR for stability
    warmup_steps = 200
    weight_decay = 0.01
    
    lora_r = 16  # Reduced for stability
    lora_alpha = 32
    lora_dropout = 0.05
    # Target only decoder for text generation
    lora_target_modules = ["q_proj", "v_proj", "out_proj"]
    
    sample_rate = 16000
    chunk_duration = 4.0  # Longer chunks for better context
    hop_duration = 2.0
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)
    
    max_grad_norm = 1.0
    clear_cache_steps = 25
    
    logging_steps = 25
    save_steps = 250
    eval_steps = 250
    log_sample_predictions = 3
    
    train_split = "train"
    val_split = "dev"
    max_val_samples = 50
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    target_lang = "cym"
    source_lang = "eng"
    
    # Generation parameters
    max_new_tokens = 100
    num_beams = 4
    length_penalty = 1.0

config = Config()

for dir_path in [config.checkpoints_dir, config.logs_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# ==================== LOGGING ====================
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.logs_dir) / f"streaming_{timestamp}.log"
    
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
    logger.info("STREAMING S2ST TRAINING (CORRECTED WORKFLOW)")
    logger.info("Using Speech-to-Text fine-tuning approach")
    logger.info("="*70)
    return logger

logger = setup_logging()

# ==================== MODELS ====================
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

def load_models():
    logger.info("\nLoading models...")
    
    processor = AutoProcessor.from_pretrained(config.s2st_model_name)
    dtype = torch.bfloat16 if config.use_bfloat16 else torch.float32
    
    # Use Speech-to-Text model for fine-tuning
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        config.s2st_model_name,
        torch_dtype=dtype
    )
    
    # Apply LoRA only to text decoder
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(config.device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"   Model: SeamlessM4Tv2ForSpeechToText")
    logger.info(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    logger.info(f"   Target: {config.source_lang} -> {config.target_lang}")
    
    return {"model": model, "processor": processor}

# ==================== DATASET ====================
class StreamingChunkDataset:
    def __init__(self, split="train"):
        df = pd.read_csv(config.metadata_file)
        
        if split == "all":
            self.data = df
        else:
            self.data = df[df['split'] == split]
        
        if split == config.val_split and config.max_val_samples:
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
                
                # Only create chunks if audio is long enough
                if audio_len > config.chunk_samples:
                    num_chunks = (audio_len - config.chunk_samples) // config.hop_samples + 1
                else:
                    num_chunks = 1
                
                for chunk_idx in range(num_chunks):
                    self.chunk_indices.append({
                        'data_idx': idx,
                        'chunk_idx': chunk_idx,
                        'total_chunks': num_chunks,
                    })
            except Exception as e:
                logger.warning(f"Error processing {row['eng_audio']}: {e}")
        
        logger.info(f"   {len(self.data)} files -> {len(self.chunk_indices)} chunks ({split})")
    
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
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
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
        
        # Improved text chunking - proportional to audio
        words = row['welsh_transcript'].split()
        total_chunks = chunk_info['total_chunks']
        chunk_idx = chunk_info['chunk_idx']
        
        if total_chunks == 1:
            chunk_text = row['welsh_transcript']
        else:
            # Proportional chunking based on audio position
            total_words = len(words)
            start_ratio = chunk_idx / total_chunks
            end_ratio = (chunk_idx + 1) / total_chunks
            
            start_word = int(start_ratio * total_words)
            end_word = int(end_ratio * total_words)
            
            if start_word >= total_words:
                chunk_text = words[-1] if words else ""
            else:
                chunk_text = ' '.join(words[start_word:end_word])
        
        return {
            'audio': chunk,
            'text': chunk_text if chunk_text.strip() else row['welsh_transcript'],
            'audio_id': f"{row['audio_id']}_c{chunk_idx}",
        }

def collate_fn(batch):
    return {
        'audio': [s['audio'].numpy() for s in batch],  # List of numpy arrays
        'text': [s['text'] for s in batch],
        'audio_id': [s['audio_id'] for s in batch],
    }

# ==================== GENERATION ====================
@torch.no_grad()
def generate_translations(model, processor, batch):
    """
    Generate translations using the Speech-to-Text model
    """
    try:
        # Process audio inputs
        audio_inputs = processor(
            audios=batch["audio"],
            return_tensors="pt",
            sampling_rate=config.sample_rate,
            padding=True,
        )
        audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                       if isinstance(v, torch.Tensor)}
        
        # Generate with proper parameters
        generated_ids = model.generate(
            **audio_inputs,
            tgt_lang=config.target_lang,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            length_penalty=config.length_penalty,
            early_stopping=True,
            forced_bos_token_id=processor.tokenizer.convert_tokens_to_ids(
                f"__{config.target_lang}__"
            ) if hasattr(processor.tokenizer, 'convert_tokens_to_ids') else None,
        )
        
        # Decode translations
        translations = []
        for ids in generated_ids:
            text = processor.decode(ids, skip_special_tokens=True)
            translations.append(text.strip())
        
        return translations
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return [""] * len(batch["audio"])

# ==================== METRICS ====================
def calculate_wer(references, hypotheses):
    try:
        valid = [(r, h) for r, h in zip(references, hypotheses) 
                if r.strip() and h.strip()]
        if not valid:
            return 1.0
        refs, hyps = zip(*valid)
        return jiwer.wer(list(refs), list(hyps))
    except:
        return 1.0

def calculate_bleu(references, hypotheses):
    try:
        valid = [(r, h) for r, h in zip(references, hypotheses) 
                if r.strip() and h.strip()]
        if not valid:
            return 0.0
        refs, hyps = zip(*valid)
        refs_list = [[r] for r in refs]
        bleu = corpus_bleu(list(hyps), refs_list)
        return bleu.score
    except:
        return 0.0

def detect_language(text):
    """Simple Welsh vs English detection"""
    if not text or not text.strip():
        return 'Empty'
    
    text_lower = text.lower()
    welsh_words = ['yn', 'y', 'yr', 'mae', 'chi', 'ar', 'o', 'ei', 'dw', 
                   'sy', 'bod', 'ddim', 'i', 'na', 'wedi', 'fi']
    english_words = ['the', 'and', 'is', 'to', 'of', 'in', 'a', 'you', 
                     'that', 'it', 'for', 'was', 'have', 'with']
    
    welsh_count = sum(1 for w in welsh_words if f' {w} ' in f' {text_lower} ' or text_lower.startswith(f'{w} ') or text_lower.endswith(f' {w}'))
    english_count = sum(1 for w in english_words if f' {w} ' in f' {text_lower} ' or text_lower.startswith(f'{w} ') or text_lower.endswith(f' {w}'))
    
    if english_count > welsh_count and english_count > 0:
        return 'English'
    elif welsh_count > 0:
        return 'Welsh'
    return 'Unknown'

# ==================== TRAINER ====================
class Trainer:
    def __init__(self):
        models = load_models()
        self.model = models["model"]
        self.processor = models["processor"]
        
        logger.info("\nCreating datasets...")
        train_dataset = StreamingChunkDataset(split=config.train_split)
        val_dataset = StreamingChunkDataset(split=config.val_split)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        logger.info(f"   Train: {len(self.train_loader)} batches")
        logger.info(f"   Val: {len(self.val_loader)} batches")
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params, 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            eps=1e-8,
        )
        
        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, config.warmup_steps, total_steps
        )
        
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_bleu = 0.0
        self.best_wer = float("inf")
    
    def train_step(self, batch):
        """
        Training step with proper label preparation
        """
        try:
            # Process audio
            audio_inputs = self.processor(
                audios=batch["audio"],
                return_tensors="pt",
                sampling_rate=config.sample_rate,
                padding=True,
            )
            audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                           if isinstance(v, torch.Tensor)}
            
            # Tokenize target text with proper language prefix
            text_inputs = self.processor.tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_new_tokens,
            )
            
            labels = text_inputs["input_ids"].to(config.device)
            # Replace padding token id's with -100 so they are ignored in loss
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            # Forward pass
            outputs = self.model(
                **audio_inputs,
                labels=labels,
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            
            return loss.item() * config.gradient_accumulation_steps
            
        except Exception as e:
            logger.error(f"Train step error: {e}", exc_info=True)
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
        
        lang_counts = {'Welsh': 0, 'English': 0, 'Unknown': 0, 'Empty': 0}
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
            try:
                # Calculate loss
                audio_inputs = self.processor(
                    audios=batch["audio"],
                    return_tensors="pt",
                    sampling_rate=config.sample_rate,
                    padding=True,
                )
                audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                               if isinstance(v, torch.Tensor)}
                
                text_inputs = self.processor.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_new_tokens,
                )
                
                labels = text_inputs["input_ids"].to(config.device)
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                
                outputs = self.model(**audio_inputs, labels=labels)
                losses.append(outputs.loss.item())
                
                # Generate translations
                translations = generate_translations(self.model, self.processor, batch)
                all_refs.extend(batch["text"])
                all_hyps.extend(translations)
                
                # Language detection
                for trans in translations:
                    lang = detect_language(trans)
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                # Collect samples
                if len(samples) < config.log_sample_predictions * 2:
                    for i in range(len(translations)):
                        if len(samples) >= config.log_sample_predictions * 2:
                            break
                        samples.append({
                            'id': batch['audio_id'][i],
                            'ref': batch['text'][i],
                            'hyp': translations[i],
                            'lang': detect_language(translations[i]),
                        })
                
            except Exception as e:
                logger.warning(f"Val batch {batch_idx} error: {e}")
        
        if not losses:
            logger.warning("No valid validation batches!")
            self.model.train()
            return float("inf"), 0.0, 1.0
        
        avg_loss = sum(losses) / len(losses)
        bleu = calculate_bleu(all_refs, all_hyps)
        wer = calculate_wer(all_refs, all_hyps)
        
        # Count empty and non-empty
        empty_count = sum(1 for h in all_hyps if not h.strip())
        non_empty = len(all_hyps) - empty_count
        
        logger.info(f"\n{'='*70}")
        logger.info(f"RESULTS - Step {self.global_step}")
        logger.info(f"{'='*70}")
        logger.info(f"Loss:  {avg_loss:.4f}")
        logger.info(f"BLEU:  {bleu:.2f}")
        logger.info(f"WER:   {wer:.4f}")
        logger.info(f"\nOutput Statistics:")
        logger.info(f"  Non-empty: {non_empty}/{len(all_hyps)} ({100*non_empty/len(all_hyps):.1f}%)")
        logger.info(f"  Empty:     {empty_count}/{len(all_hyps)} ({100*empty_count/len(all_hyps):.1f}%)")
        
        total_detected = sum(v for k, v in lang_counts.items() if k != 'Empty')
        if total_detected > 0:
            logger.info(f"\nLanguage Detection (non-empty):")
            for lang in ['Welsh', 'English', 'Unknown']:
                count = lang_counts.get(lang, 0)
                pct = 100 * count / total_detected if total_detected > 0 else 0
                logger.info(f"  {lang:8s}: {count:3d} ({pct:.1f}%)")
        
        if samples:
            logger.info(f"\n{'='*70}")
            logger.info("SAMPLE PREDICTIONS")
            logger.info(f"{'='*70}")
            for i, s in enumerate(samples[:config.log_sample_predictions], 1):
                logger.info(f"\nSample {i}: {s['id']}")
                logger.info(f"  Language: {s['lang']}")
                logger.info(f"  REF: {s['ref']}")
                logger.info(f"  HYP: {s['hyp']}")
        
        # Save best checkpoints
        improved = False
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("best_loss")
            logger.info(f"\n✓ NEW BEST LOSS: {avg_loss:.4f}")
            improved = True
        
        if bleu > self.best_bleu and bleu > 0:
            self.best_bleu = bleu
            self.save_checkpoint("best_bleu")
            logger.info(f"✓ NEW BEST BLEU: {bleu:.2f}")
            improved = True
        
        if wer < self.best_wer and wer < 1.0:
            self.best_wer = wer
            self.save_checkpoint("best_wer")
            logger.info(f"✓ NEW BEST WER: {wer:.4f}")
            improved = True
        
        if not improved:
            logger.info(f"\nNo improvement (Best Loss: {self.best_val_loss:.4f}, BLEU: {self.best_bleu:.2f}, WER: {self.best_wer:.4f})")
        
        logger.info(f"{'='*70}\n")
        
        self.model.train()
        return avg_loss, bleu, wer
    
    def save_checkpoint(self, name):
        save_dir = Path(config.checkpoints_dir) / f"{name}_step{self.global_step}"
        save_dir.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_bleu': self.best_bleu,
            'best_wer': self.best_wer,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(state, save_dir / "training_state.pt")
        logger.info(f"  → Saved checkpoint to {save_dir}")
    
    def train(self):
        logger.info("\n" + "="*70)
        logger.info("TRAINING START")
        logger.info("="*70)
        
        for epoch in range(config.num_epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"EPOCH {epoch+1}/{config.num_epochs}")
            logger.info(f"{'='*70}")
            
            self.model.train()
            epoch_losses = []
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                # Gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Memory management
                    if self.global_step % config.clear_cache_steps == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Progress display
                recent_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                progress.set_postfix({
                    'loss': f'{recent_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Logging
                if self.global_step > 0 and self.global_step % config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {recent_loss:.4f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                    )
                
                # Checkpoint saving
                if self.global_step > 0 and self.global_step % config.save_steps == 0:
                    save_dir = Path(config.checkpoints_dir) / f"checkpoint_step{self.global_step}"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    self.model.save_pretrained(save_dir)
                    logger.info(f"Checkpoint saved: step {self.global_step}")
                
                # Validation
                if self.global_step > 0 and self.global_step % config.eval_steps == 0:
                    self.validate()
                    self.model.train()
            
            # End of epoch
            epoch_avg = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            logger.info(f"\nEpoch {epoch+1} Complete | Avg Loss: {epoch_avg:.4f}")
            
            # End of epoch validation
            self.validate()
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Best Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best BLEU: {self.best_bleu:.2f}")
        logger.info(f"Best WER:  {self.best_wer:.4f}")
        logger.info("="*70 + "\n")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    try:
        trainer = Trainer()
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n⚠ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        raise