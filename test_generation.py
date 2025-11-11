"""
COMPLETE FIXED: Streaming Speech-to-Speech Translation Training
Fixes all generation errors using correct HuggingFace methods
Based on official documentation: output_tokens[batch_idx].tolist()[0]
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
    
    s2s_model_name = "facebook/seamless-m4t-v2-large"
    
    batch_size = 2
    gradient_accumulation_steps = 8
    num_epochs = 1
    learning_rate = 2e-4
    warmup_steps = 500
    weight_decay = 0.01
    
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    sample_rate = 16000
    chunk_duration = 2.0
    hop_duration = 0.5
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)
    
    max_grad_norm = 1.0
    clear_cache_steps = 50
    
    logging_steps = 50
    save_steps = 500
    eval_steps = 500
    log_sample_predictions = 3
    
    train_split = "train"
    val_split = "dev"
    max_val_samples = 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    use_bfloat16 = True
    target_lang = "cym"

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
    logger.info("STREAMING S2S TRAINING (DOCUMENTATION-CORRECT VERSION)")
    logger.info("="*70)
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
    logger.info(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    logger.info("   SeamlessM4T supports Welsh (cym) via tgt_lang parameter")
    
    return {"s2s_model": s2s_model, "processor": processor}

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
                
                num_chunks = max(1, (audio_len - config.chunk_samples) // config.hop_samples + 1)
                
                for chunk_idx in range(num_chunks):
                    self.chunk_indices.append({
                        'data_idx': idx,
                        'chunk_idx': chunk_idx,
                        'total_chunks': num_chunks,
                    })
            except Exception as e:
                logger.warning(f"Error loading {row['eng_audio']}: {e}")
        
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
        
        # Simple text chunking
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

# ==================== CORRECT GENERATION (FIXED) ====================
def generate_translations(model, processor, batch):
    """
    CORRECT generation based on official HuggingFace documentation.
    Format: output_tokens[batch_idx].tolist()[0]
    Reference: https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2
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
        
        # Generate text only (no speech for faster validation)
        with torch.no_grad():
            output_tokens = model.generate(
                **audio_inputs,
                tgt_lang="cym",              # Welsh language code
                generate_speech=False,       # Text only
                max_new_tokens=50,
                min_new_tokens=1,
                num_beams=3,
                length_penalty=1.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        
        # Decode according to documentation: output_tokens[i].tolist()[0]
        translations = []
        for i in range(batch_size):
            try:
                # Extract token IDs for this sample - CORRECT FORMAT
                token_ids = output_tokens[i].tolist()[0]
                # Decode to text
                text = processor.decode(token_ids, skip_special_tokens=True)
                translations.append(text.strip())
            except (IndexError, TypeError, AttributeError) as e:
                logger.warning(f"Decode error for sample {i}: {e}")
                translations.append("")
        
        return translations
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return [""] * len(batch["audio"])


def generate_with_confidence(model, processor, batch):
    """
    Generate WITH confidence scores using output_scores=True.
    Confidence = average of max softmax probabilities per token.
    """
    try:
        audio = batch["audio"].to(config.device)
        batch_size = len(batch["audio"])
        
        audio_inputs = processor(
            audio=audio.cpu().numpy(),
            return_tensors="pt",
            sampling_rate=config.sample_rate,
        )
        audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                       if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            output_tokens = model.generate(
                **audio_inputs,
                tgt_lang="cym",
                generate_speech=False,
                max_new_tokens=50,
                min_new_tokens=1,
                num_beams=3,
                length_penalty=1.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                output_scores=True,           # Get scores for confidence
                return_dict_in_generate=True, # Return as dict
            )
        
        # Extract sequences from the GenerationOutput
        if hasattr(output_tokens, 'sequences'):
            sequences = output_tokens.sequences
            scores = output_tokens.scores if hasattr(output_tokens, 'scores') else None
        else:
            # Fallback if not GenerationOutput
            sequences = output_tokens
            scores = None
        
        # Decode translations
        translations = []
        for i in range(batch_size):
            try:
                token_ids = sequences[i].tolist()[0] if isinstance(sequences[i].tolist(), list) and len(sequences[i].tolist()) > 0 else sequences[i].tolist()
                text = processor.decode(token_ids, skip_special_tokens=True)
                translations.append(text.strip())
            except (IndexError, TypeError, AttributeError) as e:
                logger.warning(f"Decode error for sample {i}: {e}")
                translations.append("")
        
        # Calculate confidence from scores
        confidences = []
        if scores and len(scores) > 0:
            for i in range(batch_size):
                token_probs = []
                for step_scores in scores:
                    if i < step_scores.shape[0]:
                        probs = torch.softmax(step_scores[i], dim=-1)
                        token_probs.append(probs.max().item())
                
                avg_conf = sum(token_probs) / len(token_probs) if token_probs else 0.5
                confidences.append(avg_conf)
        else:
            confidences = [0.5] * batch_size
        
        return translations, confidences
        
    except Exception as e:
        logger.error(f"Generation with confidence error: {e}", exc_info=True)
        batch_size = len(batch["audio"])
        return [""] * batch_size, [0.0] * batch_size

# ==================== METRICS ====================
def calculate_wer(references, hypotheses):
    try:
        valid = [(r, h) for r, h in zip(references, hypotheses) if r.strip() and h.strip()]
        if not valid:
            return 1.0
        refs, hyps = zip(*valid)
        return jiwer.wer(list(refs), list(hyps))
    except:
        return 1.0

def calculate_bleu(references, hypotheses):
    try:
        valid = [(r, h) for r, h in zip(references, hypotheses) if r.strip() and h.strip()]
        if not valid:
            return 0.0
        refs, hyps = zip(*valid)
        refs_list = [[r] for r in refs]
        bleu = corpus_bleu(list(hyps), refs_list)
        return bleu.score
    except:
        return 0.0

def detect_language(text):
    """Simple Welsh vs English detection."""
    text_lower = text.lower()
    welsh_words = ['yn', 'y', 'yr', 'mae', 'chi', 'ar', 'o', 'ei', 'dw', 'sy', 'bod', 'ddim', 'i', 'na']
    english_words = ['the', 'and', 'is', 'to', 'of', 'in', 'a', 'you', 'that', 'it', 'for', 'was']
    
    welsh_count = sum(1 for w in welsh_words if f' {w} ' in f' {text_lower} ')
    english_count = sum(1 for w in english_words if f' {w} ' in f' {text_lower} ')
    
    if english_count > welsh_count and english_count > 1:
        return 'English'
    elif welsh_count > 0:
        return 'Welsh'
    return 'Unknown'

# ==================== TRAINER ====================
class Trainer:
    def __init__(self):
        models = load_models()
        self.model = models["s2s_model"]
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
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        logger.info(f"   Train: {len(self.train_loader)} batches")
        logger.info(f"   Val: {len(self.val_loader)} batches")
        
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
        self.best_confidence = 0.0
    
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
        logger.info("\nValidating...")
        self.model.eval()
        
        losses = []
        all_refs = []
        all_hyps = []
        all_confidences = []
        samples = []
        
        welsh_count = 0
        english_count = 0
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Val", leave=False)):
            try:
                audio = batch["audio"].to(config.device)
                texts = batch["text"]
                
                # Calculate loss
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
                
                # Generate with confidence
                translations, confidences = generate_with_confidence(self.model, self.processor, batch)
                all_refs.extend(texts)
                all_hyps.extend(translations)
                all_confidences.extend(confidences)
                
                # Language detection
                for trans in translations:
                    lang = detect_language(trans)
                    if lang == 'Welsh':
                        welsh_count += 1
                    elif lang == 'English':
                        english_count += 1
                
                # Collect samples
                if batch_idx < config.log_sample_predictions:
                    for i in range(min(2, len(translations))):
                        if i < len(batch['audio_id']):
                            samples.append({
                                'id': batch['audio_id'][i],
                                'ref': texts[i],
                                'hyp': translations[i],
                                'conf': confidences[i],
                                'lang': detect_language(translations[i])
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
        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        # Count empty predictions
        empty_count = sum(1 for h in all_hyps if not h.strip())
        
        # Language stats
        total_preds = welsh_count + english_count
        
        logger.info(f"\nResults (Step {self.global_step}):")
        logger.info(f"   Loss: {avg_loss:.4f}")
        logger.info(f"   BLEU: {bleu:.2f}")
        logger.info(f"   WER: {wer:.4f}")
        logger.info(f"   Avg Confidence: {avg_conf:.3f}")
        logger.info(f"   Empty: {empty_count}/{len(all_hyps)}")
        if total_preds > 0:
            logger.info(f"   Welsh: {100*welsh_count/total_preds:.1f}% | English: {100*english_count/total_preds:.1f}%")
        
        if samples:
            logger.info("\nSample Predictions:")
            for s in samples[:5]:
                logger.info(f"\n   {s['id']} | Conf: {s['conf']:.3f} | Lang: {s['lang']}")
                logger.info(f"   REF: {s['ref']}")
                logger.info(f"   HYP: {s['hyp']}")
        
        # Save best checkpoints
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("loss")
            logger.info("   NEW BEST LOSS!")
        
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self.save_checkpoint("bleu")
            logger.info("   NEW BEST BLEU!")
        
        if wer < self.best_wer:
            self.best_wer = wer
            self.save_checkpoint("wer")
            logger.info("   NEW BEST WER!")
        
        if avg_conf > self.best_confidence:
            self.best_confidence = avg_conf
            self.save_checkpoint("confidence")
            logger.info("   NEW BEST CONFIDENCE!")
        
        self.model.train()
        return avg_loss, bleu, wer
    
    def save_checkpoint(self, metric):
        save_dir = Path(config.checkpoints_dir) / f"best_{metric}"
        save_dir.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(save_dir)
        logger.info(f"   Saved to {save_dir}")
    
    def train(self):
        logger.info("\n" + "="*70)
        logger.info("   TRAINING START")
        logger.info("="*70)
        
        for epoch in range(config.num_epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"   EPOCH {epoch+1}/{config.num_epochs}")
            logger.info(f"{'='*70}")
            
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
                progress.set_postfix({'loss': f'{avg:.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
                
                if self.global_step > 0 and self.global_step % config.logging_steps == 0:
                    logger.info(f"Step {self.global_step} - Loss: {avg:.4f}")
                
                if self.global_step > 0 and self.global_step % config.save_steps == 0:
                    save_dir = Path(config.checkpoints_dir) / f"step-{self.global_step}"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    self.model.save_pretrained(save_dir)
                    logger.info(f"Checkpoint: step-{self.global_step}")
                
                if self.global_step > 0 and self.global_step % config.eval_steps == 0:
                    self.validate()
            
            epoch_avg = sum(losses) / len(losses) if losses else 0.0
            logger.info(f"\nEpoch {epoch+1} Complete - Avg Loss: {epoch_avg:.4f}")
            self.validate()
        
        logger.info("\n" + "="*70)
        logger.info("   TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Best Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best BLEU: {self.best_bleu:.2f}")
        logger.info(f"Best WER: {self.best_wer:.4f}")
        logger.info(f"Best Confidence: {self.best_confidence:.3f}")

# ==================== MAIN ====================
if __name__ == "__main__":
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    try:
        trainer = Trainer()
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        raise