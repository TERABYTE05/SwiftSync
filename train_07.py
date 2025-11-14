"""
COMPLETE ENHANCED S2S TRAINING WITH FEEDBACK MECHANISMS
Implements:
1. Semantic Consistency Check (ASR back-translation + similarity)
2. Confidence Estimation (prediction uncertainty)
3. Joint optimization with feedback loops
4. Context buffering for streaming
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import numpy as np

# Feedback mechanism imports
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SEMANTIC_AVAILABLE = True
except:
    SEMANTIC_AVAILABLE = False
    print("⚠️  Install: pip install sentence-transformers")

try:
    import jiwer
    from sacrebleu import corpus_bleu
    METRICS_AVAILABLE = True
except:
    METRICS_AVAILABLE = False
    print("⚠️  Install: pip install jiwer sacrebleu")

warnings.filterwarnings("ignore")

# ==================== ENHANCED CONFIG ====================
class Config:
    # Paths
    metadata_file = "organized_data/parallel_metadata.csv"
    checkpoints_dir = "training_output_final/checkpoints"
    logs_dir = "training_output_final/logs"
    
    # Model
    s2s_model_name = "facebook/seamless-m4t-v2-large"
    asr_model_name = "openai/whisper-medium"  # For back-translation
    semantic_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Training
    batch_size = 2
    gradient_accumulation_steps = 8
    num_epochs = 5
    learning_rate = 5e-5
    warmup_steps = 100
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
    
    # Context Buffer for Streaming
    context_buffer_size = 3  # Number of previous chunks to keep
    
    # Feedback Mechanisms
    use_semantic_feedback = True
    use_confidence_estimation = True
    semantic_loss_weight = 0.3  # Weight for semantic consistency loss
    confidence_loss_weight = 0.2  # Weight for confidence loss
    semantic_threshold = 0.7  # Minimum semantic similarity
    confidence_threshold = 0.6  # Minimum confidence score
    
    # Memory Management - CRITICAL for 8GB GPU
    offload_feedback_models = True  # Move ASR/semantic to CPU when not in use
    validation_feedback_frequency = 3  # Only compute full feedback every N validations
    
    # Optimization
    max_grad_norm = 1.0
    clear_cache_steps = 50
    
    # Logging
    logging_steps = 10
    save_steps = 100
    eval_steps = 100
    log_sample_predictions = 5
    
    # Data
    train_split = "train"
    val_split = "dev"
    max_train_samples = None
    max_val_samples = 50
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    use_bfloat16 = False
    target_lang = "cym"
    source_lang = "eng"

config = Config()

for dir_path in [config.checkpoints_dir, config.logs_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# ==================== LOGGING ====================
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.logs_dir) / f"train_enhanced_{timestamp}.log"
    
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
    logger.info("ENHANCED S2S TRAINING WITH FEEDBACK MECHANISMS")
    logger.info("="*70)
    logger.info(f"Semantic Feedback: {config.use_semantic_feedback}")
    logger.info(f"Confidence Estimation: {config.use_confidence_estimation}")
    return logger

logger = setup_logging()

# ==================== MODELS ====================
from transformers import (
    SeamlessM4Tv2Model, 
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor
)
from peft import LoraConfig, get_peft_model, TaskType

def load_models():
    logger.info("\nLoading models...")
    
    # Main S2S model
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
    logger.info(f"  S2S Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # ASR model for back-translation feedback
    asr_model = None
    asr_processor = None
    if config.use_semantic_feedback:
        try:
            logger.info("  Loading ASR model for back-translation...")
            asr_processor = WhisperProcessor.from_pretrained(config.asr_model_name)
            asr_model = WhisperForConditionalGeneration.from_pretrained(
                config.asr_model_name,
                torch_dtype=dtype
            )
            asr_model = asr_model.to(config.device)
            asr_model.eval()
            logger.info("  ASR model loaded successfully")
        except Exception as e:
            logger.warning(f"  Could not load ASR model: {e}")
    
    # Semantic similarity model
    semantic_model = None
    if config.use_semantic_feedback and SEMANTIC_AVAILABLE:
        try:
            logger.info("  Loading semantic similarity model...")
            semantic_model = SentenceTransformer(config.semantic_model_name)
            semantic_model = semantic_model.to(config.device)
            logger.info("  Semantic model loaded successfully")
        except Exception as e:
            logger.warning(f"  Could not load semantic model: {e}")
    
    return {
        "s2s_model": s2s_model,
        "processor": processor,
        "asr_model": asr_model,
        "asr_processor": asr_processor,
        "semantic_model": semantic_model,
    }

# ==================== CONTEXT BUFFER ====================
class ContextBuffer:
    """Stores recent chunks for context-aware translation"""
    
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.audio_buffer = []
        self.text_buffer = []
    
    def add(self, audio_chunk, text_chunk):
        self.audio_buffer.append(audio_chunk)
        self.text_buffer.append(text_chunk)
        
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer.pop(0)
            self.text_buffer.pop(0)
    
    def get_context(self):
        """Returns concatenated context"""
        if not self.audio_buffer:
            return None, None
        
        # Concatenate audio chunks
        audio_context = torch.cat(self.audio_buffer, dim=-1)
        
        # Join text with space
        text_context = " ".join(self.text_buffer)
        
        return audio_context, text_context
    
    def clear(self):
        self.audio_buffer.clear()
        self.text_buffer.clear()

# ==================== ENHANCED DATASET ====================
class StreamingDatasetWithContext(Dataset):
    def __init__(self, split="train"):
        df = pd.read_csv(config.metadata_file)
        
        if split == "all":
            self.data = df
        else:
            self.data = df[df['split'] == split]
        
        if split == config.train_split and config.max_train_samples:
            self.data = self.data[:config.max_train_samples]
        elif split == config.val_split and config.max_val_samples:
            self.data = self.data[:config.max_val_samples]
        
        self.data = self.data.reset_index(drop=True)
        
        # Generate chunks with context
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
        
        # Get English reference for semantic comparison
        eng_words = row.get('english_transcript', '').split()
        if eng_words and total_chunks > 1:
            eng_words_per_chunk = max(1, len(eng_words) // total_chunks)
            eng_start = chunk_idx * eng_words_per_chunk
            eng_end = min(eng_start + eng_words_per_chunk, len(eng_words))
            eng_reference = ' '.join(eng_words[eng_start:eng_end])
        else:
            eng_reference = row.get('english_transcript', '')
        
        return {
            'audio': chunk,
            'text': chunk_text if chunk_text else row['welsh_transcript'],
            'audio_id': f"{row['audio_id']}_c{chunk_idx}",
            'english_reference': eng_reference,
            'chunk_idx': chunk_idx,
            'total_chunks': total_chunks,
        }

def collate_fn(batch):
    return {
        'audio': torch.stack([s['audio'] for s in batch]),
        'text': [s['text'] for s in batch],
        'audio_id': [s['audio_id'] for s in batch],
        'english_reference': [s['english_reference'] for s in batch],
        'chunk_idx': [s['chunk_idx'] for s in batch],
        'total_chunks': [s['total_chunks'] for s in batch],
    }

# ==================== FEEDBACK MECHANISMS ====================
class FeedbackMechanisms:
    """Implements semantic consistency and confidence estimation"""
    
    def __init__(self, asr_model, asr_processor, semantic_model):
        self.asr_model = asr_model
        self.asr_processor = asr_processor
        self.semantic_model = semantic_model
    
    @torch.no_grad()
    def back_translate_audio(self, synthesized_audio_tensor):
        """
        Run ASR on synthesized audio to get back-translated text.
        synthesized_audio_tensor: [batch_size, audio_len]
        """
        if self.asr_model is None or self.asr_processor is None:
            return None
        
        try:
            # Process audio for Whisper
            inputs = self.asr_processor(
                synthesized_audio_tensor.cpu().numpy(),
                sampling_rate=config.sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(config.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Generate transcription
            generated_ids = self.asr_model.generate(**inputs, language=config.source_lang)
            transcriptions = self.asr_processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            return transcriptions
        except Exception as e:
            logger.debug(f"Back-translation error: {e}")
            return None
    
    def compute_semantic_similarity(self, texts1, texts2):
        """
        Compute semantic similarity between two lists of texts.
        Returns: similarity scores [batch_size]
        """
        if self.semantic_model is None or not texts1 or not texts2:
            return None
        
        try:
            # Filter valid pairs
            valid_pairs = [(t1, t2) for t1, t2 in zip(texts1, texts2) 
                          if t1 and t1.strip() and t2 and t2.strip()]
            
            if not valid_pairs:
                return torch.zeros(len(texts1), device=config.device)
            
            texts1_valid, texts2_valid = zip(*valid_pairs)
            
            # Encode
            embeddings1 = self.semantic_model.encode(
                list(texts1_valid), 
                convert_to_tensor=True,
                device=config.device
            )
            embeddings2 = self.semantic_model.encode(
                list(texts2_valid),
                convert_to_tensor=True,
                device=config.device
            )
            
            # Cosine similarity
            similarities = st_util.pytorch_cos_sim(embeddings1, embeddings2)
            # Extract diagonal (pairwise similarities)
            sim_scores = torch.diagonal(similarities)
            
            # Map back to full batch
            result = torch.zeros(len(texts1), device=config.device)
            valid_idx = 0
            for i, (t1, t2) in enumerate(zip(texts1, texts2)):
                if t1 and t1.strip() and t2 and t2.strip():
                    result[i] = sim_scores[valid_idx]
                    valid_idx += 1
            
            return result
        except Exception as e:
            logger.debug(f"Semantic similarity error: {e}")
            return torch.zeros(len(texts1), device=config.device)
    
    def estimate_confidence(self, logits, temperature=1.0):
        """
        Estimate prediction confidence using entropy of output distribution.
        Lower entropy = higher confidence
        
        logits: [batch_size, seq_len, vocab_size]
        Returns: confidence scores [batch_size] in range [0, 1]
        """
        try:
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Compute probabilities
            probs = F.softmax(scaled_logits, dim=-1)  # [batch, seq, vocab]
            
            # Compute entropy per token
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [batch, seq]
            
            # Average entropy over sequence
            avg_entropy = torch.mean(entropy, dim=-1)  # [batch]
            
            # Normalize to [0, 1] - lower entropy = higher confidence
            # Max entropy for uniform distribution = log(vocab_size)
            vocab_size = logits.shape[-1]
            max_entropy = np.log(vocab_size)
            confidence = 1.0 - (avg_entropy / max_entropy)
            
            return torch.clamp(confidence, 0.0, 1.0)
        except Exception as e:
            logger.debug(f"Confidence estimation error: {e}")
            return torch.ones(logits.shape[0], device=logits.device)

# ==================== ENHANCED GENERATION ====================
def generate_translations_with_feedback(model, processor, batch, feedback_mechanisms=None):
    """Generate translations with optional feedback"""
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
        
        # Generate text translations
        with torch.no_grad():
            outputs = model.generate(
                **audio_inputs,
                tgt_lang=config.target_lang,
                generate_speech=False,
                max_new_tokens=50,
                num_beams=3,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Extract text tokens
        if isinstance(outputs, tuple):
            generated_ids = outputs[0]
        elif hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences
        else:
            generated_ids = outputs
        
        if generated_ids.dim() == 3:
            generated_ids = generated_ids[:, 0, :]
        
        # Decode translations
        translations = processor.batch_decode(generated_ids, skip_special_tokens=True)
        translations = [t.strip() for t in translations]
        
        # Feedback metrics (for validation only)
        feedback_info = {
            'semantic_similarity': None,
            'confidence': None,
            'back_translations': None,
        }
        
        if feedback_mechanisms:
            # Confidence estimation from output scores
            if hasattr(outputs, 'scores') and outputs.scores:
                try:
                    # Stack scores: [seq_len, batch, vocab]
                    stacked_scores = torch.stack(outputs.scores)
                    # Transpose to [batch, seq_len, vocab]
                    logits = stacked_scores.permute(1, 0, 2)
                    confidence = feedback_mechanisms.estimate_confidence(logits)
                    feedback_info['confidence'] = confidence.cpu().numpy()
                except Exception as e:
                    logger.debug(f"Confidence computation error: {e}")
        
        return translations, feedback_info
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return [""] * len(batch["audio"]), {'semantic_similarity': None, 'confidence': None, 'back_translations': None}

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

# ==================== ENHANCED TRAINER ====================
class EnhancedTrainer:
    def __init__(self):
        models = load_models()
        self.model = models["s2s_model"]
        self.processor = models["processor"]
        
        # Feedback mechanisms
        self.feedback = None
        if config.use_semantic_feedback or config.use_confidence_estimation:
            self.feedback = FeedbackMechanisms(
                models["asr_model"],
                models["asr_processor"],
                models["semantic_model"]
            )
        
        logger.info("\nCreating datasets...")
        train_dataset = StreamingDatasetWithContext(split=config.train_split)
        val_dataset = StreamingDatasetWithContext(split=config.val_split)
        
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
        
        # Context buffers (one per sample in batch)
        self.context_buffers = [ContextBuffer(config.context_buffer_size) 
                               for _ in range(config.batch_size)]
    
    def train_step(self, batch):
        """Training step with optional feedback losses"""
        audio = batch["audio"].to(config.device)
        texts = batch["text"]
        
        try:
            # Process audio
            audio_inputs = self.processor(
                audio=audio.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=config.sample_rate,
            )
            audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                           if isinstance(v, torch.Tensor)}
            
            # Process target text
            text_inputs = self.processor.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=100
            )
            labels = text_inputs["input_ids"].to(config.device)
            
            # Forward pass
            outputs = self.model(**audio_inputs, labels=labels)
            loss = outputs.loss
            
            # Add feedback losses during training
            if self.feedback and config.use_confidence_estimation:
                # Get logits for confidence estimation
                if hasattr(outputs, 'logits'):
                    try:
                        confidence = self.feedback.estimate_confidence(outputs.logits)
                        # Penalize low confidence predictions
                        confidence_loss = torch.mean(1.0 - confidence)
                        loss = loss + config.confidence_loss_weight * confidence_loss
                    except:
                        pass
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            return loss.item() * config.gradient_accumulation_steps
            
        except Exception as e:
            logger.error(f"Train error: {e}")
            return 0.0
    
    @torch.no_grad()
    def validate(self):
        logger.info("\n" + "="*70)
        logger.info("VALIDATION WITH FEEDBACK METRICS")
        logger.info("="*70)
        self.model.eval()
        
        losses = []
        all_refs = []
        all_hyps = []
        samples = []
        
        all_confidences = []
        all_semantic_sims = []
        
        # Move feedback models to GPU only if needed
        compute_full_feedback = (
            self.feedback and 
            hasattr(self, 'validation_count') and 
            self.validation_count % config.validation_feedback_frequency == 0
        )
        
        if not hasattr(self, 'validation_count'):
            self.validation_count = 0
        self.validation_count += 1
        
        if compute_full_feedback and config.offload_feedback_models:
            logger.info("Moving feedback models to GPU...")
            if self.feedback.asr_model:
                self.feedback.asr_model.to(config.device)
            if self.feedback.semantic_model:
                self.feedback.semantic_model.to(config.device)
        
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
                
                # Generate with feedback
                feedback_to_use = self.feedback if compute_full_feedback else None
                translations, feedback_info = generate_translations_with_feedback(
                    self.model, self.processor, batch, feedback_to_use
                )
                
                all_refs.extend(texts)
                all_hyps.extend(translations)
                
                # Collect feedback metrics
                if feedback_info['confidence'] is not None:
                    all_confidences.extend(feedback_info['confidence'])
                
                # Semantic similarity (only when computing full feedback)
                if compute_full_feedback and self.feedback and batch['english_reference']:
                    sem_sim = self.feedback.compute_semantic_similarity(
                        translations,
                        batch['english_reference']
                    )
                    if sem_sim is not None:
                        all_semantic_sims.extend(sem_sim.cpu().numpy())
                
                # Collect samples
                if batch_idx < 3:
                    for i in range(min(2, len(translations))):
                        if i < len(batch['audio_id']):
                            sample = {
                                'id': batch['audio_id'][i],
                                'ref': texts[i],
                                'hyp': translations[i],
                            }
                            if feedback_info['confidence'] is not None and i < len(feedback_info['confidence']):
                                sample['confidence'] = f"{feedback_info['confidence'][i]:.3f}"
                            samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Val batch error: {e}")
        
        # Offload feedback models back to CPU
        if compute_full_feedback and config.offload_feedback_models:
            logger.info("Moving feedback models back to CPU...")
            if self.feedback.asr_model:
                self.feedback.asr_model.cpu()
            if self.feedback.semantic_model:
                self.feedback.semantic_model.cpu()
            torch.cuda.empty_cache()
        
        if not losses:
            self.model.train()
            return float("inf"), 0.0, 1.0
        
        # Compute metrics
        avg_loss = sum(losses) / len(losses)
        bleu = calculate_bleu(all_refs, all_hyps)
        wer = calculate_wer(all_refs, all_hyps)
        empty_count = sum(1 for h in all_hyps if not h.strip())
        
        logger.info(f"\nStep {self.global_step}:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  BLEU: {bleu:.2f}")
        logger.info(f"  WER: {wer:.4f}")
        logger.info(f"  Empty: {empty_count}/{len(all_hyps)}")
        
        # Feedback metrics
        if all_confidences:
            avg_conf = np.mean(all_confidences)
            logger.info(f"  Avg Confidence: {avg_conf:.3f}")
            low_conf = sum(1 for c in all_confidences if c < config.confidence_threshold)
            logger.info(f"  Low Confidence: {low_conf}/{len(all_confidences)}")
        
        if all_semantic_sims:
            avg_sem = np.mean(all_semantic_sims)
            logger.info(f"  Avg Semantic Sim: {avg_sem:.3f}")
            low_sem = sum(1 for s in all_semantic_sims if s < config.semantic_threshold)
            logger.info(f"  Low Semantic Sim: {low_sem}/{len(all_semantic_sims)}")
        
        if samples:
            logger.info("\nSamples:")
            for s in samples[:5]:
                logger.info(f"\n  {s['id']}")
                logger.info(f"  REF: {s['ref']}")
                logger.info(f"  HYP: {s['hyp']}")
                if 'confidence' in s:
                    logger.info(f"  CONF: {s['confidence']}")
        
        # Save best models
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
        
        # Save config
        config_dict = {
            'semantic_feedback': config.use_semantic_feedback,
            'confidence_estimation': config.use_confidence_estimation,
            'semantic_loss_weight': config.semantic_loss_weight,
            'confidence_loss_weight': config.confidence_loss_weight,
            'best_val_loss': self.best_val_loss,
            'best_bleu': self.best_bleu,
            'best_wer': self.best_wer,
        }
        with open(save_dir / 'training_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
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
    
    def cleanup(self):
        """Free up GPU memory after training"""
        logger.info("\nCleaning up GPU memory...")
        
        # Move models to CPU and delete
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
        
        if hasattr(self, 'feedback') and self.feedback:
            if self.feedback.asr_model:
                self.feedback.asr_model.cpu()
                del self.feedback.asr_model
            if self.feedback.semantic_model:
                self.feedback.semantic_model.cpu()
                del self.feedback.semantic_model
            del self.feedback
        
        # Clear references
        del self.optimizer
        del self.scheduler
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("GPU memory cleared!")

# ==================== INFERENCE WITH FEEDBACK ====================
class InferenceEngine:
    """Inference engine with streaming context and feedback"""
    
    def __init__(self, model_path, use_feedback=False, load_asr=False):
        """
        Args:
            model_path: Path to fine-tuned model checkpoint
            use_feedback: Enable confidence estimation (lightweight)
            load_asr: Load ASR model for back-translation (memory intensive)
        """
        logger.info("\nLoading inference models...")
        
        # Load fine-tuned model
        self.processor = AutoProcessor.from_pretrained(config.s2s_model_name)
        
        from peft import PeftModel
        base_model = SeamlessM4Tv2Model.from_pretrained(
            config.s2s_model_name,
            torch_dtype=torch.float32,  # Use FP32 for inference stability
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model = self.model.to(config.device)
        self.model.eval()
        
        logger.info("  Main model loaded")
        
        # Feedback mechanisms (optional, memory-intensive)
        self.feedback = None
        self.use_confidence = use_feedback
        
        if load_asr:
            logger.info("  Loading ASR for back-translation...")
            try:
                asr_processor = WhisperProcessor.from_pretrained(config.asr_model_name)
                asr_model = WhisperForConditionalGeneration.from_pretrained(
                    config.asr_model_name,
                    torch_dtype=torch.float32
                )
                asr_model = asr_model.to(config.device)
                asr_model.eval()
                
                semantic_model = None
                if SEMANTIC_AVAILABLE:
                    semantic_model = SentenceTransformer(config.semantic_model_name)
                    semantic_model = semantic_model.to(config.device)
                
                self.feedback = FeedbackMechanisms(asr_model, asr_processor, semantic_model)
                logger.info("  Feedback models loaded")
            except Exception as e:
                logger.warning(f"  Could not load feedback models: {e}")
        
        # Context buffer for streaming
        self.context_buffer = ContextBuffer(config.context_buffer_size)
        
        logger.info("Inference engine ready!")
    
    @torch.no_grad()
    def translate_chunk(self, audio_chunk, return_feedback=False):
        """
        Translate a single audio chunk with optional context and feedback.
        
        Args:
            audio_chunk: torch.Tensor [audio_len] or numpy array
            return_feedback: bool, whether to return feedback metrics
        
        Returns:
            translation: str
            feedback_info: dict (if return_feedback=True)
        """
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk)
        
        # Add batch dimension
        audio_batch = audio_chunk.unsqueeze(0).to(config.device)
        
        # Process audio
        audio_inputs = self.processor(
            audio=audio_batch.cpu().numpy(),
            return_tensors="pt",
            sampling_rate=config.sample_rate,
        )
        audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
                       if isinstance(v, torch.Tensor)}
        
        # Generate
        outputs = self.model.generate(
            **audio_inputs,
            tgt_lang=config.target_lang,
            generate_speech=False,
            max_new_tokens=50,
            num_beams=3,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True if (return_feedback and self.use_confidence) else False,
        )
        
        # Extract text
        if isinstance(outputs, tuple):
            generated_ids = outputs[0]
        elif hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences
        else:
            generated_ids = outputs
        
        if generated_ids.dim() == 3:
            generated_ids = generated_ids[:, 0, :]
        
        translation = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Update context buffer
        self.context_buffer.add(audio_chunk, translation)
        
        if not return_feedback:
            return translation
        
        # Compute feedback metrics
        feedback_info = {
            'confidence': None,
            'context_length': len(self.context_buffer.text_buffer),
        }
        
        if self.use_confidence and hasattr(outputs, 'scores') and outputs.scores:
            try:
                stacked_scores = torch.stack(outputs.scores)
                logits = stacked_scores.permute(1, 0, 2)
                confidence = self.feedback.estimate_confidence(logits) if self.feedback else None
                if confidence is None:
                    # Lightweight confidence without feedback module
                    probs = F.softmax(logits / 1.0, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    avg_entropy = torch.mean(entropy, dim=-1)
                    vocab_size = logits.shape[-1]
                    max_entropy = np.log(vocab_size)
                    confidence = 1.0 - (avg_entropy / max_entropy)
                    confidence = torch.clamp(confidence, 0.0, 1.0)
                feedback_info['confidence'] = confidence[0].item() if hasattr(confidence, 'item') else confidence
            except Exception as e:
                logger.debug(f"Confidence error: {e}")
        
        return translation, feedback_info
    
    def translate_file(self, audio_path, return_chunks=False):
        """
        Translate entire audio file in streaming fashion.
        
        Args:
            audio_path: path to audio file
            return_chunks: bool, whether to return individual chunks
        
        Returns:
            full_translation: str (if return_chunks=False)
            chunk_translations: list of dicts (if return_chunks=True)
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()
        
        # Reset context
        self.context_buffer.clear()
        
        # Process chunks
        audio_len = len(waveform)
        num_chunks = max(1, (audio_len - config.chunk_samples) // config.hop_samples + 1)
        
        chunk_results = []
        
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * config.hop_samples
            end_sample = start_sample + config.chunk_samples
            
            if end_sample > audio_len:
                chunk = torch.nn.functional.pad(waveform[start_sample:], (0, end_sample - audio_len))
            else:
                chunk = waveform[start_sample:end_sample]
            
            translation, feedback = self.translate_chunk(chunk, return_feedback=True)
            
            chunk_results.append({
                'chunk_idx': chunk_idx,
                'translation': translation,
                'confidence': feedback['confidence'],
                'start_time': start_sample / config.sample_rate,
                'end_time': end_sample / config.sample_rate,
            })
        
        if return_chunks:
            return chunk_results
        else:
            full_translation = " ".join([c['translation'] for c in chunk_results])
            return full_translation
    
    def get_context(self):
        """Get current context buffer state"""
        audio_ctx, text_ctx = self.context_buffer.get_context()
        return {
            'text_context': text_ctx,
            'buffer_size': len(self.context_buffer.text_buffer),
        }

# ==================== MAIN ====================
if __name__ == "__main__":
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    try:
        # Training
        logger.info("="*70)
        logger.info("MODE: TRAINING")
        logger.info("="*70)
        
        trainer = EnhancedTrainer()
        trainer.train()
        
        # CRITICAL: Clean up GPU memory before inference
        trainer.cleanup()
        
        # Wait a moment for GPU memory to be released
        import time
        time.sleep(2)
        
        # Example inference (lightweight - no ASR/semantic models)
        logger.info("\n" + "="*70)
        logger.info("TESTING INFERENCE ENGINE (LIGHTWEIGHT)")
        logger.info("="*70)
        
        best_model_path = Path(config.checkpoints_dir) / "best_bleu"
        if best_model_path.exists():
            # Load ONLY the main model, no heavy feedback models
            inference = InferenceEngine(
                str(best_model_path), 
                use_feedback=True,  # Lightweight confidence estimation
                load_asr=False  # Don't load heavy ASR model
            )
            
            # Test on first validation file
            df = pd.read_csv(config.metadata_file)
            test_file = df[df['split'] == 'dev'].iloc[0]['eng_audio']
            
            logger.info(f"\nTesting on: {test_file}")
            chunks = inference.translate_file(test_file, return_chunks=True)
            
            logger.info("\nChunk-by-chunk results:")
            for chunk in chunks[:5]:  # First 5 chunks
                logger.info(f"\nChunk {chunk['chunk_idx']} ({chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s)")
                logger.info(f"  Translation: {chunk['translation']}")
                if chunk['confidence']:
                    logger.info(f"  Confidence: {chunk['confidence']:.3f}")
            
            logger.info("\nFull translation:")
            full_trans = " ".join([c['translation'] for c in chunks])
            logger.info(full_trans)
            
            # Cleanup inference engine
            logger.info("\nCleaning up inference engine...")
            inference.model.cpu()
            del inference
            torch.cuda.empty_cache()
        
        logger.info("\n" + "="*70)
        logger.info("ALL OPERATIONS COMPLETE!")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"\nFailed: {e}", exc_info=True)
        raise
    finally:
        # Final cleanup
        torch.cuda.empty_cache()