import torch
import torch.nn.functional as F
import numpy as np
import logging
from transformers import SeamlessM4Tv2Model, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer, util as st_util

logger = logging.getLogger(__name__)

def load_models():
    from config import config
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
    logger.info(f"  S2S Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # ASR model for back-translation feedback
    asr_model = None
    asr_processor = None
    if config.use_semantic_feedback:
        try:
            logger.info("Loading ASR model for back-translation...")
            asr_processor = WhisperProcessor.from_pretrained(config.asr_model_name)
            asr_model = WhisperForConditionalGeneration.from_pretrained(
                config.asr_model_name,
                torch_dtype=dtype
            )
            asr_model = asr_model.to(config.device)
            asr_model.eval()
            logger.info("ASR model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ASR model: {e}")
    
    # Semantic similarity model
    semantic_model = None
    if config.use_semantic_feedback:
        logger.info("Loading semantic similarity model...")
        semantic_model = SentenceTransformer(config.semantic_model_name)
        semantic_model = semantic_model.to(config.device)
        logger.info("Semantic model loaded successfully")
    
    return {
        "s2s_model": s2s_model,
        "processor": processor,
        "asr_model": asr_model,
        "asr_processor": asr_processor,
        "semantic_model": semantic_model,
    }

### Context Buffer
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

### Feedback Mechanisms
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
        from config import config
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
        from config import config
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
            
            # Normalize and Max entropy for uniform distribution = log(vocab_size)
            vocab_size = logits.shape[-1]
            max_entropy = np.log(vocab_size)
            confidence = 1.0 - (avg_entropy / max_entropy)
            
            return torch.clamp(confidence, 0.0, 1.0)
        except Exception as e:
            logger.debug(f"Confidence estimation error: {e}")
            return torch.ones(logits.shape[0], device=logits.device)