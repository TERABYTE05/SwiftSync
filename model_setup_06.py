"""
Phase 1: Model Setup
Load SeamlessM4T with LoRA adapters and Welsh ASR feedback model.
"""

import torch
from transformers import (
    SeamlessM4Tv2Model,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)
from peft import LoraConfig, get_peft_model, TaskType
from training_config_04 import get_config


class S2SModelWithFeedback:
    """
    Wrapper class for SeamlessM4T with LoRA and feedback loop.
    """
    
    def __init__(self, config):
        """
        Initialize models:
        - Main S2S model with LoRA adapters
        - Feedback ASR model (frozen)
        """
        self.config = config
        self.device = config.device
        
        print("=" * 70)
        print("   LOADING MODELS")
        print("=" * 70)
        
        # Load main S2S model
        self._load_s2s_model()
        
        # Load feedback model
        self._load_feedback_model()
        
        # Print parameter counts
        self._print_trainable_parameters()
        
        print("=" * 70)
    
    def _load_s2s_model(self):
        """Load SeamlessM4T model with LoRA adapters."""
        print(f"\n📥 Loading S2S model: {self.config.s2s_model_name}")
        
        # Load processor (handles tokenization)
        self.processor = AutoProcessor.from_pretrained(self.config.s2s_model_name)
        
        # Load model
        dtype = torch.bfloat16 if self.config.use_bfloat16 else torch.float32
        self.s2s_model = SeamlessM4Tv2Model.from_pretrained(
            self.config.s2s_model_name,
            torch_dtype=dtype,
        )
        
        print(f"   ✅ Base model loaded")
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            print(f"\n🔧 Applying LoRA adapters...")
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            
            # Apply LoRA to model
            self.s2s_model = get_peft_model(self.s2s_model, lora_config)
            print(f"   ✅ LoRA applied (rank={self.config.lora_r})")
        
        # Move to device
        self.s2s_model = self.s2s_model.to(self.device)
        print(f"   ✅ Model moved to {self.device}")
    
    def _load_feedback_model(self):
        """Load Welsh ASR model for feedback loop (frozen)."""
        print(f"\n📥 Loading feedback model: {self.config.welsh_asr_model}")
        
        # Load processor
        self.feedback_processor = Wav2Vec2Processor.from_pretrained(
            self.config.welsh_asr_model
        )
        
        # Load model
        self.feedback_model = Wav2Vec2ForCTC.from_pretrained(
            self.config.welsh_asr_model
        )
        
        # Freeze all parameters (no training)
        for param in self.feedback_model.parameters():
            param.requires_grad = False
        
        # Move to device
        self.feedback_model = self.feedback_model.to(self.device)
        self.feedback_model.eval()  # Always in eval mode
        
        print(f"   ✅ Feedback model loaded (frozen)")
    
    def _print_trainable_parameters(self):
        """Print number of trainable vs total parameters."""
        print(f"\n📊 Model Parameters:")
        
        # Count S2S model parameters
        trainable_params = 0
        all_params = 0
        
        for _, param in self.s2s_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_params
        
        print(f"   • Total params: {all_params:,}")
        print(f"   • Trainable params: {trainable_params:,}")
        print(f"   • Trainable %: {trainable_percent:.2f}%")
        
        # Memory estimate
        param_memory = (trainable_params * 4) / (1024**3)  # 4 bytes per param (fp32)
        print(f"   • Estimated trainable memory: {param_memory:.2f} GB")
    
    def get_models(self):
        """Return all models and processors."""
        return {
            's2s_model': self.s2s_model,
            'processor': self.processor,
            'feedback_model': self.feedback_model,
            'feedback_processor': self.feedback_processor,
        }


def load_models(config):
    """
    Load all models required for training.
    
    Args:
        config: Training configuration
    
    Returns:
        Dictionary with all models and processors
    """
    model_wrapper = S2SModelWithFeedback(config)
    return model_wrapper.get_models()


# ==================== TEST MODEL LOADING ====================
if __name__ == "__main__":
    print("=" * 70)
    print("   TESTING MODEL LOADING WITH REAL DATA")
    print("=" * 70)
    
    # Load config
    config = get_config()
    
    # Load models
    models = load_models(config)
    
    print("\n✅ All models loaded successfully!")
    print("\n📦 Available models:")
    for key in models.keys():
        print(f"   • {key}")
    
    # Load real data
    print("\n🧪 Testing with real data samples...")
    from dataloader_05 import create_dataloaders
    
    # Create dataloader
    _, val_loader = create_dataloaders(config)
    
    # Get one real batch
    batch = next(iter(val_loader))
    real_audio = batch['audio'].to(config.device)
    real_texts = batch['text']
    
    print(f"\n📊 Real data batch:")
    print(f"   • Audio shape: {real_audio.shape}")
    print(f"   • Batch size: {len(real_texts)}")
    print(f"   • Sample texts:")
    for i, text in enumerate(real_texts[:2]):
        print(f"      [{i}] '{text}'")
    
    # Test S2S model
    print("\n🔄 Testing S2S model forward pass...")
    s2s_model = models['s2s_model']
    processor = models['processor']
    
    # Prepare inputs for speech encoder
    inputs = processor(
        audios=real_audio.cpu().numpy(),
        return_tensors="pt",
        sampling_rate=16000
    )
    
    # Move to device
    inputs = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Forward pass through speech encoder
    with torch.no_grad():
        encoder_outputs = s2s_model.speech_encoder(**inputs)
    
    print(f"   ✅ Speech encoder output shape: {encoder_outputs.last_hidden_state.shape}")
    
    # Test text decoder with target text
    print("\n🔄 Testing text decoder...")
    
    # Tokenize Welsh text
    text_inputs = processor.tokenizer(
        real_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200
    )
    
    # Move to device
    text_inputs = {k: v.to(config.device) for k, v in text_inputs.items()}
    
    # Forward through decoder
    with torch.no_grad():
        decoder_outputs = s2s_model.text_decoder(
            input_ids=text_inputs['input_ids'],
            encoder_hidden_states=encoder_outputs.last_hidden_state,
        )
    
    print(f"   ✅ Text decoder output shape: {decoder_outputs.last_hidden_state.shape}")
    
    # Test feedback model
    print("\n🔄 Testing feedback ASR model...")
    feedback_model = models['feedback_model']
    feedback_processor = models['feedback_processor']
    
    # Process audio for feedback model
    feedback_inputs = feedback_processor(
        real_audio[0].cpu().numpy(),  # Single sample
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    feedback_inputs = {k: v.to(config.device) for k, v in feedback_inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        feedback_outputs = feedback_model(**feedback_inputs)
    
    print(f"   ✅ Feedback model output shape: {feedback_outputs.logits.shape}")
    
    # Decode feedback prediction
    predicted_ids = torch.argmax(feedback_outputs.logits, dim=-1)
    transcription = feedback_processor.decode(predicted_ids[0])
    print(f"   • Predicted Welsh text: '{transcription}'")
    print(f"   • Actual Welsh text: '{real_texts[0]}'")
    
    # Check memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - memory_reserved
        
        print(f"\n💾 GPU Memory Status:")
        print(f"   • Allocated: {memory_allocated:.2f} GB")
        print(f"   • Reserved: {memory_reserved:.2f} GB")
        print(f"   • Free: {memory_free:.2f} GB")
        print(f"   • Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Memory health check
        if memory_allocated > 10.0:
            print("\n   ⚠️  WARNING: High memory usage! May cause OOM during training.")
        elif memory_free > 5.0:
            print("\n   ✅ Good memory headroom for training!")
        else:
            print("\n   ⚠️  Tight memory, may need to reduce batch size if OOM occurs.")
    
    print("\n" + "=" * 70)
    print("✅ MODEL TEST COMPLETE WITH REAL DATA")
    print("=" * 70)
    print("\nNext steps:")
    print("   1. If memory looks good (>5GB free), proceed to training")
    print("   2. If memory is tight, reduce batch_size in config")
    print("   3. Ready to run: python phase1_training.py")
    print("=" * 70)