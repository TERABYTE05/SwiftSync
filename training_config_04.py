"""
Phase 1: Training Configuration
Define all hyperparameters and settings for training.
"""

import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for training the speech translation model."""
    
    # ==================== PATHS ====================
    # Data directories
    organized_data_dir: str = "organized_data"
    english_audio_dir: str = "organized_data/english_audio"
    welsh_audio_dir: str = "organized_data/welsh_audio"
    welsh_transcripts_dir: str = "organized_data/welsh_transcripts"
    welsh_alignments_dir: str = "organized_data/welsh_alignments"
    metadata_file: str = "organized_data/parallel_metadata.csv"
    
    # Output directories
    output_dir: str = "training_output"
    checkpoints_dir: str = "training_output/checkpoints"
    logs_dir: str = "training_output/logs"
    
    # ==================== MODEL CONFIGURATION ====================
    # Base model - Using medium for better fit on 12GB GPU
    s2s_model_name: str = "facebook/seamless-m4t-v2-large"
    welsh_asr_model: str = "techiaith/wav2vec2-xlsr-ft-cy-en"
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 32  # Increased rank for medium model
    lora_alpha: int = 64  # Increased alpha
    lora_dropout: float = 0.05
    lora_target_modules: list = None
    
    # Model precision
    use_bfloat16: bool = True  # Use bfloat16 for training
    use_8bit: bool = False  # Not needed with medium model
    
    # ==================== TRAINING HYPERPARAMETERS ====================
    # Batch and epochs - Conservative settings to avoid OOM
    batch_size: int = 2  # Small batch to be safe (can increase if stable)
    gradient_accumulation_steps: int = 8  # Effective batch = 2 * 8 = 16
    num_epochs: int = 1
    
    # Learning rate
    learning_rate: float = 2e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Audio chunking - Keep memory low
    audio_chunk_duration: float = 2.0  # seconds per chunk
    sample_rate: int = 16000  # Hz
    chunk_samples: int = 32000  # Max frames (2 seconds at 16kHz)
    
    # Loss weights
    lambda_consistency: float = 0.5  # Weight for consistency loss
    
    # ==================== OPTIMIZATION ====================
    # Gradient settings
    max_grad_norm: float = 1.0
    empty_cache_steps: int = 50  # Clear CUDA cache every N steps to prevent OOM
    
    # Optimizer
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # ==================== TRAINING SETTINGS ====================
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory management
    clear_cache_steps: int = 50  # Clear CUDA cache every N steps
    pin_memory: bool = False  # Disable to reduce memory pressure
    num_workers: int = 0  # Single process loading (safer for GPU memory)
    
    # Checkpoint saving
    save_steps: int = 500  # Save every 500 steps
    eval_steps: int = 500  # Evaluate every 500 steps
    save_total_limit: int = 2  # Keep only last 2 checkpoints (save disk space)
    
    # Logging
    logging_steps: int = 50  # Log every 10 steps
    log_sample_predictions: int = 3
    
    # Data
    train_split: str = "train"  # Use all matched pairs for training
    val_split: str = "dev"
    # test_split: str = "test"
    max_train_samples: int = None  # Use all training data
    max_val_samples: int = 100  # Limit validation for speed
    
    # Reproducibility
    seed: int = 42
    
    # ==================== FEEDBACK LOOP SETTINGS ====================
    use_feedback_loop: bool = True  # Enable consistency loss
    feedback_frequency = 5  # Apply feedback every N steps
    
    def __post_init__(self):
        """Create output directories and set device info."""
        # Create directories
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.checkpoints_dir).mkdir(exist_ok=True)
        Path(self.logs_dir).mkdir(exist_ok=True)
        
        # Set LoRA target modules for SeamlessM4T
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc1",
                "fc2",
            ]
        
        # Print configuration
        self.print_config()
    
    def print_config(self):
        """Print training configuration."""
        print("=" * 70)
        print("   TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"\n🤖 Models:")
        print(f"   • S2S Model: {self.s2s_model_name}")
        print(f"   • Feedback Model: {self.welsh_asr_model}")
        
        print(f"\n🎮 Device:")
        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name(0)}")
            print(f"   • Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"   • CPU (Warning: Training will be very slow)")
        
        print(f"\n📊 Training:")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"   • Effective batch: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"   • Epochs: {self.num_epochs}")
        print(f"   • Learning rate: {self.learning_rate}")
        
        print(f"\n🔧 LoRA:")
        print(f"   • Enabled: {self.use_lora}")
        print(f"   • Rank (r): {self.lora_r}")
        print(f"   • Alpha: {self.lora_alpha}")
        print(f"   • Dropout: {self.lora_dropout}")
        
        print(f"\n🔄 Feedback Loop:")
        print(f"   • Enabled: {self.use_feedback_loop}")
        print(f"   • Lambda (λ): {self.lambda_consistency}")
        
        print(f"\n💾 Checkpoints:")
        print(f"   • Save every: {self.save_steps} steps")
        print(f"   • Output dir: {self.checkpoints_dir}")
        
        print("=" * 70)


# ==================== CREATE CONFIG INSTANCE ====================
def get_config():
    """Get training configuration instance."""
    return TrainingConfig()


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("\n✅ Configuration loaded successfully!")