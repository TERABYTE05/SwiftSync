import torch
from pathlib import Path

class Config:
    # Paths
    metadata_file = "organized_data/parallel_metadata.csv"
    checkpoints_dir = "training_output/checkpoints"
    logs_dir = "training_output/logs"
    
    # Models
    s2s_model_name = "facebook/seamless-m4t-v2-large"
    asr_model_name = "openai/whisper-medium" 
    semantic_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Training Parameters
    batch_size = 1 
    gradient_accumulation_steps = 16  
    num_epochs = 10
    learning_rate = 5e-5
    warmup_steps = 100
    weight_decay = 0.01
    
    # LoRA Configuration
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    # Audio Settings
    sample_rate = 16000
    chunk_duration = 2.0  # seconds
    hop_duration = 0.5  # seconds
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)
    
    # Streaming Context
    context_buffer_size = 3  # Number of previous chunks to keep
    
    # Feedback Mechanisms
    use_semantic_feedback = True  # ASR back-translation + similarity
    use_confidence_estimation = True  # Prediction uncertainty
    semantic_loss_weight = 0.3  # Weight for semantic consistency loss
    confidence_loss_weight = 0.2  # Weight for confidence loss
    semantic_threshold = 0.7  # Minimum semantic similarity
    confidence_threshold = 0.6  # Minimum confidence score
    
    # Memory Management
    offload_feedback_models = True  # Move ASR/semantic to CPU when not in use
    validation_feedback_frequency = 3  # Compute full feedback every N validations
    aggressive_memory_cleanup = True  # Enable aggressive cleanup
    
    # Optimization
    max_grad_norm = 1.0
    clear_cache_steps = 50
    
    # Logging
    logging_steps = 10
    save_steps = 100
    eval_steps = 100
    log_sample_predictions = 5
    
    # Data Splits
    train_split = "all"
    val_split = "dev"
    max_train_samples = None  
    max_val_samples = 50 
    
    # Device & Precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    use_bfloat16 = False  
    
    # Language 
    target_lang = "cym"  
    source_lang = "eng"  
    
config = Config()

for dir_path in [config.checkpoints_dir, config.logs_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)