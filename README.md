# Swiftsync - S2ST with Feedback Mechanism for Low Resource Languages

Complete modular implementation of Enhanced Speech-to-Speech Translation training with feedback mechanisms.

## Installation
### 1. Conda Environment Setup

Create the environment using the provided `.yml` file:

```bash
conda env create -f environment.yml
conda activate swiftsync
```
Install dependencies:

```bash
pip install -r requirements.txt
```
### 2. Create Dataset
```bash
python extract_files.py && extract_and_organize.py && forced_alignment.py
```
This will:
- Extract raw audio/text files from ZIP files and convert audio to standard format
- Structure dataset into English and Welsh data and create CSV metadata 
- Pair audio with transcripts
- Perform forced alignment and generate word/phoneme timestamps
- Split long audio into segments to produce REF–audio aligned data

## Training

```bash
python train.py
```

This will:
- Load configuration from `config.py`
- Create dataloaders from your metadata
- Load S2S model with LoRA
- Train for 10 epochs with feedback mechanisms
- Save best checkpoints to `training_output_final/checkpoints/`

### Modify Configuration

Edit `config.py` to change settings and adjust according to your hardware:

```python
# Memory optimization
batch_size = 1
gradient_accumulation_steps = 16

# Training duration
num_epochs = 10
learning_rate = 5e-5

# Feedback mechanisms
use_semantic_feedback = True
use_confidence_estimation = True

# Validation frequency
eval_steps = 100
save_steps = 100
```


## Module Descriptions

### `config.py`
- All configuration parameters
- Paths, model names, training settings
- Memory management options
- Easy to modify without touching code

### `dataloader.py`
- `StreamingDatasetWithContext`: Chunked audio dataset
- `collate_fn`: Batch collation
- `create_dataloaders()`: Creates train/val loaders
- Handles audio preprocessing and text chunking

### `model_setup.py`
- `load_s2s_model()`: Load SeamlessM4T with LoRA
- `load_feedback_models()`: Load ASR and semantic models
- `FeedbackMechanisms`: Confidence and similarity estimation
- Model configuration and device placement

### `metrics.py`
- `calculate_wer()`: Word Error Rate
- `calculate_bleu()`: BLEU score
- `calculate_cer()`: Character Error Rate
- `get_all_metrics()`: All metrics at once

### `generation.py`
- `generate_translations_with_feedback()`: Text generation
- `generate_with_speech()`: Text + speech generation
- Handles output formats and decoding
- Feedback metric computation

### `train.py`
- `EnhancedTrainer`: Main training class
- Training loop with gradient accumulation
- Validation with feedback metrics
- Checkpoint saving
- Memory management

## Usage Examples

### Basic Training
```bash
python train.py
```

### Monitor Training
```bash
# Watch logs in real-time
tail -f training_output_final/logs/train_*.log
```

### Resume from Checkpoint
Edit `train.py` and add:
```python
# After model loading
checkpoint_path = "training_output_final/checkpoints/best_bleu"
trainer.model.load_state_dict(torch.load(checkpoint_path + "/adapter_model.bin"))
```

### Adjust Batch Size for Memory
Edit `config.py`:
```python
# Reduce batch size
batch_size = 1

# Increase gradient accumulation to compensate
gradient_accumulation_steps = 32  
```

### Disable Feedback Mechanisms
Edit `config.py`:
```python
# Disable heavy models
use_semantic_feedback = False

# Keep lightweight confidence
use_confidence_estimation = True
```

## Output Structure

```
training_output_final/
├── checkpoints/
│   ├── best_bleu/          # Best BLEU checkpoint
│   ├── best_wer/           # Best WER checkpoint
│   └── best_loss/          # Best loss checkpoint
└── logs/
    └── train_YYYYMMDD_HHMMSS.log
```

Each checkpoint contains:
- `adapter_model.bin`: LoRA weights
- `adapter_config.json`: LoRA configuration
- `training_config.json`: Training metadata

## Troubleshooting

### CUDA Out of Memory
1. Reduce `batch_size` in `config.py`
2. Increase `gradient_accumulation_steps` proportionally
3. Reduce `max_val_samples`
4. Set `use_semantic_feedback = False`
5. Increase `validation_feedback_frequency`

### Slow Training
1. Increase `batch_size` if memory allows
2. Reduce `eval_steps` (validate less frequently)
3. Reduce `save_steps` (save less frequently)
4. Set `num_workers=2` in `dataloader.py` (Linux only)

### Empty Translations
1. Check if model is loading correctly
2. Verify `target_lang = "cym"` in config
3. Check dataset has valid Welsh text
4. Increase training epochs

## Monitoring Training

### Key Metrics
- **Loss**: Should decrease over time
- **BLEU**: Should increase (higher is better)
- **WER**: Should decrease (lower is better)
- **Confidence**: Average prediction confidence
- **Empty Count**: Should be low/zero

### Expected Performance
- Initial BLEU: ~5-15
- Final BLEU: ~30-50 (depends on dataset)
- Training time: ~2-4 hours per epoch (8GB GPU)

## Important Notes

1. **Memory**: Always start with `batch_size=1` on 8GB GPUs
2. **Checkpoints**: Best models saved automatically
3. **Logs**: All training details in logs directory
4. **Feedback**: Can be disabled to save memory
5. **Validation**: More frequent = slower but better monitoring

Check the logs:
```bash
cat training_output_final/logs/train_*.log | grep -i error
```

Common issues:
- OOM - Reduce batch size
- Slow - Reduce validation frequency  
- Empty outputs - Check target language setting
- No improvement - Increase learning rate or epochs

## Output 
To check the streaming output

```bash
python streaming.py
```