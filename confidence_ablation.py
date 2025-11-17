import torch
import torch.nn.functional as F
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
import numpy as np
import sys

from config import config
from dataloader import StreamingDatasetWithContext, collate_fn
from model_setup import load_models
from generation import calculate_wer, calculate_bleu

# Override config
config.use_semantic_feedback = False
config.use_confidence_estimation = True
config.semantic_loss_weight = 0.0
config.confidence_loss_weight = 0.2
config.checkpoints_dir = "ablation_results/confidence_only/checkpoints"
config.logs_dir = "ablation_results/confidence_only/logs"

for dir_path in [config.checkpoints_dir, config.logs_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.logs_dir) / f"confidence_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),  # Force stdout
        ],
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("ABLATION STUDY 2: CONFIDENCE ESTIMATION ONLY")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("="*70)
    return logger

print("="*70)
print("INITIALIZING CONFIDENCE ESTIMATION ABLATION STUDY")
print("="*70)
sys.stdout.flush()

logger = setup_logging()

def estimate_confidence(logits, temperature=1.0):
    """Standalone confidence estimation"""
    try:
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        avg_entropy = torch.mean(entropy, dim=-1)
        vocab_size = logits.shape[-1]
        max_entropy = np.log(vocab_size)
        confidence = 1.0 - (avg_entropy / max_entropy)
        return torch.clamp(confidence, 0.0, 1.0)
    except Exception as e:
        logger.warning(f"Confidence estimation error: {e}")
        return torch.ones(logits.shape[0], device=logits.device)

class ConfidenceTrainer:
    def __init__(self):
        print("\n" + "="*70)
        print("LOADING MODELS...")
        print("="*70)
        sys.stdout.flush()
        
        logger.info("Loading models...")
        models = load_models()
        self.model = models["s2s_model"]
        self.processor = models["processor"]
        
        print("Models loaded successfully")
        sys.stdout.flush()
        
        print("\n" + "="*70)
        print("CREATING DATASETS...")
        print("="*70)
        sys.stdout.flush()
        
        logger.info("Creating datasets...")
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
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        sys.stdout.flush()
        
        logger.info(f"  Train: {len(self.train_loader)} batches")
        logger.info(f"  Val: {len(self.val_loader)} batches")
        
        print("\n" + "="*70)
        print("SETTING UP OPTIMIZER...")
        print("="*70)
        sys.stdout.flush()
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, config.warmup_steps, total_steps
        )
        
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {config.warmup_steps}")
        sys.stdout.flush()
        
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_bleu = 0.0
        self.best_wer = float("inf")
        
        print("\nTrainer initialized successfully!")
        print("="*70)
        sys.stdout.flush()
    
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
            loss = outputs.loss
            
            # Add confidence loss
            if hasattr(outputs, 'logits'):
                try:
                    confidence = estimate_confidence(outputs.logits)
                    confidence_loss = torch.mean(1.0 - confidence)
                    loss = loss + config.confidence_loss_weight * confidence_loss
                except Exception as e:
                    logger.debug(f"Confidence loss error: {e}")
            
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            return loss.item() * config.gradient_accumulation_steps
            
        except Exception as e:
            logger.error(f"Train step error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    @torch.no_grad()
    def validate(self):
        print("\n" + "-"*70)
        print(f"VALIDATION - Step {self.global_step}")
        print("-"*70)
        sys.stdout.flush()
        
        logger.info(f"Validation at step {self.global_step}")
        self.model.eval()
        
        losses = []
        all_refs = []
        all_hyps = []
        all_confidences = []
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            if batch_idx % 10 == 0:
                print(f"  Validation batch {batch_idx}/{len(self.val_loader)}")
                sys.stdout.flush()
            
            try:
                audio = batch["audio"].to(config.device)
                texts = batch["text"]
                
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
                
                # Generate with scores
                gen_outputs = self.model.generate(
                    **audio_inputs,
                    tgt_lang=config.target_lang,
                    generate_speech=False,
                    max_new_tokens=50,
                    num_beams=3,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
                generated_ids = gen_outputs.sequences if hasattr(gen_outputs, 'sequences') else gen_outputs[0]
                if generated_ids.dim() == 3:
                    generated_ids = generated_ids[:, 0, :]
                
                translations = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                translations = [t.strip() for t in translations]
                
                all_refs.extend(texts)
                all_hyps.extend(translations)
                
                # Compute confidence
                if hasattr(gen_outputs, 'scores') and gen_outputs.scores:
                    try:
                        stacked_scores = torch.stack(gen_outputs.scores)
                        logits = stacked_scores.permute(1, 0, 2)
                        confidence = estimate_confidence(logits)
                        all_confidences.extend(confidence.cpu().numpy())
                    except Exception as e:
                        logger.debug(f"Confidence computation error: {e}")
                
            except Exception as e:
                logger.warning(f"Val batch {batch_idx} error: {e}")
        
        if not losses:
            print("No validation losses computed!")
            self.model.train()
            return float("inf"), 0.0, 1.0
        
        avg_loss = sum(losses) / len(losses)
        bleu = calculate_bleu(all_refs, all_hyps)
        wer = calculate_wer(all_refs, all_hyps)
        
        print(f"\nResults:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  BLEU: {bleu:.2f}")
        print(f"  WER: {wer:.4f}")
        
        logger.info(f"Loss: {avg_loss:.4f} | BLEU: {bleu:.2f} | WER: {wer:.4f}")
        
        if all_confidences:
            avg_conf = np.mean(all_confidences)
            std_conf = np.std(all_confidences)
            low_conf = sum(1 for c in all_confidences if c < config.confidence_threshold)
            
            print(f"  Avg Confidence: {avg_conf:.3f} ± {std_conf:.3f}")
            print(f"  Low Confidence: {low_conf}/{len(all_confidences)}")
            
            logger.info(f"Avg Confidence: {avg_conf:.3f}")
            logger.info(f"Low Confidence: {low_conf}/{len(all_confidences)}")
        
        sys.stdout.flush()
        
        # Save best models
        improved = False
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("loss")
            print("  ✨ New best loss!")
            improved = True
        
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self.save_checkpoint("bleu")
            print("  ✨ New best BLEU!")
            improved = True
        
        if wer < self.best_wer:
            self.best_wer = wer
            self.save_checkpoint("wer")
            print("New best WER!")
            improved = True
        
        if not improved:
            print("No improvement this validation")
        
        sys.stdout.flush()
        self.model.train()
        return avg_loss, bleu, wer
    
    def save_checkpoint(self, metric):
        save_dir = Path(config.checkpoints_dir) / f"best_{metric}"
        save_dir.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(save_dir)
        
        results = {
            'ablation': 'confidence_only',
            'best_val_loss': float(self.best_val_loss),
            'best_bleu': float(self.best_bleu),
            'best_wer': float(self.best_wer),
            'global_step': self.global_step,
        }
        with open(save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Checkpoint saved: {save_dir}")
    
    def train(self):
        print("\n" + "="*70)
        print("TRAINING START")
        print("="*70)
        print(f"Total epochs: {config.num_epochs}")
        print(f"Batches per epoch: {len(self.train_loader)}")
        print(f"Validation every {config.eval_steps} steps")
        print("="*70)
        sys.stdout.flush()
        
        logger.info("Training started")
        
        for epoch in range(config.num_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{config.num_epochs}")
            print(f"{'='*70}")
            sys.stdout.flush()
            
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
            
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            losses = []
            
            for step, batch in enumerate(progress):
                if step % 50 == 0:
                    print(f"  Training step {step}/{len(self.train_loader)}")
                    sys.stdout.flush()
                
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
            
            epoch_loss = sum(losses) / len(losses) if losses else 0.0
            print(f"\nEpoch {epoch+1} Complete - Avg Loss: {epoch_loss:.4f}")
            sys.stdout.flush()
            
            logger.info(f"Epoch {epoch+1} Avg Loss: {epoch_loss:.4f}")
            self.validate()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Best Loss: {self.best_val_loss:.4f}")
        print(f"Best BLEU: {self.best_bleu:.2f}")
        print(f"Best WER: {self.best_wer:.4f}")
        print("="*70)
        sys.stdout.flush()
        
        logger.info("="*70)
        logger.info("CONFIDENCE ESTIMATION TRAINING COMPLETE")
        logger.info(f"Best Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best BLEU: {self.best_bleu:.2f}")
        logger.info(f"Best WER: {self.best_wer:.4f}")
        logger.info("="*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING CONFIDENCE ESTIMATION ABLATION")
    print(f"Time: {datetime.now()}")
    print("="*70)
    sys.stdout.flush()
    
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Using CPU (will be slow)")
    sys.stdout.flush()
    
    try:
        trainer = ConfidenceTrainer()
        trainer.train()
        
        print("\n Training completed successfully!")
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
        sys.stdout.flush()
    except Exception as e:
        print(f"\n Training failed: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise