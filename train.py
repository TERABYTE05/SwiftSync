import torch
import numpy as np
import logging
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
from config import config
from dataloader import StreamingDatasetWithContext, collate_fn
from model_setup import load_models, FeedbackMechanisms
from generation import generate_translations_with_feedback, calculate_wer, calculate_bleu

### Logging Setup
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
    logger.info("S2S Translation Training with Enhanced Feedback Mechanisms")
    logger.info("="*70)
    logger.info(f"Semantic Feedback: {config.use_semantic_feedback}")
    logger.info(f"Confidence Estimation: {config.use_confidence_estimation}")
    return logger

logger = setup_logging()

### Trainer
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
        logger.info("\n" + "-"*70)
        logger.info("Validation with Feedback Metrics")
        logger.info("-"*70)
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
            num_samples = min(5, len(samples))
            random_samples = random.sample(samples, num_samples)
            for s in random_samples:
                logger.info(f"\n  {s['id']}")
                logger.info(f"Reference: {s['ref']}")
                logger.info(f"Hypothesis: {s['hyp']}")
                if 'confidence' in s:
                    logger.info(f"Confidence: {s['confidence']}")
        
        # Save best models
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("loss")
            logger.info("Best loss")
        
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self.save_checkpoint("bleu")
            logger.info("Best BLEU")
        
        if wer < self.best_wer:
            self.best_wer = wer
            self.save_checkpoint("wer")
            logger.info("Best WER")
        
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
        logger.info("\n" + "-"*70)
        logger.info("Training Start")
        logger.info("-"*70)
        
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
        
        logger.info("\n" + "-"*70)
        logger.info("Training Complete")
        logger.info("-"*70)
        logger.info(f"Best Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best BLEU: {self.best_bleu:.2f}")
        logger.info(f"Best WER: {self.best_wer:.4f}")

### Main
if __name__ == "__main__":
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    try:
        trainer = EnhancedTrainer()
        trainer.train()
        
        logger.info("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        raise