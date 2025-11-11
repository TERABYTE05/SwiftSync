# """
# Phase 1: Complete Training Script with Feedback Loop and Full Metrics
# Integrated with modular config, dataloader_05, and model_setup_06
# """
# import os
# import json
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from datetime import datetime
# from transformers import get_linear_schedule_with_warmup
# import logging
# import warnings
# import jiwer
# from sacrebleu import corpus_bleu
# from pathlib import Path

# # === Imports from modular files ===
# from training_config_04 import get_config
# from dataloader_05 import create_dataloaders, collate_fn
# from model_setup_06 import load_models

# warnings.filterwarnings("ignore")

# # ==================== CONFIG ====================
# config = get_config()

# # Create directories
# for dir_path in [config.checkpoints_dir, config.logs_dir]:
#     Path(dir_path).mkdir(parents=True, exist_ok=True)

# # ==================== LOGGING ====================
# def setup_logging():
#     """Setup logging with UTF-8 encoding."""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = Path(config.logs_dir) / f"training_{timestamp}.log"

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[
#             logging.FileHandler(log_file, encoding="utf-8"),
#             logging.StreamHandler(),
#         ],
#     )

#     logger = logging.getLogger(__name__)
#     logger.info("Training Configuration : ")
#     for key, value in vars(config).items():
#         if not key.startswith("_"):
#             logger.info(f"{key:30s}: {value}")
#     return logger


# logger = setup_logging()
# logger.info(f"Device: {config.device}")
# if torch.cuda.is_available():
#     logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


# # ==================== METRICS ====================
# class MetricsCalculator:
#     """Calculate BLEU and WER metrics."""

#     @staticmethod
#     def calculate_wer(references, hypotheses):
#         """Calculate Word Error Rate."""
#         try:
#             valid_pairs = [
#                 (ref, hyp)
#                 for ref, hyp in zip(references, hypotheses)
#                 if ref.strip() and hyp.strip()
#             ]
#             if not valid_pairs:
#                 return 1.0
#             refs, hyps = zip(*valid_pairs)
#             wer = jiwer.wer(list(refs), list(hyps))
#             return wer
#         except Exception as e:
#             logger.warning(f"WER calculation failed: {e}")
#             return 1.0

#     @staticmethod
#     def calculate_bleu(references, hypotheses):
#         """Calculate BLEU score."""
#         try:
#             valid_pairs = [
#                 (ref, hyp)
#                 for ref, hyp in zip(references, hypotheses)
#                 if ref.strip() and hyp.strip()
#             ]
#             if not valid_pairs:
#                 return 0.0
#             refs, hyps = zip(*valid_pairs)
#             refs_list = [[ref] for ref in refs]
#             bleu = corpus_bleu(list(hyps), refs_list)
#             return bleu.score
#         except Exception as e:
#             logger.warning(f"BLEU calculation failed: {e}")
#             return 0.0


# # ==================== GENERATION ====================
# def generate_predictions(model, processor, batch):
#     """Generate predictions with proper decoding."""
#     try:
#         audio = batch["audio"].to(config.device)
#         audio_inputs = processor(
#             audio=audio.cpu().numpy(),
#             return_tensors="pt",
#             sampling_rate=config.sample_rate,
#         )
#         audio_inputs = {
#             k: v.to(config.device)
#             for k, v in audio_inputs.items()
#             if isinstance(v, torch.Tensor)
#         }

#         outputs = model.generate(
#             input_features=audio_inputs.get("input_features"),
#             tgt_lang=config.target_lang,
#             max_new_tokens=100,
#             num_beams=1,
#         )

#         if isinstance(outputs, tuple):
#             generated_ids = outputs[0]
#         elif hasattr(outputs, "sequences"):
#             generated_ids = outputs.sequences
#         else:
#             generated_ids = outputs

#         predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
#         return predictions

#     except Exception as e:
#         logger.warning(f"Generation error: {e}")
#         return [""] * len(batch["audio"])


# # ==================== TRAINER ====================
# class Trainer:
#     """Main trainer with feedback loop and metrics."""

#     def __init__(self):
#         # Load models and processors
#         models = load_models(config)
#         self.s2s_model = models["s2s_model"]
#         self.processor = models["processor"]
#         self.feedback_model = models["feedback_model"]
#         self.feedback_processor = models["feedback_processor"]

#         # Load dataloaders
#         logger.info("\nCreating dataloaders...")
#         self.train_loader, self.val_loader = create_dataloaders(config)
#         logger.info(f"  Train batches: {len(self.train_loader)}")
#         logger.info(f"  Val batches: {len(self.val_loader)}")

#         # Optimizer
#         trainable_params = [
#             p for p in self.s2s_model.parameters() if p.requires_grad
#         ]
#         self.optimizer = AdamW(
#             trainable_params,
#             lr=config.learning_rate,
#             weight_decay=config.weight_decay,
#         )

#         # Scheduler
#         total_steps = len(self.train_loader) * config.num_epochs
#         self.scheduler = get_linear_schedule_with_warmup(
#             self.optimizer,
#             num_warmup_steps=config.warmup_steps,
#             num_training_steps=total_steps,
#         )

#         # State tracking
#         self.global_step = 0
#         self.best_val_loss = float("inf")
#         self.best_bleu = 0.0
#         self.best_wer = float("inf")

#         # Metrics
#         self.metrics_calc = MetricsCalculator()
#         self.history = {
#             "train_loss": [],
#             "val_loss": [],
#             "bleu": [],
#             "wer": [],
#         }

#     def train_step(self, batch):
#         """Single training step with optional feedback loop."""
#         audio = batch["audio"].to(config.device)
#         texts = batch["text"]

#         try:
#             audio_inputs = self.processor(
#                 audio=audio.cpu().numpy(),
#                 return_tensors="pt",
#                 sampling_rate=config.sample_rate,
#             )
#             audio_inputs = {
#                 k: v.to(config.device)
#                 for k, v in audio_inputs.items()
#                 if isinstance(v, torch.Tensor)
#             }

#             text_inputs = self.processor.tokenizer(
#                 texts, return_tensors="pt", padding=True, truncation=True, max_length=200
#             )
#             labels = text_inputs["input_ids"].to(config.device)

#             # Forward pass
#             outputs = self.s2s_model(
#                 input_features=audio_inputs.get("input_features"),
#                 labels=labels,
#             )
#             main_loss = outputs.loss
#             total_loss = main_loss
#             consistency_loss_value = 0.0

#             # Feedback consistency
#             if (
#                 config.use_feedback_loop
#                 and (self.global_step + 1) % config.feedback_frequency == 0
#             ):
#                 try:
#                     generated_outputs = self.s2s_model.generate(
#                         input_features=audio_inputs.get("input_features"),
#                         tgt_lang=config.target_lang,
#                         generate_speech=True,
#                         max_new_tokens=200,
#                     )

#                     if hasattr(generated_outputs, "audio_wavs") and generated_outputs.audio_wavs:
#                         gen_audios = [
#                             wav.squeeze().cpu().numpy()
#                             if isinstance(wav, torch.Tensor)
#                             else wav
#                             for wav in generated_outputs.audio_wavs
#                         ]

#                         feedback_inputs = self.feedback_processor(
#                             gen_audios,
#                             sampling_rate=16000,
#                             return_tensors="pt",
#                             padding=True,
#                         )
#                         feedback_inputs = {
#                             k: v.to(config.device)
#                             for k, v in feedback_inputs.items()
#                             if isinstance(v, torch.Tensor)
#                         }

#                         with self.feedback_processor.as_target_processor():
#                             feedback_labels = self.feedback_processor(
#                                 texts, return_tensors="pt", padding=True
#                             ).input_ids.to(config.device)

#                         feedback_outputs = self.feedback_model(
#                             input_values=feedback_inputs.get("input_values"),
#                             attention_mask=feedback_inputs.get("attention_mask"),
#                             labels=feedback_labels,
#                         )

#                         consistency_loss = feedback_outputs.loss
#                         if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
#                             total_loss += config.lambda_consistency * consistency_loss
#                             consistency_loss_value = consistency_loss.item()

#                 except Exception as e:
#                     logger.debug(f"Feedback loop error: {str(e)[:150]}")

#             # Backprop
#             loss_scaled = total_loss / config.gradient_accumulation_steps
#             loss_scaled.backward()

#             return {
#                 "total_loss": total_loss.item(),
#                 "main_loss": main_loss.item(),
#                 "consistency_loss": consistency_loss_value,
#             }

#         except Exception as e:
#             logger.error(f"Train step error: {e}")
#             return {"total_loss": 0.0, "main_loss": 0.0, "consistency_loss": 0.0}

#     @torch.no_grad()
#     def validate(self):
#         """Validation loop with metrics."""
#         logger.info("\nRunning validation...")
#         self.s2s_model.eval()

#         losses = []
#         all_predictions = []
#         all_references = []
#         sample_logs = []

#         for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
#             try:
#                 audio = batch["audio"].to(config.device)
#                 texts = batch["text"]

#                 audio_inputs = self.processor(
#                     audio=audio.cpu().numpy(),
#                     return_tensors="pt",
#                     sampling_rate=config.sample_rate,
#                 )
#                 audio_inputs = {
#                     k: v.to(config.device)
#                     for k, v in audio_inputs.items()
#                     if isinstance(v, torch.Tensor)
#                 }

#                 text_inputs = self.processor.tokenizer(
#                     texts,
#                     return_tensors="pt",
#                     padding=True,
#                     truncation=True,
#                     max_length=200,
#                 )
#                 labels = text_inputs["input_ids"].to(config.device)

#                 outputs = self.s2s_model(
#                     input_features=audio_inputs.get("input_features"),
#                     labels=labels,
#                 )
#                 losses.append(outputs.loss.item())

#                 predictions = generate_predictions(self.s2s_model, self.processor, batch)
#                 all_predictions.extend(predictions)
#                 all_references.extend(texts)

#                 if batch_idx < config.log_sample_predictions:
#                     for i in range(min(len(predictions), 2)):
#                         if i < len(batch["audio_id"]):
#                             sample_logs.append(
#                                 {
#                                     "audio_id": batch["audio_id"][i],
#                                     "reference": texts[i],
#                                     "prediction": predictions[i],
#                                 }
#                             )
#             except Exception as e:
#                 logger.warning(f"Validation batch error: {str(e)[:150]}")

#         if not losses:
#             logger.warning("No valid validation batches!")
#             return float("inf"), 0.0, 1.0

#         avg_loss = sum(losses) / len(losses)
#         bleu_score = self.metrics_calc.calculate_bleu(all_references, all_predictions)
#         wer_score = self.metrics_calc.calculate_wer(all_references, all_predictions)

#         self.history["val_loss"].append(avg_loss)
#         self.history["bleu"].append(bleu_score)
#         self.history["wer"].append(wer_score)

#         logger.info(f"\nValidation Results (Step {self.global_step}):")
#         logger.info(f"  Loss: {avg_loss:.4f}")
#         logger.info(f"  BLEU: {bleu_score:.2f}")
#         logger.info(f"  WER: {wer_score:.4f}")

#         if sample_logs:
#             logger.info("\nSample Predictions:")
#             for sample in sample_logs:
#                 logger.info(f"\n  ID: {sample['audio_id']}")
#                 logger.info(f"  REF: {sample['reference']}")
#                 logger.info(f"  HYP: {sample['prediction']}")

#         # Save bests
#         if avg_loss < self.best_val_loss:
#             self.best_val_loss = avg_loss
#             self.save_checkpoint(best=True, metric="loss")
#         if bleu_score > self.best_bleu:
#             self.best_bleu = bleu_score
#             self.save_checkpoint(best=True, metric="bleu")
#         if wer_score < self.best_wer:
#             self.best_wer = wer_score
#             self.save_checkpoint(best=True, metric="wer")

#         self.s2s_model.train()
#         return avg_loss, bleu_score, wer_score

#     def save_checkpoint(self, best=False, metric="loss"):
#         """Save model checkpoint."""
#         try:
#             if best:
#                 save_dir = Path(config.checkpoints_dir) / f"best_model_{metric}"
#             else:
#                 save_dir = Path(config.checkpoints_dir) / f"step-{self.global_step}"

#             save_dir.mkdir(exist_ok=True, parents=True)
#             self.s2s_model.save_pretrained(save_dir)

#             state = {
#                 "global_step": self.global_step,
#                 "best_val_loss": self.best_val_loss,
#                 "best_bleu": self.best_bleu,
#                 "best_wer": self.best_wer,
#                 "history": self.history,
#                 "optimizer": self.optimizer.state_dict(),
#                 "scheduler": self.scheduler.state_dict(),
#             }
#             torch.save(state, save_dir / "training_state.pt")
#             logger.info(f"Checkpoint saved to {save_dir}")
#         except Exception as e:
#             logger.error(f"Error saving checkpoint: {e}")

#     def train(self):
#         """Main training loop."""
#         for epoch in range(config.num_epochs):
#             logger.info(f"   EPOCH {epoch + 1}/{config.num_epochs}")
#             progress = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
#             epoch_losses = []

#             for step, batch in enumerate(progress):
#                 losses = self.train_step(batch)
#                 epoch_losses.append(losses["total_loss"])

#                 if (step + 1) % config.gradient_accumulation_steps == 0:
#                     torch.nn.utils.clip_grad_norm_(
#                         self.s2s_model.parameters(), config.max_grad_norm
#                     )
#                     self.optimizer.step()
#                     self.scheduler.step()
#                     self.optimizer.zero_grad()
#                     self.global_step += 1

#                     if self.global_step % config.clear_cache_steps == 0:
#                         torch.cuda.empty_cache()

#                 avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
#                 progress.set_postfix(
#                     {"loss": f"{avg_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"}
#                 )

#                 if self.global_step % config.logging_steps == 0:
#                     logger.info(
#                         f"Step {self.global_step} - Loss: {losses['total_loss']:.4f}, "
#                         f"Main: {losses['main_loss']:.4f}, "
#                         f"Consistency: {losses['consistency_loss']:.4f}"
#                     )

#                 if self.global_step > 0 and self.global_step % config.save_steps == 0:
#                     self.save_checkpoint()

#                 if self.global_step > 0 and self.global_step % config.eval_steps == 0:
#                     self.validate()

#             epoch_avg = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
#             self.history["train_loss"].append(epoch_avg)
#             logger.info(f"\nEpoch {epoch + 1} Complete - Avg Loss: {epoch_avg:.4f}")
#             self.validate()

#         logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
#         logger.info(f"Best BLEU Score: {self.best_bleu:.2f}")
#         logger.info(f"Best WER: {self.best_wer:.4f}")

#         history_file = Path(config.logs_dir) / "training_history.json"
#         with open(history_file, "w") as f:
#             json.dump(self.history, f, indent=2)
#         logger.info(f"\nTraining history saved to {history_file}")


# # ==================== MAIN ====================
# if __name__ == "__main__":
#     torch.manual_seed(config.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(config.seed)

#     try:
#         trainer = Trainer()
#         trainer.train()
#     except KeyboardInterrupt:
#         logger.info("\nTraining interrupted by user")
#     except Exception as e:
#         logger.error(f"\nTraining failed: {e}", exc_info=True)
#         raise
"""
Fixed Training Script with Proper Text Generation
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

# === Imports from modular files ===
from training_config_04 import get_config
from dataloader_05 import create_dataloaders, collate_fn
from model_setup_06 import load_models

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
config = get_config()

# Create directories
for dir_path in [config.checkpoints_dir, config.logs_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# ==================== LOGGING ====================
def setup_logging():
    """Setup logging with UTF-8 encoding."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.logs_dir) / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("Training Configuration : ")
    for key, value in vars(config).items():
        if not key.startswith("_"):
            logger.info(f"{key:30s}: {value}")
    return logger


logger = setup_logging()
logger.info(f"Device: {config.device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


# ==================== METRICS ====================
class MetricsCalculator:
    """Calculate BLEU and WER metrics."""

    @staticmethod
    def calculate_wer(references, hypotheses):
        """Calculate Word Error Rate."""
        try:
            valid_pairs = [
                (ref, hyp)
                for ref, hyp in zip(references, hypotheses)
                if ref.strip() and hyp.strip()
            ]
            if not valid_pairs:
                return 1.0
            refs, hyps = zip(*valid_pairs)
            wer = jiwer.wer(list(refs), list(hyps))
            return wer
        except Exception as e:
            logger.warning(f"WER calculation failed: {e}")
            return 1.0

    @staticmethod
    def calculate_bleu(references, hypotheses):
        """Calculate BLEU score."""
        try:
            valid_pairs = [
                (ref, hyp)
                for ref, hyp in zip(references, hypotheses)
                if ref.strip() and hyp.strip()
            ]
            if not valid_pairs:
                return 0.0
            refs, hyps = zip(*valid_pairs)
            refs_list = [[ref] for ref in refs]
            bleu = corpus_bleu(list(hyps), refs_list)
            return bleu.score
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return 0.0


# ==================== GENERATION (FIXED) ====================
def generate_predictions(model, processor, batch, force_debug=False):
    """Generate predictions with proper decoding - FIXED VERSION."""
    try:
        audio = batch["audio"].to(config.device)
        
        # Process audio inputs
        audio_inputs = processor(
            audio=audio.cpu().numpy(),
            return_tensors="pt",
            sampling_rate=config.sample_rate,
        )
        audio_inputs = {
            k: v.to(config.device)
            for k, v in audio_inputs.items()
            if isinstance(v, torch.Tensor)
        }

        # Generate with proper parameters
        with torch.no_grad():
            generated_outputs = model.generate(
                **audio_inputs,
                tgt_lang="cym",  # CRITICAL: Must be Welsh language code
                max_new_tokens=256,  # Increased from 100
                num_beams=3,  # Use beam search for better quality
                do_sample=False,  # Greedy/beam search, no sampling
                return_dict_in_generate=True,
                output_scores=False
            )

        # Extract generated token IDs
        if hasattr(generated_outputs, 'sequences'):
            generated_ids = generated_outputs.sequences
        elif isinstance(generated_outputs, torch.Tensor):
            generated_ids = generated_outputs
        else:
            logger.warning(f"Unexpected output type: {type(generated_outputs)}")
            return [""] * len(batch["audio"])

        # Decode using the TEXT tokenizer (not speech tokenizer)
        # CRITICAL: Use processor.tokenizer for text decoding
        predictions = []
        for i, ids in enumerate(generated_ids):
            try:
                # Remove pad tokens and special tokens
                ids_filtered = ids[ids != processor.tokenizer.pad_token_id]
                
                # Decode
                text = processor.tokenizer.decode(ids_filtered, skip_special_tokens=True)
                
                # Clean up
                text = text.strip()
                
                # Debug first batch
                if force_debug and i < 2:
                    logger.info(f"\n  Generated IDs (sample {i}): {ids_filtered[:20].tolist()}")
                    logger.info(f"  Decoded text: '{text}'")
                    logger.info(f"  Text length: {len(text)}")
                
                predictions.append(text)
                
            except Exception as e:
                logger.warning(f"Decoding error for sample {i}: {e}")
                predictions.append("")

        # Check if all predictions are empty
        non_empty = [p for p in predictions if p.strip()]
        if not non_empty and force_debug:
            logger.warning(f"⚠️  ALL PREDICTIONS EMPTY!")
            logger.warning(f"  Generated IDs shape: {generated_ids.shape}")
            logger.warning(f"  Sample IDs: {generated_ids[0][:30].tolist()}")

        return predictions

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return [""] * len(batch["audio"])


# ==================== TRAINER ====================
class Trainer:
    """Main trainer with feedback loop and metrics."""

    def __init__(self):
        # Load models and processors
        models = load_models(config)
        self.s2s_model = models["s2s_model"]
        self.processor = models["processor"]
        self.feedback_model = models["feedback_model"]
        self.feedback_processor = models["feedback_processor"]

        # CRITICAL: Set target language in processor
        if hasattr(self.processor, 'set_target_lang'):
            self.processor.set_target_lang("cym")
        
        # Load dataloaders
        logger.info("\n📊 Creating dataloaders...")
        self.train_loader, self.val_loader = create_dataloaders(config)
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")

        # Optimizer
        trainable_params = [
            p for p in self.s2s_model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        # State tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_bleu = 0.0
        self.best_wer = float("inf")

        # Metrics
        self.metrics_calc = MetricsCalculator()
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "bleu": [],
            "wer": [],
        }
        
        # Debug flag
        self.first_validation = True

    def train_step(self, batch):
        """Single training step - SIMPLIFIED (no feedback for now)."""
        audio = batch["audio"].to(config.device)
        texts = batch["text"]

        try:
            audio_inputs = self.processor(
                audio=audio.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=config.sample_rate,
            )
            audio_inputs = {
                k: v.to(config.device)
                for k, v in audio_inputs.items()
                if isinstance(v, torch.Tensor)
            }

            # Tokenize with TARGET tokenizer
            text_inputs = self.processor.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256  # Increased
            )
            labels = text_inputs["input_ids"].to(config.device)

            # Forward pass
            outputs = self.s2s_model(
                **audio_inputs,
                labels=labels,
            )
            
            loss = outputs.loss
            
            # Backprop
            loss_scaled = loss / config.gradient_accumulation_steps
            loss_scaled.backward()

            return {
                "total_loss": loss.item(),
                "main_loss": loss.item(),
                "consistency_loss": 0.0,
            }

        except Exception as e:
            logger.error(f"Train step error: {e}", exc_info=True)
            return {"total_loss": 0.0, "main_loss": 0.0, "consistency_loss": 0.0}

    @torch.no_grad()
    def validate(self):
        """Validation loop with metrics."""
        logger.info("\n🔍 Running validation...")
        self.s2s_model.eval()

        losses = []
        all_predictions = []
        all_references = []
        sample_logs = []

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            try:
                audio = batch["audio"].to(config.device)
                texts = batch["text"]

                audio_inputs = self.processor(
                    audio=audio.cpu().numpy(),
                    return_tensors="pt",
                    sampling_rate=config.sample_rate,
                )
                audio_inputs = {
                    k: v.to(config.device)
                    for k, v in audio_inputs.items()
                    if isinstance(v, torch.Tensor)
                }

                text_inputs = self.processor.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                labels = text_inputs["input_ids"].to(config.device)

                outputs = self.s2s_model(
                    **audio_inputs,
                    labels=labels,
                )
                losses.append(outputs.loss.item())

                # Generate predictions with debug on first batch
                predictions = generate_predictions(
                    self.s2s_model, 
                    self.processor, 
                    batch,
                    force_debug=(batch_idx == 0 and self.first_validation)
                )
                
                all_predictions.extend(predictions)
                all_references.extend(texts)

                if batch_idx < config.log_sample_predictions:
                    for i in range(min(len(predictions), 2)):
                        if i < len(batch["audio_id"]):
                            sample_logs.append(
                                {
                                    "audio_id": batch["audio_id"][i],
                                    "reference": texts[i],
                                    "prediction": predictions[i],
                                    "pred_length": len(predictions[i]),
                                    "ref_length": len(texts[i])
                                }
                            )
            except Exception as e:
                logger.warning(f"Validation batch error: {str(e)[:150]}")

        self.first_validation = False

        if not losses:
            logger.warning("⚠️  No valid validation batches!")
            return float("inf"), 0.0, 1.0

        avg_loss = sum(losses) / len(losses)
        bleu_score = self.metrics_calc.calculate_bleu(all_references, all_predictions)
        wer_score = self.metrics_calc.calculate_wer(all_references, all_predictions)

        self.history["val_loss"].append(avg_loss)
        self.history["bleu"].append(bleu_score)
        self.history["wer"].append(wer_score)

        logger.info(f"\n📊 Validation Results (Step {self.global_step}):")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  BLEU: {bleu_score:.2f}")
        logger.info(f"  WER: {wer_score:.4f}")
        
        # Count empty predictions
        empty_preds = sum(1 for p in all_predictions if not p.strip())
        logger.info(f"  Empty predictions: {empty_preds}/{len(all_predictions)}")

        if sample_logs:
            logger.info("\n📝 Sample Predictions:")
            for sample in sample_logs:
                logger.info(f"\n  ID: {sample['audio_id']}")
                logger.info(f"  REF ({sample['ref_length']} chars): {sample['reference'][:100]}")
                logger.info(f"  HYP ({sample['pred_length']} chars): {sample['prediction'][:100]}")

        # Save bests
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(best=True, metric="loss")
        if bleu_score > self.best_bleu:
            self.best_bleu = bleu_score
            self.save_checkpoint(best=True, metric="bleu")
        if wer_score < self.best_wer:
            self.best_wer = wer_score
            self.save_checkpoint(best=True, metric="wer")

        self.s2s_model.train()
        return avg_loss, bleu_score, wer_score

    def save_checkpoint(self, best=False, metric="loss"):
        """Save model checkpoint."""
        try:
            if best:
                save_dir = Path(config.checkpoints_dir) / f"best_model_{metric}"
            else:
                save_dir = Path(config.checkpoints_dir) / f"step-{self.global_step}"

            save_dir.mkdir(exist_ok=True, parents=True)
            self.s2s_model.save_pretrained(save_dir)

            state = {
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "best_bleu": self.best_bleu,
                "best_wer": self.best_wer,
                "history": self.history,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            torch.save(state, save_dir / "training_state.pt")
            logger.info(f"💾 Checkpoint saved to {save_dir}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def train(self):
        """Main training loop."""
        logger.info("\n" + "="*70)
        logger.info("   🚀 STARTING TRAINING")
        logger.info("="*70)
        
        for epoch in range(config.num_epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"   📚 EPOCH {epoch + 1}/{config.num_epochs}")
            logger.info(f"{'='*70}")
            
            progress = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
            epoch_losses = []

            for step, batch in enumerate(progress):
                losses = self.train_step(batch)
                epoch_losses.append(losses["total_loss"])

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.s2s_model.parameters(), config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % config.clear_cache_steps == 0:
                        torch.cuda.empty_cache()

                avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                progress.set_postfix(
                    {"loss": f"{avg_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"}
                )

                if self.global_step % config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step} - Loss: {losses['total_loss']:.4f}"
                    )

                if self.global_step > 0 and self.global_step % config.save_steps == 0:
                    self.save_checkpoint()

                if self.global_step > 0 and self.global_step % config.eval_steps == 0:
                    self.validate()

            epoch_avg = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            self.history["train_loss"].append(epoch_avg)
            logger.info(f"\n📊 Epoch {epoch + 1} Complete - Avg Loss: {epoch_avg:.4f}")
            self.validate()

        logger.info("\n" + "="*70)
        logger.info("   ✅ TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"📊 Best Validation Loss: {self.best_val_loss:.4f}")
        logger.info(f"📊 Best BLEU Score: {self.best_bleu:.2f}")
        logger.info(f"📊 Best WER: {self.best_wer:.4f}")

        history_file = Path(config.logs_dir) / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"\n💾 Training history saved to {history_file}")


# ==================== MAIN ====================
if __name__ == "__main__":
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    try:
        trainer = Trainer()
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        raise