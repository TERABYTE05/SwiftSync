import torch
import logging
import jiwer
from sacrebleu import corpus_bleu

logger = logging.getLogger(__name__)

### Generation with Feedback
def generate_translations_with_feedback(model, processor, batch, feedback_mechanisms=None):
    """Generate translations with optional feedback"""
    from config import config
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

### Metrics Calculation
def calculate_wer(refs, hyps):
    try:
        valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
        if not valid:
            return 1.0
        r, h = zip(*valid)
        return jiwer.wer(list(r), list(h))
    except Exception as e:
        logger.warning(f"WER calculation error: {e}")
        return 1.0

def calculate_bleu(refs, hyps):
    try:
        valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
        if not valid:
            return 0.0
        r, h = zip(*valid)
        refs_list = [[ref] for ref in r]
        bleu = corpus_bleu(list(h), refs_list)
        return bleu.score
    except Exception as e:
        logger.warning(f"BLEU calculation error: {e}")
        return 0.0