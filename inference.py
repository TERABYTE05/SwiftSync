# """
# Phase 2: Real-Time Streaming Speech-to-Speech Translation
# - Microphone input → Welsh audio output
# - 2-second sliding window with 0.5s hop
# - Adaptive confidence-based gating (Emit/Wait/Buffer)
# - Works with your trained model
# """

# import torch
# import torchaudio
# import numpy as np
# from collections import deque
# from pathlib import Path
# import gradio as gr
# import time
# import soundfile as sf
# from transformers import SeamlessM4Tv2Model, AutoProcessor
# from peft import PeftModel

# # ==================== CONFIGURATION ====================
# class InferenceConfig:
#     # Model paths
#     base_model_name = "facebook/seamless-m4t-v2-large"
#     checkpoint_path = "training_output_final/checkpoints/best_bleu"  # Your trained model
    
#     # Audio settings
#     sample_rate = 16000
#     chunk_duration = 2.0  # seconds
#     hop_duration = 0.5  # seconds (process every 0.5s)
#     chunk_samples = int(chunk_duration * sample_rate)  # 32000
#     hop_samples = int(hop_duration * sample_rate)  # 8000
    
#     # Confidence thresholds for Adaptive Gate
#     confidence_emit = 0.85  # Emit immediately if > 0.85
#     confidence_wait = 0.70  # Wait for more context if 0.70-0.85
#     confidence_buffer = 0.70  # Skip if < 0.70
    
#     # Generation settings
#     max_new_tokens = 50
#     num_beams = 3
    
#     # Device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     use_8bit = False  # Set True if GPU memory is tight
    
#     # Languages
#     source_lang = "eng"
#     target_lang = "cym"

# config = InferenceConfig()

# # ==================== MODEL LOADING ====================
# class TranslationModel:
#     """Load trained model with LoRA adapters."""
    
#     def __init__(self):
#         print("="*70)
#         print("   LOADING REAL-TIME TRANSLATION MODEL")
#         print("="*70)
#         print(f"Device: {config.device}")
        
#         # Load processor
#         print("\n📥 Loading processor...")
#         self.processor = AutoProcessor.from_pretrained(config.base_model_name)
        
#         # Load base model
#         print(f"📥 Loading base model...")
#         if config.use_8bit and config.device == "cuda":
#             self.model = SeamlessM4Tv2Model.from_pretrained(
#                 config.base_model_name,
#                 load_in_8bit=True,
#                 device_map="auto"
#             )
#             print("   ✅ Loaded with 8-bit quantization")
#         else:
#             self.model = SeamlessM4Tv2Model.from_pretrained(
#                 config.base_model_name,
#                 torch_dtype=torch.float32
#             )
#             self.model = self.model.to(config.device)
#             print(f"   ✅ Loaded on {config.device}")
        
#         # Load trained LoRA adapters
#         checkpoint_path = Path(config.checkpoint_path)
#         if checkpoint_path.exists():
#             print(f"\n📥 Loading trained LoRA adapters from {checkpoint_path}")
#             self.model = PeftModel.from_pretrained(self.model, str(checkpoint_path))
#             print("   ✅ Fine-tuned model loaded!")
#         else:
#             print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
#             print("   Using base model without fine-tuning")
        
#         self.model.eval()
        
#         # GPU memory info
#         if config.device == "cuda":
#             memory_allocated = torch.cuda.memory_allocated() / 1024**3
#             print(f"\n💾 GPU Memory: {memory_allocated:.2f} GB")
        
#         print("="*70)
    
#     def translate_chunk(self, audio_chunk):
#         """
#         Translate audio chunk to Welsh text and speech.
        
#         Returns:
#             dict with 'text', 'audio', 'confidence'
#         """
#         # Ensure correct shape
#         if audio_chunk.dim() == 1:
#             audio_chunk = audio_chunk.unsqueeze(0)
        
#         # Process audio
#         audio_inputs = self.processor(
#             audio=audio_chunk.cpu().numpy(),
#             sampling_rate=config.sample_rate,
#             return_tensors="pt"
#         )
#         audio_inputs = {k: v.to(config.device) for k, v in audio_inputs.items() 
#                        if isinstance(v, torch.Tensor)}
        
#         # Generate translation with speech
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **audio_inputs,
#                 tgt_lang=config.target_lang,
#                 generate_speech=True,  # Generate audio output
#                 max_new_tokens=config.max_new_tokens,
#                 num_beams=config.num_beams,
#                 return_intermediate_token_ids=True,
#                 output_scores=True,
#                 return_dict_in_generate=True,
#             )
        
#         # Extract text
#         if hasattr(outputs, 'sequences'):
#             text_ids = outputs.sequences
#         else:
#             text_ids = outputs[0] if isinstance(outputs, tuple) else outputs
        
#         translated_text = self.processor.batch_decode(text_ids, skip_special_tokens=True)[0]
        
#         # Extract audio
#         audio_output = None
#         if hasattr(outputs, 'audio_wavs') and outputs.audio_wavs:
#             audio_wav = outputs.audio_wavs[0]
#             if isinstance(audio_wav, torch.Tensor):
#                 audio_output = audio_wav.cpu().numpy()
#             else:
#                 audio_output = np.array(audio_wav)
        
#         # Calculate confidence from scores
#         confidence = 0.5  # Default
#         if hasattr(outputs, 'scores') and outputs.scores:
#             token_probs = []
#             for step_scores in outputs.scores:
#                 probs = torch.softmax(step_scores[0], dim=-1)
#                 token_probs.append(probs.max().item())
#             confidence = sum(token_probs) / len(token_probs) if token_probs else 0.5
        
#         return {
#             'text': translated_text.strip(),
#             'audio': audio_output,
#             'confidence': confidence
#         }


# # ==================== AUDIO BUFFER ====================
# class SlidingWindowBuffer:
#     """Manages 2-second sliding window with 0.5s hop."""
    
#     def __init__(self):
#         self.buffer = deque(maxlen=config.chunk_samples)
#         self.sample_rate = config.sample_rate
#         self.last_process_time = time.time()
    
#     def add_audio(self, audio_data):
#         """Add new audio data to buffer."""
#         if isinstance(audio_data, torch.Tensor):
#             audio_data = audio_data.cpu().numpy()
        
#         # Flatten and add
#         audio_data = audio_data.flatten()
#         self.buffer.extend(audio_data.tolist())
    
#     def get_chunk(self):
#         """Get current 2-second chunk."""
#         if len(self.buffer) < config.chunk_samples:
#             # Pad with zeros
#             chunk = list(self.buffer) + [0.0] * (config.chunk_samples - len(self.buffer))
#         else:
#             chunk = list(self.buffer)[-config.chunk_samples:]
        
#         return torch.tensor(chunk, dtype=torch.float32)
    
#     def should_process(self):
#         """Check if enough time has passed (0.5s hop)."""
#         current_time = time.time()
#         if current_time - self.last_process_time >= config.hop_duration:
#             self.last_process_time = current_time
#             return True
#         return False
    
#     def clear(self):
#         """Clear buffer."""
#         self.buffer.clear()


# # ==================== ADAPTIVE DECISION GATE ====================
# class AdaptiveGate:
#     """
#     Implements Emit/Wait/Buffer logic based on confidence.
#     """
    
#     def __init__(self):
#         self.last_emit_time = time.time()
#         self.waiting_buffer = []
#         self.max_wait_time = 1.5  # Maximum wait time in seconds
    
#     def decide(self, translation_result):
#         """
#         Decide action based on confidence.
        
#         Returns:
#             dict with 'action' (emit/wait/buffer), 'text', 'audio', 'confidence'
#         """
#         confidence = translation_result['confidence']
#         text = translation_result['text']
#         audio = translation_result['audio']
        
#         current_time = time.time()
        
#         # HIGH CONFIDENCE: EMIT immediately
#         if confidence >= config.confidence_emit:
#             self.last_emit_time = current_time
#             self.waiting_buffer.clear()
#             return {
#                 'action': 'emit',
#                 'text': text,
#                 'audio': audio,
#                 'confidence': confidence
#             }
        
#         # MEDIUM CONFIDENCE: WAIT for more context
#         elif confidence >= config.confidence_wait:
#             self.waiting_buffer.append(translation_result)
            
#             # If waited too long, emit best result
#             if current_time - self.last_emit_time > self.max_wait_time:
#                 if self.waiting_buffer:
#                     best = max(self.waiting_buffer, key=lambda x: x['confidence'])
#                     self.last_emit_time = current_time
#                     self.waiting_buffer.clear()
#                     return {
#                         'action': 'emit',
#                         'text': best['text'],
#                         'audio': best['audio'],
#                         'confidence': best['confidence']
#                     }
            
#             return {
#                 'action': 'wait',
#                 'text': '',
#                 'audio': None,
#                 'confidence': confidence
#             }
        
#         # LOW CONFIDENCE: BUFFER (skip)
#         else:
#             return {
#                 'action': 'buffer',
#                 'text': '',
#                 'audio': None,
#                 'confidence': confidence
#             }


# # ==================== REAL-TIME TRANSLATOR ====================
# class RealtimeTranslator:
#     """Main real-time translation engine."""
    
#     def __init__(self):
#         print("\n🔄 Initializing real-time translator...")
#         self.model = TranslationModel()
#         self.audio_buffer = SlidingWindowBuffer()
#         self.adaptive_gate = AdaptiveGate()
#         self.accumulated_output = []
#         print("✅ Ready for real-time translation!\n")
    
#     def process_audio_stream(self, audio_chunk, sample_rate):
#         """
#         Process incoming audio chunk from microphone.
        
#         Returns:
#             Welsh text and audio to display/play
#         """
#         if audio_chunk is None:
#             return "", None
        
#         # Resample if needed
#         if sample_rate != config.sample_rate:
#             audio_tensor = torch.from_numpy(audio_chunk).float()
#             if audio_tensor.dim() == 1:
#                 audio_tensor = audio_tensor.unsqueeze(0)
#             resampler = torchaudio.transforms.Resample(sample_rate, config.sample_rate)
#             audio_chunk = resampler(audio_tensor).squeeze().numpy()
        
#         # Add to buffer
#         self.audio_buffer.add_audio(audio_chunk)
        
#         # Check if time to process (every 0.5s)
#         if not self.audio_buffer.should_process():
#             return "", None
        
#         # Get current chunk
#         chunk = self.audio_buffer.get_chunk()
        
#         # Translate
#         translation = self.model.translate_chunk(chunk)
        
#         # Adaptive decision
#         decision = self.adaptive_gate.decide(translation)
        
#         # Handle decision
#         if decision['action'] == 'emit':
#             output_text = f"[{decision['confidence']:.2f}] {decision['text']}"
#             self.accumulated_output.append(output_text)
            
#             # Return text and audio
#             return "\n".join(self.accumulated_output), (config.sample_rate, decision['audio'])
        
#         # Wait or buffer - return empty
#         return "\n".join(self.accumulated_output), None


# # ==================== GRADIO INTERFACE ====================
# def create_gradio_app():
#     """Create Gradio streaming interface."""
    
#     translator = RealtimeTranslator()
    
#     def process_audio(audio, state):
#         """Process streaming audio."""
#         if audio is None:
#             return state, None
        
#         sample_rate, audio_data = audio
        
#         # Process chunk
#         text_output, audio_output = translator.process_audio_stream(
#             audio_data,
#             sample_rate
#         )
        
#         return text_output, audio_output
    
#     # Create Gradio interface
#     with gr.Blocks(title="Welsh Real-Time Translation", theme=gr.themes.Soft()) as app:
#         gr.Markdown("""
#         # 🎙️ Real-Time English → Welsh Speech Translation
        
#         **How to use:**
#         1. Click "Record" and speak English
#         2. Welsh translation appears in real-time
#         3. Confidence scores show model certainty
        
#         **Confidence Guide:**
#         - 🟢 [0.85+] High confidence - Emitted immediately
#         - 🟡 [0.70-0.85] Medium - Waited for context
#         - 🔴 [<0.70] Low - Buffered/skipped
#         """)
        
#         with gr.Row():
#             with gr.Column(scale=1):
#                 audio_input = gr.Audio(
#                     sources=["microphone"],
#                     streaming=True,
#                     label="🎤 Speak English",
#                     type="numpy"
#                 )
                
#                 gr.Markdown("""
#                 ### Settings
#                 - **Chunk**: 2.0s sliding window
#                 - **Hop**: 0.5s processing interval
#                 - **Model**: Your fine-tuned SeamlessM4T
#                 """)
            
#             with gr.Column(scale=1):
#                 text_output = gr.Textbox(
#                     label="📝 Welsh Translation (with confidence)",
#                     lines=15,
#                     max_lines=20
#                 )
                
#                 audio_output = gr.Audio(
#                     label="🔊 Welsh Audio Output",
#                     autoplay=True
#                 )
        
#         # Stream processing
#         audio_input.stream(
#             fn=process_audio,
#             inputs=[audio_input, text_output],
#             outputs=[text_output, audio_output]
#         )
        
#         gr.Markdown("""
#         ---
#         ### Architecture
#         - **Streaming**: 2-second sliding window with 0.5s hop
#         - **Adaptive Gate**: Emit/Wait/Buffer based on confidence
#         - **Fine-tuned**: Your trained LoRA model
#         """)
    
#     return app


# # ==================== MAIN ====================
# def main():
#     """Launch the real-time translation app."""
    
#     print("\n" + "="*70)
#     print("   LAUNCHING REAL-TIME WELSH TRANSLATION")
#     print("="*70)
    
#     # Create and launch app
#     app = create_gradio_app()
    
#     print("\n🌐 Starting web interface...")
#     print("   Open your browser and start speaking!")
#     print("\n" + "="*70)
    
#     app.launch(
#         share=False,  # Set True to create public link
#         server_name="127.0.0.1",  # Changed from 0.0.0.0
#         server_port=7860,  # Changed from 7860 in case port is busy
#         show_error=True,
#         inbrowser=True  # Auto-open browser
#     )


# if __name__ == "__main__":
#     main()


"""
Simple Non-Streaming Version - Process Complete Audio Files
Fixed based on official HuggingFace documentation
"""

import torch
import torchaudio
import gradio as gr
from transformers import SeamlessM4Tv2Model, AutoProcessor
from peft import PeftModel
from pathlib import Path
import time
import numpy as np

# Configuration
MODEL_NAME = "facebook/seamless-m4t-v2-large"
CHECKPOINT = "training_output_final/checkpoints/best_bleu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {DEVICE}...")

# Load model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SeamlessM4Tv2Model.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)

# Load LoRA
if Path(CHECKPOINT).exists():
    print(f"Loading LoRA from {CHECKPOINT}")
    model = PeftModel.from_pretrained(model, CHECKPOINT)
    print("✅ Fine-tuned model loaded!")

model.eval()

def translate_audio(audio):
    """Translate complete audio recording."""
    if audio is None:
        return "Please record some audio first!", None
    
    sample_rate, audio_data = audio
    
    print(f"\n{'='*50}")
    print(f"Processing audio: {len(audio_data)} samples at {sample_rate}Hz")
    print(f"Duration: {len(audio_data)/sample_rate:.2f}s")
    
    # RESAMPLE TO 16kHz if needed
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz...")
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Handle stereo/mono
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 2:
            # If stereo, take mean of channels
            audio_tensor = audio_tensor.mean(dim=1, keepdim=True)
        
        # Resample
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_data = resampler(audio_tensor).squeeze().numpy()
        sample_rate = 16000
        print(f"✅ Resampled to {len(audio_data)} samples")
    
    # Ensure mono and correct dtype
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)
    
    start_time = time.time()
    
    # Process audio
    inputs = processor(
        audios=audio_data,  # Use 'audios' parameter
        sampling_rate=16000,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # ===== SPEECH-TO-SPEECH TRANSLATION =====
    print("Generating speech translation...")
    with torch.no_grad():
        # Returns (audio_waveform, text_tokens) tuple
        audio_outputs = model.generate(
            **inputs,
            tgt_lang="cym",
            generate_speech=True,
            num_beams=5
        )
    
    # Extract audio - first element of tuple
    audio_array = audio_outputs[0].cpu().numpy().squeeze()
    print(f"Generated audio shape: {audio_array.shape}")
    
    # ===== TEXT TRANSLATION (for display) =====
    print("Generating text translation...")
    with torch.no_grad():
        text_outputs = model.generate(
            **inputs,
            tgt_lang="cym",
            generate_speech=False,
            num_beams=5
        )
    
    # Decode text - note the double indexing [0].tolist()[0]
    translated_text = processor.decode(text_outputs[0].tolist()[0], skip_special_tokens=True)
    
    elapsed = time.time() - start_time
    
    print(f"✅ Translation: {translated_text}")
    print(f"⏱️ Time: {elapsed:.2f}s")
    print(f"{'='*50}\n")
    
    result_text = f"Translation: {translated_text}\n\nProcessing time: {elapsed:.2f}s"
    
    # Normalize audio to float32 [-1, 1]
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        audio_array = audio_array / max_val
    
    return result_text, (16000, audio_array)

# Create interface
with gr.Blocks() as app:
    gr.Markdown("# 🎙️ Welsh Translation - Simple Version")
    gr.Markdown("Record audio, click submit, wait for result")
    
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Record English")
    
    with gr.Row():
        submit_btn = gr.Button("🔄 Translate", variant="primary")
    
    with gr.Row():
        text_out = gr.Textbox(label="Welsh Translation", lines=5)
        audio_out = gr.Audio(label="Welsh Audio")
    
    submit_btn.click(
        fn=translate_audio,
        inputs=audio_in,
        outputs=[text_out, audio_out]
    )

print("\n✅ Launching interface...")
app.launch(share=True, inbrowser=True)