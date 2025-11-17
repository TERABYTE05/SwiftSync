import gradio as gr
import torch
import torchaudio
import numpy as np
from pathlib import Path
from transformers import SeamlessM4Tv2Model, AutoProcessor
from peft import PeftModel
import tempfile
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Configuration 
class InferenceConfig:
    """Configuration for inference"""
    # Model paths
    base_model_name = "facebook/seamless-m4t-v2-large"
    checkpoint_path = "training_output_final/checkpoints/best_bleu"
    
    # Audio settings
    sample_rate = 16000
    chunk_duration = 2.0
    chunk_samples = int(chunk_duration * sample_rate)
    hop_duration = 0.5
    hop_samples = int(hop_duration * sample_rate)
    
    # Generation settings
    target_lang = "cym"  # Welsh
    max_new_tokens = 100
    num_beams = 5
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = InferenceConfig()

### Model Loading
class S2STranslator:
    """English to Welsh Speech-to-Speech Translator"""
    
    def __init__(self):
        logger.info("Loading models...")
        self.device = config.device
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(config.base_model_name)
        
        # Load fine-tuned model
        checkpoint_path = Path(config.checkpoint_path)
        
        if checkpoint_path.exists():
            logger.info(f"Loading fine-tuned model from {checkpoint_path}")
            base_model = SeamlessM4Tv2Model.from_pretrained(
                config.base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        else:
            logger.warning("Fine-tuned model not found! Using base model.")
            self.model = SeamlessM4Tv2Model.from_pretrained(
                config.base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz
            if sample_rate != config.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, config.sample_rate)
                waveform = resampler(waveform)
            
            return waveform.squeeze()
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def mix_stereo_audio(self, left_audio, right_audio):
        """
        Mix two mono audio streams into stereo (left=English, right=Welsh)
        for true simultaneous playback
        """
        try:
            # Ensure both are same length (pad shorter one)
            max_len = max(len(left_audio), len(right_audio))
            
            if len(left_audio) < max_len:
                left_audio = torch.nn.functional.pad(left_audio, (0, max_len - len(left_audio)))
            if len(right_audio) < max_len:
                right_audio = torch.nn.functional.pad(right_audio, (0, max_len - len(right_audio)))
            
            # Create stereo: [2, samples]
            stereo = torch.stack([left_audio, right_audio], dim=0)
            
            return stereo
        except Exception as e:
            logger.error(f"Stereo mixing error: {e}")
            return None
    
    @torch.no_grad()
    def translate_chunk(self, audio_chunk):
        """Translate a single audio chunk"""
        try:
            # Add batch dimension
            audio_batch = audio_chunk.unsqueeze(0)
            
            # Process audio
            audio_inputs = self.processor(
                audio=audio_batch.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=config.sample_rate,
            )
            audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items() 
                           if isinstance(v, torch.Tensor)}
            
            # Generate both text AND speech
            outputs = self.model.generate(
                **audio_inputs,
                tgt_lang=config.target_lang,
                generate_speech=True,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                return_intermediate_token_ids=True,
            )
            
            # Extract outputs
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                text_ids = outputs[0]
                audio_waveform = outputs[1]
            else:
                if hasattr(outputs, 'sequences'):
                    text_ids = outputs.sequences
                else:
                    text_ids = outputs
                
                if hasattr(outputs, 'waveform'):
                    audio_waveform = outputs.waveform
                elif hasattr(outputs, 'audio_waveform'):
                    audio_waveform = outputs.audio_waveform
                else:
                    audio_waveform = None
            
            # Decode text
            if isinstance(text_ids, torch.Tensor):
                if text_ids.ndim == 3:
                    text_ids = text_ids[:, 0, :]
            
            translation_text = self.processor.batch_decode(text_ids, skip_special_tokens=True)[0].strip()
            
            # Process audio waveform
            if audio_waveform is not None:
                if isinstance(audio_waveform, torch.Tensor):
                    while audio_waveform.ndim > 1 and audio_waveform.shape[0] == 1:
                        audio_waveform = audio_waveform.squeeze(0)
                    
                    if audio_waveform.ndim == 2:
                        audio_waveform = torch.mean(audio_waveform, dim=0)
                    
                    if audio_waveform.ndim == 1 and audio_waveform.numel() > 0:
                        return translation_text, audio_waveform.cpu()
            
            return translation_text, None
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", None
    
    def translate_file(self, audio_path, progress=gr.Progress()):
        """Translate entire audio file with streaming chunks"""
        try:
            start_time = time.time()
            
            # Load audio
            progress(0, desc="Loading audio...")
            input_waveform = self.load_audio(audio_path)
            
            audio_len = len(input_waveform)
            duration = audio_len / config.sample_rate
            num_chunks = max(1, (audio_len - config.chunk_samples) // config.hop_samples + 1)
            
            logger.info(f"Audio length: {duration:.2f}s, Chunks: {num_chunks}")
            
            all_translations = []
            all_audio_chunks = []
            
            # Process each chunk
            for chunk_idx in range(num_chunks):
                progress_pct = (chunk_idx + 1) / num_chunks
                elapsed = time.time() - start_time
                eta = (elapsed / progress_pct) - elapsed if progress_pct > 0 else 0
                
                progress(
                    progress_pct, 
                    desc=f"Translating chunk {chunk_idx+1}/{num_chunks} | ETA: {eta:.0f}s"
                )
                
                # Extract chunk
                start_sample = chunk_idx * config.hop_samples
                end_sample = start_sample + config.chunk_samples
                
                if end_sample > audio_len:
                    chunk = torch.nn.functional.pad(input_waveform[start_sample:], (0, end_sample - audio_len))
                else:
                    chunk = input_waveform[start_sample:end_sample]
                
                # Translate chunk
                translation_text, audio_output = self.translate_chunk(chunk)
                
                if translation_text and not translation_text.startswith("Error"):
                    all_translations.append(translation_text)
                    
                    if audio_output is not None and isinstance(audio_output, torch.Tensor):
                        if audio_output.ndim == 1 and audio_output.numel() > 0:
                            all_audio_chunks.append(audio_output)
            
            # Combine results
            progress(1.0, desc="Simultaneous translation...")
            full_translation = " ".join(all_translations) if all_translations else "Translation failed"
            
            # Concatenate Welsh audio
            welsh_audio = None
            if all_audio_chunks:
                try:
                    welsh_audio = torch.cat(all_audio_chunks, dim=0)
                    output_duration = welsh_audio.shape[0] / config.sample_rate
                    logger.info(f"Welsh audio: {output_duration:.2f}s")
                except Exception as e:
                    logger.error(f"Audio concatenation error: {e}")
            
            # Create simultaneous stereo mix (English left, Welsh right)
            simultaneous_audio_file = None
            if welsh_audio is not None:
                stereo_mix = self.mix_stereo_audio(input_waveform, welsh_audio)
                
                if stereo_mix is not None:
                    simultaneous_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='_simultaneous.wav')
                    torchaudio.save(
                        simultaneous_audio_file.name,
                        stereo_mix,
                        config.sample_rate
                    )
                    simultaneous_audio_file = simultaneous_audio_file.name
                    logger.info(f"Simultaneous audio created: {simultaneous_audio_file}")
            
            total_time = time.time() - start_time
            
            return full_translation, simultaneous_audio_file, duration, total_time
                
        except Exception as e:
            logger.error(f"File translation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", None, 0, 0

# Initialize translator
logger.info("Initializing translator...")
translator = S2STranslator()
logger.info("Translator ready!")

# ==================== GRADIO INTERFACE ====================
def translate_audio_interface(audio_file):
    """
    Main translation function with automatic simultaneous playback
    """
    if audio_file is None:
        return "Please upload an audio file.", None, ""
    
    try:
        translation_text, simultaneous_audio, input_duration, total_time = translator.translate_file(audio_file)
        
        # Format output
        output_text = f"""
## 🏴󠁧󠁢󠁷󠁬󠁳󠁿 Welsh Translation:

**{translation_text}**
"""
        
        if simultaneous_audio is None:
            output_text += "\n\n**Warning:** Audio generation failed. Text-only translation available."
        
        return output_text, simultaneous_audio, "Translation complete! Audio playing automatically..."
        
    except Exception as e:
        logger.error(f"Interface error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None, "...Translation failed"

# Create Gradio interface
def create_interface():
    """Create auto-play simultaneous translation interface"""
    
    with gr.Blocks(
        title="Simultaneous Translation with Auto-Play",
        theme=gr.themes.Soft(),
        css="""
        .simultaneous-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .audio-player {
            border: 3px solid #4CAF50;
            border-radius: 12px;
            padding: 20px;
            background: #f0f8ff;
        }
        .status-box {
            background: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # **Simultaneous** English to Welsh Speech Translation
            ## **Auto-Play Mode**
    
            ---
            
            ### How It Works:
            1. **Upload** English audio file
            2. **Wait** for translation (~2-5 seconds per second of audio)
            3. **Listen** automatically to simultaneous playback:
            
            **Use headphones for the full simultaneous translation experience!**
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload English Audio")
                audio_input = gr.Audio(
                    label="Select Audio File",
                    type="filepath",
                    sources=["upload"],
                    elem_classes="audio-player"
                )
                
                gr.Markdown(
                    """
                    **Supported formats:**  
                    WAV, MP3, FLAC, OGG, M4A
                    """
                )
                
                status_box = gr.Markdown(
                    value="**Status:** Waiting for audio upload...",
                    elem_classes="status-box"
                )
        
        # Main output section
        gr.Markdown("---")
        gr.Markdown("## Simultaneous Translation Output")
        
        with gr.Row():
            with gr.Column(scale=1):
                simultaneous_audio = gr.Audio(
                    label="SIMULTANEOUS PLAYBACK (Auto-Play)",
                    type="filepath",
                    interactive=False,
                    autoplay=True,  # AUTO-PLAY ENABLED
                    elem_classes="simultaneous-box"
                )
        
        gr.Markdown(
            """
            **Tips:**  
            - With **headphones**: Hear English in left ear, Welsh in right ear simultaneously
            - With **speakers**: Hear both mixed together (stereo effect)
            - **Best experience:** Use stereo headphones! 
            """
        )
        
        # Translation text and stats
        gr.Markdown("---")
        translation_output = gr.Markdown(
            value="",
            elem_classes="simultaneous-box"
        )
        
        # AUTO-TRIGGER
        audio_input.change(
            fn=translate_audio_interface,
            inputs=[audio_input],
            outputs=[translation_output, simultaneous_audio, status_box]
        )
        
        # Examples
        gr.Markdown("---")
        gr.Markdown("### Example Translations")
        
        gr.DataFrame(
            value=[
                ["Good morning, how are you?", "Bore da, sut wyt ti?"],
                ["Thank you very much", "Diolch yn fawr iawn"],
                ["Welcome to Wales", "Croeso i Gymru"],
                ["What time is it?", "Faint o'r gloch yw hi?"],
                ["I love learning Welsh", "Rwy'n caru dysgu Cymraeg"]
            ],
            headers=[" English", " Welsh (Cymraeg)"],
            interactive=False
        )
        
        # Technical details
        gr.Markdown("---")
        gr.Markdown(
            """
            ### 🔧 Technical Architecture
            
            | Component | Specification |
            |-----------|--------------|
            | **Model** | SeamlessM4T-v2-Large (2.3B parameters) |
            | **Fine-tuning** | LoRA adapters (rank=32, alpha=64) |
            | **Trainable Parameters** | 134M (5.8% of total) |
            | **Translation Mode** | Streaming with 2-second chunks |
            | **Overlap** | 0.5 seconds for context preservation |
            | **Audio Output** | Stereo mix (L=English, R=Welsh) |
            | **Sample Rate** | 16kHz |
            | **Beam Search** | 5 beams for quality |
            | **Device** | """ + f"{config.device.upper()}" + """ |
            | **Auto-Play** | Enabled (plays immediately after translation) |
            
            ---
            
            ### 🎓 Research Details
            
            **Thesis Title:** Simultaneous Speech-to-Speech Translation for Low-Resource Languages  
            **Focus:** English to Welsh real-time translation  
            **Institution:** Indian Institute of Technology Bhilai
            **Made by:** Teesha and Suraj  
            **Advisor:** Dr. Rajesh Kumar Mundotiya 
            
            **Key Innovation:** Streaming translation with overlap for coherent output in simultaneous mode.
            
            ---
            
            ### Performance Metrics (From Training)
            
            - **BLEU Score:** 42.73
            - **WER:** 0.61
            - **Translation Speed:** ~1.5-2.0x real-time
            - **Latency:** <1 second per 2-second chunk
            
            **Dataset:** CVSS-C and CV English-Welsh parallel corpus
            """
        )
    
    return demo

### Main
if __name__ == "__main__":
    print(f"Device: {config.device.upper()}")
    print(f"Model: {config.base_model_name}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Mode: Auto-play stereo (L=English, R=Welsh)")
    print("-"*70)
    
    # Create and launch
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        inbrowser=True
    )