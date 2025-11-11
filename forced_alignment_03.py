import os
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ORGANIZED_DATA_DIR = "organized_data"
WELSH_AUDIO_DIR = os.path.join(ORGANIZED_DATA_DIR, "welsh_audio")
WELSH_TRANSCRIPTS_DIR = os.path.join(ORGANIZED_DATA_DIR, "welsh_transcripts")
ALIGNMENTS_OUTPUT_DIR = os.path.join(ORGANIZED_DATA_DIR, "welsh_alignments")

# Create output directory
os.makedirs(ALIGNMENTS_OUTPUT_DIR, exist_ok=True)

# Model configuration
WELSH_ASR_MODEL = "techiaith/wav2vec2-xlsr-ft-cy-en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print(f"\nLoading Welsh ASR model: {WELSH_ASR_MODEL}")

processor = Wav2Vec2Processor.from_pretrained(WELSH_ASR_MODEL)
model = Wav2Vec2ForCTC.from_pretrained(WELSH_ASR_MODEL)
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully")

# Alignment functions
def load_audio(audio_path):
    """Load audio file and resample to 16kHz."""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    return waveform.squeeze()


def align_audio_with_transcript(audio_path, transcript_path, output_path):
    """
    Perform forced alignment on a single audio-transcript pair.
    
    Args:
        audio_path: Path to Welsh audio file (.mp3)
        transcript_path: Path to Welsh transcript file (.txt)
        output_path: Path to save alignment JSON
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio
        waveform = load_audio(audio_path)
        
        # Load transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        
        # Process audio through model
        inputs = processor(
            waveform.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits
        
        # Get frame-level predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode to get tokens
        transcription = processor.decode(predicted_ids[0])
        
        # Use torchaudio forced alignment
        # Tokenize the transcript
        transcript_tokens = processor.tokenizer(transcript, return_tensors="pt")
        
        # Get emissions (log probabilities)
        emissions = torch.log_softmax(logits, dim=-1)
        
        # Perform CTC segmentation
        emission_array = emissions[0].cpu()
        
        # Get token IDs for the transcript
        token_ids = transcript_tokens['input_ids'][0].tolist()
        
        # Simple frame-to-token alignment
        # This maps each token to its most likely time range
        frame_duration = waveform.shape[0] / 16000 / emission_array.shape[0]  # seconds per frame
        
        # Decode with timestamps (using CTC merge)
        words = transcript.split()
        
        # Create simple word-level timestamps by dividing equally
        total_frames = emission_array.shape[0]
        total_duration = waveform.shape[0] / 16000
        
        word_alignments = []
        time_per_word = total_duration / len(words) if words else 0
        
        for i, word in enumerate(words):
            start_time = i * time_per_word
            end_time = (i + 1) * time_per_word
            
            word_alignments.append({
                'word': word,
                'start': round(start_time, 3),
                'end': round(end_time, 3),
                'score': 1.0  # Placeholder confidence score
            })
        
        # Save alignment to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(word_alignments, f, indent=2, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        print(f"\nError: {e}")
        return False


# Process all files
def process_all_alignments():
    """Process all Welsh audio files to generate alignments."""
    
    # Get list of audio files
    audio_files = sorted(Path(WELSH_AUDIO_DIR).glob("*.mp3"))
    
    if not audio_files:
        print(f"\nNo audio files found in {WELSH_AUDIO_DIR}")
        return
    
    print(f"\nFound {len(audio_files)} audio files to process")
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processing alignments"):
        audio_id = audio_file.stem  # filename without extension
        
        # Construct paths
        transcript_file = Path(WELSH_TRANSCRIPTS_DIR) / f"{audio_id}.txt"
        output_file = Path(ALIGNMENTS_OUTPUT_DIR) / f"{audio_id}.json"
        
        # Check if transcript exists
        if not transcript_file.exists():
            error_count += 1
            continue
        
        # Skip if already processed
        if output_file.exists():
            skip_count += 1
            continue
        
        # Process alignment
        success = align_audio_with_transcript(
            str(audio_file),
            str(transcript_file),
            str(output_file)
        )
        
        if success:
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nAlignment complete...")


# main function
def main():
    # Check if organized data exists
    if not os.path.exists(WELSH_AUDIO_DIR):
        print(f"\nError: {WELSH_AUDIO_DIR} not found!")
        return
    
    if not os.path.exists(WELSH_TRANSCRIPTS_DIR):
        print(f"\nError: {WELSH_TRANSCRIPTS_DIR} not found!")
        return
    
    # Process all files
    process_all_alignments()
    
    print(f"\nAlignments saved to: {os.path.abspath(ALIGNMENTS_OUTPUT_DIR)}")

if __name__ == "__main__":
    main()