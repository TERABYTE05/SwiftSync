"""
Simple test to verify Welsh token forcing works correctly
Tests different approaches to force Welsh output
"""

import torch
from transformers import SeamlessM4Tv2Model, AutoProcessor
import torchaudio

# Load model and processor
print("Loading model...")
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load test audio
audio_path = "audio.wav"
print(f"\nLoading audio: {audio_path}")
waveform, sr = torchaudio.load(audio_path)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)
audio = waveform.squeeze().numpy()

print(f"Audio duration: {len(audio)/16000:.2f}s")

# Get Welsh token ID
welsh_token_id = processor.tokenizer.convert_tokens_to_ids("__cym__")
eng_token_id = processor.tokenizer.convert_tokens_to_ids("__eng__")

print(f"\n{'='*70}")
print("TOKEN IDs:")
print(f"{'='*70}")
print(f"Welsh (__cym__): {welsh_token_id}")
print(f"English (__eng__): {eng_token_id}")

# Check if tokens exist
if welsh_token_id == processor.tokenizer.unk_token_id:
    print("❌ Welsh token not found!")
else:
    print("✅ Welsh token found")

if eng_token_id == processor.tokenizer.unk_token_id:
    print("❌ English token not found!")
else:
    print("✅ English token found")

# Process audio
inputs = processor(
    audios=audio,
    sampling_rate=16000,
    return_tensors="pt"
)
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"\n{'='*70}")
print("TESTING DIFFERENT APPROACHES:")
print(f"{'='*70}")

# Test 1: Just tgt_lang
print("\n--- Test 1: tgt_lang='cym' only ---")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        tgt_lang="cym",
        generate_speech=False,
        max_new_tokens=20,
    )

tokens = outputs[0].tolist()
if isinstance(tokens[0], list):
    tokens = tokens[0]
text = processor.decode(tokens, skip_special_tokens=True)
print(f"Tokens (first 10): {tokens[:10]}")
print(f"Text: '{text}'")
print(f"Language: {'Welsh ✅' if any(w in text.lower() for w in ['yn', 'mae', 'dw']) else 'English ❌'}")

# Test 2: tgt_lang + forced_bos_token_id
print("\n--- Test 2: tgt_lang='cym' + forced_bos_token_id ---")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        tgt_lang="cym",
        generate_speech=False,
        max_new_tokens=20,
        forced_bos_token_id=welsh_token_id,
    )

tokens = outputs[0].tolist()
if isinstance(tokens[0], list):
    tokens = tokens[0]
text = processor.decode(tokens, skip_special_tokens=True)
print(f"Tokens (first 10): {tokens[:10]}")
print(f"Text: '{text}'")
print(f"Language: {'Welsh ✅' if any(w in text.lower() for w in ['yn', 'mae', 'dw']) else 'English ❌'}")

# Test 3: Add src_lang to processor
print("\n--- Test 3: processor with src_lang='eng' ---")
inputs2 = processor(
    audios=audio,
    sampling_rate=16000,
    return_tensors="pt",
    src_lang="eng"  # Explicitly set source language
)
inputs2 = {k: v.to(device) for k, v in inputs2.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs2,
        tgt_lang="cym",
        generate_speech=False,
        max_new_tokens=20,
        forced_bos_token_id=welsh_token_id,
    )

tokens = outputs[0].tolist()
if isinstance(tokens[0], list):
    tokens = tokens[0]
text = processor.decode(tokens, skip_special_tokens=True)
print(f"Tokens (first 10): {tokens[:10]}")
print(f"Text: '{text}'")
print(f"Language: {'Welsh ✅' if any(w in text.lower() for w in ['yn', 'mae', 'dw']) else 'English ❌'}")

# Test 4: Use text_inputs with tgt_lang prefix
print("\n--- Test 4: Using decoder_input_ids with Welsh prefix ---")

# Create a manual Welsh prefix
prefix_ids = torch.tensor([[welsh_token_id]], device=device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        tgt_lang="cym",
        generate_speech=False,
        max_new_tokens=20,
        decoder_input_ids=prefix_ids,  # Force decoder to start with Welsh token
    )

tokens = outputs[0].tolist()
if isinstance(tokens[0], list):
    tokens = tokens[0]
text = processor.decode(tokens, skip_special_tokens=True)
print(f"Tokens (first 10): {tokens[:10]}")
print(f"Text: '{text}'")
print(f"Language: {'Welsh ✅' if any(w in text.lower() for w in ['yn', 'mae', 'dw']) else 'English ❌'}")

# Test 5: Check model's actual language mapping
print(f"\n{'='*70}")
print("MODEL CONFIGURATION CHECK:")
print(f"{'='*70}")

print("\nChecking model.config for language mappings...")
if hasattr(model.config, 'lang2id'):
    print(f"Available languages: {list(model.config.lang2id.keys())[:20]}...")  # Show first 20
    if 'cym' in model.config.lang2id:
        print(f"✅ 'cym' (Welsh) found in lang2id: {model.config.lang2id['cym']}")
    else:
        print("❌ 'cym' NOT in lang2id!")
    
    if 'eng' in model.config.lang2id:
        print(f"✅ 'eng' (English) found in lang2id: {model.config.lang2id['eng']}")

if hasattr(model.config, 'id2lang'):
    if welsh_token_id in model.config.id2lang:
        print(f"✅ Welsh token {welsh_token_id} maps to: {model.config.id2lang[welsh_token_id]}")
    else:
        print(f"❌ Welsh token {welsh_token_id} NOT in id2lang!")

print(f"\n{'='*70}")
print("SUMMARY:")
print(f"{'='*70}")
print("Run this script and check which approach produces Welsh output.")
print("If ALL approaches produce English, the issue is likely:")
print("1. The audio content (is it actually English speech?)")
print("2. Model behavior with Welsh - may need fine-tuning")
print("3. Welsh token mapping in the model")