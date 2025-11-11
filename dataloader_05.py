"""
Phase 1: Data Loader
Creates streaming dataset that generates 2-second audio chunks with aligned Welsh text.
"""

import os
import json
import random
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

from training_config_04 import get_config


class StreamingS2SDataset(Dataset):
    """
    Streaming dataset for speech-to-speech translation.
    
    Randomly samples 2-second chunks from English audio and finds the
    corresponding Welsh words from the alignment JSON files.
    """
    
    def __init__(self, config, split="train"):
        """
        Args:
            config: Training configuration
            split: Data split ('train', 'dev', 'test', or 'all')
        """
        self.config = config
        self.split = split
        
        # Load metadata
        metadata_df = pd.read_csv(config.metadata_file)
        print(f"\n🔍 Debug: Total rows in CSV: {len(metadata_df)}")
        print(f"🔍 Debug: Requested split: '{split}'")
        
        # Filter by split
        if split == "all":
            # Use all data for training (combine all splits)
            self.data = metadata_df.reset_index(drop=True)
            print(f"📊 Using all {len(self.data)} matched pairs for training")
        else:
            # Use specific split
            self.data = metadata_df[metadata_df['split'] == split].reset_index(drop=True)
            print(f"📊 Loaded {len(self.data)} samples for '{split}' split")
        
        # Limit samples if specified
        if split in ["train", "all"] and config.max_train_samples:
            self.data = self.data[:config.max_train_samples]
            print(f"   (Limited to {len(self.data)} samples)")
        elif split == "dev" and config.max_val_samples:
            self.data = self.data[:config.max_val_samples]
            print(f"   (Limited to {len(self.data)} samples)")
        
        if len(self.data) == 0:
            raise ValueError(f"No data loaded for split '{split}'! Check your metadata file.")
        
        # Audio settings
        self.sample_rate = config.sample_rate
        self.chunk_duration = config.audio_chunk_duration
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)  # 32000 samples
        
        # Random seed for reproducibility
        random.seed(config.seed)
    
    def __len__(self):
        return len(self.data)
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze()  # Return 1D tensor
    
    def load_alignment(self, alignment_path):
        """Load word-level alignment JSON."""
        with open(alignment_path, 'r', encoding='utf-8') as f:
            alignment = json.load(f)
        return alignment
    
    def get_chunk_with_alignment(self, eng_audio_path, alignment_path):
        """
        Sample a 2-second chunk from English audio and find corresponding Welsh words.
        
        Returns:
            eng_chunk: 2-second audio chunk (torch.Tensor)
            welsh_text: Corresponding Welsh text (str)
            start_time: Chunk start time in seconds (float)
        """
        # Load English audio
        eng_waveform = self.load_audio(eng_audio_path)
        total_samples = eng_waveform.shape[0]
        total_duration = total_samples / self.sample_rate
        
        # If audio is shorter than chunk duration, pad it
        if total_samples < self.chunk_samples:
            padding = self.chunk_samples - total_samples
            eng_chunk = torch.nn.functional.pad(eng_waveform, (0, padding))
            start_time = 0.0
        else:
            # Randomly sample a start position
            max_start_sample = total_samples - self.chunk_samples
            start_sample = random.randint(0, max_start_sample)
            
            # Extract chunk
            eng_chunk = eng_waveform[start_sample:start_sample + self.chunk_samples]
            start_time = start_sample / self.sample_rate
        
        end_time = start_time + self.chunk_duration
        
        # Load Welsh alignment
        alignment = self.load_alignment(alignment_path)
        
        # Find all words that fall within the time window
        welsh_words = []
        for word_info in alignment:
            word_start = word_info['start']
            word_end = word_info['end']
            
            # Include word if it overlaps at all with our time window
            # More lenient: include if word is within or near the window
            if (word_start <= end_time and word_end >= start_time):
                welsh_words.append(word_info['word'])
        
        # Join words to create target text
        welsh_text = " ".join(welsh_words) if welsh_words else ""
        
        # If no words found, use full transcript as fallback
        if not welsh_text:
            # Use all words from alignment
            all_words = [w['word'] for w in alignment]
            if all_words:
                # Take a reasonable chunk from the middle
                total_words = len(all_words)
                words_per_chunk = max(5, total_words // 10)  # At least 5 words
                mid = total_words // 2
                start_idx = max(0, mid - words_per_chunk // 2)
                end_idx = min(total_words, start_idx + words_per_chunk)
                welsh_text = " ".join(all_words[start_idx:end_idx])
        
        return eng_chunk, welsh_text, start_time
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            dict with keys:
                - 'audio': English audio chunk (Tensor)
                - 'text': Welsh target text (str)
                - 'audio_id': Audio file identifier (str)
        """
        row = self.data.iloc[idx]
        
        audio_id = row['audio_id']
        eng_audio_path = row['eng_audio']
        alignment_path = os.path.join(
            self.config.welsh_alignments_dir,
            f"{audio_id}.json"
        )
        
        # Get chunk and aligned text
        try:
            eng_chunk, welsh_text, start_time = self.get_chunk_with_alignment(
                eng_audio_path,
                alignment_path
            )
        except Exception as e:
            print(f"\n⚠ Error processing {audio_id}: {e}")
            # Return a fallback sample
            eng_chunk = torch.zeros(self.chunk_samples)
            welsh_text = ""
        
        return {
            'audio': eng_chunk,
            'text': welsh_text,
            'audio_id': audio_id,
        }


def collate_fn(batch):
    """
    Custom collate function to batch samples.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary
    """
    # Stack audio tensors
    audios = torch.stack([sample['audio'] for sample in batch])
    
    # Keep texts as list (will be tokenized by model)
    texts = [sample['text'] for sample in batch]
    
    # Keep audio IDs
    audio_ids = [sample['audio_id'] for sample in batch]
    
    return {
        'audio': audios,  # Shape: (batch_size, audio_length)
        'text': texts,    # List of strings
        'audio_id': audio_ids,
    }


def create_dataloaders(config):
    """
    Create train and validation dataloaders.
    
    Args:
        config: Training configuration
    
    Returns:
        train_loader, val_loader
    """
    print("=" * 70)
    print("   CREATING DATALOADERS")
    print("=" * 70)
    
    # Create datasets
    train_dataset = StreamingS2SDataset(config, split=config.train_split)
    val_dataset = StreamingS2SDataset(config, split=config.val_split)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    print(f"\n✅ Dataloaders created!")
    print(f"   • Train batches: {len(train_loader)}")
    print(f"   • Val batches: {len(val_loader)}")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print("=" * 70)
    
    return train_loader, val_loader



# ==================== TEST THE DATALOADER ====================
# This section only runs when script is executed directly
def test_dataloader():
    """Test function - only runs when called explicitly."""
    print("=" * 70)
    print("   TESTING DATALOADER")
    print("=" * 70)
    
    # Load config
    config = get_config()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Test loading one batch
    print("\n🧪 Testing batch loading...")
    batch = next(iter(train_loader))
    
    print(f"\n📦 Batch contents:")
    print(f"   • Audio shape: {batch['audio'].shape}")
    print(f"   • Number of texts: {len(batch['text'])}")
    print(f"   • Audio IDs: {batch['audio_id']}")
    
    print(f"\n📝 Sample texts:")
    for i, text in enumerate(batch['text'][:3]):  # Show first 3
        print(f"   [{i}] {text[:100]}...")  # First 100 chars
    
    print("\n✅ Dataloader test successful!")
    print("=" * 70)


if __name__ == "__main__":
    test_dataloader()