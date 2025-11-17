import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class StreamingDatasetWithContext(Dataset):
    def __init__(self, split="train"):
        from config import config
        self.config = config
        
        df = pd.read_csv(config.metadata_file)
        
        if split == "all":
            self.data = df
        else:
            self.data = df[df['split'] == split]
        
        if split == config.train_split and config.max_train_samples:
            self.data = self.data[:config.max_train_samples]
        elif split == config.val_split and config.max_val_samples:
            self.data = self.data[:config.max_val_samples]
        
        self.data = self.data.reset_index(drop=True)
        
        # Generate chunks
        self.chunk_indices = []
        for idx, row in self.data.iterrows():
            try:
                waveform, sr = torchaudio.load(row['eng_audio'])
                audio_len = waveform.shape[-1]
                if sr != config.sample_rate:
                    audio_len = int(audio_len * config.sample_rate / sr)
                
                num_chunks = max(1, (audio_len - config.chunk_samples) // config.hop_samples + 1)
                
                for chunk_idx in range(num_chunks):
                    self.chunk_indices.append({
                        'data_idx': idx,
                        'chunk_idx': chunk_idx,
                        'total_chunks': num_chunks,
                    })
            except Exception as e:
                logger.warning(f"Skip {row['eng_audio']}: {e}")
        
        logger.info(f"  {len(self.data)} files → {len(self.chunk_indices)} chunks ({split})")
    
    def __len__(self):
        return len(self.chunk_indices)
    
    def load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)
            return waveform.squeeze()
        except:
            return torch.zeros(self.config.chunk_samples)
    
    def __getitem__(self, idx):
        chunk_info = self.chunk_indices[idx]
        row = self.data.iloc[chunk_info['data_idx']]
        
        audio = self.load_audio(row['eng_audio'])
        
        start_sample = chunk_info['chunk_idx'] * self.config.hop_samples
        end_sample = start_sample + self.config.chunk_samples
        
        if end_sample > len(audio):
            chunk = torch.nn.functional.pad(audio[start_sample:], (0, end_sample - len(audio)))
        else:
            chunk = audio[start_sample:end_sample]
        
        # Text chunking
        words = row['welsh_transcript'].split()
        total_chunks = chunk_info['total_chunks']
        chunk_idx = chunk_info['chunk_idx']
        
        if total_chunks == 1:
            chunk_text = row['welsh_transcript']
        else:
            words_per_chunk = max(1, len(words) // total_chunks)
            start_word = chunk_idx * words_per_chunk
            end_word = min(start_word + words_per_chunk, len(words))
            chunk_text = ' '.join(words[start_word:end_word])
        
        # Get English reference for semantic comparison
        eng_words = row.get('english_transcript', '').split()
        if eng_words and total_chunks > 1:
            eng_words_per_chunk = max(1, len(eng_words) // total_chunks)
            eng_start = chunk_idx * eng_words_per_chunk
            eng_end = min(eng_start + eng_words_per_chunk, len(eng_words))
            eng_reference = ' '.join(eng_words[eng_start:eng_end])
        else:
            eng_reference = row.get('english_transcript', '')
        
        return {
            'audio': chunk,
            'text': chunk_text if chunk_text else row['welsh_transcript'],
            'audio_id': f"{row['audio_id']}_c{chunk_idx}",
            'english_reference': eng_reference,
            'chunk_idx': chunk_idx,
            'total_chunks': total_chunks,
        }

def collate_fn(batch):
    return {
        'audio': torch.stack([s['audio'] for s in batch]),
        'text': [s['text'] for s in batch],
        'audio_id': [s['audio_id'] for s in batch],
        'english_reference': [s['english_reference'] for s in batch],
        'chunk_idx': [s['chunk_idx'] for s in batch],
        'total_chunks': [s['total_chunks'] for s in batch],
    }