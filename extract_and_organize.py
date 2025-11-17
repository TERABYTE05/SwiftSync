import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "data"

# English data paths (CVSS dataset)
ENG_DIR = os.path.join(DATA_DIR, "english")
ENG_TRAIN_AUDIO = os.path.join(ENG_DIR, "train")
ENG_TEST_AUDIO = os.path.join(ENG_DIR, "test")
ENG_DEV_AUDIO = os.path.join(ENG_DIR, "dev")
ENG_TRAIN_TSV = os.path.join(ENG_DIR, "train.tsv")
ENG_TEST_TSV = os.path.join(ENG_DIR, "test.tsv")
ENG_DEV_TSV = os.path.join(ENG_DIR, "dev.tsv")

# Welsh data paths (Common Voice dataset)
WELSH_DIR = os.path.join(DATA_DIR, "welsh", "cv-corpus-22.0-2025-06-20", "cy")
WELSH_CLIPS_DIR = os.path.join(WELSH_DIR, "clips")
WELSH_TRAIN_TSV = os.path.join(WELSH_DIR, "train.tsv")
WELSH_TEST_TSV = os.path.join(WELSH_DIR, "test.tsv")
WELSH_DEV_TSV = os.path.join(WELSH_DIR, "dev.tsv")

# Output organized data paths
OUTPUT_DIR = "organized_data"
OUTPUT_ENG_AUDIO = os.path.join(OUTPUT_DIR, "english_audio")
OUTPUT_WELSH_AUDIO = os.path.join(OUTPUT_DIR, "welsh_audio")
OUTPUT_WELSH_TRANSCRIPTS = os.path.join(OUTPUT_DIR, "welsh_transcripts")
OUTPUT_METADATA = os.path.join(OUTPUT_DIR, "parallel_metadata.csv")

# Create output directories
os.makedirs(OUTPUT_ENG_AUDIO, exist_ok=True)
os.makedirs(OUTPUT_WELSH_AUDIO, exist_ok=True)
os.makedirs(OUTPUT_WELSH_TRANSCRIPTS, exist_ok=True)

# Read English transcripts
def read_english_tsvs():
    """Read all English TSV files and combine them"""
    print("\nReading English TSV files...")
    
    eng_data = []
    
    splits = {
        'train': (ENG_TRAIN_TSV, ENG_TRAIN_AUDIO),
        'test': (ENG_TEST_TSV, ENG_TEST_AUDIO),
        'dev': (ENG_DEV_TSV, ENG_DEV_AUDIO)
    }
    
    for split_name, (tsv_path, audio_dir) in splits.items():
        if not os.path.exists(tsv_path):
            print(f"Problem: {tsv_path} not found, skipping...")
            continue
        
        try:
            df = pd.read_csv(tsv_path, sep='\t', header=None, names=['audio', 'transcript'])
            df['split'] = split_name
            df['audio_dir'] = audio_dir
            eng_data.append(df)
            print(f"Loaded {len(df)} entries from {split_name}.tsv")
        except Exception as e:
            print(f"Error reading {tsv_path}: {e}")
    
    if not eng_data:
        raise ValueError("No English TSV files could be read...")
    
    # Combine all splits
    eng_df = pd.concat(eng_data, ignore_index=True)
    print(f"Total English transcripts: {len(eng_df)}")
    
    return eng_df

# Read Welsh transcripts
def read_welsh_tsvs():
    """Read all Welsh TSV files and combine them."""
    print("\nReading Welsh TSV files...")
    
    welsh_data = []
    
    splits = {
        'train': WELSH_TRAIN_TSV,
        'test': WELSH_TEST_TSV,
        'dev': WELSH_DEV_TSV
    }
    
    for split_name, tsv_path in splits.items():
        if not os.path.exists(tsv_path):
            print(f"Problem: {tsv_path} not found, skipping...")
            continue
        
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            df['split'] = split_name
            welsh_data.append(df)
            print(f"Loaded {len(df)} entries from {split_name}.tsv")
        except Exception as e:
            print(f"Error reading {tsv_path}: {e}")
    
    if not welsh_data:
        raise ValueError("No Welsh TSV files could be read...")
    
    # Combine all splits
    welsh_df = pd.concat(welsh_data, ignore_index=True)
    print(f"Total Welsh transcripts: {len(welsh_df)}")
    
    return welsh_df


def normalize_audio_name(filename):
    """Normalize audio filename by removing extensions"""
    name = Path(filename).stem  # Remove last extension
    # Remove additional extensions if present
    while '.' in name:
        name = Path(name).stem
    return name


# Create parallel dataset
def create_parallel_dataset(eng_df, welsh_df):
    """Match English and Welsh audio files by normalized filename."""
    print("\nMatching English-Welsh pairs...")
    
    # Create lookup dictionaries
    eng_lookup = {}
    for _, row in eng_df.iterrows():
        audio_filename = row['audio']
        normalized_name = normalize_audio_name(audio_filename)
        
        # Construct actual file path with .wav extension
        actual_filename = f"{audio_filename}.wav"
        
        eng_lookup[normalized_name] = {
            'audio_path': os.path.join(row['audio_dir'], actual_filename),
            'transcript': row['transcript'],
            'split': row['split']
        }
    
    welsh_lookup = {}
    for _, row in welsh_df.iterrows():
        normalized_name = normalize_audio_name(row['path'])
        welsh_lookup[normalized_name] = {
            'audio_path': os.path.join(WELSH_CLIPS_DIR, row['path']),
            'transcript': row['sentence'],
            'split': row['split']
        }
    
    # Find matching pairs
    matched_pairs = []
    
    print(f"Scanning {len(eng_lookup)} English files...")
    print("\n   Checking for matches...")
    
    for audio_name in tqdm(eng_lookup.keys(), desc="Matching"):
        if audio_name in welsh_lookup:
            eng_info = eng_lookup[audio_name]
            welsh_info = welsh_lookup[audio_name]
            
            # Check if files actually exist
            eng_exists = os.path.exists(eng_info['audio_path'])
            welsh_exists = os.path.exists(welsh_info['audio_path'])
            
            if not eng_exists:
                print(f"\nEnglish file missing: {eng_info['audio_path']}")
            if not welsh_exists:
                print(f"\nWelsh file missing: {welsh_info['audio_path']}")
            
            if eng_exists and welsh_exists:
                matched_pairs.append({
                    'audio_id': audio_name,
                    'eng_audio': eng_info['audio_path'],
                    'eng_transcript': eng_info['transcript'],
                    'welsh_audio': welsh_info['audio_path'],
                    'welsh_transcript': welsh_info['transcript'],
                    'split': eng_info['split']  # Use English split
                })
    
    print(f"\nFound {len(matched_pairs)} matching pairs")
    print(f"English-only: {len(eng_lookup) - len(matched_pairs)}")
    print(f"Welsh-only: {len(welsh_lookup) - len(matched_pairs)}")
    
    if len(matched_pairs) == 0:
        print("\nError : No matching pairs found!")
        print("Showing sample filenames for debugging:")
        print(f"\nSample English files:")
        for name in list(eng_lookup.keys())[:5]:
            print(f"{name}")
        print(f"\nSample Welsh files:")
        for name in list(welsh_lookup.keys())[:5]:
            print(f"{name}")
    
    return pd.DataFrame(matched_pairs)


# Copy files to a new folder
def organize_files(parallel_df):
    """Copy matched files to organized structure"""
    
    if len(parallel_df) == 0:
        print("No files to organize (no matched pairs found)")
        return
    
    for idx, row in tqdm(parallel_df.iterrows(), total=len(parallel_df), desc="Copying files"):
        audio_id = row['audio_id']
        
        try:
            # Copy English audio (keep as .wav)
            eng_dest = os.path.join(OUTPUT_ENG_AUDIO, f"{audio_id}.wav")
            if not os.path.exists(eng_dest):
                shutil.copy2(row['eng_audio'], eng_dest)
            
            # Copy Welsh audio (keep as .mp3)
            welsh_dest = os.path.join(OUTPUT_WELSH_AUDIO, f"{audio_id}.mp3")
            if not os.path.exists(welsh_dest):
                shutil.copy2(row['welsh_audio'], welsh_dest)
            
            # Save Welsh transcript as .txt file
            transcript_dest = os.path.join(OUTPUT_WELSH_TRANSCRIPTS, f"{audio_id}.txt")
            with open(transcript_dest, 'w', encoding='utf-8') as f:
                f.write(row['welsh_transcript'])
        
        except Exception as e:
            print(f"\nError processing {audio_id}: {e}")
            continue
    
    print(f"Organized {len(parallel_df)} file pairs")


# main function
def main():
    # Read English TSVs
    eng_df = read_english_tsvs()
    
    # Read Welsh TSVs
    welsh_df = read_welsh_tsvs()
    
    # Match pairs
    parallel_df = create_parallel_dataset(eng_df, welsh_df)
    
    if len(parallel_df) == 0:
        print("No matching pairs found")
        return
    
    # Save metadata
    parallel_df.to_csv(OUTPUT_METADATA, index=False)
    print(f"\nSaved metadata to: {OUTPUT_METADATA}")
    
    # Organize files
    organize_files(parallel_df)

    print(f"Total matched pairs: {len(parallel_df)}")
    print(f"Train: {len(parallel_df[parallel_df['split'] == 'train'])}")
    print(f"Test: {len(parallel_df[parallel_df['split'] == 'test'])}")
    print(f"Dev: {len(parallel_df[parallel_df['split'] == 'dev'])}")
    print(f"\nOutput location: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()