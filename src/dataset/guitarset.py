import os
import pandas as pd
import librosa

# Directory of the WAV files
audio_dir = 'audio_hex-pickup_debleeded'

# List all files in the directory
files = os.listdir(audio_dir)

# Metadata list
metadata = []

# Iterate over each file and collect metadata
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_dir, filename)
        
        # Load the audio file to get duration
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Append metadata (assuming filename might contain label info)
        metadata.append({
            "file_name": filename,
            "file_path": file_path,
            "duration": duration,
            "sample_rate": sr
            # Add additional columns as needed, like "speaker_id" if applicable
        })

# Convert to DataFrame and save
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv("metadata.csv", index=False)

import torch
from torch.utils.data import Dataset
import librosa
import pandas as pd

class AudioDataset(Dataset):
    def __init__(self, metadata_file, transform=None, sample_rate=16000):
        """
        Args:
            metadata_file (string): Path to the metadata CSV file.
            transform (callable, optional): Optional transform to apply on a sample.
            sample_rate (int): Desired sample rate for loading audio.
        """
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_length = target_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Retrieve metadata for audio file
        file_path = self.metadata.iloc[idx]["file_path"]
        label = self.metadata.iloc[idx].get("label", None)  # Modify if labels are included

        # Load the audio file
        audio, sr = librosa.load(file_path, sr=self.sample_rate)

        # Convert to a PyTorch tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Apply any transforms if provided
        if self.transform:
            for t in self.transform:
                audio_tensor = t(audio_tensor)

        sample = {"audio": audio_tensor, "label": label} if label else {"audio": audio_tensor}
        return sample

from torch.utils.data import DataLoader

import torch

class NormalizeAudio:
    def __call__(self, audio):
        return (audio - torch.mean(audio)) / torch.std(audio)

class FixedLengthAudio:
    def __init__(self, target_length):
        self.target_length = target_length

    def __call__(self, audio_tensor):
        length = len(audio_tensor)
        if length > self.target_length:
            return audio_tensor[:self.target_length]
        else:
            padding = self.target_length - length
            return torch.cat([audio_tensor, torch.zeros(padding)], dim=0)

# Specify the target length (e.g., 16000 samples, equivalent to 1 second of audio at 16kHz)
target_length = 16000

# Initialize the dataset with both transforms
dataset = AudioDataset(
    metadata_file='metadata.csv',
    transform=[NormalizeAudio(), FixedLengthAudio(target_length=target_length)],
    sample_rate=16000
)

# Create DataLoader to manage batching and shuffling
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Iterate through the dataset
for batch in dataloader:
    audio = batch['audio']
    print(audio.shape)
