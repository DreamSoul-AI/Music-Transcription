import os
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

class MaestroDataset(Dataset):
    def __init__(self, config, split):
        """
        Initializes the dataset.
        
        Args:
            config (dict): Configuration dictionary.
            split (str): Dataset split to use ('train', 'val', 'test').
        """
        self.config = config
        self.split = split
        
        # Load configuration for the specific split
        self.data_split = self.config[split]
        self.root = os.path.expandvars(self.data_split['root'])
        self.ext_audio = self.data_split['ext_audio']
        
        # List audio files in the directory
        self.audio_files = [f for f in os.listdir(self.root) if f.endswith(self.ext_audio)]
        
        # Optional: Preload data into memory if specified
        self.preloaded_data = []
        if self.data_split['preload']:
            self.preload_data()

    def preload_data(self):
        """Preloads audio data into memory."""
        for file in self.audio_files:
            file_path = os.path.join(self.root, file)
            audio, _ = librosa.load(file_path, sr=self.data_split['sampling_rate'])
            self.preloaded_data.append(audio)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.audio_files)

    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing the audio data and additional information.
        """
        if self.data_split['preload']:
            audio = self.preloaded_data[index]
        else:
            file_path = os.path.join(self.root, self.audio_files[index])
            audio, _ = librosa.load(file_path, sr=self.data_split['sampling_rate'])

        # Additional processing can be done here, e.g., padding, slicing
        return {
            'audio': audio,
            'file_path': self.audio_files[index]
        }

# Configuration for MAESTRO dataset (same as before)
config = {
    'name': 'MAESTRO',
    'train': {
        'root': '${data_root}',
        'groups': 'train',
        'download': False,
        'preload': False,
        'sequence_length': 2048,  # Set a realistic value
        'seed': 42,
        'hop_length': 256,
        'max_midi': 108,
        'min_midi': 21,
        'ext_audio': '.wav',
        'sampling_rate': 16000
    },
    'val': {
        'root': '${data_root}',
        'groups': 'validation',
        'download': False,
        'preload': False,
        'sequence_length': 2048,  # Set a realistic value
        'seed': 42,
        'hop_length': 256,
        'max_midi': 108,
        'min_midi': 21,
        'ext_audio': '.wav',
        'sampling_rate': 16000
    },
    'test': {
        'root': '${data_root}',
        'groups': 'test',
        'download': False,
        'preload': False,
        'sequence_length': 2048,  # Set a realistic value
        'seed': 42,
        'hop_length': 256,
        'max_midi': 108,
        'min_midi': 21,
        'ext_audio': '.wav',
        'sampling_rate': 16000
    }
}

# Example usage
train_dataset = MaestroDataset(config, 'train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Iterate through the DataLoader
for batch in train_loader:
    audio = batch['audio']
    file_paths = batch['file_path']
    print(f"Loaded batch of audio with {len(audio)} samples.")
