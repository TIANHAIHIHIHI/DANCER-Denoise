# import torch
# from torch.utils.data import Dataset
# import os
# import numpy as np
# import json

# import os
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils import compute_metrics


# class ECGDataset(Dataset):
#     def __init__(
#         self,
#         split: str = "train",
#         noise_type: str = "bw",
#         snr_db: int = 0,
#         split_dir: str = "./data_split",
#     ):
#         super().__init__()
#         self.split = split
#         self.split_dir = split_dir
#         self.noise_type = noise_type
#         self.snr_db = snr_db

#         if snr_db not in [-4, -2, 0, 2, 4]:
#             raise ValueError(f"Unsupported SNR level: {snr_db}")

#         if noise_type not in ["bw", "em", "ma", "emb"]:
#             raise ValueError(f"Unsupported noise type: {noise_type}")

#         split_path = os.path.join(split_dir, "split_info.json")
#         if not os.path.exists(split_path):
#             raise FileNotFoundError(f"Split file not found: {split_path}")

#         with open(split_path, "r") as f:
#             self.split_data = json.load(f)

#         if split == "train":
#             self.indices = self.split_data["train_indices"]
#         elif split == "test":
#             self.indices = self.split_data["test_indices"]
#         else:
#             raise ValueError(f"Unknown split: {split}")

#         self.noisy_signals = np.load(
#             os.path.join(split_dir, f"noisy_{noise_type}_snr_{snr_db}.npy")
#         )
#         self.clean_signals = np.load(os.path.join(split_dir, "clean_signals.npy"))

#         train_noisy = self.noisy_signals[self.split_data["train_indices"]]

#         self.__mean = np.mean(train_noisy, axis=(0, 1), keepdims=True)
#         self.__std = np.std(train_noisy, axis=(0, 1), keepdims=True)

#         if split == "train":
#             self.noisy_signals = (self.noisy_signals - self.__mean) / self.__std
#             self.clean_signals = (self.clean_signals - self.__mean) / self.__std
#         else:
#             self.noisy_signals = (self.noisy_signals - self.__mean) / self.__std

#         self.noisy_signals = self.noisy_signals.transpose(
#             0, 2, 1
#         )  # (num_samples, 2, window_size)
#         self.clean_signals = self.clean_signals.transpose(
#             0, 2, 1
#         )  # (num_samples, 2, window_size)

#         # print(f"Loaded {split} dataset with {len(self.indices)} samples")

#     def get_stats(self):
#         return torch.FloatTensor(self.__mean), torch.FloatTensor(self.__std)

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         data_idx = self.indices[idx]

#         noisy_signal = self.noisy_signals[data_idx]
#         clean_signal = self.clean_signals[data_idx]

#         noisy_tensor = torch.FloatTensor(noisy_signal)
#         clean_tensor = torch.FloatTensor(clean_signal)

#         return noisy_tensor, clean_tensor


# if __name__ == "__main__":

#     train_dataset = ECGDataset(split="train", split_dir="./data_split")
#     test_dataset = ECGDataset(split="test", split_dir="./data_split")

#     clean = train_dataset[0][1]
#     print(f"sample shape: {clean.shape}")

#     print(f"trainset shape: {len(train_dataset)}")
#     print(f"testset shape: {len(test_dataset)}")

#     noisy, clean = train_dataset[0]
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(12, 6))
#     plt.subplot(2, 1, 1)
#     plt.plot(noisy[0].numpy(), label="Noisy ECG")
#     plt.plot(clean[0].numpy(), label="Clean ECG")
#     plt.legend()
#     plt.title("ECG Signal Sample from Training Set, channel 0")

#     plt.subplot(2, 1, 2)
#     plt.plot(noisy[1].numpy(), label="Noisy ECG")
#     plt.plot(clean[1].numpy(), label="Clean ECG")
#     plt.legend()
#     plt.title("ECG Signal Sample from Training Set, channel 1")

#     plt.tight_layout()
#     plt.show()


# eeg_dataset.py
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json


class EEGDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        noise_type: str = "emg",
        snr_db: int = 0,
        split_dir: str = "./eeg_data_split",
    ):
        super().__init__()
        self.split = split
        self.split_dir = split_dir
        self.noise_type = noise_type
        self.snr_db = snr_db
        

        # Validate SNR and noise type
        if snr_db not in [-4, -2, 0, 2, 4]:
            raise ValueError(f"Unsupported SNR level: {snr_db}")

        if noise_type not in ["emg", "eog", "mog"]:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        # Load split info
        split_path = os.path.join(split_dir, "split_info.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            self.split_data = json.load(f)

        # Set indices based on split
        if split == "train":
            self.indices = self.split_data["train_indices"]
        elif split == "test":
            self.indices = self.split_data["test_indices"]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load signals
        self.noisy_signals = np.load(
            os.path.join(split_dir, f"noisy_{noise_type}_snr_{snr_db}.npy")
        )  # (10000, T, 1)
        self.clean_signals = np.load(os.path.join(split_dir, "clean_signals.npy"))  # (10000, T, 1)

        # Compute normalization stats from TRAINING noisy data only
        train_noisy = self.noisy_signals[self.split_data["train_indices"]]  # (N_train, T, 1)
        self.__mean = np.mean(train_noisy, axis=(0, 1), keepdims=True)      # (1, 1, 1)
        self.__std = np.std(train_noisy, axis=(0, 1), keepdims=True)        # (1, 1, 1)

        # Apply normalization to entire dataset (consistent with training stats)
        self.noisy_signals = (self.noisy_signals - self.__mean) / (self.__std + 1e-8)
        if split == "train":
            self.clean_signals = (self.clean_signals - self.__mean) / (self.__std + 1e-8)

        # Transpose to PyTorch convention: (batch, channel, time)
        self.noisy_signals = self.noisy_signals.transpose(0, 2, 1)   # (N, 1, T)
        self.clean_signals = self.clean_signals.transpose(0, 2, 1)   # (N, 1, T)

        print(f"Loaded {split} EEG dataset with {len(self.indices)} samples "
              f"(noise={noise_type}, SNR={snr_db}dB)")

    def get_stats(self):
        """Return mean and std used for normalization (as tensors)."""
        return torch.FloatTensor(self.__mean), torch.FloatTensor(self.__std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        noisy_signal = self.noisy_signals[data_idx]      # (1, T)
        clean_signal = self.clean_signals[data_idx]      # (1, T)

        noisy_tensor = torch.FloatTensor(noisy_signal)
        clean_tensor = torch.FloatTensor(clean_signal)

        return noisy_tensor, clean_tensor


if __name__ == "__main__":
    # Example usage
    train_dataset = EEGDataset(split="train", noise_type="emg", snr_db=0, split_dir="./eeg_data_split")
    test_dataset = EEGDataset(split="test", noise_type="emg", snr_db=0, split_dir="./eeg_data_split")

    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    noisy, clean = train_dataset[0]
    print(f"Sample shape: {noisy.shape}")  # e.g., torch.Size([1, 512])
    

    # Optional: plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        time_axis = np.arange(noisy.shape[1])
        plt.plot(time_axis, noisy[0].numpy(), label="Noisy EEG", alpha=0.8)
        plt.plot(time_axis, clean[0].numpy(), label="Clean EEG", alpha=0.8)
        plt.legend()
        plt.title(f"EEG Sample (Noise: emg, SNR: 0dB) - Channel 0")
        plt.xlabel("Time (samples)")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not available; skipping plot.")
        
