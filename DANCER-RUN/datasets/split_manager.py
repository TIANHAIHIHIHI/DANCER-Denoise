# # split_manager.py
# import os
# import json
# import numpy as np
# from typing import Dict, List
# import wfdb


# class SplitManager:
#     def __init__(
#         self,
#         mitdb_dir: str,
#         nstdb_dir: str,
#         window_size: int = 256,
#         seed: int = 42,
#     ):
#         self.mitdb_dir = mitdb_dir
#         self.nstdb_dir = nstdb_dir
#         self.window_size = window_size
#         self.seed = seed
#         np.random.seed(seed)

#     def _get_all_records(self) -> List[str]:
#         return [
#             f.split(".")[0] for f in os.listdir(self.mitdb_dir) if f.endswith(".dat")
#         ]

#     def _load_noise_segments(self) -> Dict[str, np.ndarray]:
#         nstdb_files = ["bw", "em", "ma"]
#         noise_segments = {}

#         for noise_type in nstdb_files:
#             rec = wfdb.rdrecord(os.path.join(self.nstdb_dir, noise_type))
#             noise_signal = rec.p_signal

#             # 随机采样10000个片段
#             segments = []
#             max_start = len(noise_signal) - self.window_size

#             for _ in range(10000):
#                 start_idx = np.random.randint(0, max_start)
#                 seg = noise_signal[start_idx : start_idx + self.window_size, :]
#                 segments.append(seg)

#             noise_segments[noise_type] = np.array(segments, dtype=np.float32)
#             print(f"Loaded {len(segments)} segments for noise type: {noise_type}")

#         return noise_segments

#     def _load_clean_segments(self) -> np.ndarray:
#         records = self._get_all_records()
#         clean_segments = []

#         for record_id in records:
#             rec = wfdb.rdrecord(os.path.join(self.mitdb_dir, record_id))
#             ecg_signal = rec.p_signal

#             max_start = len(ecg_signal) - self.window_size
#             num_segments_per_record = max(1, len(ecg_signal) // (self.window_size * 10))

#             for _ in range(num_segments_per_record):
#                 start_idx = np.random.randint(0, max_start)
#                 seg = ecg_signal[start_idx : start_idx + self.window_size, :]
#                 clean_segments.append(seg)

#         # 随机选择10000个片段
#         if len(clean_segments) > 10000:
#             indices = np.random.choice(len(clean_segments), 10000, replace=False)
#             clean_segments = [clean_segments[i] for i in indices]
#         elif len(clean_segments) < 10000:
#             # 如果样本不足，使用替换采样
#             indices = np.random.choice(len(clean_segments), 10000, replace=True)
#             clean_segments = [clean_segments[i] for i in indices]

#         return np.array(clean_segments, dtype=np.float32)

#     def _calculate_snr_adjustment(
#         self, clean_signal: np.ndarray, noise: np.ndarray, target_snr_db: float
#     ) -> float:
#         clean_power = np.mean(clean_signal**2, axis=(0, 1), keepdims=True)
#         noise_power = np.mean(noise**2, axis=(0, 1), keepdims=True)

#         target_noise_power = clean_power / (10 ** (target_snr_db / 10))
#         scale_factor = np.sqrt(target_noise_power / noise_power)
#         return scale_factor

#     def _create_noisy_signals(
#         self,
#         clean_segments: np.ndarray,
#         noise_segments: Dict[str, np.ndarray],
#         snr_db: float,
#     ) -> Dict[str, np.ndarray]:
#         noisy_signals = {}

#         for noise_type, noise_data in noise_segments.items():
#             noisy_segments = []

#             for i, clean_sig in enumerate(clean_segments):
#                 noise_idx = i % len(noise_data)
#                 noise_segment = noise_data[noise_idx].copy()

#                 scale_factor = self._calculate_snr_adjustment(
#                     clean_sig, noise_segment, snr_db
#                 )
#                 adjusted_noise = noise_segment * scale_factor
#                 noisy_sig = clean_sig + adjusted_noise
#                 noisy_segments.append(noisy_sig)

#             noisy_signals[noise_type] = np.array(noisy_segments, dtype=np.float32)
#             print(
#                 f"Created {len(noisy_segments)} {noise_type} noisy segments at SNR {snr_db}dB"
#             )

#         mixed_noisy_segments = []
#         for i, clean_sig in enumerate(clean_segments):
#             mixed_noise = np.zeros((self.window_size, 2), dtype=np.float32)

#             for noise_type in ["bw", "em", "ma"]:
#                 noise_idx = i % len(noise_segments[noise_type])
#                 noise_segment = noise_segments[noise_type][noise_idx].copy()

#                 mixed_noise += noise_segment

#             scale_factor = self._calculate_snr_adjustment(
#                 clean_sig, mixed_noise, snr_db
#             )
#             mixed_noise *= scale_factor
#             noisy_sig = clean_sig + mixed_noise
#             mixed_noisy_segments.append(noisy_sig)

#         noisy_signals["emb"] = np.array(mixed_noisy_segments, dtype=np.float32)
#         print(
#             f"Created {len(mixed_noisy_segments)} emb (mixed) noisy segments at SNR {snr_db}dB"
#         )

#         return noisy_signals

#     def _zscore_normalize(
#         self, signals: np.ndarray, mean: np.ndarray, std: np.ndarray
#     ) -> np.ndarray:
#         normalized = (signals - mean) / (std)
#         return normalized

#     def save_split(
#         self,
#         split_dir: str,
#         train_ratio: float = 0.8,
#         snr_levels: List[float] = None,
#     ):
#         if snr_levels is None:
#             snr_levels = [-4, -2, 0, 2, 4]

#         print("Loading noise segments...")
#         noise_segments = self._load_noise_segments()  # (10000, window_size, 2)

#         print("Loading clean segments...")
#         clean_segments = self._load_clean_segments()  # (10000, window_size, 2)
#         print(f"Using {len(clean_segments)} clean segments")

#         n_total = len(clean_segments)
#         indices = np.random.permutation(n_total)

#         n_train = int(n_total * train_ratio)
#         train_indices = indices[:n_train]
#         test_indices = indices[n_train:]

#         os.makedirs(split_dir, exist_ok=True)

#         np.save(os.path.join(split_dir, "clean_signals.npy"), clean_segments)

#         for snr_db in snr_levels:
#             noisy_signals_dict = self._create_noisy_signals(
#                 clean_segments, noise_segments, snr_db
#             )
#             for noise_type, noisy_data in noisy_signals_dict.items():
#                 filename = f"noisy_{noise_type}_snr_{snr_db}.npy"
#                 np.save(os.path.join(split_dir, filename), noisy_data)

#         split_info = {
#             "train_indices": train_indices.tolist(),
#             "test_indices": test_indices.tolist(),
#             "total_samples": n_total,
#             "train_ratio": train_ratio,
#             "test_ratio": 1.0 - train_ratio,
#             "window_size": self.window_size,
#             "snr_levels": snr_levels,
#             "noise_types": ["bw", "em", "ma", "emb"],
#             "seed": self.seed,
#         }

#         split_path = os.path.join(split_dir, "split_info.json")
#         with open(split_path, "w") as f:
#             json.dump(split_info, f, indent=2)

#         print(f"\nSaved split info to {split_path}")
#         print(f"Train samples: {len(train_indices)}")
#         print(f"Test samples: {len(test_indices)}")
#         print(f"SNR levels: {snr_levels}")
#         print(f"Noise types: bw, em, ma, emb")

#         return split_info


# if __name__ == "__main__":
#     mitdb_dir = "./mit-bih-arrhythmia-database"
#     nstdb_dir = "./mit-bih-noise-stress-test-database"
#     split_dir = "./data_split"

#     manager = SplitManager(mitdb_dir, nstdb_dir)

#     manager.save_split(
#         split_dir=split_dir, train_ratio=0.8, snr_levels=[-4, -2, 0, 2, 4]
#     )




# # split_manager_eeg.py
# import os
# import json
# import numpy as np
# from typing import Dict, List
# import scipy.io as sio


# class SplitManager:
#     def __init__(
#         self,
#         data_dir: str,
#         window_size: int = None,  # 可选，若为 None 则自动取自数据
#         seed: int = 42,
#     ):
#         self.data_dir = data_dir
#         self.seed = seed
#         np.random.seed(seed)

#         # Load all data upfront to determine window_size if not given
#         eeg_path = os.path.join(data_dir, "EEG_all_epochs.mat")
#         emg_path = os.path.join(data_dir, "EMG_all_epochs.mat")
#         eog_path = os.path.join(data_dir, "EOG_all_epochs.mat")

#         self.eeg_data = sio.loadmat(eeg_path)["EEG_all_epochs"]  # (N, T)
#         self.emg_data = sio.loadmat(emg_path)["EMG_all_epochs"]  # (M, T)
#         self.eog_data = sio.loadmat(eog_path)["EOG_all_epochs"]  # (K, T)

#         assert self.eeg_data.shape[1] == self.emg_data.shape[1] == self.eog_data.shape[1], \
#             "All signals must have same time length!"
        
#         self.window_size = window_size or self.eeg_data.shape[1]
#         assert self.window_size == self.eeg_data.shape[1], "window_size must match signal length"

#     def _load_clean_segments(self) -> np.ndarray:
#         """Load EEG as clean segments; ensure exactly 10,000 samples"""
#         eeg = self.eeg_data  # (N, T)
#         n_total = eeg.shape[0]

#         if n_total >= 10000:
#             indices = np.random.choice(n_total, 10000, replace=False)
#         else:
#             indices = np.random.choice(n_total, 10000, replace=True)
        
#         clean_segments = eeg[indices]  # (10000, T)
#         # Add channel dim: (10000, T, 1) to match original code's (..., 2) style
#         clean_segments = np.expand_dims(clean_segments, axis=-1).astype(np.float32)
#         print(f"Loaded {len(clean_segments)} clean EEG segments")
#         return clean_segments

#     def _load_noise_segments(self) -> Dict[str, np.ndarray]:
#         """Load EMG and EOG as noise sources; each has 10,000 segments"""
#         noise_segments = {}

#         for noise_type, raw_data in [("emg", self.emg_data), ("eog", self.eog_data)]:
#             n_total = raw_data.shape[0]
#             if n_total >= 10000:
#                 indices = np.random.choice(n_total, 10000, replace=False)
#             else:
#                 indices = np.random.choice(n_total, 10000, replace=True)
            
#             segs = raw_data[indices]  # (10000, T)
#             segs = np.expand_dims(segs, axis=-1).astype(np.float32)  # (10000, T, 1)
#             noise_segments[noise_type] = segs
#             print(f"Loaded {len(segs)} segments for noise type: {noise_type}")

#         return noise_segments

#     def _calculate_snr_adjustment(
#         self, clean_signal: np.ndarray, noise: np.ndarray, target_snr_db: float
#     ) -> float:
#         # Power over time and channel (keepdims for broadcasting)
#         clean_power = np.mean(clean_signal ** 2, axis=(1, 2), keepdims=True)  # (B,1,1)
#         noise_power = np.mean(noise ** 2, axis=(1, 2), keepdims=True)        # (B,1,1)

#         target_noise_power = clean_power / (10 ** (target_snr_db / 10))
#         scale_factor = np.sqrt(target_noise_power / noise_power)
#         return scale_factor

#     def _create_noisy_signals(
#         self,
#         clean_segments: np.ndarray,
#         noise_segments: Dict[str, np.ndarray],
#         snr_db: float,
#     ) -> Dict[str, np.ndarray]:
#         noisy_signals = {}

#         # Single noise types: emg, eog
#         for noise_type in ["emg", "eog"]:
#             noise_data = noise_segments[noise_type]  # (10000, T, 1)
#             noisy_segments = []

#             for i in range(len(clean_segments)):
#                 clean_sig = clean_segments[i:i+1]  # (1, T, 1)
#                 noise_idx = i % len(noise_data)
#                 noise_seg = noise_data[noise_idx:noise_idx+1]  # (1, T, 1)

#                 scale = self._calculate_snr_adjustment(clean_sig, noise_seg, snr_db)
#                 adjusted_noise = noise_seg * scale
#                 noisy_sig = clean_sig + adjusted_noise
#                 noisy_segments.append(noisy_sig[0])

#             noisy_signals[noise_type] = np.array(noisy_segments, dtype=np.float32)
#             print(f"Created {len(noisy_segments)} {noise_type} noisy segments at SNR {snr_db}dB")

#         # Mixed noise: mog = emg + eog
#         mixed_noisy_segments = []
#         for i in range(len(clean_segments)):
#             clean_sig = clean_segments[i:i+1]  # (1, T, 1)
#             mixed_noise = np.zeros_like(clean_sig)

#             for nt in ["emg", "eog"]:
#                 noise_idx = i % len(noise_segments[nt])
#                 mixed_noise += noise_segments[nt][noise_idx:noise_idx+1]

#             scale = self._calculate_snr_adjustment(clean_sig, mixed_noise, snr_db)
#             mixed_noise *= scale
#             noisy_sig = clean_sig + mixed_noise
#             mixed_noisy_segments.append(noisy_sig[0])

#         noisy_signals["mog"] = np.array(mixed_noisy_segments, dtype=np.float32)
#         print(f"Created {len(mixed_noisy_segments)} mog (mixed) noisy segments at SNR {snr_db}dB")

#         return noisy_signals

#     def save_split(
#         self,
#         split_dir: str,
#         train_ratio: float = 0.8,
#         snr_levels: List[float] = None,
#     ):
#         if snr_levels is None:
#             snr_levels = [-4, -2, 0, 2, 4]

#         print("Loading clean EEG segments...")
#         clean_segments = self._load_clean_segments()  # (10000, T, 1)

#         print("Loading noise segments (EMG, EOG)...")
#         noise_segments = self._load_noise_segments()  # each: (10000, T, 1)

#         n_total = len(clean_segments)
#         indices = np.random.permutation(n_total)
#         n_train = int(n_total * train_ratio)
#         train_indices = indices[:n_train]
#         test_indices = indices[n_train:]

#         os.makedirs(split_dir, exist_ok=True)

#         # Save clean signals
#         np.save(os.path.join(split_dir, "clean_signals.npy"), clean_segments)

#         # Generate and save noisy signals for each SNR level
#         for snr_db in snr_levels:
#             noisy_dict = self._create_noisy_signals(clean_segments, noise_segments, snr_db)
#             for noise_type, data in noisy_dict.items():
#                 filename = f"noisy_{noise_type}_snr_{snr_db}.npy"
#                 np.save(os.path.join(split_dir, filename), data)

#         # Save split info
#         split_info = {
#             "train_indices": train_indices.tolist(),
#             "test_indices": test_indices.tolist(),
#             "total_samples": n_total,
#             "train_ratio": train_ratio,
#             "test_ratio": 1.0 - train_ratio,
#             "window_size": self.window_size,
#             "snr_levels": snr_levels,
#             "noise_types": ["emg", "eog", "mog"],
#             "seed": self.seed,
#         }

#         split_path = os.path.join(split_dir, "split_info.json")
#         with open(split_path, "w") as f:
#             json.dump(split_info, f, indent=2)

#         print(f"\nSaved split info to {split_path}")
#         print(f"Train samples: {len(train_indices)}")
#         print(f"Test samples: {len(test_indices)}")
#         print(f"SNR levels: {snr_levels}")
#         print(f"Noise types: emg, eog, mog")

#         return split_info


# if __name__ == "__main__":
#     data_dir = "./data"          # contains EEG_all_epochs.mat etc.
#     split_dir = "./eeg_data_split"

#     manager = SplitManager(data_dir=data_dir, seed=42)
#     manager.save_split(
#         split_dir=split_dir,
#         train_ratio=0.8,
#         snr_levels=[-4, -2, 0, 2, 4]
#     )
    


# # split_manager_eeg.py
# import os
# import json
# import numpy as np
# from typing import Dict, List
# import scipy.io as sio


# class SplitManager:
#     def __init__(
#         self,
#         data_dir: str,
#         window_size: int = None,  # 可选，若为 None 则自动取自数据
#         seed: int = 42,
#     ):
#         self.data_dir = data_dir
#         self.seed = seed
#         np.random.seed(seed)

#         # Load all data upfront to determine window_size if not given
#         eeg_path = os.path.join(data_dir, "EEG_all_epochs.mat")
#         emg_path = os.path.join(data_dir, "EMG_all_epochs.mat")
#         eog_path = os.path.join(data_dir, "EOG_all_epochs.mat")

#         self.eeg_data = sio.loadmat(eeg_path)["EEG_all_epochs"]  # (N, T)
#         self.emg_data = sio.loadmat(emg_path)["EMG_all_epochs"]  # (M, T)
#         self.eog_data = sio.loadmat(eog_path)["EOG_all_epochs"]  # (K, T)

#         assert self.eeg_data.shape[1] == self.emg_data.shape[1] == self.eog_data.shape[1], \
#             "All signals must have same time length!"
        
#         self.window_size = window_size or self.eeg_data.shape[1]
#         assert self.window_size == self.eeg_data.shape[1], "window_size must match signal length"

#         # Record original min/max for optional denormalization
#         self.eeg_min, self.eeg_max = float(self.eeg_data.min()), float(self.eeg_data.max())
#         self.emg_min, self.emg_max = float(self.emg_data.min()), float(self.emg_data.max())
#         self.eog_min, self.eog_max = float(self.eog_data.min()), float(self.eog_data.max())

#         print(f"Original signal ranges:")
#         print(f"  EEG: [{self.eeg_min:.3f}, {self.eeg_max:.3f}]")
#         print(f"  EMG: [{self.emg_min:.3f}, {self.emg_max:.3f}]")
#         print(f"  EOG: [{self.eog_min:.3f}, {self.eog_max:.3f}]")

#     def _normalize_to_voltage(self, signal: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
#         """
#         Linearly scale signal to [-15, 15] volts.
#         Preserves waveform shape with no distortion (float32 precision).
#         """
#         range_orig = max_val - min_val
#         if range_orig == 0:
#             return np.zeros_like(signal, dtype=np.float32)
#         scaled = (signal - min_val) / range_orig * 30.0 - 15.0  # 30 = 15 - (-15)
#         return scaled.astype(np.float32)

#     def _load_clean_segments(self) -> np.ndarray:
#         """Load EEG as clean segments; ensure exactly 10,000 samples"""
#         eeg = self.eeg_data  # (N, T)
#         n_total = eeg.shape[0]

#         if n_total >= 10000:
#             indices = np.random.choice(n_total, 10000, replace=False)
#         else:
#             indices = np.random.choice(n_total, 10000, replace=True)
        
#         clean_segments = eeg[indices]  # (10000, T)
#         # Normalize to [-15, 15] V
#         clean_segments = self._normalize_to_voltage(clean_segments, self.eeg_min, self.eeg_max)
#         # Add channel dim: (10000, T, 1)
#         clean_segments = np.expand_dims(clean_segments, axis=-1).astype(np.float32)
#         print(f"Loaded and normalized {len(clean_segments)} clean EEG segments to [-15, 15]V")
#         return clean_segments

#     def _load_noise_segments(self) -> Dict[str, np.ndarray]:
#         """Load EMG and EOG as noise sources; each has 10,000 segments"""
#         noise_segments = {}

#         for noise_type, raw_data in [("emg", self.emg_data), ("eog", self.eog_data)]:
#             n_total = raw_data.shape[0]
#             if n_total >= 10000:
#                 indices = np.random.choice(n_total, 10000, replace=False)
#             else:
#                 indices = np.random.choice(n_total, 10000, replace=True)
            
#             segs = raw_data[indices]  # (10000, T)
#             # Normalize to [-15, 15] V
#             if noise_type == "emg":
#                 segs = self._normalize_to_voltage(segs, self.emg_min, self.emg_max)
#             else:  # eog
#                 segs = self._normalize_to_voltage(segs, self.eog_min, self.eog_max)
            
#             segs = np.expand_dims(segs, axis=-1).astype(np.float32)  # (10000, T, 1)
#             noise_segments[noise_type] = segs
#             print(f"Loaded and normalized {len(segs)} segments for noise type: {noise_type}")

#         return noise_segments

#     def _calculate_snr_adjustment(
#         self, clean_signal: np.ndarray, noise: np.ndarray, target_snr_db: float
#     ) -> float:
#         # Power over time and channel (keepdims for broadcasting)
#         clean_power = np.mean(clean_signal ** 2, axis=(1, 2), keepdims=True)  # (B,1,1)
#         noise_power = np.mean(noise ** 2, axis=(1, 2), keepdims=True)        # (B,1,1)

#         target_noise_power = clean_power / (10 ** (target_snr_db / 10))
#         scale_factor = np.sqrt(target_noise_power / noise_power)
#         return scale_factor

#     def _create_noisy_signals(
#         self,
#         clean_segments: np.ndarray,
#         noise_segments: Dict[str, np.ndarray],
#         snr_db: float,
#     ) -> Dict[str, np.ndarray]:
#         noisy_signals = {}

#         # Single noise types: emg, eog
#         for noise_type in ["emg", "eog"]:
#             noise_data = noise_segments[noise_type]  # (10000, T, 1)
#             noisy_segments = []

#             for i in range(len(clean_segments)):
#                 clean_sig = clean_segments[i:i+1]  # (1, T, 1)
#                 noise_idx = i % len(noise_data)
#                 noise_seg = noise_data[noise_idx:noise_idx+1]  # (1, T, 1)

#                 scale = self._calculate_snr_adjustment(clean_sig, noise_seg, snr_db)
#                 adjusted_noise = noise_seg * scale
#                 noisy_sig = clean_sig + adjusted_noise
#                 noisy_segments.append(noisy_sig[0])

#             noisy_signals[noise_type] = np.array(noisy_segments, dtype=np.float32)
#             print(f"Created {len(noisy_segments)} {noise_type} noisy segments at SNR {snr_db}dB")

#         # Mixed noise: mog = emg + eog
#         mixed_noisy_segments = []
#         for i in range(len(clean_segments)):
#             clean_sig = clean_segments[i:i+1]  # (1, T, 1)
#             mixed_noise = np.zeros_like(clean_sig)

#             for nt in ["emg", "eog"]:
#                 noise_idx = i % len(noise_segments[nt])
#                 mixed_noise += noise_segments[nt][noise_idx:noise_idx+1]

#             scale = self._calculate_snr_adjustment(clean_sig, mixed_noise, snr_db)
#             mixed_noise *= scale
#             noisy_sig = clean_sig + mixed_noise
#             mixed_noisy_segments.append(noisy_sig[0])

#         noisy_signals["mog"] = np.array(mixed_noisy_segments, dtype=np.float32)
#         print(f"Created {len(mixed_noisy_segments)} mog (mixed) noisy segments at SNR {snr_db}dB")

#         return noisy_signals

#     def save_split(
#         self,
#         split_dir: str,
#         train_ratio: float = 0.8,
#         snr_levels: List[float] = None,
#     ):
#         if snr_levels is None:
#             snr_levels = [-4, -2, 0, 2, 4]

#         print("Loading clean EEG segments...")
#         clean_segments = self._load_clean_segments()  # (10000, T, 1)

#         print("Loading noise segments (EMG, EOG)...")
#         noise_segments = self._load_noise_segments()  # each: (10000, T, 1)

#         n_total = len(clean_segments)
#         indices = np.random.permutation(n_total)
#         n_train = int(n_total * train_ratio)
#         train_indices = indices[:n_train]
#         test_indices = indices[n_train:]

#         os.makedirs(split_dir, exist_ok=True)

#         # Save clean signals
#         np.save(os.path.join(split_dir, "clean_signals.npy"), clean_segments)

#         # Generate and save noisy signals for each SNR level
#         for snr_db in snr_levels:
#             noisy_dict = self._create_noisy_signals(clean_segments, noise_segments, snr_db)
#             for noise_type, data in noisy_dict.items():
#                 filename = f"noisy_{noise_type}_snr_{snr_db}.npy"
#                 np.save(os.path.join(split_dir, filename), data)

#         # Save split info including voltage range
#         split_info = {
#             "train_indices": train_indices.tolist(),
#             "test_indices": test_indices.tolist(),
#             "total_samples": n_total,
#             "train_ratio": train_ratio,
#             "test_ratio": 1.0 - train_ratio,
#             "window_size": self.window_size,
#             "snr_levels": snr_levels,
#             "noise_types": ["emg", "eog", "mog"],
#             "seed": self.seed,
#             "voltage_range": [-15.0, 15.0],
#             "original_ranges": {
#                 "EEG": [self.eeg_min, self.eeg_max],
#                 "EMG": [self.emg_min, self.emg_max],
#                 "EOG": [self.eog_min, self.eog_max]
#             }
#         }

#         split_path = os.path.join(split_dir, "split_info.json")
#         with open(split_path, "w") as f:
#             json.dump(split_info, f, indent=2)

#         print(f"\n✅ Saved split info to {split_path}")
#         print(f"Train samples: {len(train_indices)}")
#         print(f"Test samples: {len(test_indices)}")
#         print(f"SNR levels: {snr_levels}")
#         print(f"Noise types: emg, eog, mog")
#         print(f"All signals normalized to [-15, 15] volts")


# if __name__ == "__main__":
#     data_dir = "./data"          # contains EEG_all_epochs.mat etc.
#     split_dir = "./eeg_data_split"

#     manager = SplitManager(data_dir=data_dir, seed=42)
#     manager.save_split(
#         split_dir=split_dir,
#         train_ratio=0.8,
#         snr_levels=[-4, -2, 0, 2, 4]
#     )
    

# split_manager_eeg.py
import os
import json
import numpy as np
from typing import Dict, List
import scipy.io as sio


class SplitManager:
    def __init__(
        self,
        data_dir: str,
        window_size: int = None,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.seed = seed
        np.random.seed(seed)

        eeg_path = os.path.join(data_dir, "EEG_all_epochs.mat")
        emg_path = os.path.join(data_dir, "EMG_all_epochs.mat")
        eog_path = os.path.join(data_dir, "EOG_all_epochs.mat")

        self.eeg_data = sio.loadmat(eeg_path)["EEG_all_epochs"]
        self.emg_data = sio.loadmat(emg_path)["EMG_all_epochs"]
        self.eog_data = sio.loadmat(eog_path)["EOG_all_epochs"]

        assert self.eeg_data.shape[1] == self.emg_data.shape[1] == self.eog_data.shape[1], \
            "All signals must have same time length!"
        
        self.window_size = window_size or self.eeg_data.shape[1]
        assert self.window_size == self.eeg_data.shape[1], "window_size must match signal length"

        self.eeg_min, self.eeg_max = float(self.eeg_data.min()), float(self.eeg_data.max())
        self.emg_min, self.emg_max = float(self.emg_data.min()), float(self.emg_data.max())
        self.eog_min, self.eog_max = float(self.eog_data.min()), float(self.eog_data.max())

        print(f"Original signal ranges:")
        print(f"  EEG: [{self.eeg_min:.3f}, {self.eeg_max:.3f}]")
        print(f"  EMG: [{self.emg_min:.3f}, {self.emg_max:.3f}]")
        print(f"  EOG: [{self.eog_min:.3f}, {self.eog_max:.3f}]")

    def _normalize_to_voltage(self, signal: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        range_orig = max_val - min_val
        if range_orig == 0:
            return np.zeros_like(signal, dtype=np.float32)
        scaled = (signal - min_val) / range_orig * 30.0 - 15.0
        return scaled.astype(np.float32)

    def _load_clean_segments(self) -> np.ndarray:
        eeg = self.eeg_data
        n_total = eeg.shape[0]

        if n_total >= 10000:
            indices = np.random.choice(n_total, 10000, replace=False)
        else:
            indices = np.random.choice(n_total, 10000, replace=True)
        
        clean_segments = eeg[indices]
        clean_segments = self._normalize_to_voltage(clean_segments, self.eeg_min, self.eeg_max)
        clean_segments = np.expand_dims(clean_segments, axis=-1).astype(np.float32)
        print(f"Loaded and normalized {len(clean_segments)} clean EEG segments to [-15, 15]V")
        return clean_segments

    def _load_noise_segments(self) -> Dict[str, np.ndarray]:
        noise_segments = {}

        for noise_type, raw_data in [("emg", self.emg_data), ("eog", self.eog_data)]:
            n_total = raw_data.shape[0]
            if n_total >= 10000:
                indices = np.random.choice(n_total, 10000, replace=False)
            else:
                indices = np.random.choice(n_total, 10000, replace=True)
            
            segs = raw_data[indices]
            if noise_type == "emg":
                segs = self._normalize_to_voltage(segs, self.emg_min, self.emg_max)
            else:
                segs = self._normalize_to_voltage(segs, self.eog_min, self.eog_max)
            
            segs = np.expand_dims(segs, axis=-1).astype(np.float32)
            noise_segments[noise_type] = segs
            print(f"Loaded and normalized {len(segs)} segments for noise type: {noise_type}")

        return noise_segments

    def _calculate_snr_adjustment(
        self, clean_signal: np.ndarray, noise: np.ndarray, target_snr_db: float
    ) -> float:
        clean_power = np.mean(clean_signal ** 2, axis=(1, 2), keepdims=True)
        noise_power = np.mean(noise ** 2, axis=(1, 2), keepdims=True)
        target_noise_power = clean_power / (10 ** (target_snr_db / 10))
        scale_factor = np.sqrt(target_noise_power / noise_power)
        return scale_factor

    def _create_noisy_signals(
        self,
        clean_segments: np.ndarray,
        noise_segments: Dict[str, np.ndarray],
        snr_db: float,
    ) -> Dict[str, np.ndarray]:
        noisy_signals = {}

        for noise_type in ["emg", "eog"]:
            noise_data = noise_segments[noise_type]
            noisy_segments = []

            for i in range(len(clean_segments)):
                clean_sig = clean_segments[i:i+1]
                noise_idx = i % len(noise_data)
                noise_seg = noise_data[noise_idx:noise_idx+1]

                scale = self._calculate_snr_adjustment(clean_sig, noise_seg, snr_db)
                adjusted_noise = noise_seg * scale
                noisy_sig = clean_sig + adjusted_noise
                noisy_segments.append(noisy_sig[0])

            noisy_signals[noise_type] = np.array(noisy_segments, dtype=np.float32)
            print(f"Created {len(noisy_segments)} {noise_type} noisy segments at SNR {snr_db}dB")

        mixed_noisy_segments = []
        for i in range(len(clean_segments)):
            clean_sig = clean_segments[i:i+1]
            mixed_noise = np.zeros_like(clean_sig)

            for nt in ["emg", "eog"]:
                noise_idx = i % len(noise_segments[nt])
                mixed_noise += noise_segments[nt][noise_idx:noise_idx+1]

            scale = self._calculate_snr_adjustment(clean_sig, mixed_noise, snr_db)
            mixed_noise *= scale
            noisy_sig = clean_sig + mixed_noise
            mixed_noisy_segments.append(noisy_sig[0])

        noisy_signals["mog"] = np.array(mixed_noisy_segments, dtype=np.float32)
        print(f"Created {len(mixed_noisy_segments)} mog (mixed) noisy segments at SNR {snr_db}dB")

        return noisy_signals

    # ✅ 新增方法：生成纯 MOG 噪声（无 clean）
    def _create_mog_noise_only(self, noise_segments: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate pure MOG noise = EMG + EOG (both normalized), no clean signal.
        Output shape: (10000, T, 1)
        """
        emg_segs = noise_segments["emg"]  # (10000, T, 1)
        eog_segs = noise_segments["eog"]  # (10000, T, 1)

        # Ensure same length (they should be, both 10000)
        n = min(len(emg_segs), len(eog_segs))
        mog_noise = emg_segs[:n] + eog_segs[:n]  # element-wise sum

        print(f"Generated {n} pure MOG noise segments (EMG + EOG only, no clean)")
        return mog_noise.astype(np.float32)
    
        # ✅ 新增方法：生成纯 EOG 噪声（无 clean）
    def _create_eog_noise_only(self, noise_segments: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate pure EOG noise (normalized), no clean signal, no EMG.
        Output shape: (10000, T, 1)
        """
        eog_segs = noise_segments["eog"]  # (10000, T, 1)
        n = len(eog_segs)
        print(f"Generated {n} pure EOG noise segments (EOG only, no clean)")
        return eog_segs.astype(np.float32)

    def save_split(
        self,
        split_dir: str,
        train_ratio: float = 0.8,
        snr_levels: List[float] = None,
    ):
        if snr_levels is None:
            snr_levels = [-4, -2, 0, 2, 4]

        print("Loading clean EEG segments...")
        clean_segments = self._load_clean_segments()

        print("Loading noise segments (EMG, EOG)...")
        noise_segments = self._load_noise_segments()

        n_total = len(clean_segments)
        indices = np.random.permutation(n_total)
        n_train = int(n_total * train_ratio)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        os.makedirs(split_dir, exist_ok=True)

        # Save clean signals
        np.save(os.path.join(split_dir, "clean_signals.npy"), clean_segments)

        # Generate and save noisy signals for each SNR level
        for snr_db in snr_levels:
            noisy_dict = self._create_noisy_signals(clean_segments, noise_segments, snr_db)
            for noise_type, data in noisy_dict.items():
                filename = f"noisy_{noise_type}_snr_{snr_db}.npy"
                np.save(os.path.join(split_dir, filename), data)

        # ✅ Save pure MOG noise (no clean, no SNR scaling)
        mog_noise_input = self._create_mog_noise_only(noise_segments)
        np.save(os.path.join(split_dir, "mog_noise_input.npy"), mog_noise_input)

        # ✅ Save pure EOG noise (no clean)
        eog_noise_input = self._create_eog_noise_only(noise_segments)
        np.save(os.path.join(split_dir, "eog_noise_input.npy"), eog_noise_input)

        # Save split info
        split_info = {
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "total_samples": n_total,
            "train_ratio": train_ratio,
            "test_ratio": 1.0 - train_ratio,
            "window_size": self.window_size,
            "snr_levels": snr_levels,
            "noise_types": ["emg", "eog", "mog"],
            "seed": self.seed,
            "voltage_range": [-15.0, 15.0],
            "original_ranges": {
                "EEG": [self.eeg_min, self.eeg_max],
                "EMG": [self.emg_min, self.emg_max],
                "EOG": [self.eog_min, self.eog_max]
            }
        }

        split_path = os.path.join(split_dir, "split_info.json")
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)

        print(f"\n✅ Saved split info to {split_path}")
        print(f"Train samples: {len(train_indices)}")
        print(f"Test samples: {len(test_indices)}")
        print(f"SNR levels: {snr_levels}")
        print(f"Noise types: emg, eog, mog")
        print(f"Also saved pure MOG noise as 'mog_noise_input.npy'")
        print(f"All signals normalized to [-15, 15] volts")


if __name__ == "__main__":
    data_dir = "./data"
    split_dir = "./eeg_data_split"

    manager = SplitManager(data_dir=data_dir, seed=42)
    manager.save_split(
        split_dir=split_dir,
        train_ratio=0.8,
        snr_levels=[-4, -2, 0, 2, 4]
    )




