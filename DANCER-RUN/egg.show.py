# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # 设置路径
# data_dir = "./eeg_data_split"  # 替换为你的实际路径
# files = [
#     "clean_signals.npy",
#     "noisy_emg_snr_-2.npy",
#     "noisy_emg_snr_0.npy",
#     "noisy_emg_snr_2.npy",
#     "noisy_emg_snr_4.npy",
#     "noisy_eog_snr_-2.npy",
#     "noisy_eog_snr_0.npy",
#     "noisy_eog_snr_2.npy",
#     "noisy_eog_snr_4.npy",
#     "noisy_mog_snr_-2.npy",
#     "noisy_mog_snr_0.npy",
#     "noisy_mog_snr_2.npy",
#     "noisy_mog_snr_4.npy"
# ]

# # 加载所有信号的第一个样本（index=0）
# signals = []
# labels = []

# for file in files:
#     filepath = os.path.join(data_dir, file)
#     if os.path.exists(filepath):
#         data = np.load(filepath, allow_pickle=False)  # (N, T, 1)
#         signal = data[0].squeeze()  # (T,)
#         signals.append(signal)
#         # 提取标签：如 "EMG -2dB", "EOG 0dB", ...
#         if "clean" in file:
#             label = "Clean"
#         elif "emg" in file:
#             snr = file.split("_")[-1].replace(".npy", "")
#             label = f"EMG {snr}dB"
#         elif "eog" in file:
#             snr = file.split("_")[-1].replace(".npy", "")
#             label = f"EOG {snr}dB"
#         elif "mog" in file:
#             snr = file.split("_")[-1].replace(".npy", "")
#             label = f"MOG {snr}dB"
#         else:
#             label = file.replace(".npy", "")
#         labels.append(label)

# # 绘图
# plt.figure(figsize=(14, 8))

# for i, (signal, label) in enumerate(zip(signals, labels)):
#     plt.plot(signal, label=label, linewidth=1.5)

# plt.title("EEG Signals: Clean vs Noisy (EMG, EOG, MOG)", fontsize=16)
# plt.xlabel("Time Steps", fontsize=12)
# plt.ylabel("Amplitude (μV)", fontsize=12)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置路径
data_dir = "./eeg_data_split"  # 替换为你的实际路径
files_of_interest = [
    "clean_signals.npy",
    "noisy_emg_snr_0.npy"
]

# 加载信号
signals_to_plot = []
labels = []

for file in files_of_interest:
    filepath = os.path.join(data_dir, file)
    if os.path.exists(filepath):
        data = np.load(filepath, allow_pickle=False)  # (N, T, 1)
        signal = data[0].squeeze()  # 假定只取第一个样本进行绘图
        signals_to_plot.append(signal)
        
        # 根据文件名分配标签
        if "clean" in file:
            labels.append("Clean")
        elif "emg" in file and "snr_0" in file:
            labels.append("EMG 0dB")

# 绘制图形
plt.figure(figsize=(14, 8))

for i, (signal, label) in enumerate(zip(signals_to_plot, labels)):
    plt.plot(signal, label=label, linewidth=1.5)

plt.title("EEG Signals: Clean vs EMG 0dB", fontsize=16)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Amplitude (μV)", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


