

import numpy as np
import matplotlib.pyplot as plt
import os

# 数据目录和模型列表
data_dir = "./data_results"
models = ["ACDAE", "DACNN", "DANCER", "U-Net"]
sample_idx = 0
channel_idx = 0

# --- 加载 clean (N, T, C) ---
clean_path = os.path.join(data_dir, "clean_DANCER_eog_snr_-2.npy")  # 更新文件名为2dB
clean = np.load(clean_path)
clean_signal = clean[sample_idx, :, channel_idx]  # (T,)

# # --- 加载 noisy (N, C, T) ---
# noisy_path = os.path.join(data_dir, "noisy_DANCER_eog_snr_2.npy")  # 更新文件名为2dB
# noisy = np.load(noisy_path)
# noisy_signal = noisy[sample_idx, channel_idx, :]  # 注意顺序！

# --- 加载 pure MOG noise input (N, T, C) from eeg_data_split ---
noisy_path = "./eeg_data_split/eog_noise_input.npy"  # ✅ 替换为纯噪声
mog_noise = np.load(noisy_path)
noisy_signal = mog_noise[sample_idx, :, channel_idx]  # ✅ shape (T,) — same as clean

# --- 绘图 ---
plt.figure(figsize=(16, 10))  # 增大整体图形尺寸

# Clean: black dashed
plt.plot(clean_signal, label="Clean Signal", color='black', linestyle='--', linewidth=2.0)

# Noisy: gray dashed
plt.plot(noisy_signal, label="Noisy Input", color='gray', linestyle='--', linewidth=1.5)

# Model colors
model_colors = {
    "ACDAE": 'blue',
    "DACNN": 'orange',
    "DANCER": 'red',
    "U-Net": 'purple'
}

# Load and plot each denoised model (all are N, C, T)
for model in models:
    path = os.path.join(data_dir, f"denoised_{model}_eog_snr_-2.npy")  # 更新文件名为2dB
    if not os.path.exists(path):
        print(f"⚠️ {model} file not found")
        continue
    denoised = np.load(path)
    denoised_signal = denoised[sample_idx, channel_idx, :]  # ← 关键：C 在中间
    plt.plot(denoised_signal, label=f"Denoised ({model})", color=model_colors[model], linewidth=1.5)

# Labels & legend
plt.title("EEG Denoising Comparison (EOG Noise, SNR=-2dB)", fontsize=20)  # 增大标题字体
plt.xlabel("Time Points", fontsize=18)  # 增大x轴标签字体
plt.ylabel("Amplitude (uV)", fontsize=18)  # 增大y轴标签字体
plt.grid(True, alpha=0.3)

# 增大图例字体和调整位置
plt.legend(loc='upper right', ncol=3, frameon=True, fancybox=True, shadow=True, fontsize=20)

# 增大坐标轴刻度标签字体
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()

# Save
os.makedirs("./plots", exist_ok=True)
plt.savefig("./plots/multi_model_comparison_eog_snr_-2.png", dpi=600, bbox_inches='tight')  # 文件名包含snr_2

plt.show()










# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# # =====================
# # 数据目录和模型列表
# # =====================
# data_dir = "./data_results"
# models = ["ACDAE", "DACNN", "DANCER", "U-Net"]
# sample_idx = 0
# channel_idx = 0

# # =====================
# # 加载 clean (N, T, C)
# # =====================
# clean_path = os.path.join(data_dir, "clean_mog_snr_2.npy")
# clean = np.load(clean_path)
# clean_signal = clean[sample_idx, :, channel_idx]  # (T,)

# # =====================
# # 加载 noisy (N, C, T)
# # =====================
# noisy_path = os.path.join(data_dir, "noisy_mog_snr_2.npy")
# noisy = np.load(noisy_path)
# noisy_signal = noisy[sample_idx, channel_idx, :]

# # =====================
# # 主图
# # =====================
# fig, ax = plt.subplots(figsize=(16, 10))

# # Clean
# ax.plot(clean_signal,
#         label="Clean Signal",
#         color='black',
#         linestyle='--',
#         linewidth=2.0)

# # Noisy
# ax.plot(noisy_signal,
#         label="Noisy Input",
#         color='gray',
#         linestyle='--',
#         linewidth=1.5)

# # Model colors
# model_colors = {
#     "ACDAE": 'blue',
#     "DACNN": 'orange',
#     "DANCER": 'red',
#     "U-Net": 'purple'
# }

# # Denoised results
# denoised_dict = {}  # 存一份给 inset 用
# for model in models:
#     path = os.path.join(data_dir, f"denoised_{model}_mog_snr_2.npy")
#     if not os.path.exists(path):
#         print(f"⚠️ {model} file not found")
#         continue

#     denoised = np.load(path)
#     denoised_signal = denoised[sample_idx, channel_idx, :]
#     denoised_dict[model] = denoised_signal

#     ax.plot(denoised_signal,
#             label=f"Denoised ({model})",
#             color=model_colors[model],
#             linewidth=1.5)

# # =====================
# # 局部放大区域设置
# # =====================
# zoom_start = 50
# zoom_end   = 100

# # 检查切片范围是否有效
# if zoom_start >= len(clean_signal) or zoom_end > len(clean_signal) or zoom_start >= zoom_end:
#     raise ValueError("Invalid zoom range: ensure 0 <= zoom_start < zoom_end <= len(clean_signal)")

# # 获取局部数据
# zoom_data = clean_signal[zoom_start:zoom_end]

# # 确保数据不为空
# if zoom_data.size == 0:
#     raise ValueError("The zoomed data is empty. Check the zoom range.")

# # 计算局部区域的最小值和最大值
# y_min = np.min(zoom_data) - 5
# y_max = np.max(zoom_data) + 5

# # 主图中画一个矩形框
# ax.add_patch(
#     plt.Rectangle((zoom_start, y_min),
#                   zoom_end - zoom_start,
#                   y_max - y_min,
#                   fill=False,
#                   edgecolor='black',
#                   linewidth=1.5)
# )

# # =====================
# # inset 放大图
# # =====================
# axins = inset_axes(
#     ax,
#     width="30%",    # inset 宽度
#     height="30%",   # inset 高度
#     loc="down right",
#     borderpad=2
# )

# # inset 中绘制相同信号（只画局部）
# axins.plot(zoom_data,
#            color='black',
#            linestyle='--',
#            linewidth=2.0)

# for model in denoised_dict:
#     axins.plot(denoised_dict[model][zoom_start:zoom_end],
#                color=model_colors[model],
#                linewidth=1.5)

# axins.set_xlim(0, zoom_end - zoom_start)
# axins.set_ylim(y_min, y_max)

# # 去掉 inset 的刻度（更像论文图）
# axins.set_xticks([])
# axins.set_yticks([])

# # 主图和 inset 之间画连接线
# mark_inset(ax, axins,
#            loc1=2, loc2=4,
#            fc="none",
#            ec="black",
#            linewidth=1.2)

# # =====================
# # Labels & legend
# # =====================
# ax.set_title("EEG Denoising Comparison (MOG Noise, SNR=2dB)", fontsize=20)
# ax.set_xlabel("Time Points", fontsize=18)
# ax.set_ylabel("Amplitude (uV)", fontsize=18)
# ax.grid(True, alpha=0.3)

# ax.legend(loc='upper right',
#           ncol=3,
#           frameon=True,
#           fancybox=True,
#           shadow=True,
#           fontsize=15)

# ax.tick_params(axis='both', labelsize=15)

# plt.tight_layout()
# plt.show()
