#!/bin/bash


# i=4
# snr_values=2
# gpu_ids=(0 1 0 1 1)  

# echo "starting..."

# # for i in "${!snr_values[@]}"; do
#     echo "启动 SNR=${snr_values} 在 GPU ${gpu_ids[i]}"
#     python run.py \
#         --split_dir ./eeg_data_split  \
#         --model DANCER \
#         --batch_size 64 \
#         --epochs 80 \
#         --lr 1e-3 \
#         --noise_type emg \
#         --snr_db "${snr_values}" \
#         --gpu_id "${gpu_ids[i]}" \
#         --checkpoint_dir ./checkpoints \
#         --mode train 
#     # sleep 5  
# # done


# wait
# echo "所有DANCER训练任务已完成!"


# #!/bin/bash

# # SNR值和对应的GPU分配
# declare -A snr_gpu_map
# snr_gpu_map[-4]=0
# snr_gpu_map[-2]=1
# snr_gpu_map[0]=0
# snr_gpu_map[2]=1
# snr_gpu_map[4]=0

# echo "starting..."

# # 启动所有SNR值的训练
# for snr in "${!snr_gpu_map[@]}"; do
#     gpu_id=${snr_gpu_map[$snr]}
    
#     echo "启动 SNR=${snr} 在 GPU ${gpu_id}"
#     python run.py \
#         --split_dir ./eeg_data_split \
#         --model DANCER \
#         --batch_size 64 \
#         --epochs 80 \
#         --lr 1e-3 \
#         --noise_type emg \
#         --snr_db "${snr}" \
#         --gpu_id "${gpu_id}" \
#         --checkpoint_dir ./checkpoints \
#         --mode train &
    
#     sleep 2
# done

# wait
# echo "所有DANCER训练任务已完成!"




#!/bin/bash

# 噪声类型数组
noise_types=("emg" "eog" "mog")
# SNR值数组
snr_values=(-4 -2 0 2 4)

echo "开始训练所有噪声类型..."

# 遍历所有噪声类型
for noise_type in "${noise_types[@]}"; do
    echo "================================"
    echo "开始训练噪声类型: ${noise_type}"
    echo "================================"
    
    # 创建两个数组，分别对应GPU0和GPU1的任务
    gpu0_snr=()
    gpu1_snr=()
    
    # 将SNR值分配到不同的GPU
    for i in "${!snr_values[@]}"; do
        if [ $((i % 2)) -eq 0 ]; then
            gpu0_snr+=(${snr_values[i]})
        else
            gpu1_snr+=(${snr_values[i]})
        fi
    done
    
    echo "GPU0将训练SNR值: ${gpu0_snr[@]}"
    echo "GPU1将训练SNR值: ${gpu1_snr[@]}"
    
    # 启动GPU0的任务（一次一个）
    for snr in "${gpu0_snr[@]}"; do
        echo "启动 噪声类型=${noise_type}, SNR=${snr}dB 在 GPU 0"
        python run.py \
            --split_dir ./eeg_data_split \
            --model DANCER \
            --batch_size 64 \
            --epochs 80 \
            --lr 1e-3 \
            --noise_type "${noise_type}" \
            --snr_db "${snr}" \
            --gpu_id 0 \
            --checkpoint_dir ./checkpoints \
            --mode train
    done &
    
    # 启动GPU1的任务（一次一个）
    for snr in "${gpu1_snr[@]}"; do
        echo "启动 噪声类型=${noise_type}, SNR=${snr}dB 在 GPU 1"
        python run.py \
            --split_dir ./eeg_data_split \
            --model DANCER \
            --batch_size 64 \
            --epochs 80 \
            --lr 1e-3 \
            --noise_type "${noise_type}" \
            --snr_db "${snr}" \
            --gpu_id 1 \
            --checkpoint_dir ./checkpoints \
            --mode train
    done &
    
    # 等待当前噪声类型的所有训练任务完成
    echo "等待噪声类型 ${noise_type} 的所有训练任务完成..."
    wait
    echo "噪声类型 ${noise_type} 的所有训练任务已完成!"
    echo ""
    
done

echo "所有噪声类型的所有训练任务已完成!"
