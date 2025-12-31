import os
import re
import pandas as pd
from pathlib import Path

def parse_txt_file(file_path):
    """解析单个txt文件，提取性能指标"""
    data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取模型名称
        model_match = re.search(r'Model:\s*(.+)', content)
        if model_match:
            data['Model'] = model_match.group(1).strip()
        else:
            # 尝试其他可能的格式
            model_match = re.search(r'Model Name:\s*(.+)', content)
            if model_match:
                data['Model'] = model_match.group(1).strip()
        
        # 提取噪声类型
        noise_match = re.search(r'Noise Type:\s*(.+)', content)
        if noise_match:
            data['Noise Type'] = noise_match.group(1).strip().lower()
        else:
            # 尝试其他可能的格式
            noise_match = re.search(r'Noise:\s*(.+)', content)
            if noise_match:
                data['Noise Type'] = noise_match.group(1).strip().lower()
        
        # 提取输入SNR
        snr_input_match = re.search(r'SNR\s*\(dB\):\s*(-?\d+)', content)
        if snr_input_match:
            data['Input SNR'] = int(snr_input_match.group(1))
        else:
            # 尝试其他可能的格式
            snr_input_match = re.search(r'Input SNR:\s*(-?\d+)', content)
            if snr_input_match:
                data['Input SNR'] = int(snr_input_match.group(1))
        
        # 提取最终SNR
        final_snr_match = re.search(r'Final SNR:\s*([\d.]+)\s*dB', content)
        if final_snr_match:
            data['Final SNR'] = float(final_snr_match.group(1))
        else:
            # 尝试其他可能的格式
            final_snr_match = re.search(r'Output SNR:\s*([\d.]+)', content)
            if final_snr_match:
                data['Final SNR'] = float(final_snr_match.group(1))
        
        # 提取最终RMSE
        rmse_match = re.search(r'Final RMSE:\s*([\d.]+)', content)
        if rmse_match:
            data['Final RMSE'] = float(rmse_match.group(1))
        else:
            # 尝试其他可能的格式
            rmse_match = re.search(r'RMSE:\s*([\d.]+)', content)
            if rmse_match:
                data['Final RMSE'] = float(rmse_match.group(1))
        
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None
    
    # 检查必要字段是否都存在
    required_fields = ['Model', 'Noise Type', 'Input SNR', 'Final SNR', 'Final RMSE']
    for field in required_fields:
        if field not in data:
            print(f"Warning: Missing {field} in {file_path}")
            return None
    
    return data

def generate_txt_table(results_dir):
    """生成TXT格式的表格"""
    
    # 查找所有txt文件
    txt_files = list(Path(results_dir).rglob('*.txt'))
    
    if not txt_files:
        print(f"No txt files found in {results_dir}")
        return ""
    
    print(f"Found {len(txt_files)} text files")
    
    # 解析所有文件
    all_data = []
    for txt_file in txt_files:
        file_data = parse_txt_file(txt_file)
        if file_data:
            all_data.append(file_data)
    
    if not all_data:
        print("No valid data parsed")
        return ""
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 定义噪声类型（根据您的要求）
    noise_types = ['emg', 'eog', 'mog']
    models = ['U-Net', 'DACNN', 'ACDAE', 'DANCER']
    snr_levels = [-4, -2, 0, 2, 4]
    
    # 创建字典来存储数据
    # 结构: data[noise_type][model][snr] = {'SNR': value, 'RMSE': value}
    data_dict = {}
    
    for noise_type in noise_types:
        data_dict[noise_type] = {}
        for model in models:
            data_dict[noise_type][model] = {}
            for snr in snr_levels:
                data_dict[noise_type][model][snr] = {'SNR': None, 'RMSE': None}
    
    # 填充数据
    for _, row in df.iterrows():
        noise_type = row['Noise Type'].lower()
        model = row['Model']
        snr = row['Input SNR']
        
        # 只处理我们关心的噪声类型
        if noise_type in noise_types and model in models and snr in snr_levels:
            data_dict[noise_type][model][snr]['SNR'] = row['Final SNR']
            data_dict[noise_type][model][snr]['RMSE'] = row['Final RMSE']
    
    # 生成TXT表格
    txt_output = ""
    
    # 表头
    header1 = "Noise Type    Models            "
    header2 = "              "
    
    # SNR列
    for snr in snr_levels:
        header1 += f"  {snr:3} dB    "
        header2 += f"  SNR(dB)   "
    
    # RMSE列
    for snr in snr_levels:
        header1 += f"  {snr:3} dB    "
        header2 += f"   RMSE     "
    
    txt_output += header1 + "\n"
    txt_output += header2 + "\n"
    txt_output += "-" * len(header1) + "\n"
    
    # 数据行
    for noise_type in noise_types:
        # 噪声类型标题行
        noise_header = f"{noise_type.upper():<13} "
        for _ in range(len(snr_levels) * 2 + 1):
            noise_header += "            "
        txt_output += noise_header + "\n"
        
        # 每个模型的数据行
        for model in models:
            row = f"{'':<13} {model:<16}"
            
            # SNR数据
            for snr in snr_levels:
                snr_value = data_dict[noise_type][model][snr]['SNR']
                if snr_value is not None:
                    row += f"  {snr_value:8.2f}  "
                else:
                    row += f"  {'N/A':8}  "
            
            # RMSE数据
            for snr in snr_levels:
                rmse_value = data_dict[noise_type][model][snr]['RMSE']
                if rmse_value is not None:
                    row += f"  {rmse_value:8.4f}  "
                else:
                    row += f"  {'N/A':8}  "
            
            txt_output += row + "\n"
        
        # 添加空行分隔不同的噪声类型
        txt_output += "\n"
    
    return txt_output

def save_txt_table(txt_content, output_path):
    """保存TXT表格到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    print(f"TXT table saved to {output_path}")

def main():
    # 设置路径
    results_dir = 'results'  # 修改为您的results文件夹路径
    output_dir = 'output'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing results files...")
    
    # 生成TXT表格
    txt_content = generate_txt_table(results_dir)
    
    if txt_content:
        # 打印到控制台
        print("\n" + "="*80)
        print("Generated TXT Table:")
        print("="*80)
        print(txt_content)
        
        # 保存到文件
        output_path = os.path.join(output_dir, 'performance_summary.txt')
        save_txt_table(txt_content, output_path)
        
        # 也保存一个更紧凑的版本
        compact_output = generate_compact_txt_table(results_dir)
        if compact_output:
            compact_path = os.path.join(output_dir, 'performance_summary_compact.txt')
            save_txt_table(compact_output, compact_path)
    else:
        print("Failed to generate table")

def generate_compact_txt_table(results_dir):
    """生成更紧凑的TXT表格（类似您提供的格式）"""
    
    # 查找所有txt文件
    txt_files = list(Path(results_dir).rglob('*.txt'))
    
    if not txt_files:
        return ""
    
    # 解析所有文件
    all_data = []
    for txt_file in txt_files:
        file_data = parse_txt_file(txt_file)
        if file_data:
            all_data.append(file_data)
    
    if not all_data:
        return ""
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 定义噪声类型和模型
    noise_types = ['emg', 'eog', 'mog']
    models = ['U-Net', 'DACNN', 'ACDAE', 'DANCER']
    snr_levels = [-4, -2, 0, 2, 4]
    
    # 创建数据字典
    data_dict = {}
    for noise_type in noise_types:
        data_dict[noise_type] = {}
        for model in models:
            data_dict[noise_type][model] = {}
            for snr in snr_levels:
                data_dict[noise_type][model][snr] = {'SNR': None, 'RMSE': None}
    
    # 填充数据
    for _, row in df.iterrows():
        noise_type = row['Noise Type'].lower()
        model = row['Model']
        snr = row['Input SNR']
        
        if noise_type in noise_types and model in models and snr in snr_levels:
            data_dict[noise_type][model][snr]['SNR'] = row['Final SNR']
            data_dict[noise_type][model][snr]['RMSE'] = row['Final RMSE']
    
    # 生成紧凑表格
    txt_output = ""
    
    # 表头（更紧凑）
    header = "Noise Type  Models       "
    for snr in snr_levels:
        header += f" {snr:>3} dB "
    header += "   "
    for snr in snr_levels:
        header += f" {snr:>3} dB "
    
    txt_output += header + "\n"
    txt_output += "=" * len(header) + "\n\n"
    
    # 数据行
    for noise_type in noise_types:
        txt_output += f"{noise_type.upper()}\n"
        
        for model in models:
            row = f"            {model:<12}"
            
            # SNR数据
            for snr in snr_levels:
                snr_value = data_dict[noise_type][model][snr]['SNR']
                if snr_value is not None:
                    row += f" {snr_value:6.2f}"
                else:
                    row += f" {'N/A':>6}"
            
            row += "   "
            
            # RMSE数据
            for snr in snr_levels:
                rmse_value = data_dict[noise_type][model][snr]['RMSE']
                if rmse_value is not None:
                    row += f" {rmse_value:6.4f}"
                else:
                    row += f" {'N/A':>6}"
            
            txt_output += row + "\n"
        
        txt_output += "\n"
    
    return txt_output

def generate_exact_format_table(results_dir):
    """生成与您提供的示例格式完全一致的表格"""
    
    # 查找所有txt文件
    txt_files = list(Path(results_dir).rglob('*.txt'))
    
    if not txt_files:
        return ""
    
    # 解析所有文件
    all_data = []
    for txt_file in txt_files:
        file_data = parse_txt_file(txt_file)
        if file_data:
            all_data.append(file_data)
    
    if not all_data:
        return ""
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 定义噪声类型和模型
    noise_types = ['emg', 'eog', 'mog']
    models = ['U-Net', 'DACNN', 'ACDAE', 'DANCER']
    snr_levels = [-4, -2, 0, 2, 4]
    
    # 创建数据字典
    data_dict = {}
    for noise_type in noise_types:
        data_dict[noise_type] = {}
        for model in models:
            data_dict[noise_type][model] = {}
            for snr in snr_levels:
                data_dict[noise_type][model][snr] = {'SNR': None, 'RMSE': None}
    
    # 填充数据
    for _, row in df.iterrows():
        noise_type = row['Noise Type'].lower()
        model = row['Model']
        snr = row['Input SNR']
        
        if noise_type in noise_types and model in models and snr in snr_levels:
            data_dict[noise_type][model][snr]['SNR'] = row['Final SNR']
            data_dict[noise_type][model][snr]['RMSE'] = row['Final RMSE']
    
    # 生成表格
    txt_output = ""
    
    # 创建表格顶部
    top_line = "┌─────────────┬───────────────"
    for i in range(5):
        top_line += "┬──────────"
    top_line += "┬"
    for i in range(5):
        top_line += "┬──────────"
    top_line += "┐"
    
    txt_output += top_line + "\n"
    
    # 表头第一行
    header1 = "│ Noise Type  │ Models        "
    for i in range(5):
        header1 += "│ SNR (dB) "
    header1 += "│"
    for i in range(5):
        header1 += "│  RMSE    "
    header1 += "│"
    
    txt_output += header1 + "\n"
    
    # 表头第二行
    header2 = "├─────────────┼───────────────"
    for i in range(5):
        header2 += "┼──────────"
    header2 += "┼"
    for i in range(5):
        header2 += "┼──────────"
    header2 += "┤"
    
    txt_output += header2 + "\n"
    
    # 表头第三行
    header3 = "│             │               "
    for snr in snr_levels:
        header3 += f"│  {snr:3} dB  "
    header3 += "│"
    for snr in snr_levels:
        header3 += f"│  {snr:3} dB  "
    header3 += "│"
    
    txt_output += header3 + "\n"
    
    # 数据行
    for noise_type in noise_types:
        # 噪声类型分隔线
        sep_line = "├─────────────┼───────────────"
        for i in range(5):
            sep_line += "┼──────────"
        sep_line += "┼"
        for i in range(5):
            sep_line += "┼──────────"
        sep_line += "┤"
        
        txt_output += sep_line + "\n"
        
        # 噪声类型行
        noise_row = f"│ {noise_type.upper():<11} │               "
        for i in range(10):
            noise_row += "│          "
        noise_row += "│"
        
        txt_output += noise_row + "\n"
        
        # 每个模型的数据行
        for model_idx, model in enumerate(models):
            row = f"│             │ {model:<13} "
            
            # SNR数据
            for snr in snr_levels:
                snr_value = data_dict[noise_type][model][snr]['SNR']
                if snr_value is not None:
                    row += f"│ {snr_value:8.2f} "
                else:
                    row += f"│   N/A    "
            
            row += "│"
            
            # RMSE数据
            for snr in snr_levels:
                rmse_value = data_dict[noise_type][model][snr]['RMSE']
                if rmse_value is not None:
                    row += f"│ {rmse_value:8.4f} "
                else:
                    row += f"│   N/A    "
            
            row += "│"
            
            txt_output += row + "\n"
    
    # 表格底部
    bottom_line = "└─────────────┴───────────────"
    for i in range(5):
        bottom_line += "┴──────────"
    bottom_line += "┴"
    for i in range(5):
        bottom_line += "┴──────────"
    bottom_line += "┘"
    
    txt_output += bottom_line + "\n"
    
    return txt_output

if __name__ == "__main__":
    main()
    
    # 如果需要生成更精美的表格，可以取消下面的注释
    # print("\n" + "="*80)
    # print("Formatted TXT Table:")
    # print("="*80)
    # formatted_output = generate_exact_format_table('results')
    # if formatted_output:
    #     print(formatted_output)
    #     with open('output/formatted_table.txt', 'w', encoding='utf-8') as f:
    #         f.write(formatted_output)
    