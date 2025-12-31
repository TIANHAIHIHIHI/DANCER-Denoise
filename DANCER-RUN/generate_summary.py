import os
import re
import pandas as pd
from pathlib import Path
import numpy as np

def parse_txt_file_enhanced(file_path):
    """增强版解析函数，处理更多格式"""
    data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # 使用多种匹配模式
            if line.startswith('Model:'):
                data['Model'] = line.split(':', 1)[1].strip()
            elif line.startswith('Noise Type:'):
                data['Noise Type'] = line.split(':', 1)[1].strip()
            elif 'SNR' in line and 'dB' in line and not 'Final' in line:
                # 匹配输入SNR
                match = re.search(r'SNR\s*\(dB\):\s*(-?\d+)', line)
                if match:
                    data['Input SNR'] = int(match.group(1))
            elif 'Final SNR' in line:
                match = re.search(r'Final SNR:\s*([\d.]+)', line)
                if match:
                    data['Final SNR'] = float(match.group(1))
            elif 'Final RMSE' in line:
                match = re.search(r'Final RMSE:\s*([\d.]+)', line)
                if match:
                    data['Final RMSE'] = float(match.group(1))
            elif 'RMSE:' in line:
                match = re.search(r'RMSE:\s*([\d.]+)', line)
                if match:
                    data['Final RMSE'] = float(match.group(1))
                    
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    
    return data

def generate_formatted_table(results_dir):
    """生成格式化的表格，更接近原始表格样式"""
    all_data = []
    
    # 查找所有txt文件
    txt_files = list(Path(results_dir).rglob('*.txt'))
    
    print(f"Found {len(txt_files)} text files")
    
    # 解析所有文件
    for txt_file in txt_files:
        file_data = parse_txt_file_enhanced(txt_file)
        if file_data and len(file_data) >= 4:  # 至少有4个关键字段
            all_data.append(file_data)
    
    if not all_data:
        print("No valid data parsed")
        return None
    
    df = pd.DataFrame(all_data)
    print(f"Parsed {len(df)} records")
    
    # 检查数据
    print("\nData preview:")
    print(df.head())
    
    # 创建完整表格
    noise_types = ['emg', 'eog', 'mog']  # 根据您的要求
    models = ['U-Net', 'DACNN', 'ACDAE', 'DANCER']
    snr_levels = [-4, -2, 0, 2, 4]
    
    # 创建空的DataFrame来存储表格
    table_data = []
    
    for noise_idx, noise_type in enumerate(noise_types):
        # 添加噪声类型标题行
        table_data.append({
            'Noise Type': noise_type,
            'Models': '',
            **{f'SNR_{snr}': '' for snr in snr_levels},
            **{f'RMSE_{snr}': '' for snr in snr_levels}
        })
        
        # 为每个模型添加数据行
        for model in models:
            row_data = {'Noise Type': '', 'Models': model}
            
            # 获取该噪声类型和模型的所有数据
            model_data = df[(df['Noise Type'].str.lower() == noise_type.lower()) & 
                           (df['Model'].str.contains(model, case=False, na=False))]
            
            # 填充SNR数据
            for snr in snr_levels:
                snr_data = model_data[model_data['Input SNR'] == snr]
                if not snr_data.empty:
                    row_data[f'SNR_{snr}'] = f"{snr_data['Final SNR'].iloc[0]:.2f}"
                else:
                    row_data[f'SNR_{snr}'] = 'N/A'
            
            # 填充RMSE数据
            for snr in snr_levels:
                snr_data = model_data[model_data['Input SNR'] == snr]
                if not snr_data.empty:
                    row_data[f'RMSE_{snr}'] = f"{snr_data['Final RMSE'].iloc[0]:.4f}"
                else:
                    row_data[f'RMSE_{snr}'] = 'N/A'
            
            table_data.append(row_data)
    
    # 转换为DataFrame
    result_df = pd.DataFrame(table_data)
    
    # 重命名列以匹配原始表格
    column_mapping = {}
    for snr in snr_levels:
        column_mapping[f'SNR_{snr}'] = f'{snr} dB'
        column_mapping[f'RMSE_{snr}'] = f'RMSE {snr} dB'
    
    result_df = result_df.rename(columns=column_mapping)
    
    # 重新排列列
    final_columns = ['Noise Type', 'Models'] + [f'{snr} dB' for snr in snr_levels] + [f'RMSE {snr} dB' for snr in snr_levels]
    result_df = result_df[final_columns]
    
    return result_df

def create_html_with_merged_cells(df, output_path):
    """创建带有合并单元格的HTML表格"""
    
    # 创建HTML头部
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-family: Arial, sans-serif;
        }
        th, td {
            border: 1px solid #333;
            padding: 8px 12px;
            text-align: center;
            vertical-align: middle;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .noise-header {
            background-color: #2E86C1;
            color: white;
            font-weight: bold;
            font-size: 16px;
        }
        .model-name {
            text-align: left;
            font-weight: bold;
        }
        .snr-header {
            background-color: #45a049;
        }
        .rmse-header {
            background-color: #388E3C;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .best-value {
            font-weight: bold;
            color: #e74c3c;
        }
    </style>
    </head>
    <body>
    <h2 style="text-align: center; color: #2c3e50;">Denoising Performance Comparison</h2>
    <table>
    """
    
    # 添加表头
    html += """
    <thead>
        <tr>
            <th rowspan="2" style="width: 10%;">Noise Type</th>
            <th rowspan="2" style="width: 15%;">Models</th>
            <th colspan="5" class="snr-header">SNR (dB)</th>
            <th colspan="5" class="rmse-header">RMSE</th>
        </tr>
        <tr>
    """
    
    # SNR列
    snr_columns = ['-4 dB', '-2 dB', '0 dB', '2 dB', '4 dB']
    for col in snr_columns:
        html += f'<th class="snr-header">{col}</th>'
    
    # RMSE列
    for col in snr_columns:
        html += f'<th class="rmse-header">{col}</th>'
    
    html += """
        </tr>
    </thead>
    <tbody>
    """
    
    # 添加数据行
    current_noise = None
    noise_rowspan = 0
    
    # 计算每个噪声类型的行数
    noise_groups = {}
    for idx, row in df.iterrows():
        noise_type = row['Noise Type']
        if noise_type and noise_type != '':
            if noise_type not in noise_groups:
                noise_groups[noise_type] = 0
            current_noise = noise_type
        if current_noise:
            noise_groups[current_noise] += 1
    
    # 生成表格行
    current_noise = None
    rows_in_current_noise = 0
    
    for idx, row in df.iterrows():
        noise_type = row['Noise Type']
        model = row['Models']
        
        html += '<tr>'
        
        # 处理噪声类型列（合并单元格）
        if noise_type and noise_type != '':
            current_noise = noise_type
            rows_in_current_noise = noise_groups[current_noise]
            html += f'<td rowspan="{rows_in_current_noise}" class="noise-header">{noise_type.upper()}</td>'
            html += f'<td class="model-name">{model}</td>' if model else '<td></td>'
        else:
            html += f'<td class="model-name">{model}</td>' if model else '<td></td>'
        
        # 添加数据单元格
        for snr in snr_columns:
            value = row.get(snr, '')
            html += f'<td>{value}</td>'
        
        for snr in snr_columns:
            rmse_key = f'RMSE {snr}'
            value = row.get(rmse_key, '')
            html += f'<td>{value}</td>'
        
        html += '</tr>'
    
    html += """
    </tbody>
    </table>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Formatted HTML saved to {output_path}")

def main_enhanced():
    # 设置路径
    results_dir = 'results'  # 修改为您的results文件夹路径
    output_dir = 'output'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing results files with enhanced parser...")
    result_df = generate_formatted_table(results_dir)
    
    if result_df is not None and not result_df.empty:
        print("\nGenerated Table:")
        print(result_df.to_string())
        
        # 保存各种格式
        csv_path = os.path.join(output_dir, 'formatted_results.csv')
        excel_path = os.path.join(output_dir, 'formatted_results.xlsx')
        html_path = os.path.join(output_dir, 'formatted_results.html')
        
        result_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV saved to {csv_path}")
        
        result_df.to_excel(excel_path, index=False)
        print(f"Excel saved to {excel_path}")
        
        # 创建带格式的HTML
        create_html_with_merged_cells(result_df, html_path)
        
        # 显示统计信息
        print(f"\nSummary Statistics:")
        print(f"Total rows in table: {len(result_df)}")
        print(f"Noise types included: {result_df['Noise Type'].unique()}")
        print(f"Models included: {result_df['Models'].unique()}")
        
    else:
        print("Failed to generate table")

if __name__ == "__main__":
    # 运行基础版本
    # main()
    
    # 运行增强版本
    main_enhanced()
    