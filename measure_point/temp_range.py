# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:53:38 2026

@author: windows11
"""
import os
import re
import pandas as pd
from sklearn.neighbors import KDTree

# 1. 你的解析函数
def parse_params(filename):
    name = os.path.splitext(filename)[0]
    pattern = re.compile(r"steamT(?P<st>\d+)V(?P<sv>[\d.]+)gasT(?P<gt>\d+)V(?P<gv>[\d.]+)", re.I)
    match = pattern.search(name)
    return [float(match.group("st")), float(match.group("sv")), 
            float(match.group("gt")), float(match.group("gv"))] if match else None

# 2. 配置路径
data_dir = r"D:\DeepCFD2025\CNN\data"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
points_file = os.path.join(CURRENT_DIR, "measure_point.xlsx")
target_points = pd.read_excel(points_file)[['x', 'y', 'z']].values

# 3. 初始化存储容器: {测点索引: [数据字典列表]}
# 数据字典包含: st, sv, gt, gv, temp
all_results = {i: [] for i in range(len(target_points))}

# 4. 遍历处理
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print(f"正在处理 {len(files)} 个文件...")

for filename in files:
    params = parse_params(filename)
    if not params:
        continue # 文件名格式不匹配则跳过
        
    file_path = os.path.join(data_dir, filename)
    try:
        df = pd.read_csv(file_path, usecols=['x', 'y', 'z', 'temp'])
        tree = KDTree(df[['x', 'y', 'z']].values)
        
        # 寻找最近点
        _, ind = tree.query(target_points, k=1)
        
        # 存入数据
        for i, idx in enumerate(ind.flatten()):
            record = {
                'SteamT': params[0], 'SteamV': params[1], 
                'GasT': params[2], 'GasV': params[3],
                'Temp': df.iloc[idx]['temp']
            }
            all_results[i].append(record)
    except Exception as e:
        print(f"处理 {filename} 失败: {e}")

# 5. 写入 Excel (5个Sheet)
output_file = os.path.join(CURRENT_DIR, "measure_point", "all_points_data.xlsx")
with pd.ExcelWriter(output_file) as writer:
    for i in range(len(target_points)):
        df_sheet = pd.DataFrame(all_results[i])
        # 按工况参数排序，使数据更整洁
        df_sheet = df_sheet.sort_values(by=['SteamT', 'GasT'])
        df_sheet.to_excel(writer, sheet_name=f"Point_{i}", index=False)

print(f"成功保存到 {output_file}")