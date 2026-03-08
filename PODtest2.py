# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:40:08 2026

@author: windows11
"""

"""
功能：自动化批量评估 POD 模型性能。
1. 遍历 DATA_DIR 下所有原始 CSV 文件。
2. 提取文件名中的工况参数并进行模型预测。
3. 计算每个工况的 RMSE, MaxAE（最大绝对误差）和 MAE。
4. 生成详细的统计报告 txt，并对所有工况按误差进行排序。
"""

import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# =========================
# 1) 配置参数 (请根据实际情况调整)
# =========================
DATA_DIR = r"D:\DeepCFD2025\CNN\data"
# 模型文件建议放在代码同级目录的 pod_model 文件夹中
MODEL_PATH = r"D:\DeepCFD2025\jnjpcode\POD\pod_model\best_pod_temp_model.joblib"
REPORT_PATH = r"D:\DeepCFD2025\jnjpcode\POD\test_result\report.txt"

COORD_COLS = ["x", "y", "z"]
T_COL = "temp"

# =========================
# 2) 辅助功能函数
# =========================
def parse_params(filename):
    """从文件名解析工况参数"""
    pattern = re.compile(r"steamT(?P<st>[-+]?\d*\.?\d+)V(?P<sv>[-+]?\d*\.?\d+)gasT(?P<gt>[-+]?\d*\.?\d+)V(?P<gv>[-+]?\d*\.?\d+)", re.I)
    match = pattern.search(filename)
    if match:
        return [float(match.group("st")), float(match.group("sv")), 
                float(match.group("gt")), float(match.group("gv"))]
    return None

def load_and_preprocess(file_path):
    """读取并对齐数据点"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    # 统一排序逻辑必须与训练脚本完全一致 
    df = df.sort_values(by=COORD_COLS).reset_index(drop=True)
    return df

# =========================
# 3) 主评估流程
# =========================
def run_batch_evaluation():
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        return

    # 加载模型
    print("正在加载模型...")
    data_pkg = joblib.load(MODEL_PATH)
    bundle = data_pkg["model_bundle"]

    
    model = bundle["regressor"]
    scaler = bundle["x_scaler"]
    phi = bundle["basis"]
    t_mean = bundle["mean_field"]

    files = glob.glob(os.path.join(DATA_DIR, "steamT*.csv"))
    results = []

    print(f"开始遍历 {len(files)} 个工况文件...")

    for f in files:
        params = parse_params(os.path.basename(f))
        if params is None: continue

        # 1. 模型预测
       # 1. 模型预测
        x_scaled = scaler.transform([params])
        coeffs = model.predict(x_scaled)  # 预测出的 POD 系数

        # 2. 重构温度场：T = T_mean + Phi^T * a
        # 既然 phi 的形状是 (K, N)，我们需要转置它变成 (N, K) 再乘以系数 (K,)
        # 或者直接使用 np.dot(coeffs.flatten(), phi)
        t_pred = t_mean + coeffs.flatten().dot(phi)

        # 2. 读取真实值
        try:
            df_true = load_and_preprocess(f)
            t_true = df_true[T_COL].values
            
            if len(t_true) != len(t_pred):
                print(f"跳过 {os.path.basename(f)}: 点数不匹配")
                continue

            # 3. 计算误差指标
            diff = t_pred - t_true
            rmse = np.sqrt(np.mean(diff**2))
            max_ae = np.max(np.abs(diff))
            mae = np.mean(np.abs(diff))

            results.append({
                "filename": os.path.basename(f),
                "params": params,
                "rmse": rmse,
                "max_ae": max_ae,
                "mae": mae
            })
        except Exception as e:
            print(f"处理文件 {os.path.basename(f)} 出错: {e}")

    # =========================
    # 4) 生成汇总报告
    # =========================
    if not results:
        print("未成功评估任何工况。")
        return

    df_res = pd.DataFrame(results).sort_values(by="rmse")
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write("==================================================\n")
        f_out.write("          POD 降阶模型全样本评估报告\n")
        f_out.write(f"          生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_out.write("==================================================\n\n")

        f_out.write(f"评估总工况数: {len(df_res)}\n")
        f_out.write(f"全样本平均 RMSE: {df_res['rmse'].mean():.6f}\n")
        f_out.write(f"全样本最大 RMSE: {df_res['rmse'].max():.6f}\n")
        f_out.write(f"全样本最小 RMSE: {df_res['rmse'].min():.6f}\n\n")

        # 误差极端情况
        best_case = df_res.iloc[0]
        worst_case = df_res.iloc[-1]
        
        f_out.write("--- 最优预测工况 ---\n")
        f_out.write(f"文件名: {best_case['filename']}\n")
        f_out.write(f"RMSE: {best_case['rmse']:.6f}, MaxAE: {best_case['max_ae']:.6f}\n\n")

        f_out.write("--- 最差预测工况 ---\n")
        f_out.write(f"文件名: {worst_case['filename']}\n")
        f_out.write(f"RMSE: {worst_case['rmse']:.6f}, MaxAE: {worst_case['max_ae']:.6f}\n\n")

        f_out.write("--- 所有工况误差排序 (按 RMSE 从小到大) ---\n")
        f_out.write(f"{'Filename':<50} | {'RMSE':<10} | {'MaxAE':<10}\n")
        f_out.write("-" * 75 + "\n")
        for _, row in df_res.iterrows():
            f_out.write(f"{row['filename']:<50} | {row['rmse']:<10.6f} | {row['max_ae']:<10.6f}\n")

    print(f"\n评估完成！报告已生成：\n{REPORT_PATH}")

if __name__ == "__main__":
    run_batch_evaluation()