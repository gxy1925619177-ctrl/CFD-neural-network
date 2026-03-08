# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:42:36 2026

@author: windows11
"""

# -*- coding: utf-8 -*-
"""
功能：基于测点反馈的 POD-MLP 自适应寻优评估
1. 模拟 5 个测点的实时温度反馈。
2. 以 MLP 预测为起点，通过最小化测点误差来在线修正 POD 系数。
3. 比较修正前后的全场 RMSE 提升。
"""

import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

# =========================
# 1) 配置参数
# =========================
DATA_DIR = r"D:\DeepCFD2025\CNN\data"
MODEL_PATH = r"D:\DeepCFD2025\jnjpcode\POD\pod_model\best_pod_withpoint_temp_model.joblib"
REPORT_PATH = r"D:\DeepCFD2025\jnjpcode\POD\test_result\rl_adaptive_report.txt"

# 自适应优化配置 (类似 RL 的搜索策略)
MAX_ITER = 60       # 最大迭代次数 (寻找最优分数的尝试次数)
ERROR_TOL = 0.2     # 容差：若 5 点平均误差小于 0.2K，提前停止寻优

# =========================
# 2) 辅助函数
# =========================
def parse_params(filename):
    name = os.path.splitext(filename)[0]
    pattern = re.compile(r"steamT(?P<st>\d+)V(?P<sv>[\d.]+)gasT(?P<gt>\d+)V(?P<gv>[\d.]+)", re.I)
    match = pattern.search(name)
    return [float(match.group("st")), float(match.group("sv")), 
            float(match.group("gt")), float(match.group("gv"))] if match else None



# =========================
# 3) 核心：自适应评分与修正引擎
# =========================
def adaptive_optimize(mlp_coeffs, phi, t_mean, m_idx, real_m_temps):
    """
    寻优逻辑：改变 POD 系数使测点处的误差最小
    """
    # 目标函数：计算测点处的 RMSE (分数越低越好)
    def objective_score(current_coeffs):
        # 这里的矩阵乘法逻辑需与训练保持一致：T = T_mean + coeffs @ Phi
        t_pred_full = t_mean + current_coeffs @ phi
        t_pred_at_measures = t_pred_full[m_idx]
        # 计算 5 个测点的 RMSE
        rmse_score = np.sqrt(np.mean((t_pred_at_measures - real_m_temps)**2))
        return rmse_score

    # 执行寻优 (算法：L-BFGS-B)
    # 以 MLP 输出为 initial guess，在模态空间内搜索
    res = minimize(
        objective_score, 
        mlp_coeffs, 
        method='L-BFGS-B',
        options={'maxiter': MAX_ITER, 'ftol': 1e-3}
    )
    
    return res.x, res.fun

# =========================
# 4) 执行全样本评估
# =========================
def run_evaluation():
    # 加载模型
    print("正在加载模型及测点接口...")
    data_pkg = joblib.load(MODEL_PATH)
    bundle = data_pkg["model_bundle"]

    
    # 提取组件
    mlp = bundle["regressor"]
    scaler = bundle["x_scaler"]
    phi = bundle["basis"]
    t_mean = bundle["mean_field"]
    m_idx = bundle["measure_indices"] # 训练时埋好的 5 个点索引

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    eval_list = []

    print(f"开始自适应评估，总计 {len(files)} 个工况...")

    for f in files:
        fname = os.path.basename(f)
        params = parse_params(fname)
        if params is None: continue

        # 1. 读取真实场并模拟传感器反馈
        df_real = pd.read_csv(f).sort_values(by=['x', 'y', 'z'])
        t_real_full = df_real['temp'].values
        t_real_measures = t_real_full[m_idx] # 这就是你预埋的 5 个点真实温度

        # 2. 初始预报 (纯 MLP)
        x_in = scaler.transform([params])
        coeffs_mlp = mlp.predict(x_in)[0]
        t_raw = t_mean + coeffs_mlp @ phi
        rmse_raw = np.sqrt(np.mean((t_real_full - t_raw)**2))

        # 3. 自适应修正 (你的 RL 思路核心)
        coeffs_opt, final_score = adaptive_optimize(coeffs_mlp, phi, t_mean, m_idx, t_real_measures)
        t_opt = t_mean + coeffs_opt @ phi
        rmse_opt = np.sqrt(np.mean((t_real_full - t_opt)**2))

        eval_list.append({
            "filename": fname,
            "rmse_raw": rmse_raw,
            "rmse_opt": rmse_opt,
            "score": final_score
        })
        
        print(f"工况: {fname[:30]}... | 原始 RMSE: {rmse_raw:.4f} -> 修正后: {rmse_opt:.4f}")

    # 5) 数据整理与排序
    df_res = pd.DataFrame(eval_list)
    # 按原始 RMSE 从小到大排序，方便查看最优和最差
    df_res = df_res.sort_values(by="rmse_raw").reset_index(drop=True)

    # 6) 写入详细报告
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write("==================================================\n")
        f_out.write("      基于测点自适应优化 (RL思路) 评估报告\n")
        f_out.write(f"      生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_out.write("==================================================\n\n")

        # 统计信息
        f_out.write(f"评估总工况数: {len(df_res)}\n")
        f_out.write(f"原始平均 RMSE: {df_res['rmse_raw'].mean():.6f}\n")
        f_out.write(f"修正后平均 RMSE: {df_res['rmse_opt'].mean():.6f}\n")
        improvement = (df_res['rmse_raw'].mean() - df_res['rmse_opt'].mean()) / df_res['rmse_raw'].mean() * 100
        f_out.write(f"整体精度提升幅度: {improvement:.2f}%\n\n")

        # 极端工况分析
        best_case = df_res.iloc[0]
        worst_case = df_res.iloc[-1]

        f_out.write("--- [最优预测工况] (原始误差最小) ---\n")
        f_out.write(f"文件名: {best_case['filename']}\n")
        f_out.write(f"原始 RMSE: {best_case['rmse_raw']:.6f} -> 修正后: {best_case['rmse_opt']:.6f}\n")
        f_out.write(f"测点最终残差分数: {best_case['score']:.6f}\n\n")

        f_out.write("--- [最差预测工况] (原始误差最大) ---\n")
        f_out.write(f"文件名: {worst_case['filename']}\n")
        f_out.write(f"原始 RMSE: {worst_case['rmse_raw']:.6f} -> 修正后: {worst_case['rmse_opt']:.6f}\n")
        f_out.write(f"测点最终残差分数: {worst_case['score']:.6f}\n\n")

        f_out.write("--- [详细工况列表] (按原始误差从小到大排序) ---\n")
        f_out.write(f"{'文件名':<50} | {'原始RMSE':<12} | {'修正后RMSE':<12} | {'提升度'}\n")
        f_out.write("-" * 100 + "\n")
        
        for _, row in df_res.iterrows():
            local_imp = (row['rmse_raw'] - row['rmse_opt']) / row['rmse_raw'] * 100
            f_out.write(f"{row['filename'][:50]:<50} | {row['rmse_raw']:<12.6f} | {row['rmse_opt']:<12.6f} | {local_imp:>6.2f}%\n")

    print(f"\n[评估完成] 详细对比报告已生成: {REPORT_PATH}")

if __name__ == "__main__":
    run_evaluation()