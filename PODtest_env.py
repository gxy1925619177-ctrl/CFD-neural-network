# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:36:14 2026

@author: windows11
"""


    
    
import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = r"D:\DeepCFD2025\jnjpcode\POD\pod_model\best_pod_withpoint_temp_model.joblib"
REPORT_PATH = r"D:\DeepCFD2025\jnjpcode\POD\test_result\env_adaptive_report.txt"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(CURRENT_DIR, "measure_point", "measure_point.xlsx")

    
class VirtualPlant:
    def __init__(self, excel_path):
        self.features = ['SteamT', 'SteamV', 'GasT', 'GasV']
        self.models = {}
        all_residuals = [] # 用于存储所有点的预测残差
        
        ## 1. 训练每个测点的 RF（随机森林） 模型并计算残差矩阵
        for i in range(5):
            df = pd.read_excel(excel_path, sheet_name=f"Point_{i}")
            model = RandomForestRegressor(n_estimators=100)
            model.fit(df[self.features], df['Temp'])
            
            # 计算预测残差
            residuals = df['Temp'] - model.predict(df[self.features])
            all_residuals.append(residuals)
            self.models[i] = model
            
        # 2. 计算协方差矩阵 (关键：捕捉测点间的物理联动规律)
        self.cov_matrix = np.cov(np.array(all_residuals).T, rowvar=False)
        
        # 3. 记录输入特征的边界 (用于随机生成工况)
        self.bounds = {col: (df[col].min(), df[col].max()) for col in self.features}
        
    def get_random_state(self):
        # 随机生成一组参数
        return [np.random.uniform(self.bounds[col][0], self.bounds[col][1]) for col in self.features]


    
    def get_sensor_feedback(self, params):
        # 1. 确保输入形状正确 (DataFrame)
        df_input = pd.DataFrame([params], columns=self.features)
        
        # 2. 获取基准趋势 (Physics Trend)
        means = np.array([self.models[i].predict(df_input)[0] for i in range(5)])
        
        # 3. 叠加物理相关性噪声 (Sensor Reality)
        correlated_noise = np.random.multivariate_normal(mean=np.zeros(5), cov=self.cov_matrix)
        
        return means + correlated_noise
    


def get_smart_filename(params):
    """
    根据参数生成唯一的文件名标识
    params: [SteamT, SteamV, GasT, GasV]
    """
    # 使用 f-string 格式化，比如 T700_V0.5_GT1240_GV12.1
    return f"ST{params[0]:.0f}_SV{params[1]:.2f}_GT{params[2]:.0f}_GV{params[3]:.1f}"
# =========================
# 1) 配置参数
# =========================


# 自适应优化配置 (类似 RL 的搜索策略)
MAX_ITER = 60       # 最大迭代次数 (寻找最优分数的尝试次数)
ERROR_TOL = 1e-3     # 容差：若 5 点平均误差小于 1e-3K，提前停止寻优


# =========================
# 3) 核心：自适应评分与修正引擎
# =========================
def adaptive_optimize(mlp_coeffs, phi, t_mean, m_idx, real_m_temps):
    """
    寻优逻辑：直接利用 phi 和 t_mean 进行矩阵运算
    """
    def objective_score(current_coeffs):
        # 核心数学公式：T_pred = T_mean + coeffs @ Phi
        t_pred_full = t_mean + current_coeffs @ phi
        
        # 提取测点位置的预测值
        t_pred_at_measures = t_pred_full[m_idx]
        
        # 计算 RMSE
        rmse_score = np.sqrt(np.mean((t_pred_at_measures - real_m_temps)**2))
        return rmse_score

    # 执行寻优
    res = minimize(
        objective_score, 
        mlp_coeffs, 
        method='L-BFGS-B',
        options={'maxiter': MAX_ITER, 'ftol': ERROR_TOL}
    )
    
    return res.x, res.fun

# =========================
# 4) 执行全样本评估
# =========================
def run_evaluation():
    data_pkg = joblib.load(MODEL_PATH)
    bundle = data_pkg["model_bundle"]
    mlp = bundle["regressor"]
    scaler = bundle["x_scaler"]
    phi = bundle["basis"]
    t_mean = bundle["mean_field"]
    m_idx = bundle["measure_indices"]

    # --- 新增：初始化虚拟工厂 ---
    plant = VirtualPlant(r"D:\DeepCFD2025\jnjpcode\POD\measure_point\all_points_data.xlsx")
    eval_list = []
    NUM_EPISODES = 100 
    
    print(f"开始自适应仿真评估，总计 {NUM_EPISODES} 次随机工况...")

    # --- 修改循环结构 ---
    for i in range(NUM_EPISODES):
        # 1. 生成随机参数与模拟传感器读数
        params = plant.get_random_state()
        t_real_measures = plant.get_sensor_feedback(params) # 真实测点温度(测试时随机给出)
        smart_name = get_smart_filename(params)

        # 2. 初始预报 (纯 MLP)
        x_in = scaler.transform([params])
        coeffs_mlp = mlp.predict(x_in)[0]
        t_raw = t_mean + coeffs_mlp @ phi
        
        # --- 注意：在模拟环境下没有全场真值(t_real_full)，我们用测点误差代替 RMSE ---
        # 原始 RMSE (在测点处)
        rmse_raw = np.sqrt(np.mean((t_raw[m_idx] - t_real_measures)**2))

        # 3. 自适应修正 (完全保留你原来的逻辑)
        coeffs_opt, final_score = adaptive_optimize(coeffs_mlp, phi, t_mean, m_idx, t_real_measures)
        t_opt = t_mean + coeffs_opt @ phi
        rmse_opt = np.sqrt(np.mean((t_opt[m_idx] - t_real_measures)**2))

        eval_list.append({
            "filename": smart_name, # 伪装成文件名，保证后续代码不报错
            "rmse_raw": rmse_raw,
            "rmse_opt": rmse_opt,
            "score": final_score
        })
        
        print(f"工况 {i}: 原始误差: {rmse_raw:.4f} -> 修正后: {rmse_opt:.4f}")

    # 5) 数据整理与排序 (这部分不用动，保持原样)
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
    
    