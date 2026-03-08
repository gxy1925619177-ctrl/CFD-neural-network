# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:05:09 2026

@author: windows11
"""

# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from coordinate_transform_strict import xyz_to_Ltheta_strict, TOTAL_L


# ---------------- Config ----------------
MODEL_PATH = r"D:\DeepCFD2025\jnjpcode\POD\pod_model\best_pod_withpoint_temp_model.joblib"
DATA_DIR = r"D:\DeepCFD2025\CNN\data"
REPORT_PATH = r"D:\DeepCFD2025\jnjpcode\POD\test_result\adaptive_report_detailed.txt"
OUT_DIR = Path(r'D:\DeepCFD2025\jnjpcode\POD\test_result\plots')



def visualize_adaptive_comparison(L, theta, t_real, t_raw, t_opt, filename, dot_size=35, opacity=0.6):
    """
    两行三列排版：
    第一行：真实值 | 自适应前预测 | 自适应前误差
    第二行：真实值 | 自适应后预测 | 自适应后误差
    """
    # 执行三维到二维的转换
    print("正在执行三维坐标展开 (3D -> 2D)...")


    # 计算误差
    err_raw = np.abs(t_real - t_raw)
    err_opt = np.abs(t_real - t_opt)

    # 创建 2x3 画布
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharex=True, sharey=True)
    
    # 颜色范围对齐
    t_min, t_max = t_real.min(), t_real.max()
    e_max = max(err_raw.max(), err_opt.max()) # 误差图共用刻度方便对比


    def plot_sub(ax, data, title, cmap, vmin=None, vmax=None):
        # 核心：确保这里的 L, theta, data 都是 (81080,)
        sc = ax.scatter(L, theta, c=data, s=dot_size, alpha=opacity, 
                        cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='none')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim(0, TOTAL_L)
        ax.set_ylim(0, 360)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # --- 第一行：自适应前 (Raw) ---
    plot_sub(axes[0, 0], t_real, 'Original True Temp', 'jet')
    plot_sub(axes[0, 1], t_raw,  'Raw Prediction (Before)', 'jet', vmin=t_min, vmax=t_max)
    plot_sub(axes[0, 2], err_raw, 'Absolute Error (Before)', 'Reds', vmin=0, vmax=e_max)
    axes[0, 0].set_ylabel('Theta (Before-Correction)', fontsize=14)

    # --- 第二行：自适应后 (Optimized) ---
    plot_sub(axes[1, 0], t_real, 'Original True Temp', 'jet')
    plot_sub(axes[1, 1], t_opt,   'Adaptive Prediction (After)', 'jet', vmin=t_min, vmax=t_max)
    plot_sub(axes[1, 2], err_opt,  'Absolute Error (After)', 'Reds', vmin=0, vmax=e_max)
    axes[1, 0].set_ylabel('Theta (After-Correction)', fontsize=14)

    # 设置公共标签
    for i in range(3):
        axes[1, i].set_xlabel('Unfolded Length L (m)', fontsize=14)

    plt.suptitle(f"Adaptive RL-Style Correction Comparison\nCase: {filename}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 确保 OUT_DIR 是 Path 对象
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 提取纯文件名（不带后缀），例如从 "steamT690.csv" 提取出 "steamT690"
    pure_name = Path(filename).stem
    
    # 拼接保存路径
    save_path = out_path / f"Compare_{pure_name}.png"
    
    # 执行保存
    plt.savefig(save_path, dpi=200, bbox_inches='tight') # 建议加上 bbox_inches 防止标签被切掉
    print(f"对比图已保存至: {save_path}")
    plt.close() # 必须加上，否则循环画图会内存溢出
    
    
if __name__ == "__main__":
    import joblib
    import re
    from scipy.optimize import minimize

    # --- 1. 初始化：加载模型和坐标 ---
    print(f"正在加载模型: {MODEL_PATH}")
    pkg = joblib.load(MODEL_PATH)
    bundle = pkg["model_bundle"]
    
    # 2. 【核心修改】因为模型里没存坐标，我们从数据文件夹里读取一份作为参考坐标
    print("正在从数据文件夹提取参考坐标...")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"在 {DATA_DIR} 没找到参考 CSV 文件")
        
    # 读取第一个文件来获取 x, y, z 坐标
    ref_df = pd.read_csv(os.path.join(DATA_DIR, files[0])).sort_values(by=['x', 'y', 'z'])
    # --- 修正后的调用 ---
    print("执行 3D -> 2D 坐标转换...")
    results = xyz_to_Ltheta_strict(
        ref_df['x'].values, 
        ref_df['y'].values, 
        ref_df['z'].values
    )

    # 打印一下看看函数到底返回了什么，方便调试
    # print(f"DEBUG: 函数返回类型 {type(results)}, 长度 {len(results) if isinstance(results, tuple) else 'N/A'}")

    # 正常的函数应该返回 (L, theta, segment)
    print("解析坐标转换结果...")
    
    # 因为结果是字典，我们按键名提取
    if isinstance(results, dict):
        L_2d = results['L']
        theta_2d = results['theta']
        # 如果需要总长度，也可以从字典拿：actual_total_l = results['L_total']
    elif isinstance(results, tuple):
        # 兼容元组返回的情况
        L_2d = results[0]
        theta_2d = results[1]
    else:
        raise ValueError(f"无法识别的返回类型: {type(results)}")

    # 校验数据量是否为 81080
    if L_2d.size != 81080:
        print(f"警告：点数不匹配！预期 81080，实际得到 {L_2d.size}")
    
    print("坐标解析完成，开始进入工况处理循环...")
    
    # 获取测点索引和模型组件
    m_idx = bundle["measure_indices"]
    mlp = bundle["regressor"]
    x_scaler = bundle["x_scaler"]
    phi = bundle["basis"]
    t_mean = bundle["mean_field"]

    # 3. 坐标转换 (全场 81080 个点只转一次，提高效率)


    # --- 2. 扫描数据文件夹 ---
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        print(f"错误：在 {DATA_DIR} 中没找到 CSV 文件，请检查路径！")
        sys.exit()

    # --- 3. 定义寻优函数 (自适应核心) ---
    def adaptive_optimize(initial_coeffs, phi, t_mean, m_idx, t_real_measures):
        def objective(coeffs):
            t_at_measure = t_mean[m_idx] + coeffs @ phi[:, m_idx]
            return np.sqrt(np.mean((t_at_measure - t_real_measures)**2))
        res = minimize(objective, initial_coeffs, method='L-BFGS-B', options={'maxiter': 60})
        return res.x

    # --- 4. 循环处理每一个文件 ---
    for fname in files:
        full_path = os.path.join(DATA_DIR, fname)
        print(f"正在处理: {fname}")

        # a. 解析工况参数 (需匹配你的文件名格式)
        pattern = re.compile(r"steamT(?P<st>\d+)V(?P<sv>[\d.]+)gasT(?P<gt>\d+)V(?P<gv>[\d.]+)", re.I)
        match = pattern.search(fname)
        if not match: continue
        # 在 153-154 行左右修改解析逻辑
        params = [
            float(match.group("st")), 
            float(match.group("sv").rstrip('.')), # rstrip('.') 确保去掉末尾误抓的点
            float(match.group("gt")), 
            float(match.group("gv").rstrip('.'))
        ]

        # b. 读取真实值
        df_real = pd.read_csv(full_path).sort_values(by=['x', 'y', 'z'])
        t_real = df_real['temp'].values
        t_real_measures = t_real[m_idx] # 5个测点的真实值

        # c. 初始预测 (自适应前)
        # 1. 缩放输入 (正确)
        x_in = x_scaler.transform([params])
        
        # 2. 得到预测的 POD 系数 (可能是缩放后的)
        raw_coeffs_scaled = mlp.predict(x_in)
        
        # 3. 【核心点】检查是否需要反向缩放 POD 系数
        if 'c_scaler' in bundle and bundle['c_scaler'] is not None:
            # 如果训练时缩放了系数，这里要变回来
            coeffs_mlp = bundle['c_scaler'].inverse_transform(raw_coeffs_scaled)[0]
        else:
            # 如果没缩放，直接用
            coeffs_mlp = raw_coeffs_scaled[0]
        
        # 4. 重构全场温度
        t_raw = t_mean + coeffs_mlp @ phi

        # d. 自适应寻优 (自适应后)
        coeffs_opt = adaptive_optimize(coeffs_mlp, phi, t_mean, m_idx, t_real_measures)
        t_opt = t_mean + coeffs_opt @ phi

        # e. 调用你的绘图函数
        visualize_adaptive_comparison(L_2d, theta_2d, t_real, t_raw, t_opt, fname)

# ---------------- 如何集成到你的 PODtest_withpoint_RL.py 中 ----------------
# 在 eval_list.append 之后调用此函数即可：
"""
visualize_adaptive_comparison(
    ref_coords = coords,       # 模型包里存的原始 xyz
    t_real = t_real_full,      # CSV读取的真实全场
    t_raw = t_raw,             # 纯MLP预测全场
    t_opt = t_opt,             # 寻优修正后的全场
    filename = fname,
    dot_size = 30
)
"""