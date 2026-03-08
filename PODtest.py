# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:20:24 2026

@author: windows11
"""

# -*- coding: utf-8 -*-
"""
POD模型测试程序（自动匹配真实数据）
======================================================
功能：
1. 读取训练好的 best_pod_temp_model.joblib
2. 输入 4 个初始工况
3. 用模型预测所有点温度
4. 自动在原始数据文件夹中匹配对应真实工况文件
5. 若匹配成功：
   输出各点预测温度、真实温度、绝对误差、相对误差，并打印整体误差统计
6. 若匹配失败：
   只输出各点预测温度
7. 结果保存为 csv 文件

依赖：
pip install numpy pandas joblib openpyxl
"""

import os
import re
import glob
import joblib
import numpy as np
import pandas as pd


# =========================================================
# 0. 路径配置
# =========================================================
DATA_DIR = r"D:\DeepCFD2025\CNN\data"
MODEL_PATH = r"D:\DeepCFD2025\jnjpcode\POD\pod_model\best_pod_temp_model.joblib"
REPORT_PATH = r"D:\DeepCFD2025\jnjpcode\POD\test_result\report.txt"

# 工况匹配容差
MATCH_TOL = 1e-8

# 点位匹配容差
POINT_TOL = 1e-6


# =========================================================
# 1. 从文件名解析工况
# =========================================================
def parse_conditions_from_filename(filename: str):
    """
    默认支持类似：
    steamT690V0.54gasT1236V12.11_raw.csv
    """
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]

    pattern = re.compile(
        r"steamT(?P<steamT>[-+]?\d*\.?\d+)"
        r"V(?P<steamV>[-+]?\d*\.?\d+)"
        r"gasT(?P<gasT>[-+]?\d*\.?\d+)"
        r"V(?P<gasV>[-+]?\d*\.?\d+)",
        re.IGNORECASE
    )
    m = pattern.search(name)
    if m:
        return {
            "steamT": float(m.group("steamT")),
            "steamV": float(m.group("steamV")),
            "gasT": float(m.group("gasT")),
            "gasV": float(m.group("gasV")),
        }

    nums = re.findall(r"[-+]?\d*\.?\d+", name)
    if len(nums) >= 4:
        return {
            "steamT": float(nums[0]),
            "steamV": float(nums[1]),
            "gasT": float(nums[2]),
            "gasV": float(nums[3]),
        }

    return None


# =========================================================
# 2. 读取原始真实文件
# =========================================================
def read_one_file(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # 强制转换为数值并剔除空行
    for col in ["x", "y", "z", "temp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x", "y", "z", "temp"]).reset_index(drop=True)
    
    # 关键：必须执行与训练代码完全一致的排序逻辑
    df = df.sort_values(by=["x", "y", "z"]).reset_index(drop=True)
    return df


# =========================================================
# 3. 工况匹配真实文件
# =========================================================
def is_close(a, b, tol=MATCH_TOL):
    return abs(a - b) <= tol


def find_true_file(data_dir, steamT, steamV, gasT, gasV):
    files = []
    files += glob.glob(os.path.join(data_dir, "*.csv"))
    files += glob.glob(os.path.join(data_dir, "*.xlsx"))
    files += glob.glob(os.path.join(data_dir, "*.xls"))

    for fp in files:
        cond = parse_conditions_from_filename(fp)
        if cond is None:
            continue

        if (
            is_close(cond["steamT"], steamT) and
            is_close(cond["steamV"], steamV) and
            is_close(cond["gasT"], gasT) and
            is_close(cond["gasV"], gasV)
        ):
            return fp

    return None


# =========================================================
# 4. 模型预测
# =========================================================
def predict_temperature(model_bundle, ref_points, steamT, steamV, gasT, gasV):
    x = np.array([[steamT, steamV, gasT, gasV]], dtype=float)

    x_s = model_bundle["x_scaler"].transform(x)
    c_pred_s = model_bundle["regressor"].predict(x_s)
    c_pred = model_bundle["c_scaler"].inverse_transform(c_pred_s)

    temp_pred = model_bundle["mean_field"][None, :] + c_pred @ model_bundle["basis"]
    temp_pred = temp_pred.ravel()

    pred_df = ref_points.copy()
    pred_df["temp_pred"] = temp_pred
    return pred_df


# =========================================================
# 5. 检查点位是否一致
# =========================================================
def check_same_points(pred_df: pd.DataFrame, true_df: pd.DataFrame):
    if len(pred_df) != len(true_df):
        return False

    seg_equal = np.array_equal(pred_df["x"].values, true_df["x"].values)
    l_equal = np.allclose(pred_df["y"].values, true_df["y"].values, atol=POINT_TOL, rtol=0)
    z_equal = np.allclose(pred_df["z"].values, true_df["z"].values, atol=POINT_TOL, rtol=0)

    return seg_equal and l_equal and z_equal


# =========================================================
# 6. 构建输出结果
# =========================================================
def build_pred_only_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    result = pred_df.copy()
    return result


def build_compare_table(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.DataFrame:
    result = pred_df.copy()
    result["temp_true"] = true_df["temp"].values
    result["abs_error"] = np.abs(result["temp_pred"] - result["temp_true"])

    eps = 1e-12
    result["rel_error_percent"] = result["abs_error"] / (np.abs(result["temp_true"]) + eps) * 100.0
    return result


# =========================================================
# 7. 打印整体误差
# =========================================================
def print_global_metrics(result_df: pd.DataFrame):
    temp_pred = result_df["temp_pred"].values
    temp_true = result_df["temp_true"].values

    diff = temp_pred - temp_true
    abs_error = np.abs(diff)

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(abs_error)
    max_ae = np.max(abs_error)
    mape = np.mean(abs_error / (np.abs(temp_true) + 1e-12)) * 100.0

    print("\n================= 误差统计 =================")
    print(f"点数           : {len(result_df)}")
    print(f"RMSE           : {rmse:.6f}")
    print(f"MAE            : {mae:.6f}")
    print(f"最大绝对误差   : {max_ae:.6f}")
    print(f"平均相对误差%  : {mape:.6f}")
    print("==========================================\n")


# =========================================================
# 8. 生成输出文件名
# =========================================================
def make_safe_number_str(x):
    s = str(float(x))
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


def build_output_filename(prefix, steamT, steamV, gasT, gasV):
    return (
        f"{prefix}_"
        f"steamT{make_safe_number_str(steamT)}_"
        f"steamV{make_safe_number_str(steamV)}_"
        f"gasT{make_safe_number_str(gasT)}_"
        f"gasV{make_safe_number_str(gasV)}.csv"
    )


# =========================================================
# 9. 主程序
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("POD 模型测试程序（自动匹配真实数据）")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}")

    saved_obj = joblib.load(MODEL_PATH)

    if "model_bundle" in saved_obj and "reference_points" in saved_obj:
        model_bundle = saved_obj["model_bundle"]
        ref_points = saved_obj["reference_points"]
    else:
        raise ValueError("模型文件格式不正确，未找到 model_bundle / reference_points。")

    print("请输入 4 个初始工况：")
    steamT = float(input("steamT = ").strip())
    steamV = float(input("steamV = ").strip())
    gasT = float(input("gasT   = ").strip())
    gasV = float(input("gasV   = ").strip())

    # 1) 先做预测
    pred_df = predict_temperature(
        model_bundle=model_bundle,
        ref_points=ref_points,
        steamT=steamT,
        steamV=steamV,
        gasT=gasT,
        gasV=gasV
    )

    # 2) 自动匹配真实文件
    true_file = find_true_file(DATA_DIR, steamT, steamV, gasT, gasV)

    # 3) 匹配失败：只输出预测温度
    if true_file is None:
        print("\n未匹配到真实工况文件，只输出各点预测温度。")

        result_df = build_pred_only_table(pred_df)
        out_name = build_output_filename("pred_only", steamT, steamV, gasT, gasV)
        out_path = os.path.join(OUT_DIR, out_name)
        result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"结果已保存到：\n{out_path}")
        print("输出列为：x, y, z, temp_pred")
        return

    print(f"\n已匹配到真实文件：\n{true_file}")

    # 4) 读取真实文件
    true_df = read_one_file(true_file)

    # 5) 点位再校验
    if not check_same_points(pred_df, true_df):
        print("\n已找到真实文件，但点位与模型参考点不一致。")
        print("因此无法计算误差，只输出预测温度。")

        result_df = build_pred_only_table(pred_df)
        out_name = build_output_filename("pred_only_point_mismatch", steamT, steamV, gasT, gasV)
        out_path = os.path.join(OUT_DIR, out_name)
        result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"结果已保存到：\n{out_path}")
        print("输出列为：x, y, z, temp_pred")
        return

    # 6) 匹配成功：输出误差
    print("\n匹配成功，开始计算各点误差...")
    result_df = build_compare_table(pred_df, true_df)

    print_global_metrics(result_df)

    out_name = build_output_filename("compare_result", steamT, steamV, gasT, gasV)
    out_path = os.path.join(OUT_DIR, out_name)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"结果已保存到：\n{out_path}")
    print("输出列为：x, y, z, temp_pred, temp_true, abs_error, rel_error_percent")


if __name__ == "__main__":
    main()