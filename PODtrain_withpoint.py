# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:01:44 2026

@author: windows11

U形管外表面温度场 POD 建模程序（只输出训练好的模型）
====================================================

功能：
1. 读取原始数据文件夹中的 249 组 csv/xlsx/xls 文件
2. 从文件名中提取 4 个初始工况
3. 检查所有文件的点位(x, y, z)是否一致
4. 按 80% / 20% 划分训练集与测试集
5. 在训练集内部再划分训练子集/验证子集，自动搜索较优超参数
6. 用最优参数重新训练模型
7. 用测试集做最终评估
8. 只保存训练好的模型文件到原始数据文件夹下新建的模型文件夹中

模型结构：
初始工况(4维) -> MLP回归 -> POD系数 -> 重构温度场

依赖：
pip install numpy pandas scikit-learn joblib openpyxl
"""

import os
import re
import glob
import joblib
import warnings
import numpy as np
import pandas as pd

from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# =========================================================
# 0. 配置区
# =========================================================
DATA_DIR = r"D:\DeepCFD2025\CNN\data"   # 原始数据文件夹
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MEASURE_POINTS_PATH = r"D:\DeepCFD2025\jnjpcode\POD\measure_point\measure_point.xlsx" #5个测点坐标
MODEL_DIR = os.path.join(CURRENT_DIR, "pod_model")
FILE_GLOB_CSV = r"*.csv"
FILE_GLOB_XLSX = r"*.xlsx"
FILE_GLOB_XLS = r"*.xls"


# 外层划分：训练集 / 测试集
TEST_RATIO = 0.2
RANDOM_SEED = 42

# 内层划分：从训练集再分出验证集，用于超参数选择
VAL_RATIO_IN_TRAIN = 0.2

# POD累计能量阈值候选
ENERGY_THRESHOLD_CANDIDATES = [0.995, 0.999, 0.9995]

# MLP超参数候选
MLP_HIDDEN_CANDIDATES = [
    (64, 64),
    (128, 64),
    (128, 128),
    (256, 128)
]
ALPHA_CANDIDATES = [1e-5, 1e-4, 1e-3]
LEARNING_RATE_INIT_CANDIDATES = [1e-3, 5e-4]

MLP_MAX_ITER = 4000

# 点位一致性判断容差
L_TOL = 1e-10
THETA_TOL = 1e-10


# =========================================================
# 1. 文件名中提取4个工况
# =========================================================
def parse_conditions_from_filename(filename: str) -> Dict[str, float]:
    """
    默认支持类似：
        steamT690V0.54gasT1236V12.11_raw.csv

    解析结果：
        steamT = 690
        steamV = 0.54
        gasT   = 1236
        gasV   = 12.11
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

    # 宽松模式：提取前4个数字
    nums = re.findall(r"[-+]?\d*\.?\d+", name)
    if len(nums) >= 4:
        return {
            "steamT": float(nums[0]),
            "steamV": float(nums[1]),
            "gasT": float(nums[2]),
            "gasV": float(nums[3]),
        }

    raise ValueError(f"无法从文件名解析4个工况参数：{filename}")


# =========================================================
# 2. 读取单个文件
# =========================================================
def read_one_file(filepath: str) -> pd.DataFrame:
    """
    读取一个 csv/xlsx/xls 文件
    必须包含列：x, y, z, temp
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的文件类型：{filepath}")

    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["x", "y", "z", "temp"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{filepath} 缺少必要列：{col}")

    df = df[required_cols].copy()

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

    df = df.dropna(subset=["x", "y", "z", "temp"]).reset_index(drop=True)
 

    # 排序确保所有样本向量化顺序完全一致
    df = df.sort_values(by=["x", "y", "z"], ascending=[True, True, True]).reset_index(drop=True)

    return df


# =========================================================
# 3. 点位检查
# =========================================================
def build_reference_points(df: pd.DataFrame) -> pd.DataFrame:
    return df[["x", "y", "z"]].copy().reset_index(drop=True)


def check_same_points(df_ref: pd.DataFrame, df_cur: pd.DataFrame, filepath: str):
    if len(df_ref) != len(df_cur):
        raise ValueError(
            f"{filepath} 点数与参考文件不一致：参考={len(df_ref)}，当前={len(df_cur)}"
        )

    seg_equal = np.array_equal(df_ref["x"].values, df_cur["x"].values)
    l_equal = np.allclose(df_ref["y"].values, df_cur["y"].values, atol=L_TOL, rtol=0)
    theta_equal = np.allclose(df_ref["z"].values, df_cur["z"].values, atol=THETA_TOL, rtol=0)

    if not (seg_equal and l_equal and theta_equal):
        raise ValueError(f"{filepath} 的点位(x, y, z)与参考文件不一致，无法直接做POD。")


# =========================================================
# 4. 加载整个数据集
# =========================================================
def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    file_list = []
    file_list += glob.glob(os.path.join(data_dir, FILE_GLOB_CSV))
    file_list += glob.glob(os.path.join(data_dir, FILE_GLOB_XLSX))
    file_list += glob.glob(os.path.join(data_dir, FILE_GLOB_XLS))
    file_list = sorted(file_list)

    if len(file_list) == 0:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 csv/xlsx/xls 文件")

    X_list = []
    Y_list = []
    ref_points = None

    print(f"共找到 {len(file_list)} 个文件，开始读取...")

    for i, fp in enumerate(file_list):
        cond = parse_conditions_from_filename(fp)
        df = read_one_file(fp)

        if i == 0:
            ref_points = build_reference_points(df)
        else:
            check_same_points(ref_points, df, fp)

        x = np.array([cond["steamT"], cond["steamV"], cond["gasT"], cond["gasV"]], dtype=float)
        y = df["temp"].values.astype(np.float64)

        X_list.append(x)
        Y_list.append(y)

        if (i + 1) % 20 == 0 or (i + 1) == len(file_list):
            print(f"已读取 {i + 1}/{len(file_list)} 个文件")

    X = np.vstack(X_list)   # (n_samples, 4)
    Y = np.vstack(Y_list)   # (n_samples, n_points)

    return X, Y, ref_points, file_list


# =========================================================
# 5. POD
# =========================================================
def compute_pod(Y_train: np.ndarray, energy_threshold: float):
    """
    输入：
        Y_train: (n_samples, n_points)

    返回：
        mean_field: (n_points,)
        basis:      (n_modes, n_points)
        coeffs:     (n_samples, n_modes)
        n_modes
        captured_energy
    """
    mean_field = Y_train.mean(axis=0)
    Yc = Y_train - mean_field[None, :]

    # SVD
    U, S, VT = np.linalg.svd(Yc, full_matrices=False)

    energy = S ** 2
    energy_ratio = energy / np.sum(energy)
    energy_cumsum = np.cumsum(energy_ratio)

    n_modes = int(np.searchsorted(energy_cumsum, energy_threshold) + 1)
    basis = VT[:n_modes, :]
    coeffs = Yc @ basis.T

    captured_energy = float(energy_cumsum[n_modes - 1])

    return mean_field, basis, coeffs, n_modes, captured_energy


# =========================================================
# 6. 评价函数
# =========================================================
def field_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))


def build_and_eval_one_model(
    X_train_sub: np.ndarray,
    Y_train_sub: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    energy_threshold: float,
    hidden_layers: tuple,
    alpha: float,
    lr_init: float,
    random_seed: int = 42
):
    """
    在训练子集上训练，在验证集上评估
    返回验证误差和模型组件
    """
    mean_field, basis, C_train_sub, n_modes, captured_energy = compute_pod(
        Y_train_sub, energy_threshold=energy_threshold
    )

    # 输入和POD系数标准化
    x_scaler = StandardScaler()
    c_scaler = StandardScaler()

    X_train_sub_s = x_scaler.fit_transform(X_train_sub)
    X_val_s = x_scaler.transform(X_val)

    C_train_sub_s = c_scaler.fit_transform(C_train_sub)

    reg = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=MLP_MAX_ITER,
        random_state=random_seed,
        early_stopping=True,
        validation_fraction=0.15
    )
    reg.fit(X_train_sub_s, C_train_sub_s)

    # 验证集预测
    C_val_pred_s = reg.predict(X_val_s)
    C_val_pred = c_scaler.inverse_transform(C_val_pred_s)
    Y_val_pred = mean_field[None, :] + C_val_pred @ basis

    val_rmse = field_rmse(Y_val, Y_val_pred)

    bundle = {
        "mean_field": mean_field,
        "basis": basis,
        "x_scaler": x_scaler,
        "c_scaler": c_scaler,
        "regressor": reg,
        "n_modes": n_modes,
        "captured_energy": captured_energy,
        "energy_threshold": energy_threshold,
        "hidden_layers": hidden_layers,
        "alpha": alpha,
        "learning_rate_init": lr_init
    }

    return val_rmse, bundle


# =========================================================
# 7. 搜索最佳超参数
# =========================================================
def search_best_hyperparams(X_train: np.ndarray, Y_train: np.ndarray):
    """
    在训练集内部划分训练子集/验证子集，搜索较优参数
    """
    X_train_sub, X_val, Y_train_sub, Y_val = train_test_split(
        X_train, Y_train,
        test_size=VAL_RATIO_IN_TRAIN,
        random_state=RANDOM_SEED
    )

    best_rmse = np.inf
    best_bundle = None
    best_info = None
    trial_id = 0

    total_trials = (
        len(ENERGY_THRESHOLD_CANDIDATES)
        * len(MLP_HIDDEN_CANDIDATES)
        * len(ALPHA_CANDIDATES)
        * len(LEARNING_RATE_INIT_CANDIDATES)
    )

    print(f"开始超参数搜索，共 {total_trials} 组...")

    for energy_threshold in ENERGY_THRESHOLD_CANDIDATES:
        for hidden_layers in MLP_HIDDEN_CANDIDATES:
            for alpha in ALPHA_CANDIDATES:
                for lr_init in LEARNING_RATE_INIT_CANDIDATES:
                    trial_id += 1
                    print(
                        f"[{trial_id}/{total_trials}] "
                        f"energy={energy_threshold}, hidden={hidden_layers}, "
                        f"alpha={alpha}, lr={lr_init}"
                    )

                    try:
                        val_rmse, bundle = build_and_eval_one_model(
                            X_train_sub=X_train_sub,
                            Y_train_sub=Y_train_sub,
                            X_val=X_val,
                            Y_val=Y_val,
                            energy_threshold=energy_threshold,
                            hidden_layers=hidden_layers,
                            alpha=alpha,
                            lr_init=lr_init,
                            random_seed=RANDOM_SEED
                        )

                        print(f"    验证集 RMSE = {val_rmse:.6f}")

                        if val_rmse < best_rmse:
                            best_rmse = val_rmse
                            best_bundle = bundle
                            best_info = {
                                "val_rmse": val_rmse,
                                "energy_threshold": energy_threshold,
                                "hidden_layers": hidden_layers,
                                "alpha": alpha,
                                "learning_rate_init": lr_init,
                                "n_modes": bundle["n_modes"],
                                "captured_energy": bundle["captured_energy"]
                            }
                            print("    -> 当前最优")
                    except Exception as e:
                        print(f"    该组失败：{e}")

    if best_bundle is None:
        raise RuntimeError("超参数搜索失败，没有得到可用模型。")

    print("超参数搜索完成。")
    print("最优参数：", best_info)
    return best_info


# =========================================================
# 8. 用最优参数在完整训练集上重新训练
# =========================================================
def train_final_model(X_train: np.ndarray, Y_train: np.ndarray, best_info: dict):
    mean_field, basis, C_train, n_modes, captured_energy = compute_pod(
        Y_train, energy_threshold=best_info["energy_threshold"]
    )

    x_scaler = StandardScaler()
    c_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    C_train_s = c_scaler.fit_transform(C_train)

    reg = MLPRegressor(
        hidden_layer_sizes=best_info["hidden_layers"],
        activation="relu",
        solver="adam",
        alpha=best_info["alpha"],
        learning_rate_init=best_info["learning_rate_init"],
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.15
    )
    reg.fit(X_train_s, C_train_s)

    model_bundle = {
        "mean_field": mean_field,
        "basis": basis,
        "x_scaler": x_scaler,
        "c_scaler": c_scaler,
        "regressor": reg,
        "n_modes": n_modes,
        "captured_energy": captured_energy,
        "best_hyperparams": best_info,
        "data_dir": DATA_DIR
    }

    return model_bundle


# =========================================================
# 9. 测试集评估
# =========================================================
def evaluate_on_test(model_bundle: dict, X_test: np.ndarray, Y_test: np.ndarray) -> float:
    X_test_s = model_bundle["x_scaler"].transform(X_test)
    C_test_pred_s = model_bundle["regressor"].predict(X_test_s)
    C_test_pred = model_bundle["c_scaler"].inverse_transform(C_test_pred_s)
    Y_test_pred = model_bundle["mean_field"][None, :] + C_test_pred @ model_bundle["basis"]

    test_rmse = field_rmse(Y_test, Y_test_pred)
    return test_rmse


# =========================================================
# 10. 保存模型
# =========================================================
def save_model(model_bundle: dict, ref_points: pd.DataFrame):
    os.makedirs(MODEL_DIR, exist_ok=True)

    save_dict = {
        "model_bundle": model_bundle,
        "reference_points": ref_points
    }

    model_path = os.path.join(MODEL_DIR, "best_pod_withpoint_temp_model.joblib")
    joblib.dump(save_dict, model_path)

    print(f"\n训练好的模型已保存到：\n{model_path}")


# =========================================================
# 11. 主函数
# =========================================================
def main():
    print("=" * 72)
    print("U形管外表面温度场 POD 模型训练开始")
    print("=" * 72)

    # 1) 读取数据
    X, Y, ref_points, file_list = load_dataset(DATA_DIR)

    print("-" * 72)
    print(f"样本总数: {X.shape[0]}")
    print(f"输入维度: {X.shape[1]}")
    print(f"每个样本点数: {Y.shape[1]}")
    print("-" * 72)

    # 2) 划分训练集 / 测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )

    print(f"训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")

    # 3) 超参数搜索
    best_info = search_best_hyperparams(X_train, Y_train)

    # 4) 用最优参数在完整训练集上重新训练
    print("\n开始用最优参数在完整训练集上重新训练...")
    model_bundle = train_final_model(X_train, Y_train, best_info)
    # --- 修正点：将测点索引直接塞进 model_bundle ---
    from sklearn.neighbors import BallTree
    # 这里的路径对应你上传的 csv 路径
    MEASURE_POINTS_PATH = r"D:\DeepCFD2025\jnjpcode\POD\measure_point\measure_point.xlsx"
    measure_df = pd.read_excel(MEASURE_POINTS_PATH)
    
    # 建立搜索树定位 5 个点
    tree = BallTree(ref_points[['x', 'y', 'z']].values)
    _, m_idx = tree.query(measure_df[['x', 'y', 'z']].values, k=1)
    
    # 直接作为新键值对存入 model_bundle
    model_bundle["measure_indices"] = m_idx.flatten() 
    print(f"测点接口预埋成功，索引：{model_bundle['measure_indices'].tolist()}")
    # ----------------------------------------------
    

    # 5) 测试集最终评估
    print("开始在测试集上评估最终模型...")
    test_rmse = evaluate_on_test(model_bundle, X_test, Y_test)
    print(f"最终测试集 RMSE = {test_rmse:.6f}")

    model_bundle["final_test_rmse"] = test_rmse
    model_bundle["n_train_samples"] = int(X_train.shape[0])
    model_bundle["n_test_samples"] = int(X_test.shape[0])
    model_bundle["n_total_samples"] = int(X.shape[0])
    model_bundle["n_points"] = int(Y.shape[1])
    model_bundle["source_files"] = [os.path.basename(f) for f in file_list]

    # 6) 只保存训练好的模型
    save_model(model_bundle, ref_points)

    print("=" * 72)
    print("全部完成")
    print("=" * 72)


if __name__ == "__main__":
    main()