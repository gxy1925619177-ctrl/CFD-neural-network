# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:04:20 2026

@author: windows11
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 1. 读取数据
df = pd.read_excel("all_points_data.xlsx", sheet_name="Point_0")
features = ['SteamT', 'SteamV', 'GasT', 'GasV']

# 2. 绘制相关性热图
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 3. 训练模型并分析误差
model = RandomForestRegressor(n_estimators=100)
model.fit(df[features], df['Temp'])

# 预测并计算残差
df['Pred'] = model.predict(df[features])
df['Residual'] = df['Temp'] - df['Pred']

# 4. 计算 3-sigma 波动区间
std_err = df['Residual'].std()
print(f"该测点的预测误差标准差为: {std_err:.4f}")
print(f"合理的波动区间参考: Pred ± {2 * std_err:.4f}")

# 5. 可视化实际值与预测值 (观察异常点)
plt.figure(figsize=(8, 4))
plt.scatter(df.index, df['Temp'], label='Actual', alpha=0.5)
plt.plot(df.index, df['Pred'], color='red', label='Trend (Model)')
plt.fill_between(df.index, df['Pred'] - 2*std_err, df['Pred'] + 2*std_err, color='red', alpha=0.2, label='Reasonable Range')
plt.legend()
plt.show()