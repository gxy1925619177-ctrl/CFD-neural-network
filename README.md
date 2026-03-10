项目名称：基于 POD-MLP 的换热器壁面温度场自适应重构系统


项目概述

本项目提供了一套用于换热器表面热分布快速预测与实时修正的计算框架。该方案通过本征正交分解（POD）提取流体机械受热面的空间特征，并利用多层感知器（MLP）深度学习模型建立工况参数与模态系数之间的非线性映射。系统特别集成了基于物理测点反馈的自适应优化算法，通过极少量壁面传感器数据即可实现高自由度全场温度分布的精准重构。


主要技术特征

降维重构模型：利用 POD 技术将拥有 81,080 个节点的复杂三维空间数据压缩至低维流形，大幅降低计算资源消耗。

自适应修正逻辑：引入实时反馈机制，通过最小化传感器实测值与预测值之间的残差，动态调整 POD 系数，增强了模型在变工况下的泛化能力。

严格解析变换：内置坐标转换模块，支持将换热管束的三维复杂几何精确展开为二维平面坐标（L-Theta 体系），解决了热物理过程中的可视化难题。


一、项目结构
```text
|- data/                (原始数据存放区)
|- measure_point/       (测点配置与数据分析)
|- pod_model/           (训练好的模型文件)
|- test_result/         (评估报告与可视化图表)
|
|- 标准 POD 建模分支 (Standard POD Workflow)
|  |--- PODtrain.py     (基础训练程序)
|  |--- PODtest.py      (基础预测程序)
|  |--- PODtest2.py     (批量评估工具)
|
|- 自适应测点修正分支 (Adaptive POD-with-Point Workflow)
|  |--- PODtrain_withpoint.py (进阶训练：融合测点信息)
|  |--- POD_test_withpoint_RL.py (核心自适应寻优评估)
|  |--- PODtest_env.py        (仿真环境与自适应评估集成工具)
|  |--- POD_visual.py         (可视化系统)
|
|- 公共算法模块
   |--- coordinate_transform_strict.py (3D 到 2D 几何解析展开)
```


二、 模块功能分类说明

（1）测点配置与数据分析模块 (measure_point/)

该模块负责测点物理信息的管理与初步的数据特征探索，为自适应修正提供数据底座。

数据配置：包含 measure_point.xlsx（测点三维物理坐标）及 all_points_data.xlsx（各测点历史温度数据集）。

空间可视化：提供 measure_point_position.fig 与 .png，直观展示传感器在 U 型管上的具体分布位置。

数据分析工具：

temp_range.py：负责从全量数据中提取 5 个核心测点的数据，进行统计分析，确定各点的温度变化范围。

env_random_temp.py：计算变量间（如 SteamT/GasV 与温度）的相关性，并分析预测残差特征，用于揭示测点间的耦合物理规律。


（2）标准 POD 建模分支 (Standard POD Workflow)
该分支专注于基础数据驱动建模，主要服务于无实时反馈下的常规工况预测任务。

PODtrain.py：训练基础 POD-MLP 模型。

PODtest.py：执行单工况推演与对比。

PODtest2.py：自动化批量处理工具，生成性能评估报告。


（3）自适应测点修正分支 (Adaptive POD-with-Point Workflow)
该分支引入了在线修正机制，通过数学优化算子动态调整模型系数。

PODtrain_withpoint.py：在建模阶段融入测点信息。

POD_test_withpoint_RL.py：核心修正引擎，模拟实时传感器输入，通过数学寻优在线微调 POD 系数。

PODtest_env.py：基于 VirtualPlant 类的集成仿真环境。它能够根据参数自动生成带有“物理相关性噪声”的传感器反馈，用于对自适应算法进行更接近工业现场的压力测试。

POD_visual.py：可视化展示系统。


三、环境配置要求
运行环境：Python 3.8 及以上版本。

核心依赖：NumPy, Pandas, Matplotlib, Scikit-learn, SciPy, Joblib。

四、引用说明
如果您在学术研究中使用了本项目的代码或方法，请引用如下信息：

[谷雪莹], "基于 POD-MLP 的发电动力设备壁面热场自适应预测研究", 2026. GitHub 仓库地址: [ https://github.com/gxy1925619177-ctrl/CFD-neural-network.git]。
