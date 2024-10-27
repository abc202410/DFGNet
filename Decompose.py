import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD

def SaveAndPrintIMFs(SourcePath, TargetPath, params):
    """
    保存并打印IMF分量
    :param SourcePath: 原csv文件路径
    :param TargetPath: 目标csv文件路径
    :param params: 要分解的变量的表头、K、α、tau
    :return:
    """
    data = pd.read_csv(SourcePath)

    # x = data[params[0]].values[:10000]
    x = data[params[0]].values[:params[4]] # 数据长度根据需求修改

    # VMD分解参数
    K = int(params[1])  # 分解模态（IMF）个数
    alpha = int(params[2])  # 带宽限制参数，用于控制每个固有模式（IMF）的带宽。较小的alpha值会导致更宽的带宽，而较大的值会导致较窄的带宽。
    tau = params[3] # 噪声容限参数，指定允许分解后的信号与原始信号之间的差异。较小的tau值会允许较大的差异，较大的值会限制差异。
    DC = 1  # 是否保留直流分量（或趋势项），设置为0表示不保留，设置为1表示保留。
    init = 1  # IMF中心频率初始化标志，设置为1会进行均匀初始化，即每个IMF的中心频率均匀分布在频率范围内。
    tol = 1e-7  # 控制误差大小的常数，决定分解的精度和迭代次数.

    # 进行VMD分解
    u, u_hat, omega = VMD(x, alpha, tau, K, DC, init, tol)
    # 创建列名列表
    column_names = [f"IMF{i + 1}" for i in range(K)]

    # 将分解后的各个IMF分量保存到DataFrame中
    imf_df = pd.DataFrame(u.T, columns=column_names)

    # 将DataFrame保存为CSV文件
    imf_df.to_csv(TargetPath, index=False)

    # 绘制分解结果图像
    t = np.arange(len(x))
    # 设置字体为微软雅黑
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.figure(figsize=(10, 7))
    plt.subplot(K + 1, 1, 1)
    plt.plot(t, x, color='blue', label='原始信号')
    plt.legend()
    for i, mode in enumerate(u):
        color = plt.cm.jet(float(i) / K)
        plt.subplot(K + 1, 1, i + 2)
        plt.plot(t, mode, color=color, label=f'IMF{i + 1}')
        plt.legend()
    plt.suptitle('原始输入信号及其分量')
    plt.show()
