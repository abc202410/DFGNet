import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import opfunu
from mealpy import get_optimizer_by_name
import matplotlib.cm as cm


output_folder = r""
os.makedirs(output_folder, exist_ok=True)

# 算法名称列表
algorithm_names = [

    "SLO", "EOA", "SCA", "ArchOA", "ASO", "WaOA", "TSO", "ARO", "CircleSA"
    ]


for algorithm_name in algorithm_names:

    # 为每个算法创建一个单独的文件夹
    algorithm_name_output_folder = os.path.join(output_folder, f"{algorithm_name}")
    os.makedirs(algorithm_name_output_folder, exist_ok=True)

    # 每个算法都执行一遍F1-F29,其中每个函数执行30遍
    for func_num in range(1, 30):
        dim = 30  # 问题维度
        epoch = 200  # 最大迭代次数
        pop_size = 50  # 种群数量
        print('=====================>>>>>>>>>>>>>>>>>>>' + f"F{func_num}-2017")

        # 定义CEC函数
        def cec_fun(x):
            funcs = opfunu.get_functions_by_classname(f"F{func_num}2017")
            func = funcs[0](ndim=dim)
            F = func.evaluate(x)
            return F

        # 定义问题字典
        problem_dict = {
            "fit_func": cec_fun,
            "lb": opfunu.get_functions_by_classname(f"F{func_num}2017")[0](ndim=dim).lb.tolist(),
            "ub": opfunu.get_functions_by_classname(f"F{func_num}2017")[0](ndim=dim).ub.tolist(),
            "minmax": "min",
        }

        # 设置循环次数
        num_iterations = 30
        # 存储每次迭代的最佳Fitness
        fitness_results = []
        for i in range(num_iterations):
            if algorithm_name == "DE":
                optimizer_model = get_optimizer_by_name("BaseDE")(epoch, pop_size)
            else:
                optimizer_model = get_optimizer_by_name(f"Original{algorithm_name}")(epoch, pop_size)
            '''求解 cec函数 '''
            best_x, best_f = optimizer_model.solve(problem_dict)
            print(f"Iteration {i + 1}: Solution: {best_x}, Fitness: {best_f}")
            # 存储最佳Fitness
            fitness_results.append(best_f)
        # 将Fitness结果保存到CSV文件
        fitness_df = pd.DataFrame({"Fitness": fitness_results})
        fitness_df.to_csv(algorithm_name_output_folder+f"\{algorithm_name}-F{func_num}-循环30次结果.csv",index=False)