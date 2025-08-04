import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from optimizer import optimize_r


def worker_function(eps_params, fixed_args):
    """
    这个函数负责计算外层网格中的【一个点】。
    它被设计成可以被多进程调用的形式。

    :param eps_params: 一个元组，包含变化的参数 (eps1, eps2)
    :param fixed_args: 一个字典，包含所有固定的参数 (alpha, beta, etc.)
    :return: 一个字典，包含该点的完整计算结果
    """
    eps1, eps2 = eps_params

    # 调用内层网格搜索
    r1_opt, r2_opt, max_reward = optimize_r(
        alpha=fixed_args['alpha'],
        beta=fixed_args['beta'],
        eta=fixed_args['eta'],
        gamma=fixed_args['gamma'],
        grid_density=fixed_args['grid_density']
    )

    return {
        'eps1': eps1,
        'eps2': eps2,
        'r1_opt': r1_opt,
        'r2_opt': r2_opt,
        'max_reward': max_reward
    }


if __name__ == '__main__':
    # --- 1. 定义实验参数 ---
    # 固定参数
    fixed_params = {
        'alpha': 0.2,
        'beta': 0.2,
        'eta': 0.2,
        'gamma': 0.5,
        'grid_density': 101  # 内层(r1,r2)网格密度。51比较快，101更精确
    }

    # 外层网格参数 (eps1, eps2)
    eps_grid_density = 101  # 外层网格密度。21x21=441个点。可以根据需要调整
    epsilon1_range = np.linspace(0, 1, eps_grid_density)
    epsilon2_range = np.linspace(0, 1, eps_grid_density)  # eps2可能需要大于1

    # 创建所有任务参数的列表
    tasks = [(eps1, eps2) for eps2 in epsilon2_range for eps1 in epsilon1_range]

    # --- 2. 执行多进程计算 ---
    # 使用除一个核心外的所有核心，避免电脑卡死
    num_cores = max(1, cpu_count() - 1)
    # 使用 partial 将固定参数绑定到工作函数上
    worker_with_fixed_args = partial(worker_function, fixed_args=fixed_params)

    print(f"开始在 {num_cores} 个CPU核心上执行 {len(tasks)} 个嵌套网格搜索任务...")
    print(
        f"外层网格: {eps_grid_density}x{eps_grid_density}, 内层网格: {fixed_params['grid_density']}x{fixed_params['grid_density']}")

    results = []
    with Pool(num_cores) as p:
        # 使用tqdm显示进度条
        for result in tqdm(p.imap_unordered(worker_with_fixed_args, tasks), total=len(tasks)):
            results.append(result)

    print("所有计算任务完成。")

    # --- 3. 保存结果到文件 ---
    # 将结果列表转换为pandas DataFrame
    results_df = pd.DataFrame(results)

    # 按照eps1和eps2排序，方便查看
    results_df = results_df.sort_values(by=['eps2', 'eps1']).reset_index(drop=True)

    output_filename = 'nested_search_results.csv'
    results_df.to_csv(output_filename, index=False)

    print(f"结果已成功保存到文件: {output_filename}")
    print("\n文件内容预览:")
    print(results_df.head())