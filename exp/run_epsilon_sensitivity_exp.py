import numpy as np
from tqdm import tqdm
from simulator import Simulator
from visualization import plot_heatmap
from multiprocessing import Pool, cpu_count
from functools import partial

def run_epsilon_sensitivity_exp(params, fixed_args):
    """
    这个函数只负责计算热力图中的【一个点】。
    它被设计成可以被多进程调用的形式。

    :param params: 一个元组，包含变化的参数 (eps1, eps2)
    :param fixed_args: 一部字典，包含所有固定的参数
    :return: 一个元组，包含计算结果 (eps1, eps2, rer_a, rer_t)
    """
    eps1, eps2 = params

    # 1. 创建模拟器实例
    simulator = Simulator(
        alpha=fixed_args['alpha'],
        beta=fixed_args['beta'],
        eta=fixed_args['eta'],
        gamma=fixed_args['gamma'],
        epsilon1=eps1,
        epsilon2=eps2,
        grid_density=fixed_args['grid_density']
    )

    # 2. 运行模拟
    avg_r_a_bm_paw, avg_r_t_bm_paw, avg_r_a_paw, avg_r_t_paw = simulator.run_simulation(num_rounds=fixed_args['num_rounds'])

    # 3. 计算RER
    rer_a = (avg_r_a_bm_paw - avg_r_a_paw) / avg_r_a_paw if avg_r_a_paw > 0 else 0

    # 我们也顺便计算一下rer_t
    rer_t = (avg_r_t_bm_paw - avg_r_t_paw) / avg_r_t_paw if avg_r_t_paw > 0 else 0

    return eps1, eps2, rer_a, rer_t

if __name__ == '__main__':
    # --- 1. 定义实验参数 ---
    fixed_params = {
        'alpha': 0.2, 'beta': 0.2, 'eta': 0.2, 'gamma': 0.5,
        'num_rounds': 10000,
        'grid_density':101
    }

    # 变化的参数 (网格密度，可以调整以平衡速度和精度)
    grid_density = 200 # 50x50的网格，总共2500次模拟。如果想更快，可以改成20
    epsilon1_range = np.linspace(0, 1, grid_density)
    epsilon2_range = np.linspace(0, 1, grid_density)
    # 创建所有任务参数的列表
    tasks = [(eps1, eps2) for eps2 in epsilon2_range for eps1 in epsilon1_range]
    # 使用除一个核心外的所有核心，避免电脑卡死
    num_cores = max(1, cpu_count() - 1)
    worker_func = partial(run_epsilon_sensitivity_exp, fixed_args=fixed_params)

    print(f"开始在 {num_cores} 个CPU核心上执行 {len(tasks)} 个实验点...")
    with Pool(num_cores) as p:
        # p.imap_unordered 会立即返回一个迭代器，tqdm可以很好地包裹它来显示进度
        results = list(tqdm(p.imap_unordered(worker_func, tasks), total=len(tasks)))
    print("所有实验点计算完成。")
    # --- 2. 整理实验结果 ---
    rer_a_grid = np.zeros((grid_density, grid_density))
    eps1_map = {val: i for i, val in enumerate(epsilon1_range)}
    eps2_map = {val: i for i, val in enumerate(epsilon2_range)}

    for eps1, eps2, rer_a, rer_t in results:
        i = eps2_map[eps2]
        j = eps1_map[eps1]
        rer_a_grid[i, j] = rer_a
    # --- 3. 绘制图表 ---
    plot_heatmap(
        x_range=epsilon1_range,
        y_range=epsilon2_range,
        data_grid=rer_a_grid,
        title="Attacker's RER as a function of Bribe Ratios",
        x_label=r'Bribe Ratio in Case 5-2 ($\epsilon_1$)',  # 使用LaTeX语法渲染漂亮的希腊字母
        y_label=r'Bribe Ratio in Case 5-4 ($\epsilon_2$)',
        colorbar_label='Relative Extra Reward ($RER_a$)'
    )