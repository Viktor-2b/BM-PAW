import pandas as pd
from optimizer import optimize_r
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def worker_task(params):
    """
    工作函数，被每个子进程调用来执行单个优化任务。
    它接收一个包含所有参数的元组。
    """
    alpha, beta, eta, gamma, grid_density = params
    r1_opt, r2_opt, _ = optimize_r(alpha, beta, eta, gamma, grid_density)
    # 返回所有必要信息，以便主进程可以识别这个结果属于哪个单元格
    return alpha, beta, r1_opt, r2_opt


if __name__ == '__main__':
    # --- 实验参数配置 ---
    # 使用一组最接近论文的(eta, gamma)进行最终验证
    ETA_VALUES = [0.2]
    GAMMA_VALUES = [0.5]

    ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4]
    BETA_VALUES = [0.1, 0.2, 0.3]

    GRID_DENSITY = 1001  # 使用高精度以匹配论文

    print("开始复现论文 Table 1 (使用最终正确模型)...")

    num_processes = max(1, cpu_count() - 1)
    print(f"将使用 {num_processes} 个CPU核心进行并行计算。")
    print("-" * 80)

    for eta in ETA_VALUES:
        for gamma in GAMMA_VALUES:

            tasks = [(alpha, beta, eta, gamma, GRID_DENSITY)
                     for beta in BETA_VALUES
                     for alpha in ALPHA_VALUES]

            print(f"\n>>> 正在计算: eta = {eta:.2f}, gamma = {gamma:.2f} ...")

            results = []
            with Pool(processes=num_processes) as pool:
                for result in tqdm(pool.imap_unordered(worker_task, tasks), total=len(tasks),
                                   desc=f"eta={eta:.1f},γ={gamma:.1f}"):
                    results.append(result)

            results_data = {}
            for alpha_res, beta_res, r1_opt, r2_opt in results:
                results_data[(alpha_res, beta_res)] = f"{r1_opt:.4f}({r2_opt:.4f})"

            table_for_df = {f"alpha={a}": [] for a in ALPHA_VALUES}
            for a in ALPHA_VALUES:
                for b in BETA_VALUES:
                    table_for_df[f"alpha={a}"].append(results_data[(a, b)])

            results_df = pd.DataFrame(table_for_df, index=BETA_VALUES)
            results_df.index.name = 'beta'

            print("\n" + "=" * 75)
            print(f" 复现结果 (eta = {eta:.2f}, gamma = {gamma:.2f})")
            print("=" * 75)
            print(results_df)
            print("-" * 75)
            print("将此结果与论文原文Table 1进行对比。")