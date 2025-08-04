import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import time


def calculate_c(alpha, beta, eta, gamma, delta, r2):
    """
    根据给定的r2，计算在不同分叉场景下攻击者获胜的概率c。
    :param alpha: 攻击者算力
    :param beta: 受害者算力
    :param eta: 目标算力
    :param gamma: 其他矿工在分叉时选择攻击者分支的比例
    :param delta: 其他算力
    :param r2: 扣块后的渗透算力比例
    :returns: 各种情况下的攻击者获胜概率c
    """
    denominator = 1 - r2 * alpha  # 扣块后的总和算力
    if denominator <= 0: return None  # 避免除以零

    # 目标接受贿赂 (BM-PAW)，受害者会支持攻击者渗透算力发布在受害者矿池的扣块
    c_case_5_2_accept = ((1 - r2) * alpha + eta + beta + gamma * delta) / denominator
    c_case_5_4_accept = 1
    # 目标拒绝贿赂 (PAW)
    c_case_5_2_deny = ((1 - r2) * alpha + beta + gamma * (delta + eta)) / denominator
    c_case_5_4_deny = ((1 - r2) * alpha + beta + gamma * delta) / denominator

    return {
        "c_5_2_accept": c_case_5_2_accept, "c_5_2_deny": c_case_5_2_deny,
        "c_5_4_accept": c_case_5_4_accept, "c_5_4_deny": c_case_5_4_deny,
    }


def attacker_reward(r_vars, alpha, beta, eta, gamma,epsilon1,epsilon2):
    """
    最优化r1,r2的目标函数。
    :param r_vars: 一个包含 [r1, r2] 的Numpy数组。
    :param alpha: 攻击者算力
    :param beta: 受害者算力
    :param eta: 目标算力
    :param gamma: 其他矿工在分叉时选择攻击者分支的比例
    :param epsilon1: Case 5-2 贿赂比例
    :param epsilon2: Case 5-4 贿赂比例
    :return: 攻击者总收益
    """
    delta = max(0, 1 - alpha - beta - eta)  # 其他算力
    r1, r2 = r_vars

    c_probs = calculate_c(alpha, beta, eta, gamma, delta, r2)  # 计算在当前r2下的所有c值
    if c_probs is None: return -np.inf
    # 扣块后的总和算力
    # 根据论文约束 (alpha < 0.5, r2 <= 1)，分母 denominator = 1.0 - r2 * alpha 总是 > 0.5。
    denominator = 1 - r2 * alpha

    # 事件概率
    prob_case_1 = (1 - r1) * alpha
    prob_case_3 = beta
    prob_case_5 = r1 * alpha
    # 份额收益
    r_bar = (r1 + r2) / 2  # 攻击者平均渗透算力
    share_r1 = (r1 * alpha) / (r1 * alpha + beta) if (r1 * alpha + beta) > 0 else 0
    share_r_bar = (r_bar * alpha) / (r_bar * alpha + beta) if (r_bar * alpha + beta) > 0 else 0
    # 条件概率
    prob_case_5_1 = prob_case_5 * (1 - r2) * alpha / denominator
    prob_case_5_2 = prob_case_5 * delta / denominator
    prob_case_5_3 = prob_case_5 * beta / denominator
    prob_case_5_4 = prob_case_5 * eta / denominator

    # 1. R_IMR 无辜挖矿收益
    r_imr = prob_case_1 + prob_case_5_1
    # 2. R_SR 份额收益
    r_sr = prob_case_3 * share_r1 + prob_case_5_3 * share_r_bar
    # 3. R_FR 接受贿赂的分叉收益
    fork_reward_5_2_accept = c_probs["c_5_2_accept"] * prob_case_5_2 * share_r_bar
    fork_reward_5_4_accept = c_probs["c_5_4_accept"] * prob_case_5_4 * share_r_bar
    r_fr_accept = fork_reward_5_2_accept + fork_reward_5_4_accept  # 这里为什么不用考虑拒绝时的收益？
    # 4. R_BM 贿赂金
    r_bm = epsilon1 * fork_reward_5_2_accept + epsilon2 * fork_reward_5_4_accept
    # 5. R_FR' 拒绝贿赂的分叉收益
    fork_reward_5_2_deny = c_probs["c_5_2_deny"] * prob_case_5_2 * share_r_bar
    fork_reward_5_4_deny = c_probs["c_5_4_deny"] * prob_case_5_4 * share_r_bar
    r_fr_deny = fork_reward_5_2_deny + fork_reward_5_4_deny

    # 6. 计算收益 Ra
    total_reward_with_bribe = r_imr + r_sr + r_fr_accept - r_bm
    total_reward_without_bribe = r_imr + r_sr + r_fr_deny
    return max(total_reward_with_bribe,total_reward_without_bribe)


def optimize_r(alpha, beta, eta, gamma, epsilon1,epsilon2,grid_density):
    """
    通过网格搜索找到最优的 (r1, r2)。
    """
    r_range = np.linspace(0, 1, grid_density)

    best_r1, best_r2 = -1, -1
    max_reward = -np.inf

    for r1 in r_range:
        for r2 in r_range:
            reward = attacker_reward(
                (r1, r2), alpha, beta, eta, gamma,epsilon1,epsilon2
            )
            if reward > max_reward:
                max_reward = reward
                best_r1, best_r2 = r1, r2
    return best_r1, best_r2, max_reward


# ==============================================================================
# 2. 多进程并行化执行模块
# ==============================================================================

def worker_task(params):
    """工作函数，被每个子进程调用来执行单个优化任务。"""
    alpha, beta, eta, gamma, epsilon1, epsilon2, grid_density = params
    r1_opt, r2_opt, _ = optimize_r(alpha, beta, eta, gamma, epsilon1, epsilon2, grid_density)
    return alpha, beta, r1_opt, r2_opt


# ==============================================================================
# 3. 主执行模块 (扫描 epsilon1 和 epsilon2)
# ==============================================================================

if __name__ == '__main__':
    start_time = time.time()

    # --- 实验参数配置 ---
    # 最外层循环的参数
    EPSILON_VALUES = np.linspace(0.01, 1, 10)  # 从0到1取5个点 (0, 0.25, 0.5, 0.75, 1.0)

    # 内层表格的行列定义
    ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4]
    BETA_VALUES = [0.1, 0.2, 0.3]

    # 固定参数
    ETA = 0.2
    GAMMA = 0.5

    # 搜索精度
    GRID_DENSITY = 101  # 注意：这是一个三重循环，总计算量巨大。建议从较低的密度开始测试。

    print("开始对 epsilon1 和 epsilon2 进行参数扫描...")
    print(f"Epsilon 将取: {[f'{v:.2f}' for v in EPSILON_VALUES]}")
    print(f"固定参数: eta = {ETA}, gamma = {GAMMA}")
    print(f"搜索精度: grid_density = {GRID_DENSITY}")

    num_processes = max(1, cpu_count() - 1)
    print(f"将使用 {num_processes} 个CPU核心进行并行计算。")
    print("-" * 80)

    # --- 外层循环，遍历epsilon1和epsilon2 ---
    for eps1 in EPSILON_VALUES:
        for eps2 in EPSILON_VALUES:

            tasks = [(alpha, beta, ETA, GAMMA, eps1, eps2, GRID_DENSITY)
                     for beta in BETA_VALUES
                     for alpha in ALPHA_VALUES]

            print(f"\n>>> 正在计算: epsilon1 = {eps1:.2f}, epsilon2 = {eps2:.2f} ...")

            results = []
            with Pool(processes=num_processes) as pool:
                for result in tqdm(pool.imap_unordered(worker_task, tasks), total=len(tasks),
                                   desc=f"ε1={eps1:.2f},ε2={eps2:.2f}"):
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
            print(f" 复现结果 (epsilon1 = {eps1:.2f}, epsilon2 = {eps2:.2f})")
            print("=" * 75)
            print(results_df)
            print("-" * 75)

    end_time = time.time()
    print(f"\n所有计算任务完成，总耗时: {end_time - start_time:.2f} 秒。")