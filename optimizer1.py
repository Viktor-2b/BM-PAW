import numpy as np


# 我们不再需要 scipy.optimize 了
# from scipy.optimize import minimize

def calculate_c(alpha, beta, eta, gamma, delta, r2):
    """
    计算不同分叉场景下攻击者获胜的概率c。
    """
    denominator = 1.0 - r2 * alpha
    if denominator <= 1e-9: denominator = 1e-9
    c_case_5_2_accept = ((1 - r2) * alpha + eta + beta + gamma * delta) / denominator
    c_case_5_2_deny = ((1 - r2) * alpha + beta + gamma * (delta + eta)) / denominator
    c_case_5_4_accept = 1.0
    c_case_5_4_deny = ((1 - r2) * alpha + beta + gamma * delta) / denominator
    return {
        "c_5_2_accept": c_case_5_2_accept,
        "c_5_2_deny": c_case_5_2_deny,
        "c_5_4_accept": c_case_5_4_accept,
        "c_5_4_deny": c_case_5_4_deny,
    }


def attacker_total_reward_objective(r_vars, alpha, beta, eta, gamma, epsilon1, epsilon2):
    """
    计算攻击者在BM-PAW策略下的“总期望收益”。
    """
    r1, r2 = r_vars
    delta = max(0, 1.0 - alpha - beta - eta)
    c_probs = calculate_c(alpha, beta, eta, gamma, delta, r2)
    denominator = 1.0 - r2 * alpha
    if denominator <= 1e-9: denominator = 1e-9

    r_bar = (r1 + r2) / 2.0
    share_r1 = (r1 * alpha) / (r1 * alpha + beta) if (r1 * alpha + beta) > 0 else 0
    share_r_bar = (r_bar * alpha) / (r_bar * alpha + beta) if (r_bar * alpha + beta) > 0 else 0
    block_reward = 1

    # 1. 无辜挖矿收益 (R_IMR) - 机会成本项
    r_imr = (1 - r1) * alpha * block_reward + \
            (r1 * alpha) * (((1 - r2) * alpha) / denominator) * block_reward

    # 2. 份额收益 (R_SR)
    r_sr = beta * share_r1 + \
           (r1 * alpha) * (beta / denominator) * share_r_bar

    # 3. 净分叉收益 (Net Forking Reward)
    net_forking_reward = 0

    prob_case_5_2 = (r1 * alpha) * (delta / denominator)
    net_forking_reward += prob_case_5_2 * c_probs["c_5_2_accept"] * share_r_bar * (1 - epsilon1)

    prob_case_5_4 = (r1 * alpha) * (eta / denominator)
    if epsilon2 > (1 - c_probs["c_5_4_deny"]):
        net_forking_reward += prob_case_5_4 * c_probs["c_5_4_accept"] * share_r_bar * (1 - epsilon2)
    else:
        net_forking_reward += prob_case_5_4 * c_probs["c_5_4_deny"] * share_r_bar

    total_reward = r_imr + r_sr + net_forking_reward

    return -total_reward


def grid_search_for_optimal_r(objective_func, bounds, args, grid_density=100):
    """
    使用网格搜索来寻找最优的 r1 和 r2。

    :param objective_func: 需要被优化的目标函数。
    :param bounds: r1 和 r2 的取值范围，例如 ((0, 1), (0, 1))。
    :param args: 传递给目标函数的固定参数。
    :param grid_density: 网格密度，即每个变量划分的点数。
    :return: 最优的 [r1, r2] numpy 数组。
    """
    r1_min, r1_max = bounds[0]
    r2_min, r2_max = bounds[1]

    # 创建 r1 和 r2 的搜索网格
    r1_values = np.linspace(r1_min, r1_max, grid_density)
    r2_values = np.linspace(r2_min, r2_max, grid_density)

    best_score = float('inf')
    best_r_vars = np.array([0.5, 0.5])  # 默认初始值

    # 遍历网格上的每一个点
    for r1 in r1_values:
        for r2 in r2_values:
            current_r_vars = np.array([r1, r2])
            # 计算当前点的函数值（分数）
            current_score = objective_func(current_r_vars, *args)

            # 如果找到了一个更好的分数，就更新记录
            if current_score < best_score:
                best_score = current_score
                best_r_vars = current_r_vars

    return best_r_vars


def optimize_r1_r2(alpha, beta, eta, gamma, epsilon1, epsilon2):
    """
    为给定的系统参数和贿赂率，找到最优的r1, r2。
    这个函数现在调用我们自己的网格搜索算法。
    """
    bounds = ((0, 1), (0, 1))
    args = (alpha, beta, eta, gamma, epsilon1, epsilon2)

    # ******************************************************************
    # ** 注意：网格搜索非常慢！**
    # ** 为了快速得到结果，可以暂时降低 grid_density，例如设为 50。**
    # ** 为了得到精确结果，可以设为 100 或更高，但这会花费很长时间。**
    # ******************************************************************
    grid_density = 100

    # 调用网格搜索优化器
    optimal_r = grid_search_for_optimal_r(
        attacker_total_reward_objective,
        bounds,
        args,
        grid_density=grid_density
    )

    return optimal_r