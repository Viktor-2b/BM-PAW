import numpy as np


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
        "5_2_accept": c_case_5_2_accept, "5_2_deny": c_case_5_2_deny,
        "5_4_accept": c_case_5_4_accept, "5_4_deny": c_case_5_4_deny,
    }


def attacker_reward(r_vars, alpha, beta, eta, gamma):
    """
    最优化r1,r2的目标函数。
    :param r_vars: 一个包含 [r1, r2] 的Numpy数组。
    :param alpha: 攻击者算力
    :param beta: 受害者算力
    :param eta: 目标算力
    :param gamma: 其他矿工在分叉时选择攻击者分支的比例
    :return: 攻击者总收益
    """
    delta = max(0, 1 - alpha - beta - eta)  # 其他算力
    r1, r2 = r_vars

    c = calculate_c(alpha, beta, eta, gamma, delta, r2)  # 计算在当前r2下的所有c值
    if c is None: return -np.inf
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
    fork_reward_5_2_accept = c["5_2_accept"] * prob_case_5_2 * share_r_bar
    fork_reward_5_4_accept = c["5_4_accept"] * prob_case_5_4 * share_r_bar
    r_fr_accept = fork_reward_5_2_accept + fork_reward_5_4_accept  # 这里为什么不用考虑拒绝时的收益？
    # 4. R_BM 贿赂金
    epsilon1 = c["5_2_accept"]-c["5_2_deny"]
    epsilon2 = c["5_4_accept"]-c["5_4_deny"]
    r_bm = epsilon1 * fork_reward_5_2_accept + epsilon2 * fork_reward_5_4_accept
    #    epsilon2 = (1 - c["5_4_deny"]) / (c["5_4_accept"] * share_r_bar)

    # 5. R_FR' 拒绝贿赂的分叉收益
    fork_reward_5_2_deny = c["5_2_deny"] * prob_case_5_2 * share_r_bar
    fork_reward_5_4_deny = c["5_4_deny"] * prob_case_5_4 * share_r_bar
    r_fr_deny = fork_reward_5_2_deny + fork_reward_5_4_deny

    # 6. 计算收益 Ra
    total_reward_with_bribe = r_imr + r_sr + r_fr_accept - r_bm
    total_reward_without_bribe = r_imr + r_sr + r_fr_deny
    if total_reward_with_bribe > total_reward_without_bribe and np.isfinite(epsilon2):
        return total_reward_with_bribe
    else:
        return total_reward_without_bribe


def optimize_r(alpha, beta, eta, gamma, grid_density):
    """
    通过网格搜索找到最优的 (r1, r2)。
    """
    r_range = np.linspace(0, 1, grid_density)

    best_r1, best_r2 = -1, -1
    max_reward = -np.inf

    for r1 in r_range:
        for r2 in r_range:
            reward = attacker_reward(
                (r1, r2), alpha, beta, eta, gamma
            )
            if reward > max_reward:
                max_reward = reward
                best_r1, best_r2 = r1, r2
    return best_r1, best_r2, max_reward

# from scipy.optimize import minimize
# def optimize_r1_r2(alpha, beta, eta, gamma, epsilon1, epsilon2):
#     """
#     为给定的系统参数和贿赂率，找到最优的r1, r2。
#     :param alpha: 攻击者算力
#     :param beta: 受害者算力
#     :param eta: 目标算力
#     :param gamma: 其他矿工在分叉时选择攻击者分支的比例
#     :param epsilon1: Case 5-2中的贿赂比例
#     :param epsilon2: Case 5-4中的贿赂比例
#
#     """
#     # 优化问题的初始猜测值
#     initial_guess = np.array([0.5, 0.5])
#
#     # r1和r2的取值范围是 [0, 1]
#     bounds = ((0, 1), (0, 1))
#
#     # 传递给目标函数的固定参数
#     args = (alpha, beta, eta, gamma, epsilon1, epsilon2)
#
#     # 调用优化器
#     result = minimize(
#         attacker_total_reward,
#         initial_guess,
#         args=args,
#         method='L-BFGS-B',  # 一个支持边界约束的高效算法
#         bounds=bounds
#     )
#
#     if result.success:
#         return result.x  # 返回最优的 [r1, r2]
#     else:
#         # 如果优化失败，返回一个默认值
#         # 在实践中，这很少发生，除非参数设置不合理
#         return np.array(initial_guess)
