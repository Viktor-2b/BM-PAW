import numpy as np


def calculate_c(alpha, beta, eta, gamma, r2):
    """
    计算不同分叉场景下攻击者获胜的概率c

    :param alpha: 攻击者算力比例 (0 < alpha < 0.5)
    :param beta: 受害者算力比例
    :param eta: 目标矿池算力比例
    :param gamma: 其他矿工在分叉时选择攻击者分支的比例
    :param r2: 第二阶段攻击者渗透算力比例 (0 <= r2 <= 1)

    :returns: 包含四种场景下攻击者获胜概率的字典:
      * c_5_2_accept: Case 5-2中目标接受贿赂时的获胜概率
      * c_5_2_deny: Case 5-2中目标拒绝贿赂时的获胜概率
      * c_5_4_accept: Case 5-4中目标接受贿赂时的获胜概率
      * c_5_4_deny: Case 5-4中目标拒绝贿赂时的获胜概率
    """
    delta = max(0, 1.0 - alpha - beta - eta)  # 其他算力比例
    denominator = 1.0 - r2 * alpha  # 扣块后的总和算力比例
    if denominator <= 1e-6:  # 避免除以零
        return None

    # 目标接受贿赂 (BM-PAW)
    # Case 5-2: 目标矿池接受贿赂，支持攻击者分支
    c_5_2_accept = ((1.0 - r2) * alpha + eta + beta + gamma * delta) / denominator
    # Case 5-4: 目标矿池接受贿赂，攻击者获胜概率为1
    c_5_4_accept = 1.0

    # 目标拒绝贿赂 (PAW)
    # Case 5-2: 目标矿池拒绝贿赂，加入受害者分支
    c_5_2_deny = ((1.0 - r2) * alpha + beta + gamma * (delta + eta)) / denominator
    # Case 5-4: 目标矿池拒绝贿赂，加入其他分支
    c_5_4_deny = ((1.0 - r2) * alpha + beta + gamma * delta) / denominator

    return {
        "5_2_accept": c_5_2_accept, "5_2_deny": c_5_2_deny,
        "5_4_accept": c_5_4_accept, "5_4_deny": c_5_4_deny,
    }


def calculate_min_epsilon2(alpha, beta, eta, gamma, r1, r2):
    """
    根据R_t(r1, r2) = 0 计算最小ε2

    核心思想: 不失一般性，设ε1 = 0，ε2取被贿赂矿池选择接受贿赂时的总奖励
    刚好等于不接受贿赂时总奖励的情况，此时敌手和被贿赂矿池均获得正收益。

    参数说明:
    - alpha, beta, eta, gamma: 系统参数
    - r1: 第一阶段攻击者渗透算力比例
    - r2: 第二阶段攻击者渗透算力比例

    返回值:
    - 最小ε2，如果计算失败返回np.inf
    """
    c = calculate_c(alpha, beta, eta, gamma, r2)  # 计算在当前r2下的所有c值
    if c is None:
        return np.inf

    # 计算攻击者平均渗透算力
    r_bar = (r1 + r2) / 2.0
    share_ratio_r_bar = (r_bar * alpha) / (r_bar * alpha + beta) if (r_bar * alpha + beta) > 0 else 0

    # 接受贿赂时攻击者的分叉收益
    fork_reward = share_ratio_r_bar
    # 拒绝贿赂时目标的预期损失
    potential_loss = 1.0 - c["5_4_deny"]

    epsilon2 = potential_loss/fork_reward + 0.01 if fork_reward > 0 else 0

    return epsilon2


def attacker_reward(r_vars, alpha, beta, eta, gamma, epsilon1, epsilon2):
    """
    最优化r1,r2的目标函数。

    :param r_vars: 一个包含 [r1, r2] 的Numpy数组。
    :param alpha: 攻击者算力
    :param beta: 受害者算力
    :param eta: 目标算力
    :param gamma: 其他矿工在分叉时选择攻击者分支的比例
    :param epsilon1: Case 5-2中的贿赂比例
    :param epsilon2: Case 5-4中的贿赂比例

    :return: 攻击者总收益
    """
    r1, r2 = r_vars
    c = calculate_c(alpha, beta, eta, gamma, r2)  # 计算在当前r2下的所有c值
    if c is None:
        return -np.inf

    delta = max(0, 1.0 - alpha - beta - eta)  # 其他算力比例
    denominator = 1.0 - r2 * alpha  # 扣块后的总和算力
    if denominator <= 1e-6:
        return np.inf

    # 事件概率
    prob_case_1 = (1.0 - r1) * alpha  # Case 1: 攻击者无辜挖矿
    prob_case_3 = beta  # Case 3: 受害者挖矿
    prob_case_5 = r1 * alpha  # Case 5: 攻击者渗透挖矿

    # 条件概率
    prob_case_5_1 = prob_case_5 * (1.0 - r2) * alpha / denominator  # Case 5-1: 攻击者无辜挖矿
    prob_case_5_2 = prob_case_5 * delta / denominator  # Case 5-2: 其他矿工挖矿
    prob_case_5_3 = prob_case_5 * beta / denominator  # Case 5-3: 受害者挖矿
    prob_case_5_4 = prob_case_5 * eta / denominator  # Case 5-4: 目标矿池挖矿

    # 份额收益
    r_bar = (r1 + r2) / 2.0  # 攻击者平均渗透算力
    share_ratio_r1 = (r1 * alpha) / (r1 * alpha + beta) if (r1 * alpha + beta) > 0 else 0
    share_ratio_r_bar = (r_bar * alpha) / (r_bar * alpha + beta) if (r_bar * alpha + beta) > 0 else 0

    # 1. R_IMR 无辜挖矿收益
    r_imr = prob_case_1 + prob_case_5_1
    # 2. R_SR 份额收益
    r_sr = prob_case_3 * share_ratio_r1 + prob_case_5_3 * share_ratio_r_bar

    # 3. R_FR 接受贿赂的分叉收益
    fork_reward_5_2_accept = c["5_2_accept"] * prob_case_5_2 * share_ratio_r_bar
    fork_reward_5_4_accept = c["5_4_accept"] * prob_case_5_4 * share_ratio_r_bar
    r_fr_accept = fork_reward_5_2_accept + fork_reward_5_4_accept

    # 4. R_BM 贿赂金
    r_bm = epsilon1 * fork_reward_5_2_accept + epsilon2 * fork_reward_5_4_accept

    # 5. R_FR' 拒绝贿赂的分叉收益
    fork_reward_5_2_deny = c["5_2_deny"] * prob_case_5_2 * share_ratio_r_bar
    fork_reward_5_4_deny = c["5_4_deny"] * prob_case_5_4 * share_ratio_r_bar
    r_fr_deny = fork_reward_5_2_deny + fork_reward_5_4_deny

    # 6. 计算收益 Ra
    total_reward_with_bribe = r_imr + r_sr + r_fr_accept - r_bm
    total_reward_without_bribe = r_imr + r_sr + r_fr_deny
    return r_imr + r_sr + r_fr_accept


def optimize_r(alpha, beta, eta, gamma, grid_density):
    """
    通过网格搜索找到最优的 (r1, r2)。
    """
    r_range = np.linspace(0.01, 0.99, grid_density)  # 避免边界值0和1

    best_r1, best_r2 = -1, -1
    max_reward = -np.inf

    for r1 in r_range:
        for r2 in r_range:
            eps1 = 0
            eps2 = calculate_min_epsilon2(alpha, beta, eta, gamma, r1, r2)

            reward = attacker_reward(
                (r1, r2), alpha, beta, eta, gamma, eps1, eps2
            )
            if reward > max_reward:
                max_reward = reward
                best_r1, best_r2 = r1, r2

    return best_r1, best_r2, max_reward