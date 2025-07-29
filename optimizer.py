import numpy as np
from scipy.optimize import minimize


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
    denominator = 1.0 - r2 * alpha  # 扣块后的总和算力
    # if denominator <= 1e-9: denominator = 1e-9  # 避免除以零 TODO 数值太小

    # Case 5-2: 与"其他矿工"分叉
    # 目标接受贿赂 (BM-PAW)，受害者会支持攻击者渗透算力发布在受害者矿池的扣块
    c_case_5_2_accept = ((1 - r2) * alpha + eta + beta + gamma * delta) / denominator
    # 目标拒绝贿赂 (PAW)
    c_case_5_2_deny = ((1 - r2) * alpha + beta + gamma * (delta + eta)) / denominator

    # Case 5-4: 与"目标矿池"分叉
    # 目标接受贿赂 (BM-PAW)，攻击者发布自己的块，目标不发布，攻击者必胜
    c_case_5_4_accept = 1.0
    # 目标拒绝贿赂 (PAW)
    c_case_5_4_deny = ((1 - r2) * alpha + beta + gamma * delta) / denominator

    return {
        "c_5_2_accept": c_case_5_2_accept,
        "c_5_2_deny": c_case_5_2_deny,
        "c_5_4_accept": c_case_5_4_accept,
        "c_5_4_deny": c_case_5_4_deny,
    }


def attacker_total_reward(r_vars, alpha, beta, eta, gamma, epsilon1, epsilon2):
    """
    计算并返回负的总收益，Ra = R_BM-PAW  = R_IMR + R_SR + R_FR - R_BM。
    scipy.optimize.minimize 会最小化这个函数，因此我们返回负的收益，优化变量必须位于第一个参数。
    :param r_vars: 一个包含 [r1, r2] 的Numpy数组。
    :param alpha: 攻击者算力
    :param beta: 受害者算力
    :param eta: 目标算力
    :param gamma: 其他矿工在分叉时选择攻击者分支的比例
    :param epsilon1: Case 5-2中的贿赂比例
    :param epsilon2: Case 5-4中的贿赂比例
    :return: 攻击者总收益的负数
    """
    delta = max(0, 1.0 - alpha - beta - eta)  # 其他算力
    r1, r2 = r_vars
    r_bar = (r1 + r2) / 2.0  # 攻击者平均渗透算力
    c_probs = calculate_c(alpha, beta, eta, gamma, delta, r2)  # 计算在当前r2下的所有c值
    denominator = 1.0 - r2 * alpha  # 扣块后的总和算力
    # if denominator <= 1e-9: denominator = 1e-9  # 避免除以零

    block_reward = 1 # 区块奖励归一化
    # 扣块前攻击者的份额收益
    share_r1 = (r1 * alpha) / (r1 * alpha + beta) if (r1 * alpha + beta) > 0 else 0
    # 扣块后攻击者的份额收益
    share_r_bar = (r_bar * alpha) / (r_bar * alpha + beta) if (r_bar * alpha + beta) > 0 else 0
    # P(Case 5-x)=P(Case 5)*P(Case 5-x|Case 5)
    prob_case_5_1 = (r1 * alpha) * (((1 - r2) * alpha) / denominator)
    prob_case_5_2 = (r1 * alpha) * (delta / denominator)
    prob_case_5_3 = (r1 * alpha) * (beta / denominator)
    prob_case_5_4 = (r1 * alpha) * (eta / denominator)

    # 1. R_IMR 无辜挖矿收益
    r_imr = (1 - r1) * alpha * block_reward + prob_case_5_1 * block_reward
    # 2. R_SR 份额收益
    r_sr = beta * share_r1 + prob_case_5_3 * share_r_bar
    # 3. R_FR 分叉收益 R_BM 贿赂金
    r_fr = 0
    r_bm = 0
    # Case 5-2: 假设总能成功贿赂
    fork_reward_5_2 = c_probs["c_5_2_accept"] * prob_case_5_2 * share_r_bar
    r_fr += fork_reward_5_2
    r_bm += epsilon1 * fork_reward_5_2
    # Case 5-4: 检查贿赂是否会被接受
    cond = epsilon2 *share_r_bar> 1 - c_probs["c_5_4_deny"]
    if cond:
        fork_reward_5_4_accept = c_probs["c_5_4_accept"] * prob_case_5_4 * share_r_bar
        r_fr += fork_reward_5_4_accept
        r_bm += epsilon2 * fork_reward_5_4_accept
    else:
        fork_reward_5_4_deny = c_probs["c_5_4_deny"] * prob_case_5_4 * share_r_bar
        r_fr += fork_reward_5_4_deny

    # 3. 计算总收益 Ra
    total_reward = r_imr + r_sr + r_fr - r_bm
    return -total_reward


def optimize_r1_r2(alpha, beta, eta, gamma, epsilon1, epsilon2):
    """
    为给定的系统参数和贿赂率，找到最优的r1, r2。
    :param alpha: 攻击者算力
    :param beta: 受害者算力
    :param eta: 目标算力
    :param gamma: 其他矿工在分叉时选择攻击者分支的比例
    :param epsilon1: Case 5-2中的贿赂比例
    :param epsilon2: Case 5-4中的贿赂比例

    """
    # 优化问题的初始猜测值
    initial_guess = np.array([0.5, 0.5])

    # r1和r2的取值范围是 [0, 1]
    bounds = ((0, 1), (0, 1))

    # 传递给目标函数的固定参数
    args = (alpha, beta, eta, gamma, epsilon1, epsilon2)

    # 调用优化器
    result = minimize(
        attacker_total_reward,
        initial_guess,
        args=args,
        method='L-BFGS-B',  # 一个支持边界约束的高效算法
        bounds=bounds
    )

    if result.success:
        return result.x  # 返回最优的 [r1, r2]
    else:
        # 如果优化失败，返回一个默认值
        # 在实践中，这很少发生，除非参数设置不合理
        return np.array(initial_guess)
