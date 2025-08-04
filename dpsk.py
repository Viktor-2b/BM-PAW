import numpy as np


def calculate_c(alpha, beta, eta, gamma, delta, r2):
    denominator = 1.0 - r2 * alpha
    if denominator <= 1e-6:
        return None

    # 目标接受贿赂 (BM-PAW)
    c_5_2_accept = ((1 - r2) * alpha + eta + beta + gamma * delta) / denominator
    c_5_4_accept = 1.0

    # 目标拒绝贿赂 (PAW)
    c_5_2_deny = ((1 - r2) * alpha + beta + gamma * (delta + eta)) / denominator
    c_5_4_deny = ((1 - r2) * alpha + beta + gamma * delta) / denominator

    return {
        "c_5_2_accept": c_5_2_accept, "c_5_2_deny": c_5_2_deny,
        "c_5_4_accept": c_5_4_accept, "c_5_4_deny": c_5_4_deny
    }


def calculate_min_epsilon(alpha, beta, eta, gamma, r1, r2, r_bar, c_probs):
    """根据定理3.1和3.2计算最小贿赂比例"""
    delta = max(0, 1.0 - alpha - beta - eta)
    denominator = 1.0 - r2 * alpha

    # 计算Case 5-2和5-4的概率
    prob_case5 = r1 * alpha
    prob_case5_2 = prob_case5 * delta / denominator
    prob_case5_4 = prob_case5 * eta / denominator

    # 计算攻击者在受害者矿池中的份额比例
    share_ratio_fork = (r_bar * alpha) / (r_bar * alpha + beta + 1e-9)

    # 定理3.2：攻击者收益条件
    epsilon1_max = (c_probs["c_5_2_accept"] - c_probs["c_5_2_deny"]) / c_probs["c_5_2_accept"]
    epsilon2_max = (c_probs["c_5_4_accept"] - c_probs["c_5_4_deny"]) / c_probs["c_5_4_accept"]

    # 定理3.1：目标矿池收益条件
    # 简化计算：实际实现应根据公式(8)精确计算
    epsilon2_min = (1 - c_probs["c_5_4_deny"]) / c_probs["c_5_4_accept"]

    # 确保epsilon在合理范围内
    epsilon1 = max(0.001, min(epsilon1_max - 0.001, 0.01)) if epsilon1_max > 0.001 else 0.001
    epsilon2 = max(epsilon2_min + 0.001, min(epsilon2_max - 0.001, 0.01))

    return epsilon1, epsilon2


def attacker_reward(r_vars, alpha, beta, eta, gamma):
    r1, r2 = r_vars
    delta = max(0, 1.0 - alpha - beta - eta)

    # 使用简单算术平均计算r_bar
    r_bar = (r1 + r2) / 2.0

    c_probs = calculate_c(alpha, beta, eta, gamma, delta, r2)
    if not c_probs:
        return -np.inf

    denominator = 1.0 - r2 * alpha
    if denominator < 1e-6:
        return -np.inf

    # 动态计算最小贿赂比例
    epsilon1, epsilon2 = calculate_min_epsilon(alpha, beta, eta, gamma, r1, r2, r_bar, c_probs)

    # === 事件概率计算 ===
    prob_case1 = (1 - r1) * alpha
    prob_case3 = beta
    prob_case5 = r1 * alpha

    prob_case5_1 = prob_case5 * (1 - r2) * alpha / denominator
    prob_case5_2 = prob_case5 * delta / denominator
    prob_case5_3 = prob_case5 * beta / denominator
    prob_case5_4 = prob_case5 * eta / denominator

    # === 收益计算 ===
    # 1. 无辜挖矿收益（IMR）
    r_imr = prob_case1 + prob_case5_1

    # 2. 份额收益（SR）
    share_ratio_case3 = (r1 * alpha) / (r1 * alpha + beta + 1e-9)
    share_ratio_case5_3 = (r_bar * alpha) / (r_bar * alpha + beta + 1e-9)
    r_sr = prob_case3 * share_ratio_case3 + prob_case5_3 * share_ratio_case5_3

    # 3. 分叉收益（FR）
    share_ratio_fork = (r_bar * alpha) / (r_bar * alpha + beta + 1e-9)

    # 接受贿赂的分叉收益
    fork_5_2_accept = prob_case5_2 * c_probs["c_5_2_accept"] * share_ratio_fork
    fork_5_4_accept = prob_case5_4 * c_probs["c_5_4_accept"] * share_ratio_fork
    r_fr_accept = fork_5_2_accept + fork_5_4_accept

    # 拒绝贿赂的分叉收益
    fork_5_2_deny = prob_case5_2 * c_probs["c_5_2_deny"] * share_ratio_fork
    fork_5_4_deny = prob_case5_4 * c_probs["c_5_4_deny"] * share_ratio_fork
    r_fr_deny = fork_5_2_deny + fork_5_4_deny

    # 4. 贿赂金支出（BM）
    r_bm = epsilon1 * fork_5_2_accept + epsilon2 * fork_5_4_accept

    # 计算两种策略的收益
    reward_bm_paw = r_imr + r_sr + r_fr_accept - r_bm
    reward_paw = r_imr + r_sr + r_fr_deny

    # 选择收益更高的策略
    return max(reward_bm_paw, reward_paw), epsilon1, epsilon2


def optimize_r(alpha, beta, eta, gamma, grid_density=101):
    """
    通过网格搜索找到最优的 (r1, r2)
    """
    r_range = np.linspace(0.01, 0.99, grid_density)

    best_r1, best_r2 = 0.5, 0.5
    max_reward = -np.inf
    best_eps1, best_eps2 = 0.01, 0.01

    # 在关键区域增加搜索密度
    for r1 in r_range:
        for r2 in r_range:
            # 跳过无效参数组合
            if r1 * alpha < 1e-6 or (1 - r2 * alpha) < 1e-6:
                continue

            reward, eps1, eps2 = attacker_reward((r1, r2), alpha, beta, eta, gamma)
            if reward > max_reward:
                max_reward = reward
                best_r1, best_r2 = r1, r2
                best_eps1, best_eps2 = eps1, eps2

    return best_r1, best_r2, best_eps1, best_eps2


# 表1参数设置
gamma = 0.5
eta = 0.2

# 表1中的测试场景
results = {}
for beta in [0.1, 0.2, 0.3]:
    for alpha in [0.1, 0.2, 0.3, 0.4]:
        r1, r2, eps1, eps2 = optimize_r(alpha, beta, eta, gamma)
        results[(alpha, beta)] = (r1, r2, eps1, eps2)

# 打印结果
print("优化结果 (eta = 0.20, gamma = 0.50)")
print("=" * 75)
print(f"{'':10}", end="")
for alpha in [0.1, 0.2, 0.3, 0.4]:
    print(f"{'alpha=' + str(alpha):<16}", end="")
print("\n" + "-" * 75)

for beta in [0.1, 0.2, 0.3]:
    print(f"beta={beta:<7}", end="")
    for alpha in [0.1, 0.2, 0.3, 0.4]:
        r1, r2, eps1, eps2 = results.get((alpha, beta), (0, 0, 0, 0))
        print(f"{r1:.4f}({r2:.4f})".ljust(16), end="")
    print()

print("-" * 75)

# 打印epsilon值参考
print("\n贿赂比例参考值:")
print("=" * 75)
for beta in [0.1, 0.2, 0.3]:
    print(f"beta={beta}:")
    for alpha in [0.1, 0.2, 0.3, 0.4]:
        r1, r2, eps1, eps2 = results.get((alpha, beta), (0, 0, 0, 0))
        print(f"  alpha={alpha}: ε1={eps1:.6f}, ε2={eps2:.6f}")