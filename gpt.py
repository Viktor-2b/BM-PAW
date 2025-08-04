import numpy as np


def calculate_c(alpha, beta, eta, gamma, delta, r2):
    denominator = 1 - r2 * alpha
    if denominator <= 0:
        return None
    # BM-PAW 分叉胜率
    c_5_2 = ((1 - r2) * alpha + eta + beta + gamma * delta) / denominator
    c_5_4 = 1  # 目标接受时
    c_5_2_deny = ((1 - r2) * alpha + beta + gamma * (delta + eta)) / denominator
    c_5_4_deny = ((1 - r2) * alpha + beta + gamma * delta) / denominator
    return {
        'accept': {'c52': c_5_2, 'c54': c_5_4},
        'deny':   {'c52': c_5_2_deny, 'c54': c_5_4_deny}
    }


def attacker_reward(r_vars, alpha, beta, eta, gamma, epsilon1, epsilon2):
    r1, r2 = r_vars
    delta = max(0, 1 - alpha - beta - eta)

    c = calculate_c(alpha, beta, eta, gamma, delta, r2)
    if c is None:
        return -np.inf

    denom = 1 - r2 * alpha
    prob1 = (1 - r1) * alpha
    prob3 = beta
    prob5 = r1 * alpha
    r_bar = (r1 + r2) / 2
    share1 = (r1 * alpha) / (r1 * alpha + beta) if (r1 * alpha + beta) > 0 else 0
    sharebar = (r_bar * alpha) / (r_bar * alpha + beta) if (r_bar * alpha + beta) > 0 else 0

    p_51 = prob5 * (1 - r2) * alpha / denom
    p_52 = prob5 * delta / denom
    p_53 = prob5 * beta / denom
    p_54 = prob5 * eta / denom

    R_imr = prob1 + p_51
    R_sr  = prob3 * share1 + p_53 * sharebar

    # 分叉收益
    R_fr_accept = c['accept']['c52'] * p_52 * sharebar + c['accept']['c54'] * p_54 * sharebar
    R_fr_deny   = c['deny'  ]['c52'] * p_52 * sharebar + c['deny'  ]['c54'] * p_54 * sharebar

    # 贿赂支出（固定 epsilon1, epsilon2）
    R_bm = epsilon1 * p_52 * sharebar + epsilon2 * p_54 * sharebar

    # BM-PAW 与 PAW
    R_bmpaw = R_imr + R_sr + R_fr_accept - R_bm
    R_paw   = R_imr + R_sr + R_fr_deny

    # 取二者最大
    return max(R_bmpaw, R_paw)


def optimize_r(alpha, beta, eta, gamma,
               epsilon1, epsilon2, grid_density=51):
    r_vals = np.linspace(0, 1, grid_density)
    best = (0, 0, -np.inf)
    for r1 in r_vals:
        for r2 in r_vals:
            rew = attacker_reward((r1, r2), alpha, beta, eta, gamma, epsilon1, epsilon2)
            if rew > best[2]:
                best = (r1, r2, rew)
    return best

# 示例：扫描不同 epsilon1/epsilon2
if __name__ == '__main__':
    alpha = 0.2
    beta  = 0.2
    eta   = 0.2
    gamma = 0.5
    for eps1 in [0.01, 0.05, 0.1]:
        for eps2 in [0.01, 0.05, 0.1]:
            r1, r2, rew = optimize_r(alpha, beta, eta, gamma, eps1, eps2)
            print(f"eps1={eps1}, eps2={eps2} => r1={r1:.3f}, r2={r2:.3f}, reward={rew:.4f}")
