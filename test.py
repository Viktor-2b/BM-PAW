import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# ==============================================================================
# 1. 核心计算函数 (这些函数保持不变)
# ==============================================================================

def calculate_c(alpha, beta, eta, gamma, r2):
    """计算不同分叉场景下攻击者获胜的概率c"""
    delta = max(0, 1.0 - alpha - beta - eta)
    denominator = 1.0 - r2 * alpha
    if denominator <= 1e-6:
        return None
    c_5_2_accept = ((1.0 - r2) * alpha + eta + beta + gamma * delta) / denominator
    c_5_4_accept = 1.0
    c_5_2_deny = ((1.0 - r2) * alpha + beta + gamma * (delta + eta)) / denominator
    c_5_4_deny = ((1.0 - r2) * alpha + beta + gamma * delta) / denominator
    return {
        "5_2_accept": c_5_2_accept, "5_2_deny": c_5_2_deny,
        "5_4_accept": c_5_4_accept, "5_4_deny": c_5_4_deny,
    }


def calculate_epsilon_condition(alpha, beta, eta, gamma, epsilon1, epsilon2, r1, r2):
    """
    【修正版】根据论文定理2和定理3，精确检查博弈条件是否成立。
    """
    c = calculate_c(alpha, beta, eta, gamma, r2)
    if c is None: return False

    delta = max(0, 1.0 - alpha - beta - eta)
    r_bar = (r1 + r2) / 2.0
    share_ratio_r_bar = (r_bar * alpha) / (r_bar * alpha + beta) if (r_bar * alpha + beta) > 0 else 0
    if share_ratio_r_bar <= 1e-9: return False

    # --- 攻击者条件 (Theorem 2) ---
    # 攻击者支付的期望贿赂成本
    # 注意：这里的概率是事件发生的概率，不是单纯的算力比例
    prob_5_2 = delta / (1 - r2 * alpha)
    prob_5_4 = eta / (1 - r2 * alpha)

    # 贿赂成本 = P(5-2)*贿赂金_5-2 + P(5-4)*贿赂金_5-4
    # 这里的贿赂金是针对整个区块奖励的比例，所以要乘以分叉收益
    attacker_bribe_cost = (epsilon1 * c["5_2_accept"] * prob_5_2 * share_ratio_r_bar) + \
                          (epsilon2 * c["5_4_accept"] * prob_5_4 * share_ratio_r_bar)

    # 攻击者获得的期望额外收益 (BM-PAW收益 - PAW收益)
    gain_from_5_2 = (c["5_2_accept"] - c["5_2_deny"]) * prob_5_2 * share_ratio_r_bar
    gain_from_5_4 = (c["5_4_accept"] - c["5_4_deny"]) * prob_5_4 * share_ratio_r_bar
    attacker_reward_gain = gain_from_5_2 + gain_from_5_4

    cond_atk = attacker_bribe_cost < attacker_reward_gain

    # --- 目标矿池条件 (Theorem 3) ---
    # 目标矿池收到的期望贿赂金
    # 目标只在它自己是目标时（Case 5-4）和在其他矿工中被选中时（Case 5-2）收到贿赂
    target_bribe_gain = (epsilon1 * c["5_2_accept"] * prob_5_2 * share_ratio_r_bar) + \
                        (epsilon2 * c["5_4_accept"] * prob_5_4 * share_ratio_r_bar)

    # 目标矿池放弃的期望收益 (机会成本)
    # 根据定理3，主要考虑Case 5-4中它放弃的收益
    target_opportunity_cost = eta * (1 - c["5_4_deny"]) * (r_bar * alpha / (r_bar * alpha + beta))

    cond_tar = target_bribe_gain > target_opportunity_cost

    return cond_atk and cond_tar


# ==============================================================================
# 2. 并行计算的工作函数 (这是新添加的核心部分)
# ==============================================================================

def worker_task(args):
    """
    这是每个子进程执行的工作函数。
    它计算热力图中的一个像素点 (一个eps1, eps2组合) 的最优RER。
    """
    # 从参数元组中解包所有需要的变量
    i, j, eps1, eps2, alpha, beta, eta, gamma, r_range = args

    max_rer_for_this_eps = -np.inf

    # 内层循环：为当前的 (eps1, eps2) 寻找最优的 (r1, r2)
    for r1 in r_range:
        for r2 in r_range:
            c = calculate_c(alpha, beta, eta, gamma, r2)
            if c is None: continue

            r_bar = (r1 + r2) / 2.0
            share_ratio_r_bar = (r_bar * alpha) / (r_bar * alpha + beta) if (r_bar * alpha + beta) > 0 else 0
            if share_ratio_r_bar <= 1e-9: continue

            # 计算PAW攻击的收益 (基准收益)
            prob_5_2 = max(0, 1 - alpha - beta - eta) / (1 - r2 * alpha)
            prob_5_4 = eta / (1 - r2 * alpha)
            reward_paw = c["5_2_deny"] * prob_5_2 * share_ratio_r_bar + c["5_4_deny"] * prob_5_4 * share_ratio_r_bar


            reward_bm_paw_fork = c["5_2_accept"] * prob_5_2 * share_ratio_r_bar + c[
                "5_4_accept"] * prob_5_4 * share_ratio_r_bar
            bribe_cost = eps1 * c["5_2_accept"] * prob_5_2* share_ratio_r_bar + eps2 * c[
                "5_4_accept"] * prob_5_4* share_ratio_r_bar
            reward_bm_paw = reward_bm_paw_fork - bribe_cost

            if reward_paw > 1e-9:
                rer = (reward_bm_paw - reward_paw) / reward_paw * 100  # 转换为百分比
            else:
                rer = reward_bm_paw * 100

            if rer > max_rer_for_this_eps:
                max_rer_for_this_eps = rer

    # 返回结果和它的坐标，以便主进程可以把它放回正确的位置
    return i, j, max_rer_for_this_eps


# ==============================================================================
# 3. 修改后的数据生成函数，用于分发任务
# ==============================================================================

def generate_heatmap_data_parallel(alpha, beta, eta, gamma, eps_density=50, r_density=50):
    """
    使用多进程并行生成热力图数据。
    """
    eps_range = np.linspace(0, 1, eps_density)
    r_range = np.linspace(0.01, 0.99, r_density)
    rer_matrix = np.zeros((eps_density, eps_density))

    # 创建一个任务列表，每个任务都是一个包含所有参数的元组
    tasks = []
    for i, eps1 in enumerate(eps_range):
        for j, eps2 in enumerate(eps_range):
            tasks.append((i, j, eps1, eps2, alpha, beta, eta, gamma, r_range))

    # 设置进程数，通常为CPU核心数-1，以保留一个核心给系统
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes for parallel computation...")

    # 创建进程池并分发任务
    # 使用 with 语句可以确保进程池在使用后被正确关闭
    with Pool(processes=num_processes) as pool:
        # pool.imap_unordered 会立即返回一个迭代器，并在任务完成时产生结果
        # 这使得我们可以配合tqdm实时显示进度
        results = list(tqdm(pool.imap_unordered(worker_task, tasks), total=len(tasks), desc="Calculating Heatmap"))

    # 将返回的结果重新组装成矩阵
    for i, j, rer in results:
        # 注意imshow的坐标系，(行,列) -> (j,i)
        rer_matrix[j, i] = rer

    return rer_matrix, eps_range


# ==============================================================================
# 4. 绘图函数 (保持不变)
# ==============================================================================

def plot_heatmap(rer_matrix, eps_range, title):
    """使用matplotlib和seaborn绘制热力图。"""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(rer_matrix, origin='lower', extent=[0, 1, 0, 1],
                    cmap='viridis', aspect='auto')
    cbar = plt.colorbar(im)
    cbar.set_label("Relative Extra Reward (%)")
    plt.contour(eps_range, eps_range, rer_matrix, levels=[0], colors='white', linewidths=2)
    plt.title(title)
    plt.xlabel("The proportion of reward ε₁")
    plt.ylabel("The proportion of reward ε₂")
    plt.show()


# ==============================================================================
# 5. 主程序入口 (必须放在 if __name__ == '__main__': 下)
# ==============================================================================

if __name__ == '__main__':
    # --- 实验参数配置 (与论文图3a一致) ---
    ALPHA = 0.2
    BETA = 0.2
    ETA = 0.2
    GAMMA = 0.5

    # --- 精度配置 ---
    # 密度越高，图像越精细，但计算时间越长
    EPS_DENSITY = 50
    R_DENSITY = 50

    print("Starting heatmap data generation...")
    print(f"Parameters: α={ALPHA}, β={BETA}, η={ETA}, γ={GAMMA}")
    print(f"Grid density: eps={EPS_DENSITY}, r={R_DENSITY}")

    # 1. 并行生成数据
    rer_data, eps_coords = generate_heatmap_data_parallel(ALPHA, BETA, ETA, GAMMA, EPS_DENSITY, R_DENSITY)

    # 2. 绘制图像
    plot_title = f"Attacker's Relative Extra Reward (α={ALPHA}, β={BETA}, η={ETA})"
    plot_heatmap(rer_data, eps_coords, plot_title)