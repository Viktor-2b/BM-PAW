import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib的全局样式
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20

# RUM模型
def markov_monte_carlo_rum(alpha, N):
    p11 = 0.7698 + 0.2302 * alpha
    p12 = 0.2302 * (1 - alpha)
    p21 = 0.7698 + 0.2302 * alpha
    p22 = 0.2302 * (1 - alpha)
    P = np.array([[p11, p12],
                  [p21, p22]])
    state = 0
    state_counts = np.array([0, 0])
    for _ in range(N):
        if state == 0:
            r = np.random.rand()
            if r < P[0, 0]:
                state = 0
            else:
                state = 1
        else:
            r = np.random.rand()
            if r < P[1, 0]:
                state = 0
            else:
                state = 1
        state_counts[state] += 1
    return state_counts / N

# UUM模型
def markov_monte_carlo_uum(alpha, N):
    p11 = 0.3596 + 0.6404 * alpha
    p12 = 0.6404 * (1 - alpha)
    p21 = 0.3596 + 0.6404 * alpha
    p22 = 0.6404 * (1 - alpha)
    P = np.array([[p11, p12],
                  [p21, p22]])
    state = 0
    state_counts = np.array([0, 0])
    for _ in range(N):
        if state == 0:
            r = np.random.rand()
            if r < P[0, 0]:
                state = 0
            else:
                state = 1
        else:
            r = np.random.rand()
            if r < P[1, 0]:
                state = 0
            else:
                state = 1
        state_counts[state] += 1
    return state_counts / N

# SUUM模型
def markov_monte_carlo_suum(alpha, num_simulations, num_attack_states):
    num_total_states = 2 + num_attack_states
    state_counts = np.zeros(num_total_states)
    current_state = 0

    for _ in range(num_simulations):
        if current_state == 0:  # Deploy状态
            r = np.random.rand()
            if r < 0.3596 * (1 - alpha):
                current_state = 0
            elif r < 0.3596 * (1 - alpha) + (0.3596 + 0.6404 * alpha):
                current_state = 1
            else:
                if alpha > 0:  # 只有当alpha>0时才可能进入Attack状态
                    current_state = 2
                else:
                    current_state = 0
        elif current_state == 1:  # Downgrade状态
            r = np.random.rand()
            if r < 0.6404 * (1 - alpha):
                current_state = 1
            else:
                current_state = 0
        else:  # Attack系列状态
            if current_state < num_total_states - 1:
                r = np.random.rand()
                if r < alpha:
                    current_state += 1
                else:
                    current_state = 0
            else:
                r = np.random.rand()
                if r < alpha:
                    current_state = current_state
                else:
                    current_state = 0
        state_counts[current_state] += 1
    return state_counts / num_simulations

# 计算相对奖励的函数
def calculate_rr_attack_rum(alpha, p_attack):
    return alpha / (1 - p_attack * alpha)

def calculate_rr_attack_uum(alpha, p_attack):
    return alpha / (1 - p_attack * alpha)

def calculate_rr_attack_suum(alpha, p_attack1, p_deploy):
    denominator = alpha + (1 - p_attack1) * (1 - alpha) - (1 - p_deploy) * alpha
    return alpha / denominator

# 计算诚实挖矿的相对奖励
def calculate_rr_honest(alpha):
    return alpha  # 诚实挖矿的相对奖励为alpha

# 设定参数
N = 100000
num_attack_states = 10
alphas = np.arange(0, 0.51, 0.01)

# 计算三种模型的相对奖励
rr_rum = []
rr_uum = []
rr_suum = []
rr_honest = []

# 使用SUUM.py中的稳态概率计算
steady_state_probs = []

# 计算稳态概率
for alpha in np.arange(0, 1.01, 0.01):
    state_counts = np.zeros(2 + num_attack_states)
    current_state = 0  # 初始状态设为Deploy

    for _ in range(1000000):  # 模拟次数
        if current_state == 0:  # Deploy状态
            r = np.random.rand()
            if r < 0.3596 * (1 - alpha):
                current_state = 0
            elif r < 0.3596 * (1 - alpha) + 0.6404 * (1 - alpha):
                current_state = 1
            else:
                if alpha > 0:
                    current_state = 2
                else:
                    current_state = 0
        elif current_state == 1:  # Downgrade状态
            r = np.random.rand()
            if r < 0.6404 * (1 - alpha):
                current_state = 1
            else:
                current_state = 0
        else:  # Attack系列状态
            attack_num = current_state - 2
            if attack_num < num_attack_states - 1:
                r = np.random.rand()
                if r < alpha:
                    current_state += 1
                else:
                    current_state = 0
            else:
                r = np.random.rand()
                if r < alpha:
                    current_state = current_state
                else:
                    current_state = 0

        state_counts[current_state] += 1

    # 计算稳态概率
    total_count = 1000000
    steady_state_probs.append(state_counts / total_count)

for alpha in alphas:
    # RUM
    result_rum = markov_monte_carlo_rum(alpha, N)
    p_attack_rum = result_rum[1]  # RUM的Attack状态概率
    rr_rum.append(calculate_rr_attack_rum(alpha, p_attack_rum))
    
    # UUM
    result_uum = markov_monte_carlo_uum(alpha, N)
    p_attack_uum = result_uum[1]  # UUM的Attack状态概率
    rr_uum.append(calculate_rr_attack_uum(alpha, p_attack_uum))
    
    # SUUM
    p_attack1_suum = steady_state_probs[int(alpha * 100)][2]  # SUUM的Attack 1状态概率
    p_downgrade_suum = steady_state_probs[int(alpha * 100)][1]  # SUUM的Downgrade状态概率
    rr_suum.append(calculate_rr_attack_suum(alpha, p_attack1_suum, p_downgrade_suum))
    
    # 诚实挖矿
    rr_honest.append(calculate_rr_honest(alpha))

# 使用黄金比例设置图像尺寸
golden_ratio = (1 + np.sqrt(5)) / 2
width = 6
height = width / golden_ratio

# 创建图形
fig, ax = plt.subplots(figsize=(width, height))

# 绘制曲线
ax.plot(alphas, rr_rum, 'b-', label='RUM', linewidth=3, marker='o', markersize=10, markevery=10)
ax.plot(alphas, rr_uum, 'r--', label='UUM', linewidth=3, marker='s', markersize=10, markevery=10)
ax.plot(alphas, rr_suum, 'g:', label='SUUM', linewidth=3, marker='^', markersize=10, markevery=10)
ax.plot(alphas, rr_honest, 'k-.', label='Honest Mining', linewidth=3, marker='d', markersize=10, markevery=10)

# 设置坐标轴标签
ax.set_xlabel('Adversary Relative Power $\\alpha$', fontsize=20)
ax.set_ylabel('$RR_{Attack}$', fontsize=20)

# 设置标题
ax.set_title('Comparison of $RR_{Attack}$', fontsize=20, pad=20)

# 设置网格
ax.grid(True, linestyle='--', alpha=0.7)

# 设置图例
ax.legend(fontsize=20, frameon=True, fancybox=True, 
          loc='upper left',
          facecolor='white', edgecolor='none',
          framealpha=0.5)

# 设置坐标轴刻度
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1)

# 设置边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('relative_rewards_comparison.pdf', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()