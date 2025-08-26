import random
from optimizer import optimize_r, calculate_c


class Simulator:
    """
    一个用于BM-PAW攻击的蒙特卡洛模拟器。
    这个类的核心是run_simulation方法，它通过大量随机事件来模拟
    BM-PAW攻击过程，并计算各方收益，从而验证理论模型的正确性。
    """

    def __init__(self, alpha, beta, eta, gamma, epsilon1, epsilon2, grid_density=101):
        """
        初始化模拟器。
        :param alpha: 攻击者算力
        :param beta: 受害者算力
        :param eta: 目标算力
        :param gamma: 其他矿工在分叉时选择攻击者分支的比例
        :param epsilon1: Case 5-2中的贿赂比例
        :param epsilon2: Case 5-4中的贿赂比例
        """
        # 1. 保存核心系统参数
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.delta = max(0, 1.0 - alpha - beta - eta)

        # 2. 参数有效性检查
        if not (0 < self.alpha < 0.5):
            raise ValueError("攻击者算力 alpha 必须在 (0, 0.5) 之间")
        if self.alpha + self.beta + self.eta >= 1.0:
            raise ValueError("所有已知矿池算力之和必须小于1")

        # 3. 调用优化器，找到并保存最优的 r1 和 r2  TODO 四重网格搜索
        optimal_r1, optimal_r2, _  = optimize_r(
            self.alpha, self.beta, self.eta, self.gamma,
            self.epsilon1, self.epsilon2,
            grid_density
        )
        self.r1 = optimal_r1
        self.r2 = optimal_r2

    def run_simulation(self, num_rounds=1000000):
        """
        运行指定轮次的挖矿过程，并累积奖励。
        """
        # 初始化所有奖励计数器
        # BM-PAW 策略下的奖励
        attacker_reward_bm_paw = 0
        target_reward_bm_paw = 0
        # 作为对比基线的 PAW 策略下的奖励 (即贿赂总是不成功)
        attacker_reward_paw = 0
        target_reward_paw = 0
        # 计算平均渗透算力
        r_bar = (self.r1 + self.r2) / 2.0
        # 区块奖励归一化
        block_reward = 1
        # 计算份额奖励
        share_r1 = (self.r1 * self.alpha) / (self.r1 * self.alpha + self.beta) if (
                                                                                              self.r1 * self.alpha + self.beta) > 0 else 0
        share_r_bar = (r_bar * self.alpha) / (r_bar * self.alpha + self.beta) if (
                                                                                             r_bar * self.alpha + self.beta) > 0 else 0
        state = 'INITIAL'  # 初始状态

        for _ in range(num_rounds):
            # 1. 模拟谁挖到了下一个块
            rand = random.random()

            if state == 'INITIAL':
                # 在初始状态，所有矿工 (总算力为1) 都在竞争
                if rand < (1 - self.r1) * self.alpha:
                    # Case 1 attacker_innocent
                    attacker_reward_bm_paw += block_reward
                    attacker_reward_paw += block_reward
                elif rand < (1 - self.r1) * self.alpha + self.beta + self.eta + self.delta:
                    # Case 2 others
                    pass
                elif rand < (1 - self.r1) * self.alpha + self.beta:
                    # Case 3 victim
                    attacker_reward_bm_paw += share_r1
                    attacker_reward_paw += share_r1
                elif rand < (1 - self.r1) * self.alpha + self.beta + self.eta:
                    # Case 4 target
                    target_reward_bm_paw += block_reward
                    target_reward_paw += block_reward
                else:  # 剩余的 r1 * alpha 部分
                    # Case 5 attacker_infiltration
                    state = 'WITHHOLDING'

            elif state == 'WITHHOLDING':
                denominator = 1.0 - self.r2 * self.alpha  # 扣块后的总和算力

                # 计算攻击者获胜概率
                c_probs = calculate_c(self.alpha, self.beta, self.eta, self.gamma, self.delta, self.r2)
                # 计算每个部分的相对算力
                prob_attacker_innocent = ((1 - self.r2) * self.alpha) / denominator  # Case 5-1
                prob_others = self.delta / denominator  # Case 5-2
                prob_victim = self.beta / denominator  # Case 5-3
                # prob_target = self.eta / public_miners_power # Case 5-4

                # 掷一个新的骰子，来决定谁发现了扣块后的新块
                rand_public = random.random()

                state = 'INITIAL'  # 无论发生什么，状态都会重置

                if rand_public < prob_attacker_innocent:
                    # Case 5-1 attacker_innocent
                    attacker_reward_bm_paw += block_reward
                    attacker_reward_paw += block_reward

                elif rand_public < prob_attacker_innocent + prob_others:
                    # Case 5-2 others
                    # --- BM-PAW 逻辑 ---
                    if random.random() < c_probs["c_5_2_accept"]:
                        attacker_reward_bm_paw += share_r_bar
                    attacker_reward_bm_paw -= self.epsilon1 * share_r_bar
                    target_reward_bm_paw += self.epsilon1 * share_r_bar
                    # --- PAW 基线逻辑 ---
                    if random.random() < c_probs["c_5_2_deny"]:
                        attacker_reward_paw += share_r_bar

                elif rand_public < prob_attacker_innocent + prob_others + prob_victim:
                    # Case 5-3 victim
                    attacker_reward_bm_paw += share_r_bar
                    attacker_reward_paw += share_r_bar
                else:
                    # Case 5-4 target
                    # --- BM-PAW 逻辑 ---
                    cond = c_probs["c_5_4_accept"] > c_probs["c_5_4_deny"]
                    # cond = self.epsilon2*share_r_bar > 1 - c_probs["c_5_4_deny"]
                    if cond: # 目标接受贿赂
                        if random.random() < c_probs["c_5_4_accept"]:
                            attacker_reward_bm_paw += share_r_bar
                        attacker_reward_bm_paw -= self.epsilon2 * share_r_bar
                        target_reward_bm_paw += self.epsilon2 * share_r_bar
                    else: # 目标拒绝贿赂
                        if random.random() < c_probs["c_5_4_deny"]:
                            attacker_reward_bm_paw += 0.01*share_r_bar
                        else:
                            target_reward_bm_paw += block_reward
                    # --- PAW 基线逻辑 ---
                    if random.random() < c_probs["c_5_4_deny"]:
                        attacker_reward_paw += share_r_bar
                    else:
                        target_reward_paw += block_reward
        # 计算均值
        avg_attacker_reward_bm_paw = attacker_reward_bm_paw / num_rounds
        avg_target_reward_bm_paw = target_reward_bm_paw / num_rounds
        avg_attacker_reward_paw = attacker_reward_paw / num_rounds
        avg_target_reward_paw = target_reward_paw / num_rounds
        return avg_attacker_reward_bm_paw, avg_target_reward_bm_paw, avg_attacker_reward_paw, avg_target_reward_paw
