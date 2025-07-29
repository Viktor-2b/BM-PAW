from optimizer import optimize_r1_r2


def reproduce_table1():
    """
    复现论文中的表1，计算不同alpha和beta下的最优r1, r2。
    """
    alphas = [0.1, 0.2, 0.3, 0.4]
    betas = [0.1, 0.2, 0.3]

    # 固定的系统参数，与您设定的保持一致
    eta = 0.2
    gamma = 0.5

    # 假设：为了生成表格，使用一个固定的、较小的贿赂率
    epsilon1 = 0.001
    epsilon2 = 0.5

    print("复现表1: The attacker's optimal infiltration mining power r1 and r2")
    print("-" * 70)
    print(f"{'beta':<6} | {'alpha=0.1':<15} | {'alpha=0.2':<15} | {'alpha=0.3':<15} | {'alpha=0.4':<15}")
    print("-" * 70)

    for beta in betas:
        row_str = f"{beta:<6} | "
        for alpha in alphas:
            if 1.0 - alpha - beta - eta < 0:
                row_str += f"{'N/A':<15} | "
                continue

            r1_opt, r2_opt = optimize_r1_r2(alpha, beta, eta, gamma, epsilon1, epsilon2)

            cell_str = f"{r1_opt:.4f}({r2_opt:.4f})"
            row_str += f"{cell_str:<15} | "

        print(row_str)

    print("-" * 70)


if __name__ == '__main__':
    reproduce_table1()