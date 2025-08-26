import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(df, value_col, title):
    """
    根据DataFrame中的数据绘制热力图。
    """
    # 将长格式数据转换为网格/矩阵格式
    heatmap_data = df.pivot(index='eps2', columns='eps1', values=value_col)

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=False, fmt=".2f", cmap='viridis',
                xticklabels=np.round(heatmap_data.columns.values, 2),
                yticklabels=np.round(heatmap_data.index.values, 2))

    plt.title(title, fontsize=16)
    plt.xlabel(r'Bribe Ratio in Case 5-2 ($\epsilon_1$)', fontsize=12)
    plt.ylabel(r'Bribe Ratio in Case 5-4 ($\epsilon_2$)', fontsize=12)
    plt.gca().invert_yaxis()  # 翻转Y轴，使(0,0)在左下角
    plt.savefig(f"{value_col}_heatmap.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # 读取之前计算好的结果文件
    try:
        results_df = pd.read_csv('nested_search_results.csv')
    except FileNotFoundError:
        print("错误: 未找到 'nested_search_results.csv' 文件。")
        print("请先运行 'run_nested_grid_search.py' 来生成结果文件。")
        exit()

    print("成功读取结果文件。文件信息:")
    print(results_df.info())

    # 1. 绘制攻击者最大收益的热力图
    plot_heatmap(
        df=results_df,
        value_col='max_reward',
        title="Attacker's Max Reward vs. Bribe Ratios (ε1, ε2)"
    )

    # 2. 绘制最优 r1 的热力图
    plot_heatmap(
        df=results_df,
        value_col='r1_opt',
        title="Optimal r1 vs. Bribe Ratios (ε1, ε2)"
    )

    # 3. 绘制最优 r2 的热力图
    plot_heatmap(
        df=results_df,
        value_col='r2_opt',
        title="Optimal r2 vs. Bribe Ratios (ε1, ε2)"
    )

    # 4. 找到全局最优贿赂策略
    best_strategy = results_df.loc[results_df['max_reward'].idxmax()]
    print("\n全局最优策略分析:")
    print(best_strategy)