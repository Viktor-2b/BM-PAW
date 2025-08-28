import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 全局排版设置
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    'mathtext.fontset': 'stix',  # 使用高质量STIX字体渲染数学公式
    "font.size": 20,  # 增大基础字号以提升可读性
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 14,
    "axes.linewidth": 1.5,      # 加粗坐标轴边框
    'xtick.major.width': 1.5,   # X轴主刻度线宽度
    'ytick.major.width': 1.5,   # Y轴主刻度线宽度
    'xtick.minor.width': 1.0,   # X轴次刻度线宽度 (可以设细一点或一样粗)
    'ytick.minor.width': 1.0,   # Y轴次刻度线宽度
})

def create_plot(csv_filename, x_column, style_parser, x_label, y_label, legend_config,
                output_filename, x_ticks=None, y_ticks=None):
    """
    一个通用的函数，用于创建和保存一张点线图。
   :param csv_filename: CSV数据文件名
   :param x_column: CSV中作为X轴的列名
   :param style_parser: 一个函数，用于从列名解析出该系列的样式
   :param x_label: X轴的LaTeX标签
   :param y_label: Y轴的LaTeX标签
   :param legend_config: 包含图例所有配置的字典
   :param output_filename: 输出的PDF文件名
   :param x_ticks: (可选) X轴的刻度列表
   :param y_ticks: (可选) Y轴的刻度列表
   """
    # 加载数据
    try:
        data = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: '{csv_filename}' not found. Please create this file with the provided data.")
        return
    # 使用黄金比例设置尺寸
    golden_ratio = (1 + np.sqrt(5)) / 2
    width = 8
    height = width / golden_ratio
    fig, ax = plt.subplots(figsize=(width, height))
    # 创建一个字典，用于后续根据标签查找对应的线条对象（句柄）
    handles_dict = {}

    # 遍历数据中的每一列来绘制曲线
    for col_name in data.columns:
        # 跳过作为X轴的列
        if col_name == x_column:
            continue
        # 调用样式解析函数，获取该列对应的标签和样式字典
        label, style_dict = style_parser(col_name)
        # 从DataFrame中获取Y轴数据
        y_values = data[col_name]
        # 绘制曲线，label用于图例，**style_dict将所有样式一次性应用
        line, = ax.plot(data[x_column], y_values, label=label, **style_dict)
        # 将线条对象存入字典，以便后续手动排序
        handles_dict[label] = line

    # 格式化图表
    ax.set_xlabel(x_label) # 设置X轴标签
    ax.set_ylabel(y_label) # 设置Y轴标签
    # 如果提供了X轴刻度，则进行设置
    if x_ticks is not None:
        ax.set_xticks(x_ticks) # 设置刻度的位置
        # 设置X轴的范围，左右各留出5%的空白
        ax.set_xlim(x_ticks[0] - 0.05 * (x_ticks[-1] - x_ticks[0]),
                    x_ticks[-1] + 0.05 * (x_ticks[-1] - x_ticks[0]))
        # 如果提供了Y轴刻度，则进行设置
    if y_ticks is not None:
        ax.set_yticks(y_ticks)  # 设置刻度的位置
        # 设置Y轴的范围，上下各留出5%的空白
        ax.set_ylim(y_ticks[0] - 0.05 * (y_ticks[-1] - y_ticks[0]),
                    y_ticks[-1] + 0.05 * (y_ticks[-1] - y_ticks[0]))

    # 添加半透明的辅助网格
    ax.grid(True, linestyle='--', alpha=0.6)

    # 在y=0的位置画一条黑色的水平虚线
    ax.axhline(0, color='black', linewidth=0.75, linestyle='--')

    # 手动排序图例
    # 从字典中，按照预先定义好的顺序(legend_config['order'])提取出句柄和标签
    final_handles = [handles_dict[label] for label in legend_config['order'] if label in handles_dict]
    final_labels = [label for label in legend_config['order'] if label in handles_dict]
    # 使用排序后的列表和预设的配置来创建图例
    professional_legend_settings = {
        'fancybox': True,
        'framealpha': 0.7,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    # 将预设样式与用户自定义样式合并
    combined_settings = {**professional_legend_settings, **legend_config['settings']}
    ax.legend(final_handles, final_labels, **combined_settings)

    # 保存图表为PDF矢量图格式
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved as {output_filename}")
    # 关闭图形对象，释放内存
    plt.close(fig)


def style_parser_type1(col_name):
    """
    样式解析器1，从 'eta=0.1,beta=0.1' 这样的列名解析出样式
    :param col_name:列名
    :return:标签和样式
    """
    parts = col_name.replace(' ', '').split(',')
    eta_str, beta_str = parts[0].split('=')[1], parts[1].split('=')[1]
    # 定义样式映射
    colors = {'0.1': '#d62728', '0.2': '#1f77b4', '0.3': '#2ca02c'}
    linestyles = {'0.1': '-', '0.2': '--'}
    markers = {'0.1': 'o', '0.2': 's', '0.3': '^'}
    # 生成LaTeX格式的图例标签
    label = fr'$({float(eta_str):.1f}, {float(beta_str):.1f})$'
    # 将所有样式打包成一个字典
    style = {'color': colors[beta_str], 'marker': markers[beta_str],
             'linestyle': linestyles[eta_str], 'markersize': 8, 'linewidth': 2.5}
    return label, style

# 定义图例的显示顺序和布局
legend_config_1 = {
    'order': [fr'$(0.1, 0.1)$', fr'$(0.1, 0.2)$',
              fr'$(0.1, 0.3)$', fr'$(0.2, 0.1)$',
              fr'$(0.2, 0.2)$', fr'$(0.2, 0.3)$'],
    'settings': {'loc': 'lower right',
                 'ncol': 2,
                 'title': r'$(\eta, \beta)$',
                 'title_fontsize': 16}
}
legend_config_3 = {
    'order': [fr'$(0.1, 0.1)$', fr'$(0.1, 0.2)$',
              fr'$(0.1, 0.3)$', fr'$(0.2, 0.1)$',
              fr'$(0.2, 0.2)$', fr'$(0.2, 0.3)$'],
    'settings': {'loc': 'lower left',
                 'ncol': 2,
                 'title': r'$(\eta, \beta)$',
                 'title_fontsize': 16}
}

def style_parser_type2(col_name):
    """
    样式解析器2，从 'gamma=0.2' 这样的列名解析出样式
    :param col_name:列名
    :return:标签和样式
    """
    gamma_str = col_name.split('=')[1]
    gamma_float = float(gamma_str)

    gamma_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # 使用 tab10, viridis 等色图，它是一组鲜艳的分类颜色
    colors = mpl.colormaps['plasma'](np.linspace(1, 0, len(gamma_values)))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    try:
        # 找到当前gamma值在列表中的位置，以确定颜色和标记
        idx = gamma_values.index(gamma_float)
    except ValueError:
        # 如果遇到未知的gamma值，打印警告并使用默认样式
        print(f"Warning: Unexpected gamma value '{gamma_str}'.")
        return fr'$\gamma={gamma_float:.1f}$', {'color': 'black', 'marker': 'x', 'linestyle': '-'}

    label = fr'${gamma_float:.1f}$'
    style = {'color': colors[idx], 'marker': markers[idx],
             'linestyle': '-', 'markersize': 8, 'linewidth': 2.5}
    return label, style

# 定义图例的显示顺序和布局
legend_config_type2 = {
    'order': [fr'${g:.1f}$' for g in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]],
    'settings': {'loc': 'lower left',
                 'ncol': 2,
                 'title': r'$\gamma$',
                 'title_fontsize': 16}
}

if __name__ == '__main__':
    print("--- Generating Attacker's RER Plots ---")
    create_plot('data/attacker_gamma.csv', 'gamma', style_parser_type1,
                r"The attacker's network capability $\gamma$", r"Attacker's RER (\%)",
                legend_config_1, 'attacker_gamma.pdf', x_ticks=np.arange(0, 1.1, 0.2), y_ticks=np.arange(-0.8, 0.6, 0.2))

    create_plot('data/attacker_alpha.csv', 'alpha', style_parser_type2,
                r"The attacker's mining power $\alpha$", r"Attacker's RER (\%)",
                legend_config_type2, 'attacker_alpha.pdf', x_ticks=np.arange(0, 0.51, 0.1))

    print("\n--- Generating Target's RER Plots ---")
    create_plot('data/target_gamma.csv', 'gamma', style_parser_type1,
                r"The attacker's network capability $\gamma$", r"Target's RER (\%)",
                legend_config_3, 'target_gamma.pdf', x_ticks=np.arange(0, 1.1, 0.2))

    create_plot('data/target_alpha.csv', 'alpha', style_parser_type2,
                r"The attacker's mining power $\alpha$", r"Target's RER (\%)",
                legend_config_type2, 'target_alpha.pdf', x_ticks=np.arange(0, 0.51, 0.1))