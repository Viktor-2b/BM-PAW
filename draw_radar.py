import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- 1. 全局排版设置 ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    'mathtext.fontset': 'stix',
    "text.latex.preamble": r"\usepackage{mathptmx}",
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 14,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
})

# --- 2. 准备数据 ---
data_dict = {
    'beta': [0.1, 0.2, 0.3],
    'alpha=0.1_r1': [0.1404, 0.3039, 0.2745], 'alpha=0.1_r2': [0.9985, 0.9996, 0.6771],
    'alpha=0.2_r1': [0.1222, 0.2844, 0.3079], 'alpha=0.2_r2': [0.6787, 0.9993, 0.3098],
    'alpha=0.3_r1': [0.1254, 0.2928, 0.3691], 'alpha=0.3_r2': [0.4903, 0.7998, 0.1313],
    'alpha=0.4_r1': [0.1372, 0.3271, 0.4791], 'alpha=0.4_r2': [0.3938, 0.6334, 0.0002],
}
df = pd.DataFrame(data_dict)
alphas = [0.1, 0.2, 0.3, 0.4]
betas = df['beta'].tolist()

# --- 3. 样式定义 ---
colors = mpl.colormaps['tab10'].colors
r1_style = {'color': colors[0], 'marker': 'o', 'linestyle': '-', 'label': r'$\hat{r}_1$ (Before Adjustment)'}
r2_style = {'color': colors[3], 'marker': 's', 'linestyle': '--', 'label': r'$\hat{r}_2$ (After Adjustment)'}

# --- 4. 计算雷达图角度 ---
# 每个beta值对应雷达图上的一个轴
num_vars = len(betas)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# --- 5. 创建2x2的雷达图分面图 ---
fig, axes = plt.subplots(2, 2, figsize=(8, 8), subplot_kw=dict(polar=True))
axes_flat = axes.flatten()

for i, alpha in enumerate(alphas):
    ax = axes_flat[i]
    r1_col, r2_col = f'alpha={alpha}_r1', f'alpha={alpha}_r2'

    # 准备需要闭合的数据
    r1_data = df[r1_col].tolist() + df[r1_col].tolist()[:1]
    r2_data = df[r2_col].tolist() + df[r2_col].tolist()[:1]

    # 绘制曲线
    ax.plot(angles, r1_data, **r1_style, linewidth=1.5, markersize=6, markeredgecolor='black',
            markeredgewidth=0.5)
    ax.plot(angles, r2_data, **r2_style, linewidth=1.5, markersize=6, markeredgecolor='black',
            markeredgewidth=0.5)

    # 填充颜色，增加可读性
    ax.fill(angles, r1_data, color=r1_style['color'], alpha=0.15)
    ax.fill(angles, r2_data, color=r2_style['color'], alpha=0.15)

    # --- 6. 格式化每个子图 (雷达图风格) ---
    ax.set_title(fr'$\alpha = {alpha}$')  # 避免标题与刻度重叠

    # 设置雷达图的轴标签（beta值）
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{b}' for b in betas], fontsize=14)

    # 设置雷达图的径向轴（Y轴）范围和刻度
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax.tick_params(axis='y', labelsize=10)

    # 参考示例，设置雷达图的起始方向和旋转方向
    ax.set_theta_zero_location("N")  # 将0度设置在北方（顶部）
    ax.set_theta_direction(-1)  # 设置为顺时针方向

# --- 7. 格式化整个图形 ---
# 添加总标题
fig.suptitle(r"Optimal Infiltration Ratios Grouped by Attacker Power ($\alpha$)", fontsize=20)
fig.supxlabel(r"Victim pool's hash power $\beta$", fontsize=20, y=0.06)
# 创建图例并放置在图形中央
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
           ncol=1, frameon=True, title=r'Infiltration Ratios')

# 调整布局
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88, wspace=0.1, hspace=0.25)

# --- 8. 保存为矢量图 ---
output_filename = 'optimal_r.pdf'
plt.savefig(output_filename, format='pdf')

print(f"Figure saved as {output_filename}")