import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- 1. 全局排版设置 ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 11
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

# --- 3. 样式定义 ---
colors = mpl.colormaps['tab10'].colors
r1_style = {'color': colors[0], 'marker': 'o', 'linestyle': '-', 'label': r'$\hat{r}_1$ (Before Adjustment)'}
r2_style = {'color': colors[3], 'marker': 's', 'linestyle': '--', 'label': r'$\hat{r}_2$ (After Adjustment)'}

# --- 4. 创建2x2的分面图 ---
# 增加 figsize 的宽度，为右侧图例留出空间
fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True, sharey=True)
axes_flat = axes.flatten()

for i, alpha in enumerate(alphas):
    ax = axes_flat[i]
    r1_col, r2_col = f'alpha={alpha}_r1', f'alpha={alpha}_r2'

    # 绘制曲线
    ax.plot(df['beta'], df[r1_col], **r1_style, linewidth=1.5, markersize=6, markeredgecolor='black',
            markeredgewidth=0.5)
    ax.plot(df['beta'], df[r2_col], **r2_style, linewidth=1.5, markersize=6, markeredgecolor='black',
            markeredgewidth=0.5)

    # --- 5. 格式化每个子图 ---
    ax.set_title(fr'$\alpha = {alpha}$')
    ax.grid(True, which='major', linestyle=':', linewidth='0.5', color='gray')
    ax.set_xticks([0.1, 0.2, 0.3])
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.2))

# --- 6. 格式化整个图形 ---

# 1. 添加总标题 (Suptitle)
fig.suptitle(r"Optimal Infiltration Ratios Grouped by Attacker Power ($\alpha$)", y=1.0)

# 2. 添加统一的X轴和Y轴标签
fig.supxlabel(r"Victim pool's hash power $\beta$", y=0.06)
fig.supylabel(r'Optimal Infiltration Ratio', x=0.03)

# 3. 创建图例并放置在右侧 (loc='center right')
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5),
           ncol=1, frameon=True, title=r'Infiltration Ratios')

# 4. 调整布局，为右侧图例和总标题留出空间
plt.tight_layout()
# 调整right参数留出图例空间
# 调整left参数将supylabel移到图左侧
# 调整bottom/top参数
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.10, right=0.8, hspace=0.3, wspace=0.2)

# --- 7. 保存为矢量图 ---
output_filename = 'optimal_r.pdf'
plt.savefig(output_filename, format='pdf')

print(f"Figure saved as {output_filename}")
plt.show()