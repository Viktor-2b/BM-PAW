import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LightSource

# 样式渲染
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    'mathtext.fontset': 'stix',
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
print("LaTeX渲染已启用，将使用 Times New Roman 和 Stix 字体生成图表。")

# 图表配置
charts_config = [
    {
        "input_csv": "data/restored_data1.csv", "output_pdf": "attacker_e1_e2.pdf",
        "plot_type": "2d",
        "x_label": r'The proportion of reward $\varepsilon_1$',
        "y_label": r'The proportion of reward $\varepsilon_2$',
        "z_label": r"Attacker's RER (\%)",
        "cmap": 'viridis',
        "draw_boundary_line": True,
        "colorbar_ticks": [-4, -2, 0],
        "filter_method": 'plane_fit'  # 使用平面拟合
    },
    {
        "input_csv": "data/restored_data2.csv", "output_pdf": "attacker_alpha_eta.pdf",
        "plot_type": "3d",
        "x_label": r"The attacker's mining power $\alpha$",
        "y_label": r"The target pool's mining power $\eta$",
        "z_label": r"Attacker's RER (\%)",
        "cmap": 'coolwarm',
        "colorbar_ticks": [-4, -2, 0],
        "filter_method": 'gaussian', "filter_strength": 2.0,  # 使用高斯滤波
        "view_init_elev": 20, "view_init_azim": 65  # 观察视角
    },
    {
        "input_csv": "data/restored_data3.csv", "output_pdf": "target_e1_e2.pdf",
        "plot_type": "2d",
        "x_label": r'The proportion of reward $\varepsilon_1$',
        "y_label": r'The proportion of reward $\varepsilon_2$',
        "z_label": r"Target's RER (\%)",
        "cmap": 'viridis',
        "draw_boundary_line": True, "colorbar_ticks": [0, 2, 4],
        "filter_method": 'plane_fit'  # 使用平面拟合
    },
    {
        "input_csv": "data/restored_data4.csv", "output_pdf": "target_alpha_eta.pdf",
        "plot_type": "3d",
        "x_label": r"The attacker's mining power $\alpha$",
        "y_label": r"The target pool's mining power $\eta$",
        "z_label": r"Target's RER (\%)",
        "cmap": 'coolwarm',
        "colorbar_ticks": [3, 6, 9, 12],
        "filter_method": 'gaussian', "filter_strength": 2.0  # 使用高斯滤波
    }
]


# ==============================================================================
# --- 绘图函数 ---
# ==============================================================================

def apply_filter(x, y, z, method, strength=1.5):
    """应用指定的滤波或拟合方法"""
    if method == 'plane_fit':
        x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()
        mask = ~np.isnan(z_flat)
        a = np.c_[x_flat[mask], y_flat[mask], np.ones(len(x_flat[mask]))]

        coefficients, _, _, _ = np.linalg.lstsq(a, z_flat[mask], rcond=None)
        a, b, c = coefficients

        z_fit = a * x + b * y + c
        # 同时返回拟合后的数据和平面系数
        return z_fit, (a, b, c)

    elif method == 'gaussian':
        # 对于高斯滤波，我们需要先填充NaN值
        z_filled = pd.DataFrame(z).interpolate(method='linear', limit_direction='both', axis=0).values
        z_filtered = gaussian_filter(z_filled, sigma=strength)
        return z_filtered, None

    return z, None


def plot_2d_heatmap(config):
    df = pd.read_csv(config['input_csv'], index_col=0)
    df.columns = [float(c) for c in df.columns]
    df = df.sort_index(axis=0).sort_index(axis=1)

    x, y = np.meshgrid(df.columns.values, df.index.values)
    z = df.values

    # --- 核心修正 2：应用指定的处理方法 ---
    z_processed, plane_coefficients = apply_filter(x, y, z, config.get('filter_method', 'none'))

    golden_ratio = (1 + np.sqrt(5)) / 2
    width = 8
    height = width / golden_ratio
    fig, ax = plt.subplots(figsize=(width, height))

    # --- 使用处理后数据的范围，确保完整绘图 ---
    vmin, vmax = np.min(z_processed), np.max(z_processed)
    heatmap = ax.imshow(z_processed,
                        cmap=config['cmap'],
                        vmin=vmin,
                        vmax=vmax,
                        origin='lower',  # <-- 重要：确保Y轴的原点在下方，与contourf行为一致
                        extent=[x.min(), x.max(), y.min(), y.max()],  # <-- 重要：设置正确的坐标范围
                        aspect='auto'  # <-- 重要：允许图形根据figsize设置为长方形（黄金比例）
                        )
    if config.get("draw_boundary_line") and plane_coefficients is not None:
        a, b, c = plane_coefficients

        if b != 0:
            slope = -a / b
            intercept = -c / b

            bounds = [x.min(), x.max(), y.min(), y.max()]
            points = []
            x_vals = np.array([bounds[0], bounds[1]])
            y_vals = slope * x_vals + intercept

            if bounds[2] <= y_vals[0] <= bounds[3]: points.append((x_vals[0], y_vals[0]))
            if bounds[2] <= y_vals[1] <= bounds[3]: points.append((x_vals[1], y_vals[1]))
            if slope != 0:
                x_at_y_min = (bounds[2] - intercept) / slope
                if bounds[0] <= x_at_y_min <= bounds[1]: points.append((x_at_y_min, bounds[2]))
                x_at_y_max = (bounds[3] - intercept) / slope
                if bounds[0] <= x_at_y_max <= bounds[1]: points.append((x_at_y_max, bounds[3]))

            if len(points) >= 2:
                points = sorted(points)
                line_points = np.array(points)
                ax.plot(line_points[:, 0], line_points[:, 1], color='black', linewidth=2.5)
                legend_elements = [Line2D([0], [0], color='black', lw=2.5, label='Critical Boundary (RER=0)')]
                ax.legend(handles=legend_elements, loc='upper right', facecolor='white', framealpha=0.8)
    divider = make_axes_locatable(ax)  # 创建一个与主坐标轴关联的对象
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 从 divider 的右侧 附加 一个新的坐标轴 用于放置颜色条
    cbar = fig.colorbar(heatmap, cax=cax, ticks=config['colorbar_ticks'])
    cbar.set_label(config['z_label'])

    ax.set_xlabel(config['x_label'])
    ax.set_ylabel(config['y_label'])

    plt.savefig(config['output_pdf'], bbox_inches='tight')
    plt.close(fig)
    print(f"成功生成2D图: '{config['output_pdf']}'")


def plot_3d_surface(config):
    df = pd.read_csv(config['input_csv'], index_col=0)
    df.columns = [float(c) for c in df.columns]
    df = df.sort_index(axis=0).sort_index(axis=1)

    x, y = np.meshgrid(df.columns.values, df.index.values)
    z = df.values
    z_processed, _ = apply_filter(x, y, z, config['filter_method'], config.get('filter_strength', 1.5))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(z_processed, cmap=plt.get_cmap(config['cmap']), vert_exag=0.1, blend_mode='soft')

    vmin, vmax = np.min(z_processed), np.max(z_processed)
    surf = ax.plot_surface(x, y, z_processed, cmap=config['cmap'],facecolors=rgb,
                           linewidth=0, antialiased=True,
                           rcount=300, ccount=300,
                           vmin=vmin, vmax=vmax)
    # # 计算等高线投影
    # if z.min()>0:
    #     contourf_offset=vmin * 1.1
    # else:
    #     contourf_offset = vmax +  0.1 * (vmax - vmin)
    # ax.contourf(x, y, z_processed, zdir='z', offset=contourf_offset,
    #             cmap=config['cmap'], alpha=0.7)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)  # 移除网格线

    ax.set_xlabel(config['x_label'], labelpad=15)
    ax.set_ylabel(config['y_label'], labelpad=15)

    # 颜色条映射回原始数据
    vmin, vmax = np.nanmin(z_processed), np.nanmax(z_processed)
    norm = plt.Normalize(vmin, vmax)
    mappable = plt.cm.ScalarMappable(cmap=config['cmap'], norm=norm)
    mappable.set_array(z_processed)

    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, pad=-0.03, ticks=config['colorbar_ticks'])
    cbar.set_label(config['z_label'])

    elev = config.get('view_init_elev', 30)
    azim = config.get('view_init_azim', -120)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout(pad=1.5)
    plt.savefig(config['output_pdf'], bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"成功生成3D图: '{config['output_pdf']}'")


# --- 运行所有绘图任务 ---
for chart_config in charts_config:
    try:
        if chart_config['plot_type'] == '2d':
            plot_2d_heatmap(chart_config)
        elif chart_config['plot_type'] == '3d':
            plot_3d_surface(chart_config)
    except FileNotFoundError:
        print(f"错误：输入文件 '{chart_config['input_csv']}' 未找到，跳过该图表。")
    except Exception as e:
        print(f"处理 '{chart_config['output_pdf']}' 时发生未知错误: {e}")

print("\n所有图表生成完毕！")
