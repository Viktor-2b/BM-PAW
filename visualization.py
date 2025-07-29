import matplotlib.pyplot as plt

def plot_heatmap(x_range, y_range, data_grid, title, x_label, y_label, colorbar_label):
    """
    一个通用的函数，用于绘制热力图 (Heatmap)。
    """
    plt.figure(figsize=(8, 6.5))

    # 使用 pcolormesh 绘制热力图
    # 使用 vmin 和 vmax 来固定颜色条的范围，让不同图之间可以比较
    plt.pcolormesh(x_range, y_range, data_grid*100, shading='auto', cmap='viridis')

    plt.colorbar(label=colorbar_label)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # 绘制RER=0的等高线 (那条黑色的分界线)
    # 增加更多levels可以让等高线更平滑
    contours = plt.contour(x_range, y_range, data_grid, levels=[0], colors='black', linewidths=2)
    plt.clabel(contours, inline=True, fontsize=10, fmt='RER = %1.1f')

    plt.show()