import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import pandas as pd


# ==============================================================================
# --- 主程序 (大部分无需修改) ---
# ==============================================================================

def build_color_map(image_path, val_top, val_bottom):
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误：找不到颜色条图片 '{image_path}'")
        return None, None

    width, height = img.size
    map_colors = []
    map_values = []

    for y in range(height):
        row_pixels = [img.getpixel((x, y)) for x in range(width // 4, 3 * width // 4)]
        avg_r = sum(p[0] for p in row_pixels) / len(row_pixels)
        avg_g = sum(p[1] for p in row_pixels) / len(row_pixels)
        avg_b = sum(p[2] for p in row_pixels) / len(row_pixels)
        map_colors.append([avg_r, avg_g, avg_b])

        relative_pos = y / (height - 1)
        value = val_top + relative_pos * (val_bottom - val_top)
        map_values.append(value)

    kdtree = cKDTree(map_colors)
    print(f"颜色条 '{image_path}' 分析完成，已构建 {len(map_colors)} 个级别的查找树。")
    return kdtree, np.array(map_values)


def restore_data_from_image(main_img_path, kdtree, map_values, grid_w, grid_h):
    try:
        main_img = Image.open(main_img_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误：找不到主图图片 '{main_img_path}'")
        return None

    resized_img = main_img.resize((grid_w, grid_h), Image.Resampling.LANCZOS)
    pixel_data = np.array(resized_img)
    color_list = pixel_data.reshape(-1, 3)

    _, indices = kdtree.query(color_list)
    value_data_flat = map_values[indices]
    value_grid = value_data_flat.reshape(grid_h, grid_w)

    return value_grid


def process_chart(config):
    """
    一个通用的函数，用于处理单张图表的数据还原。
    """
    print("\n" + "=" * 50)
    print(f"开始处理: {config['output_csv']}")
    print("=" * 50)

    color_kdtree, value_array = build_color_map(
        config['colorbar_path'],
        config['value_top'],
        config['value_bottom']
    )

    if not color_kdtree:
        return

    restored_grid = restore_data_from_image(
        config['main_chart_path'],
        color_kdtree,
        value_array,
        config['grid_size_x'],
        config['grid_size_y']
    )

    if restored_grid is not None:
        x_coords = np.linspace(config['x_min'], config['x_max'], config['grid_size_x'])
        y_coords = np.linspace(config['y_min'], config['y_max'], config['grid_size_y'])

        df = pd.DataFrame(
            np.flipud(restored_grid),
            index=np.round(y_coords, 2),
            columns=np.round(x_coords, 2)
        )
        df.index.name = config['y_label']
        df.columns.name = config['x_label']

        print(f"--- 数据还原成功: {config['output_csv']} ---")
        print(f"已生成 {df.shape[0]}x{df.shape[1]} 的数据网格。")

        try:
            df.to_csv(config['output_csv'])
            print(f"完整数据已成功保存到 '{config['output_csv']}' 文件中。")
        except Exception as e:
            print(f"\n保存文件时出错: {e}")


# ==============================================================================
# --- 配置文件区域 ---
# ==============================================================================

# 为每张图创建一个配置字典
chart1_config = {
    "main_chart_path": "main_chart1.png",
    "colorbar_path": "colorbar1.png",
    "output_csv": "restored_data1.csv",
    "value_top": 1.4204,
    "value_bottom": -5.5347,
    "grid_size_x": 101,
    "grid_size_y": 101,
    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0,
    "x_label": "e1",
    "y_label": "e2",
}

chart2_config = {
    "main_chart_path": "main_chart2.png",
    "colorbar_path": "colorbar2.png",
    "output_csv": "restored_data2.csv",
    "value_top": 0.2597,
    "value_bottom": -5.6883,
    "grid_size_x": 51,  # (0.5 - 0) / 0.01 + 1 = 51
    "grid_size_y": 51,  # (0.5 - 0) / 0.01 + 1 = 51
    "x_min": 0.0,
    "x_max": 0.5,
    "y_min": 0.0,
    "y_max": 0.5,
    "x_label": "alpha",
    "y_label": "eta",
}

chart3_config = {
    "main_chart_path": "main_chart3.png",
    "colorbar_path": "colorbar3.png",
    "output_csv": "restored_data3.csv",
    "value_top": 5.0391,
    "value_bottom": -1.6512,
    "grid_size_x": 101,
    "grid_size_y": 101,
    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0,
    "x_label": "e1",
    "y_label": "e2",
}

chart4_config = {
    "main_chart_path": "main_chart4.png",
    "colorbar_path": "colorbar4.png",
    "output_csv": "restored_data4.csv",
    "value_top": 15.3630,
    "value_bottom": 0.0825,
    "grid_size_x": 51,
    "grid_size_y": 51,
    "x_min": 0.0,
    "x_max": 0.5,
    "y_min": 0.0,
    "y_max": 0.5,
    "x_label": "alpha",
    "y_label": "eta",
}

# --- 运行所有任务 ---
process_chart(chart1_config)
process_chart(chart2_config)
process_chart(chart3_config)
process_chart(chart4_config)

print("\n所有图表处理完成！")