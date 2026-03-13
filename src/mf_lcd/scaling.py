import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage import io
from fluo_ts_assay import extract_image_with_mask
import matplotlib.pyplot as plt
import openpyxl
# ==============================================================================
# 1. 核心计算函数定义
# ==============================================================================

def find_max(x_values, y_values):
    """
    找到曲线y最大值及其对应的x坐标。

    参数:
    x_values (np.array): x坐标数组。
    y_values (np.array): y坐标数组。

    返回:
    tuple: (最大y值, 对应的x坐标)。
    """
    try:
        max_y = np.max(y_values)
        max_y_index = np.argmax(y_values)
        max_x = x_values[max_y_index]
        return max_y, max_x
    except ValueError:
        # 处理空数组的情况
        return np.nan, np.nan

def _interpolate_x(x1, y1, x2, y2, target_y):
    """
    在两点之间线性插值，找到target_y对应的x值。
    这是一个辅助函数。
    """
    if y2 == y1:
        return x1
    return x1 + (x2 - x1) * (target_y - y1) / (y2 - y1)

def find_front(x_values, y_values, target_y):
    """
    找到指定y值在曲线上升沿的交点x坐标。

    参数:
    x_values (np.array): x坐标数组。
    y_values (np.array): y坐标数组。
    target_y (float): 目标y值。

    返回:
    float: 交叉点的x坐标，如果找不到则返回np.nan。
    """
    try:
        max_y_index = np.argmax(y_values)
        # 只考虑最大值点左侧（上升沿）的数据
        rising_x = x_values[:max_y_index + 1]
        rising_y = y_values[:max_y_index + 1]

        # 找到穿过target_y的点的索引
        # np.diff(np.sign(...)) 不为0的地方就是符号改变的地方，即穿越点
        cross_indices = np.where(np.diff(np.sign(rising_y - target_y)))[0]

        if len(cross_indices) > 0:
            # 取第一个穿越点
            idx1 = cross_indices[0]
            idx2 = idx1 + 1
            return _interpolate_x(rising_x[idx1], rising_y[idx1], rising_x[idx2], rising_y[idx2], target_y)
        else:
            return np.nan # 没有找到交点
    except (ValueError, IndexError):
        return np.nan

def find_width(x_values, y_values):
    """
    计算曲线的半高全宽 (Full Width at Half Maximum)。
    半高位置定义为 y = (y_max + y_min) / 2。这里简化为 (y_max + 1)/2，因为数据已经归一化。
    
    参数:
    x_values (np.array): x坐标数组。
    y_values (np.array): y坐标数组。

    返回:
    float: 半高宽的值，如果无法计算则返回np.nan。
    """
    try:
        max_y_index = np.argmax(y_values)
        max_y = y_values[max_y_index]
        
        # 定义半高位置的Y值
        half_max_y = (max_y + 1) / 2

        # 寻找穿过半高Y值的点
        cross_indices = np.where(np.diff(np.sign(y_values - half_max_y)))[0]

        if len(cross_indices) < 2:
            # 至少需要两个交点才能计算宽度
            return np.nan

        # 找到上升沿和下降沿的两个交点
        # 上升沿的交点在最大值左侧
        rising_cross_idx = cross_indices[cross_indices < max_y_index]
        # 下降沿的交点在最大值右侧
        falling_cross_idx = cross_indices[cross_indices >= max_y_index]

        if len(rising_cross_idx) == 0 or len(falling_cross_idx) == 0:
            return np.nan

        # 插值计算精确的x坐标
        # 上升沿
        idx1_rise = rising_cross_idx[-1] # 取最接近峰值的那个
        idx2_rise = idx1_rise + 1
        x_half1 = _interpolate_x(x_values[idx1_rise], y_values[idx1_rise], x_values[idx2_rise], y_values[idx2_rise], half_max_y)

        # 下降沿
        idx1_fall = falling_cross_idx[0] # 取最接近峰值的那个
        idx2_fall = idx1_fall + 1
        x_half2 = _interpolate_x(x_values[idx1_fall], y_values[idx1_fall], x_values[idx2_fall], y_values[idx2_fall], half_max_y)
        
        return abs(x_half2 - x_half1)
    except (ValueError, IndexError):
        return np.nan

# ==============================================================================
# 2. 辅助函数（来自您的原代码）
# ==============================================================================

def extract_submatrix(matrix):
    # 获取矩阵的形状
    m, n = matrix.shape[:2]
    
    # 找到矩形区域的左上角和右下角坐标
    top_left = None
    bottom_right = None
    
    for i in range(m):
        for j in range(n):
            if matrix[i, j][3] == 255:
                if top_left is None:
                    top_left = (i, j)
                bottom_right = (i, j)
    
    if top_left is None or bottom_right is None:
        raise ValueError("没有找到符合条件的区域")

    p, q = bottom_right[0] - top_left[0] + 1, bottom_right[1] - top_left[1] + 1   

    submatrix = np.zeros((p, q), dtype=int)

    for i in range(p):
        for j in range(q):
            submatrix[i, j] = matrix[top_left[0] + i, top_left[1] + j][1]

    return submatrix.T

#提取box中荧光强度的值用于归一化
def normed_1(image_path,nor_mask_path):
    img=np.array(extract_image_with_mask(image_path, nor_mask_path))
    norm_area=extract_submatrix(img)
    #取得norm_area的平均值
    norm_value=np.mean(norm_area)
    return norm_value

def color_generator(cmap_name='tab20', num_colors=100):
    cmap = plt.get_cmap(cmap_name)  # 获取指定的colormap
    colors = [cmap(i) for i in range(num_colors)]  # 提取指定数量的颜色
    for color in colors:  # 遍历生成颜色
        yield color


# 定义一个函数来检测水平线与曲线的第一个交点，并返回交点的横坐标
def find_first_intersection(x, y, horizontal_y):
    """
    查找曲线与水平线的第一个交点的横坐标

    参数:
    - x: 曲线的 x 坐标数组
    - y: 曲线的 y 坐标数组
    - horizontal_y: 水平线的 y 坐标

    返回:
    - 第一个交点的 x 坐标，如果没有交点，返回 None
    """
    for i in range(1, len(x)):
        if (y[i-1] - horizontal_y) * (y[i] - horizontal_y) < 0:  # 检测跨过水平线
            # 线性插值求第一个交点的横坐标
            x_inter = x[i-1] + (horizontal_y - y[i-1]) * (x[i] - x[i-1]) / (y[i] - y[i-1])
            return x_inter  # 找到第一个交点后立即返回
    return None  # 如果没有找到交点，返回 None

#将DataFrame中大于k的值替换为None
def replace_outliers(df, k):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x if x < k else None
                                if x > k else x)
    return df

def length_calculate(L,intersection_list):
    length_list=[]
    for i in range(len(intersection_list)-1):
        length_list.append(intersection_list[i]*L)
    #平滑处理
    length_list=savgol_filter(length_list, 31, 3)
    #所有值减去第一个值
    length_list=[x-length_list[0] for x in length_list]
    return length_list
# ==============================================================================
# 3. 主执行流程
# ==============================================================================

# --- 用户需修改的参数 ---
# 文件位置
dir_name = 'bsa_200_1'
base_path = 'D:/Research/mf_series/'
# timegap, 每个文件之间的时间间隔
time_gap = 15
target_y_for_front = 0.6
# --- 参数修改结束 ---

# 构建完整路径
path = os.path.join(base_path, dir_name, 'fluo/')
mask_path = os.path.join(base_path, dir_name, 'mask/Mask.tif')
normmask_path = os.path.join(base_path, dir_name, 'mask/nMask.tif')

# 检查路径是否存在
if not os.path.isdir(path):
    print(f"错误：目录 '{path}' 不存在。请检查您的路径设置。")
else:
    files_list = os.listdir(path)
    files_list.sort() # 确保文件按顺序处理

    # 初始化列表以存储所有文件的结果
    max_results = []
    front_results = []
    width_results = []
    
    # 初始化时间变量
    current_time = 0

    print(f"开始处理目录 '{dir_name}' 下的文件...")

    for file in files_list:
        file_path = os.path.join(path, file)
        if not os.path.isfile(file_path):
            continue
        
        print(f"  正在处理: {file} (时间: {current_time})")

        # 1. 数据提取和预处理 (您的原代码)
        try:
            result = extract_image_with_mask(file_path, mask_path)
            result = extract_submatrix(np.array(result))
            if result.size == 0:
                print(f"    警告: 文件 '{file}' 提取的子矩阵为空，已跳过。")
                continue
            
            df = pd.DataFrame(result)
            normed_value = normed_1(file_path, normmask_path)
            row_means = df.mean(axis=1)

            if normed_value == 0:
                print(f"    警告: 文件 '{file}' 的归一化值为0，已跳过。")
                continue
                
            row_means = row_means / normed_value
            x_values = np.arange(0, 1, 1 / len(row_means))
            smoothed_means2 = savgol_filter(row_means, 51, 3)
        except Exception as e:
            print(f"    错误: 处理文件 '{file}' 时发生错误: {e}")
            continue

        # 2. 调用新函数计算指标
        # 计算最大值
        y_max, x_max = find_max(x_values, smoothed_means2)
        max_results.append({'file': file, 'time': current_time, 'x_max': x_max, 'y_max': y_max})

        
        front_x = find_front(x_values, smoothed_means2, target_y_for_front)
        front_results.append({'file': file, 'time': current_time, f'front_x_at_y={target_y_for_front}': front_x})
        # 注意：由于不确定您要指定的y值是什么，暂时注释掉这一部分。
        # 这里我们用一个占位符
        #front_results.append({'file': file, 'time': current_time, 'front_x': np.nan})

        # 计算半高宽
        width = find_width(x_values, smoothed_means2)
        width_results.append({'file': file, 'time': current_time, 'width': width})

        # 为下一个文件增加时间
        current_time += time_gap

    # 3. 将结果保存到Excel文件
    if max_results: # 仅当处理了至少一个文件后才保存
        df_max = pd.DataFrame(max_results)
        df_front = pd.DataFrame(front_results)
        df_width = pd.DataFrame(width_results)

        # 定义输出的Excel文件名
        output_excel_path = os.path.join(base_path, dir_name, f'{dir_name}_analysis_results.xlsx')
        
        try:
            with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
                df_max.to_excel(writer, sheet_name='max', index=False)
                df_front.to_excel(writer, sheet_name='front', index=False)
                df_width.to_excel(writer, sheet_name='width', index=False)
            
            print(f"\n处理完成！结果已保存至: {output_excel_path}")
        except Exception as e:
            print(f"\n错误：无法保存Excel文件。原因: {e}")
    else:
        print("\n没有处理任何文件，未生成Excel报告。")
