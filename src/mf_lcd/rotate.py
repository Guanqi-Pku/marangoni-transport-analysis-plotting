#对指定文件夹的图片批量旋转相同的角度，并覆盖原图像
import os
import cv2
import numpy


def rotate_images_in_folder(folder_path, angle):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件后缀
        if filename.endswith('.tif'):
            # 读取图像为彩色
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)


            
            # 获取图像的高度和宽度
            height, width = image.shape[:2]
            
            # 计算旋转中心
            center = (width // 2, height // 2)
            
            # 获取旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 应用旋转变换
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # 保存旋转后的图像
            cv2.imwrite(image_path, rotated_image)
            
            print(f'Rotated {filename} by {angle} degrees')
    
    print('Image rotation complete.')

tar='20dex120'

# 指定文件夹路径和旋转角度
folder_path = 'D:/Research/mf_series/'+tar+'/fluo'
angle = -1.42

# 旋转文件夹中的所有图像
rotate_images_in_folder(folder_path, angle)
