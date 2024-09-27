"""
提取特定标签的掩码
"""

import os  # 导入操作系统相关的库，用于文件操作
import cv2 as cv  # 导入 OpenCV 库，用于图像处理
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条
import argparse  # 导入 argparse 库，用于解析命令行参数


# 定义一个函数，用于列出指定目录下所有以指定后缀结尾的文件
def listFilewithsuffix(dir, suffix):
    # 使用列表推导，遍历指定目录下的所有文件
    return [
        os.path.join(dir, entry.name)  # 将文件路径与文件名拼接成完整路径
        for entry in os.scandir(dir)  # 遍历指定目录下的所有文件
        if entry.name.lower().endswith(suffix.lower())  # 判断文件名是否以指定后缀结尾
    ]


# 定义一个函数，用于提取掩码中的特定值
def extract_mask(folder_path, mask_value=[2, 3, 4, 5, 6, 7]):
    # 使用 listFilewithsuffix 函数获取指定目录下所有以 .png 为后缀的文件路径
    masks_path_list = listFilewithsuffix(folder_path, "png")
    # 使用 tqdm 库显示进度条，遍历所有掩码文件
    for mask in tqdm(masks_path_list):
        # 获取掩码文件名，去掉后缀
        mask_name = mask.split(".")[-2]
        # 使用 OpenCV 库读取掩码文件
        image = cv.imread(mask)
        # 遍历指定的 mask_value 列表
        for value in mask_value:
            # 创建掩码文件的副本
            image_copy = image.copy()
            # 将掩码中不等于指定值的像素值设置为 0，等于指定值的像素值设置为 255
            image_copy[image_copy != value] = 0
            image_copy[image_copy == value] = 255

            # 保存处理后的掩码文件
            mask_suffix = f"mask_{value}.jpg"
            if (image_copy == 0).all():  # 全黑则标记
                mask_suffix = f"mask_{value}_black.jpg"
            # 如果白色区域的比例在30%以下，标记为全黑
            elif (image_copy == 255).sum() / (
                image_copy.shape[0] * image_copy.shape[1]
            ) < 0.3:
                mask_suffix = f"mask_{value}_black.jpg"
            cv.imwrite(mask_name + "_" + mask_suffix, image_copy)


# 创建 argparse 对象，用于解析命令行参数
parser = argparse.ArgumentParser()
# 添加一个名为 --path 的参数，用于指定数据集路径，默认路径为 "D:\\Datasets\\LoveDA\\Train\\Rural\\masks_png"
parser.add_argument(
    "--path",
    type=str,
    default="D:\\Datasets\\LoveDA\\Train\\Rural\\masks_png",
    help="path of the dataset",
)
# 解析命令行参数
args = parser.parse_args()

# 判断是否为脚本执行
if __name__ == "__main__":
    # 调用 extract_mask 函数，提取掩码中的特定值
    extract_mask(args.path)
