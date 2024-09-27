import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image


def apply_colored_mask(image, mask, color=(255, 128, 0), border_thickness=3):
    """
    将mask覆盖在image上并附加颜色，主要是为了可视化。
    image可以是路径也可以是cv的实例，mask的值必须在0-255之间
    """
    if isinstance(image, str): # 如果image是路径
        image = np.array(Image.open(image).convert("RGB"))

    if isinstance(mask, str): # 如果mask是路径
        mask = np.array(Image.open(mask).convert("1"))
        
    if isinstance(image, Image.Image): # 如果image是PIL的实例
        image = np.array(image.convert("RGB"))

    if isinstance(mask, Image.Image): # 如果mask是PIL的实例
        mask = np.array(mask.convert("1"))

    assert isinstance(image, np.ndarray) and isinstance(
        mask, np.ndarray
    ), "image and mask must be numpy arrays"

    # 确保Mask是二值图像
    mask = mask > 0

    # 创建自定义颜色的覆盖Mask
    custom_color_mask = np.zeros_like(image)
    custom_color_mask[mask] = color

    # 将覆盖Mask应用到图像上
    overlayed_image = image.copy()
    overlayed_image[mask] = (
        overlayed_image[mask] * 0.5 + custom_color_mask[mask] * 0.5
    ).astype(np.uint8)

    # 找到Mask的边界并加粗
    edges = ndimage.binary_erosion(mask) ^ mask
    kernel = np.ones((border_thickness, border_thickness), dtype=bool)
    thick_edges = ndimage.binary_dilation(edges, structure=kernel)

    # 在图像上绘制加粗的边界线
    overlayed_image[thick_edges] = color

    plt.imshow(overlayed_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    apply_colored_mask(
        "./assets/images/1_bottom.png", "./assets/images/1_mask_bottom.png"
    )
