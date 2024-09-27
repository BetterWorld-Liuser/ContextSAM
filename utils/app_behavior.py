from PIL import Image
import numpy as np
from scipy import ndimage
from typing import Union
import torch
from utils.data_prepare import (
    get_batch_image_embeddings_with_model,
    get_image_embeddings_PIL,
    get_image_embeddings_with_model,
)
from utils.file import ULFile
import unittest
import os
from transformers import SamModel, AutoProcessor
from torch.functional import F


def get_overlayed_image(mask, image):
    """获取覆盖Mask的图像"""
    border_thickness = 3
    color = (255, 128, 0)
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
    return overlayed_image


class ImageInstance:
    def __init__(self, image_path):
        self.image_ul = ULFile(image_path)
        # feature embeddings 和 mask缓存文件
        self.cache_file_path = os.path.join(
            self.image_ul.dir, self.image_ul.name + ".im"
        )
        self.mask_save_path = os.path.join(
            self.image_ul.dir, self.image_ul.name + "_mask." + self.image_ul.extension
        )
        self.image_pil = Image.open(self.image_ul.file_path)
        self.w, self.h = self.image_pil.size

        # 获取 PIL 实例
        self.image_pil = self.image_pil.convert("RGB").resize((1024, 1024))
        # 获取 np 实例
        self.image_np = np.array(self.image_pil)
        self.mask = None

    def get_embedding(self, processer, model):
        """获取图像的embedding，附带缓存功能"""

        if os.path.exists(self.cache_file_path):
            image_cache = torch.load(self.cache_file_path)
            return image_cache["feature embeddings"][
                0
            ]  # 规定feature embeddings的shape为 b,c,h,w
        else:
            embedding = get_batch_image_embeddings_with_model(
                self.image_pil, processer, model
            )
            torch.save({"feature embeddings": embedding}, self.cache_file_path)
            return embedding[0]

    def get_first_mask(self) -> Union[torch.Tensor, None]:
        # 从内存中加载
        if self.mask is not None:
            return self.mask[0]

        # 从磁盘中加载
        if os.path.exists(self.cache_file_path):
            image_cache = torch.load(self.cache_file_path)
            if "mask" in image_cache:  # 还需判断是否有mask
                self.mask = image_cache["mask"]
                return self.mask[0]
            else:
                return None
        # 实在没有就返回None
        else:
            return None

    def get_all_mask(self) -> Union[torch.Tensor, None]:
        # 从内存中加载
        if self.mask is not None:
            return self.mask

        # 从磁盘中加载
        if os.path.exists(self.cache_file_path):
            image_cache = torch.load(self.cache_file_path)
            if "mask" in image_cache:  # 还需判断是否有mask
                self.mask = image_cache["mask"]
                return self.mask
            else:
                return None
        # 实在没有就返回None
        else:
            return None

    def set_mask(self, mask: torch.Tensor):  # mask 1024 1024
        # 要不保存的时候就旋转好吧免得麻烦
        mask = mask.unsqueeze(0)
        masks = []
        for i in (0, 1, 2, 3):
            masks.append(torch.rot90(mask, i, (1, 2)))
        mask = torch.cat(masks, dim=0)

        self.mask = mask
        if os.path.exists(self.cache_file_path):
            image_cache = torch.load(self.cache_file_path)
            image_cache["mask"] = mask  # [4, 1, 1024, 1024]
            torch.save(image_cache, self.cache_file_path)
        else:
            print("出现未知错误，应当是有缓存文件的")


def get_image_mask(model, image_embedding, guide_points, guide_labels) -> np.ndarray:
    mask_threshold = 2  # 设置mask阈值，大于这个数值的才可以作为mask
    with torch.no_grad():
        outputs = model(
            image_embeddings=image_embedding.unsqueeze(0),
            input_points=guide_points,
            input_labels=guide_labels,
            multimask_output=False,
        )
    mask = (
        F.interpolate(
            outputs.pred_masks[0],
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        > mask_threshold
    )
    mask = mask.cpu().squeeze().numpy()

    return mask


class TestImageInstance(unittest.TestCase):
    def test_image_instance(self):
        image = ImageInstance(
            "assets/trainsets/wheat_clean/DJI_20240603091918_0002_D.JPG"
        )
        processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
        model = SamModel.from_pretrained("facebook/sam-vit-base")

        image.get_embedding(processor, model)


if __name__ == "__main__":
    unittest.main()
