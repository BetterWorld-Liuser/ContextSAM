"""
预先处理数用的方法
"""

import unittest
from PIL import Image
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, SamModel
import torchshow as ts


def prepare_mask(ref_mask_path, resize=(64, 64)) -> torch.Tensor:
    """
    将对应路径的Mask处理成对应形状的Tensor
    """
    ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)

    return torch.tensor(cv2.resize(ref_mask, resize)) == 255  # (64,64)


def prepare_mask_rotate(ref_mask_path, resize=(64, 64), rotate=0) -> torch.Tensor:
    """
    将对应路径的Mask处理成对应形状的Tensor并旋转
    """
    ref_mask = Image.open(ref_mask_path).convert("L")
    ref_mask = ref_mask.rotate(rotate)
    return torch.tensor(np.array(ref_mask.resize(resize))) > 0  # (64,64)


def get_image_embeddings(image_path, rotate=0):
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    image = Image.open(image_path).convert("RGB").resize((1024, 1024))
    image = image.rotate(rotate)

    # 预处理参考图像
    pixel_values = processor(
        images=image, return_tensors="pt"
    ).pixel_values  # 会自动将图像变成1024x1024

    # 预测
    with torch.no_grad():
        feat = model.get_image_embeddings(pixel_values.to(device)).squeeze(
            0
        )  # 256 64 64
    return feat


def get_image_embeddings_PIL(image: Image.Image, rotate=0):
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if rotate != 0:  # 跳过流程，加速一下
        image = image.rotate(rotate)

    # 预处理参考图像
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # 预测
    with torch.no_grad():
        feat = model.get_image_embeddings(pixel_values.to(device)).squeeze(
            0
        )  # 256 64 64
    return feat


def get_image_embeddings_with_model(image: Image.Image, processor, model, rotate=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if rotate != 0:  # 跳过流程，加速一下
        image = image.rotate(rotate)

    # 预处理参考图像
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    # 预测
    with torch.no_grad():
        feat = model.get_image_embeddings(pixel_values).squeeze(0)  # 256 64 64
    return feat  # 256 64 64


def get_batch_image_embeddings_with_model(image: Image.Image, processor, model):
    """获取图像四角旋转的embedding"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = (0, 90, 180, 270)
    batch_feat = None
    for i in rs:
        image = image.rotate(i)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
            device
        )
        # 预测
        with torch.no_grad():
            feat = model.get_image_embeddings(pixel_values)  # 1 256 64 64
            if batch_feat is None:
                batch_feat = feat  # 1 256 64 64
            else:
                batch_feat = torch.cat([batch_feat, feat], dim=0)
    return batch_feat  # 4 256 64 64


def get_ref_embedding(ref_image_path, ref_mask_path):
    """获取参考图像的embedding"""
    # 加载图像和Mask
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ref_image = Image.open(ref_image_path).convert("RGB")
    ref_mask = prepare_mask(ref_mask_path)

    # 预处理参考图像
    ref_pixel_values = processor(images=ref_image, return_tensors="pt").pixel_values

    # 预测
    with torch.no_grad():
        ref_feat = model.get_image_embeddings(ref_pixel_values.to(device)).squeeze(
            0
        )  # 256 64 64

    ref_embedding = ref_feat.permute(1, 2, 0)[ref_mask].mean(0, keepdim=True)  # 1 256
    return ref_embedding


def get_ref_embedding_with_model_name(
    ref_image_path, ref_mask_path, model_name="facebook/sam-vit-base"
):
    """获取参考图像的embedding"""
    # 加载图像和Mask
    processor = AutoProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ref_image = Image.open(ref_image_path).convert("RGB")
    ref_mask = prepare_mask(ref_mask_path)

    # 预处理参考图像
    ref_pixel_values = processor(images=ref_image, return_tensors="pt").pixel_values

    # 预测
    with torch.no_grad():
        ref_feat = model.get_image_embeddings(ref_pixel_values.to(device)).squeeze(
            0
        )  # 256 64 64

    ref_embedding = ref_feat.permute(1, 2, 0)[ref_mask].mean(0, keepdim=True)  # 1 256
    return ref_embedding


def get_test_feat_and_true_mask(test_feat_path, true_mask_path, rotate=90):
    """
    获得测试图片和测试图片的真实mask
    """
    test_feat = get_image_embeddings(test_feat_path, rotate=rotate)
    ture_mask = prepare_mask_rotate(true_mask_path, rotate=rotate)
    return test_feat, ture_mask


def get_batch_test_feat_and_true_mask(test_feat_path, true_mask_path):
    """
    获得测试图片和测试图片的真实mask,以batch的形式
    """
    batch_test_feat = None
    batch_ture_mask = None
    for r in (0, 90, 180, 270):
        test_feat = get_image_embeddings(test_feat_path, rotate=r)
        ture_mask = prepare_mask_rotate(true_mask_path, rotate=r)
        if batch_test_feat is None:
            batch_test_feat = test_feat.unsqueeze(0)
            batch_ture_mask = ture_mask.unsqueeze(0)
        else:
            batch_ture_mask = torch.cat(
                [batch_ture_mask, ture_mask.unsqueeze(0)], dim=0
            )
            batch_test_feat = torch.cat(
                [batch_test_feat, test_feat.unsqueeze(0)], dim=0
            )
    return batch_ture_mask, batch_test_feat


def get_processor():
    return AutoProcessor.from_pretrained("facebook/sam-vit-base")


def get_SamModel():
    return SamModel.from_pretrained("facebook/sam-vit-base")


class MethodTest(unittest.TestCase):
    def test_prepare_mask_rotate(self):
        self.assertEqual(prepare_mask_rotate("assets/masks/1_left.png").shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
