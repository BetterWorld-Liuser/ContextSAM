import torch
import argparse
from transformers import SamModel

import numpy as np
import torch
import torch.nn.functional as F
import torchshow as ts
from utils.distance_mlp import DistanceMLP
from utils.data_prepare import (
    get_image_embeddings,
    get_image_embeddings_PIL,
)
from utils.data_train import val_memory, load_model
from utils.data_internal import get_guide_points
from scripts.picture_mask import apply_colored_mask
from utils.file import ULFile
from PIL import Image
import pdb

# 创建ArgumentParser()对象
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="facebook/sam-vit-base")
parser.add_argument("--target-im", type=str, default="", required=True)
parser.add_argument("--pos-gate", type=float, default=0.8)
parser.add_argument("--neg-gate", type=float, default=-0.1)
parser.add_argument("--mlp-path", type=str, default="mlps/classes.pt")
parser.add_argument("--sp", type=bool, default=True)  # single points模式
parser.add_argument("--have", type=float, default=0)  # mask阈值


args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
target_image_ulfile = ULFile(args.target_im)
target_im_pil = Image.open(args.target_im)
w, h = target_im_pil.size  # 获取原始图像的宽度和高度
mlp_ulfile = ULFile(args.mlp_path)
single_points_models = args.sp
mask_threshold = 0
have = args.have
pos_gate = args.pos_gate
neg_gate = args.neg_gate


def MemorySAM():
    global \
        w, \
        h, \
        target_im_pil, \
        target_image_ulfile, \
        mlp_ulfile, \
        single_points_models, \
        mask_threshold, \
        have, \
        pos_gate, \
        neg_gate
    # 准备模型
    print("正在准备模型...")
    memory_mlp = DistanceMLP(in_channels=256, out_channels=1).to(device)
    load_model(memory_mlp, mlp_ulfile.file_path)
    memory_mlp.eval()

    # 获取参考图像和目标图像的特征
    print("正在获取参考图像和目标图像的特征...")
    target = get_image_embeddings_PIL(target_im_pil.resize((1024, 1024)))

    # 计算相似度
    print("正在计算相似度...")
    sim_mlp = val_memory(memory_mlp, target).cpu()
    # 下采样
    # sim_mlp = F.interpolate(sim_mlp.unsqueeze(0).unsqueeze(0),size=(32,32),mode="bilinear",align_corners=False).squeeze().cpu()
    # 输出相似度的最大值和最小值
    print("相似度最大值：", sim_mlp.max().item())
    print("相似度最小值：", sim_mlp.min().item())
    ts.show(sim_mlp)

    # 获取guide_points
    print("正在获取guide_points...")
    if have != 0:  # 如果have不为0，则根据have的值来调整pos_gate
        pos_gate = sim_mlp.max().item() * (1 - have)
    guide_points, guide_labels = get_guide_points(
        sim_mlp, threshold_active=pos_gate, threshold_negative=neg_gate
    )
    guide_points = torch.tensor(guide_points).unsqueeze(0).unsqueeze(0).to(device)
    guide_labels = torch.tensor(guide_labels).unsqueeze(0).unsqueeze(0).to(device)
    print("guide_points:", guide_points.shape)
    print(guide_points)
    print(guide_labels)

    # 获取mask
    print("正在获取mask...")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    if not single_points_models:  # 如果不是单点模式，就把所有guide_points都输入进去
        with torch.no_grad():
            outputs = model(
                image_embeddings=target.unsqueeze(0),
                input_points=guide_points,
                input_labels=guide_labels,
                multimask_output=False,
            )
        ts.show(outputs.pred_masks[0])
        mask = (
            F.interpolate(
                outputs.pred_masks[0],
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            > mask_threshold
        )
        cv_mask = mask.cpu().squeeze().numpy().astype(np.uint8) * 255
    else:  # 如果是单点模式，则依次将每个guide_point输入进去，拼接最后的mask
        mask_output = None
        for i in range(guide_points.shape[-2]):
            single_guide_labels = guide_labels[:, :, i]
            if single_guide_labels.item() == 1:
                single_guide_points = guide_points[:, :, i]
                with torch.no_grad():
                    outputs = model(
                        image_embeddings=target.unsqueeze(0),
                        input_points=single_guide_points.unsqueeze(-2),
                        input_labels=single_guide_labels.unsqueeze(-2),
                        multimask_output=False,
                    )
                if mask_output is None:
                    mask_output = outputs.pred_masks[0]
                else:
                    cache = outputs.pred_masks[0]
                    mask_output[cache > mask_threshold] = cache[cache > mask_threshold]
        if mask_output is None:
            print("没有找到mask")
            return
        mask = (
            F.interpolate(
                mask_output, size=(1024, 1024), mode="bilinear", align_corners=False
            ).squeeze(0)
            > mask_threshold
        )
        cv_mask = mask.cpu().squeeze().numpy().astype(np.uint8) * 255

    # 将mask调整到原图像的大小
    mask = Image.fromarray(cv_mask).resize((w, h))
    apply_colored_mask(target_im_pil, mask)


if __name__ == "__main__":
    MemorySAM()
