import os
import torch
import argparse
import torch
from utils.distance_mlp import MemoryMLP
from utils.data_prepare import (
    get_batch_test_feat_and_true_mask,
)
import torchshow as ts
from utils.data_train import (
    little_btrain_memory,
    little_train_memory,
    save_model,
    load_model,
)

# Done：将方法抽调到utils中
# Positive-negative location prior
from utils.file import ULFile, listFilewithsuffix
from tqdm import tqdm

# 创建ArgumentParser()对象
parser = argparse.ArgumentParser()

parser.add_argument("--model-name", type=str, default="facebook/sam-vit-base")
parser.add_argument(
    "--folder",
    type=str,
    default="e:/ContextSAM/assets/trainsets/wheat_clean_test",
)
parser.add_argument("--masktype", type=str, default="png")

# 参数准备
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = args.model_name  # example: "facebook/sam-vit-base"
folder = args.folder  # 获取需要训练的文件夹
mlp_weight_path = os.path.join(folder, "weight.pt")
masktype = args.masktype


def Train_ContextSAM_folder():
    mask_file_list = listFilewithsuffix(folder, f"mask.{masktype}")
    image_file_list = [s.replace("_mask", "") for s in mask_file_list]  # 替换mask字符串
    memory_mlp = MemoryMLP(in_channels=256, out_channels=1).to(device)

    if os.path.exists(mlp_weight_path):
        print("检测到模型已经存在，正在加载原有模型权重...")
        load_model(memory_mlp, mlp_weight_path)
    target, target_mask = None, None
    print("正在准备训练的数据...")
    for image_im, target_mask_path in tqdm(zip(image_file_list, mask_file_list)):
        batch_target_mask, batch_target = get_batch_test_feat_and_true_mask(
            image_im, target_mask_path
        )
        if target is None:
            target = batch_target
            target_mask = batch_target_mask
        else:
            target = torch.cat([target, batch_target], dim=0)
            target_mask = torch.cat([target_mask, batch_target_mask], dim=0)
    print("准备完成，开始训练...")
    print("target shape: ", target.shape, "target_mask shape: ", target_mask.shape)

    little_btrain_memory(memory_mlp, target, target_mask, 100)
    save_model(memory_mlp, mlp_weight_path)
    print(f"训练完成，模型存储在{mlp_weight_path}")


if __name__ == "__main__":
    Train_ContextSAM_folder()
