import os
import torch
import argparse
import torch
from utils.distance_mlp import MemoryMLP
from utils.data_prepare import (
    get_test_feat_and_true_mask,
)
import torchshow as ts
from utils.data_train import little_train_memory,save_model,load_model
# TODO：将方法抽调到utils中
# Positive-negative location prior
from utils.file import ULFile
# 创建ArgumentParser()对象
parser = argparse.ArgumentParser()

parser.add_argument("--model-name",type=str,default="facebook/sam-vit-base")
parser.add_argument("--target-im",type=str,default="",required=True)
parser.add_argument("--target-mask",type=str,default="",required=True)

# 参数准备
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = args.model_name # example: "facebook/sam-vit-base"
target_ulfile = ULFile(args.target_im)
folder = target_ulfile.dir
mlp_weight_path = os.path.join(folder,"weight.pt")
target_image_path = args.target_im
target_mask_path = args.target_mask


def Train_ContextSAM():
    # 准备模型
    print("正在准备模型...")
    memory_mlp = MemoryMLP(in_channels=256, out_channels=1).to(device)
    if os.path.exists(mlp_weight_path):
        print("检测到模型已经存在，正在加载原有模型权重...")
        load_model(memory_mlp, mlp_weight_path)
    memory_mlp.eval()
    

    for angle in (0,90,180,270):
        print(f"正在训练角度为{angle}的图像和掩码...")
        target,target_mask = get_test_feat_and_true_mask(args.target_im,args.target_mask,rotate=angle)
        little_train_memory(memory_mlp,target,target_mask)
    
    save_model(memory_mlp,mlp_weight_path)

if __name__ == "__main__":
    Train_ContextSAM()