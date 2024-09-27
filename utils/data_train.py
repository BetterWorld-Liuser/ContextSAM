import os
import torch
import torch.nn as nn
from einops import rearrange
from einops import repeat
import torchshow as ts
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class batch_train(Dataset):
    def __init__(self, train_folder, suffix=".png"):
        self.train_folder = train_folder
        # 列出所有mask的路径
        self.mask_paths = listFilewithsuffix(self.train_folder, suffix)

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        return mask


def listFilewithsuffix(dir, suffix):
    """
    列出所有符合此后缀的文件

    用例
    ------
    ```python
    file_list = utils.listFilewithsuffix("./Data_raw", ".jpg")
        for file in file_list:
            print(file)
    ```
    ------
    """
    return [
        os.path.join(dir, entry.name)
        for entry in os.scandir(dir)
        if entry.name.lower().endswith(suffix.lower())
    ]


# 准备训练的方法
def little_train(
    distance_mlp, test_feat, ref_embedding, ture_mask, times=20, threshold=1e-2
):
    """
    稍微训练一下 distance_mlp 模型
    distance_mlp: MLP 模型
    test_feat: 测试特征 256x64x64
    ref_embedding: 参考特征 1x256
    ture_mask: 真实掩码 64x64
    times: 训练次数 默认5次
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(distance_mlp.parameters(), lr=1e-3)
    loss_fun = nn.MSELoss()
    test_feat_ready = rearrange(test_feat, "c h w -> (h w) c")
    target_embedding_ready = repeat(ref_embedding, "1 c -> n c", n=4096)
    data = torch.cat([test_feat_ready, target_embedding_ready], dim=1).to(device)
    true_mask_tensor = ture_mask.to(device).float()
    print("start training the model")
    for i in range(times):
        optimizer.zero_grad()
        y = distance_mlp(data)
        y = rearrange(y, "(h w) 1 -> h w", h=64, w=64)
        loss = loss_fun(y, true_mask_tensor)
        loss.backward()
        optimizer.step()
        print(loss)
        if loss.item() < threshold:
            print("loss 达到预定值, early stop training")
            break


# 准备训练的方法
def little_train_memory(memory_mlp, test_feat, ture_mask, times=20):
    """
    稍微训练一下 memory_mlp 模型
    memory_mlp: MLP 模型
    test_feat: 测试特征 256x64x64
    ture_mask: 真实掩码 64x64
    times: 训练次数 默认5次
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(memory_mlp.parameters(), lr=1e-3)
    loss_fun = nn.MSELoss()
    data = rearrange(test_feat, "c h w -> (h w) c")
    true_mask_tensor = ture_mask.to(device).float()
    print("start training the model")
    for i in range(times):
        optimizer.zero_grad()
        y = memory_mlp(data)
        y = rearrange(y, "(h w) 1 -> h w", h=64, w=64)
        loss = loss_fun(y, true_mask_tensor)
        loss.backward()
        optimizer.step()
        print(loss)


def little_btrain_memory(memory_mlp, test_feat, ture_mask, times=100):
    """
    稍微训练一下 memory_mlp 模型
    memory_mlp: MLP 模型
    test_feat: 测试特征 b 256 64 64
    ture_mask: 真实掩码 b 64 64
    times: 训练次数 默认5次
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("test_feat shape:", test_feat.shape)
    print("ture_mask shape:", ture_mask.shape)
    test_feat = test_feat.to(device)
    ture_mask = ture_mask.to(device)
    memory_mlp = memory_mlp.to(device)
    optimizer = torch.optim.Adam(memory_mlp.parameters(), lr=1e-3)
    loss_fun = nn.MSELoss()
    data = rearrange(test_feat, "b c h w -> b (h w) c")
    true_mask_tensor = ture_mask.to(device).float()
    print("start training the model")
    for i in tqdm(range(times)):
        optimizer.zero_grad()
        y = memory_mlp(data)
        y = rearrange(y, "b (h w) 1 ->b h w", h=64, w=64)
        loss = loss_fun(y, true_mask_tensor)
        loss.backward()
        optimizer.step()


def val_model(memory_mlp, test_feat, ref_embedding):
    """
    验证模型
    memory_mlp: MLP 模型
    test_feat: 测试特征 1024x64x64
    ref_embedding: 参考特征 1x1024
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_feat_ready = rearrange(test_feat, "c h w -> (h w) c")
    target_embedding_ready = repeat(ref_embedding, "1 c -> n c", n=4096)
    data = torch.cat([test_feat_ready, target_embedding_ready], dim=1).to(device)
    y = memory_mlp(data)
    sim = y.reshape(64, 64)
    return sim


def val_memory(memory_mlp, test_feat):
    """
    验证模型
    distance_mlp: MLP 模型
    test_feat: 测试特征 1024x64x64
    ref_embedding: 参考特征 1x1024
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = rearrange(test_feat, "c h w -> (h w) c")
    y = memory_mlp(data)
    sim = y.reshape(64, 64)
    return sim


def save_model(distance_mlp, weights_path="./weights/distance_mlp.pt"):  # 64 64  1
    """
    保存模型
    distance_mlp: MLP 模型
    """
    torch.save(distance_mlp.state_dict(), weights_path)
    print(f"model saved to {weights_path}")


def load_model(distance_mlp, weights_path="./weights/distance_mlp.pt"):
    """
    加载模型
    distance_mlp: MLP 模型
    """
    distance_mlp.load_state_dict(torch.load(weights_path))
    print(f"model loaded from {weights_path}")
