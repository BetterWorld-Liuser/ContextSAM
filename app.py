import os
from joblib import memory
from kornia import image
import matplotlib.pyplot as plt
import argparse

from torchgen import local
from torchvision.datasets import folder

from utils.data_internal import get_guide_points
from utils.data_prepare import get_image_embeddings_PIL
import torch
import torch.nn.functional as F
import numpy as np
import torchshow as ts
from utils.data_train import val_memory
from utils.file import ULFile, listFilewithsuffix
from utils.app_behavior import ImageInstance, get_image_mask, get_overlayed_image
from PIL import Image
from transformers import AutoProcessor, SamModel
from utils.distance_mlp import MemoryMLP
from utils.data_train import little_btrain_memory

# 用户输入变量
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="")
parser.add_argument("--folder", type=str, default="")
args = parser.parse_args()
print("应用启动中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = args.image  # 需要标注的文件
folder_path = args.folder  # 需要标注的文件夹, 如果为空则单独标注某个文件
file_types = (".jpg", ".jpeg", ".png")

# 初始化SAM模型
print("正在初始化SAM模型...")
processor = AutoProcessor.from_pretrained(
    "./assets/sam-vit-base", local_files_only=True
)
model = SamModel.from_pretrained("./assets/sam-vit-base", local_files_only=True).to(
    device
)

# 初始化MLP模型
print("正在初始化MLP模型...")
weight_path = os.path.join(folder_path, "weight.pt")
memory_mlp = MemoryMLP(in_channels=256, out_channels=1).to(device)
if os.path.exists(weight_path):
    memory_mlp.load_state_dict(torch.load(weight_path))


# 获取图片的列表
print("正在获取图片列表...")
if folder_path != "":
    images_list = []
    for file_type in file_types:
        images = listFilewithsuffix(folder_path, file_type)
        images_list.extend(images)
    # 消除其中mask文件
    images_list = list(filter(lambda x: "mask" not in x, images_list))
elif image_path != "":
    images_list = [image_path]
else:
    print("请指定要标注的文件或文件夹")
    exit(0)


# 初始化第一张图片
print("正在初始化第一张图片...")
fig, ax = plt.subplots()
fig.tight_layout(pad=0)  # 消除图片周围的 padding
ax.set_axis_off()  # 消除坐标轴
image_index = 0
image0 = ImageInstance(images_list[image_index])

image_instance_p = image0
image_show = image0.image_np  # 显示在图像上的图像
mask_show = image0.get_first_mask()
if mask_show is not None:
    mask_show = mask_show.numpy()
    previous_masks = mask_show
    overlay = get_overlayed_image(mask_show, image_show)
else:
    overlay = image_show
ax.imshow(overlay)

# 处理图像得到embedding
image_embedding = image0.get_embedding(processor, model)

# 多mask合成机制
previous_masks = mask_show  # 缓存之前的mask


# plt显示的需要的对象
input_points = []  # 用户输入的标记点
input_labels = []  # 用户输入的标记点的类别


def onclick(event):
    global \
        input_points, \
        input_labels, \
        mask_show, \
        previous_masks, \
        image_show, \
        image_embedding
    print(f"点击坐标: x={event.xdata}, y={event.ydata}")
    # 判断左键还是右键
    if event.button == 1:  # 左键
        if event.xdata is None or event.ydata is None:
            return
        input_points.append([event.xdata, event.ydata])
        input_labels.append(1)
        # 在图像中对应位置显示一个淡蓝色的带有加粗边缘的小圆点
        ax.plot(
            event.xdata,
            event.ydata,
            "o",
            color="#4287f5",
            markersize=3,
            markeredgecolor="white",
            markeredgewidth=1,
        )
        plt.draw()

    elif event.button == 3:  # 右键
        if event.xdata is None or event.ydata is None:
            return
        input_points.append([event.xdata, event.ydata])
        input_labels.append(0)
        ax.plot(
            event.xdata,
            event.ydata,
            "o",
            color="#f54842",
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=1,
        )
        plt.draw()
    guide_points = torch.tensor(input_points).unsqueeze(0).unsqueeze(0).to(device)
    guide_labels = torch.tensor(input_labels).unsqueeze(0).unsqueeze(0).to(device)
    # 获取图像mask
    print("正在获取图像mask...")
    print(image_embedding.shape)
    mask = get_image_mask(model, image_embedding, guide_points, guide_labels)
    if previous_masks is not None:  # 如果之前有mask，则进行融合
        mask = np.logical_or(previous_masks, mask)
    mask_show = mask
    print("获取mask成功，正在创建自定义颜色的覆盖Mask...")
    # 创建自定义颜色的覆盖Mask
    overlayed_image = get_overlayed_image(mask_show, image_show)
    ax.imshow(overlayed_image)
    plt.draw()


def on_key_press(event):
    global \
        input_points, \
        input_labels, \
        mask_show, \
        previous_masks, \
        image_show, \
        image_index, \
        image_embedding, \
        image_instance_p, \
        processor, \
        model

    if event.key == "r":  # 只重置标记点，与此同时需要撤回此时这个标记点形成的mask
        input_points = []
        input_labels = []
        ax.clear()
        ax.set_axis_off()
        if previous_masks is not None:
            overlay = get_overlayed_image(previous_masks, image_show)
        else:
            overlay = image_show
        ax.imshow(overlay)
        plt.draw()
    elif event.key == "ctrl+r":  # 重置标记点、图像和mask
        input_points = []
        input_labels = []
        previous_masks = None
        mask_show = None
        ax.clear()
        ax.set_axis_off()
        ax.imshow(image_show)
        plt.draw()
    elif event.key == "m":  # 保存mask到磁盘
        if mask_show is not None:
            image_instance_p.set_mask(torch.tensor(mask_show))
            print(f"保存mask成功,路径为：{image_instance_p.cache_file_path}")
            save_mask = mask_show.astype(np.uint8) * 255
            save_mask = Image.fromarray(save_mask).resize(
                (image_instance_p.w, image_instance_p.h)
            )
            save_path = image_instance_p.mask_save_path
            save_mask.save(save_path)
    elif event.key == "q":  # 退出程序
        plt.close()
    elif event.key == "c":  # 多mask合成
        previous_masks = mask_show
        input_points = []
        input_labels = []
        ax.clear()  # 还是要清楚图像，因为上一次的图像中存在部分标记点
        ax.set_axis_off()
        overlay = get_overlayed_image(mask_show, image_show)
        ax.imshow(overlay)
        plt.draw()
    elif event.key == "a":
        if image_index == 0:
            return
        print("切换到上一张图像")
        image_index = image_index - 1
        image_instance_p = ImageInstance(images_list[image_index])
        image_show = image_instance_p.image_np
        mask_show = image_instance_p.get_first_mask()
        if mask_show is not None:
            mask_show = mask_show.numpy()
            previous_masks = mask_show
            overlay = get_overlayed_image(mask_show, image_show)
        else:
            previous_masks = None
            overlay = image_show

        ax.clear()
        ax.set_axis_off()
        ax.imshow(overlay)
        plt.draw()
        image_embedding = image_instance_p.get_embedding(processor, model)
        input_points = []
        input_labels = []
    elif event.key == "h":  # memory模型来帮忙
        sim = val_memory(memory_mlp, image_embedding).cpu()  # 64 64
        # 转化为32 32
        sim = (
            F.interpolate(
                sim.unsqueeze(0).unsqueeze(0),
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
        )
        # 获取最可能的一些标记点
        relative_pos_gate = sim.max().item() * 0.8  # 选择20%的标记点阈值
        absolute_posgate = 0.8  # 以0.8为界限

        pos_gate = max(relative_pos_gate, absolute_posgate)  # 以上两个值谁大取谁

        guide_points, guide_labels = get_guide_points(
            sim, threshold_active=pos_gate, threshold_negative=0
        )
        pos_guide_points = guide_points[np.array(guide_labels) == 1]

        guide_points = torch.tensor(guide_points).unsqueeze(0).unsqueeze(0).to(device)
        guide_labels = torch.tensor(guide_labels).unsqueeze(0).unsqueeze(0).to(device)

        # 单点模式
        mask = None
        for i in range(guide_points.shape[-2]):
            single_guide_labels = guide_labels[:, :, i]
            if single_guide_labels.item() == 1:
                single_guide_points = guide_points[:, :, i]
                with torch.no_grad():
                    outputs = model(
                        image_embeddings=image_embedding.unsqueeze(0),
                        input_points=single_guide_points.unsqueeze(-2),
                        input_labels=single_guide_labels.unsqueeze(-2),
                        multimask_output=False,
                    )
                if mask is None:
                    mask = outputs.pred_masks[0]
                else:
                    cache = outputs.pred_masks[0]
                    mask[cache > 2] = cache[cache > 2]
        if mask is None:
            print("图像中可能没有目标")
            return
        mask = (
            F.interpolate(
                mask,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            > 0
        )

        mask = mask.cpu().squeeze().numpy()
        if previous_masks is not None:
            mask_show = mask | previous_masks
        else:
            mask_show = mask
        overlay = get_overlayed_image(mask_show, image_show)
        ax.clear()
        ax.set_axis_off()
        ax.imshow(overlay)
        ax.plot(
            pos_guide_points[:, 0].tolist(),
            pos_guide_points[:, 1].tolist(),
            "o",
            color="#4287f5",
            markersize=3,
            markeredgecolor="white",
            markeredgewidth=1,
        )
        plt.draw()
    elif event.key == "ctrl+t":  # train模型,包含以前的mask，开始train memory model
        # 加载所有im文件
        im_list = listFilewithsuffix(folder_path, "im")
        if len(im_list) == 0:
            print("没有已标注的文件")
            return
        batch_train_feat_embedding = []
        batch_train_label_embedding = []
        for im_path in im_list:
            im = torch.load(im_path)
            train_feat_embedding = im["feature embeddings"]
            if "mask" not in im:
                continue
            train_label_embedding = F.interpolate(
                im["mask"].unsqueeze(0).float(),
                size=(64, 64),
                mode="bilinear",
            ).squeeze()  # 规划到 64 64
            batch_train_feat_embedding.append(train_feat_embedding)
            batch_train_label_embedding.append(train_label_embedding)

        batch_train_feat_embedding = torch.cat(batch_train_feat_embedding)
        batch_train_label_embedding = torch.cat(batch_train_label_embedding)
        little_btrain_memory(
            memory_mlp, batch_train_feat_embedding, batch_train_label_embedding
        )
        torch.save(memory_mlp.state_dict(), weight_path)
        print("微调完成，已保存模型")
    elif event.key == "t":  # train模型，不包含以前的mask，开始train memory model
        # 加载当前文件对应的im文件
        image_path = images_list[image_index]
        image_ulfile = ULFile(image_path)
        im_path = os.path.join(image_ulfile.dir, image_ulfile.name + ".im")
        im = torch.load(im_path)
        train_feat_embedding = im["feature embeddings"]  # [4, 256, 64, 64]
        if "mask" not in im:
            print("没有标注的文件")
            return
        train_label_embedding = F.interpolate(
            im["mask"].unsqueeze(0).float(),
            size=(64, 64),
            mode="bilinear",
        ).squeeze(0)  # 规划到 [1,4,64,64]

        little_btrain_memory(memory_mlp, train_feat_embedding, train_label_embedding)
        torch.save(memory_mlp.state_dict(), weight_path)
        print("微调完成，已保存模型")

    elif event.key == "d":
        if image_index == len(images_list) - 1:
            print("已经是最后一张图像")
            return
        print("切换到下一张图像")
        image_index = image_index + 1
        image_instance_p = ImageInstance(images_list[image_index])
        image_show = image_instance_p.image_np
        mask_show = image_instance_p.get_first_mask()
        if mask_show is not None:
            mask_show = mask_show.numpy()
            previous_masks = mask_show
            overlay = get_overlayed_image(mask_show, image_show)
        else:
            previous_masks = None
            overlay = image_show

        ax.clear()
        ax.set_axis_off()
        ax.imshow(overlay)
        plt.draw()
        image_embedding = image_instance_p.get_embedding(processor, model)
        input_points = []
        input_labels = []


# 监听按键
fig.canvas.mpl_connect("key_press_event", on_key_press)
# 连接鼠标点击事件
cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
