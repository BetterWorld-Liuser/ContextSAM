import torch
import numpy as np


def point_selection(mask_sim, topk=1):
    """选择

    Args:
        mask_sim (tensor): 64x64 相似度矩阵
        topk (int, optional): 选择最上层的几个点. Defaults to 1.

    Returns:
        (相似度最大的xy坐标，
        )
    """
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = last_xy - last_x * h
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    # 返回的分别是 最大值的位置和最小值的位置
    return topk_xy, topk_label, last_xy, last_label


def point_selection_rev(mask_sim, topk=1):
    """选择

    Args:
        mask_sim (tensor): 64x64 相似度矩阵
        topk (int, optional): 选择最上层的几个点. Defaults to 1.

    Returns:
        (相似度最大的xy坐标，
        )
    """
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = last_xy - last_x * h
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_xy = last_xy.cpu().numpy()
    # 返回的分别是 最大值的位置和最小值的位置
    return topk_xy, last_xy


def get_guide_points(sim,threshold_active = 0.6,threshold_negative=0.2):
    """获取正负指导点的坐标和标签"""
    sim_w,sim_h = sim.shape
    rate = int(1024/sim_w)
    x,y = torch.meshgrid(torch.arange(sim_w),torch.arange(sim_h),indexing='xy')
    xy_index = torch.stack([x,y],dim=-1)
    # 获取sim_cos>0.5 位置的坐标
    active_points = xy_index[sim>threshold_active]*rate
    active_labels = [1] * len(active_points)
    negative_points = xy_index[sim<threshold_negative]*rate
    negative_labels = [0] * len(negative_points)
    return torch.cat([active_points,negative_points],dim=0),active_labels+negative_labels


if __name__ == "__main__":
    sim = torch.randn(32,32)
    points,labels = get_guide_points(sim)
    print(points)
    print(labels)