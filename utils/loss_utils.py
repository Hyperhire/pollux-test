#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from utils.general_utils import knn_pcl
# from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

#================== scale loss add (ljw, lsj) ========================
def scale_loss(scales, lambda_flatten = 100.0):
    # lambda_flatten: Weight for the flatten loss
    # scales: Retrieve the scaling factors for the Gaussians

    # Find the second and largest scale for each set of scales
    _min_scale_largest, _ = torch.kthvalue(scales, 3)
    _min_scale_second_largest, _ = torch.kthvalue(scales, 2)

    # Clamp the second smallest scale between 0 and 30
    _min_scale_largest = torch.clamp(_min_scale_largest, 1, 30) 
    _min_scale_second_largest = torch.clamp(_min_scale_second_largest, 1, 30) 

    # Calculate the shape of the gaussians and add second largest scale
    shape_scale = _min_scale_largest/_min_scale_second_largest - 1
    shape_scale += _min_scale_second_largest - 1

    # Calculate the mean absolute value of the clamped second smallest scales
    flatten_loss = torch.abs(shape_scale).mean()


    # Add the weighted flatten loss to the total loss
    return lambda_flatten * flatten_loss

#=====================================================================


def l1_loss(network_output, gt, weight=1):
    return torch.abs((network_output - gt)).mean()

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def bce_loss(output, mask=1):
    bce = output * torch.log(output) + (1 - output) * torch.log(1 - output)
    loss = (-bce * mask).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def knn_smooth_loss(gaussian, K):
    xyz = gaussian._xyz
    normal = gaussian.get_normal
    nn_xyz, nn_normal = knn_pcl(xyz, normal, K)
    dist_prj = torch.sum((xyz - nn_xyz) * normal, -1, True).abs()
    loss_prj = dist_prj.mean()

    nn_normal = torch.nn.functional.normalize(nn_normal)
    loss_normal = cos_loss(normal, nn_normal, thrsh=np.pi / 3)
    return loss_prj, loss_normal