import os
import numpy as np
import torch
from torch.nn import functional as F
import oflibpytorch as of
import matplotlib.pyplot as plt

# return warped frame
def findWarped(img1,img2,flow, isTorch=False, img_size=256):
    # select indices of black pixels which need to be tracked
    edge1_idx = (img1[0,:,:]==0).nonzero(as_tuple=False)
    edge2_idx = (img2[0,:,:]==0).nonzero(as_tuple=False)

    # compute warped pixel indices
    tracked_pts = of.track_pts(flow,ref=None,pts=edge1_idx,int_out=True)

    # avoid rounding error i.e exceeding boundary
    tracked_pts[tracked_pts>=img_size]=img_size-1
    
    if isTorch:
        warped = torch.ones((3, img_size, img_size))*255
        warped[:, tracked_pts[:, 0], tracked_pts[:, 1]] = 0
    else:
        warped = np.ones((img_size, img_size, 3))*255
        warped[tracked_pts[:, 0], tracked_pts[:, 1], :] = 0

    #plt.imshow(warped)
    #plt.show()

    return warped

def optical_flow_warping(x, flo, pad_mode='zeros'):
    '''
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    pad_mode (optional): ref to https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        "zeros": use 0 for out-of-bound grid locations,
        "border": use border values for out-of-bound grid locations
    '''
    B, C, H, W = x.size()
    
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    # visualize grid
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid - flo #+ or -

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=pad_mode, align_corners=True)

    mask = torch.ones(x.size())
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output*mask
