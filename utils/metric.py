import cv2
import torch
import torch.nn as nn
import numpy as np
from chamferdist import ChamferDistance

'''
Metrics of consistency loss.

    Triplet loss is implemented and is identical to Torch version.
    Quadruplet loss is implemented, no Torch version available.
    The IoU is not used, we used torchmetric library.
    Distance Transform uses cv2 distance transform in Numpy.

    Deprecated: chamfer distance
'''

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = nn.functional.pairwise_distance(anchor,positive)
        distance_negative = nn.functional.pairwise_distance(anchor,negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class QuadrupletLoss(torch.nn.Module):
    def __init__(self, margin1=1.0, margin2=0.5):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):

        squarred_distance_pos = nn.functional.pairwise_distance(anchor, positive)
        squarred_distance_neg = nn.functional.pairwise_distance(anchor, negative1)
        squarred_distance_neg_b = nn.functional.pairwise_distance(negative1, negative2)

        quadruplet_loss = torch.relu(squarred_distance_pos - squarred_distance_neg + self.margin1) + \
                          torch.relu(squarred_distance_pos - squarred_distance_neg_b + self.margin2)

        return quadruplet_loss.mean()

# IoU for binary image
def IoU(x, y, smooth = 1e-6):
    x = x.squeeze(1)
    
    intersection = (x & y).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (x | y).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + smooth) / (union + smooth)
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # thresholded.mean() if you are interested in average across the batch


'''input must be binary image, input is (h,w,c)'''
def distanceTransform(img,isTensor=False):
    if isTensor:
        img = img.numpy()
    
    grey = img[0,:,:] # convert to grey scale image
    grey = grey.astype(np.uint8)

    # Apply distance transform using open-cv
    trans = cv2.distanceTransform(grey, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
    trans = cv2.normalize(trans, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if isTensor:
        trans = torch.from_numpy(trans)

    return trans

def chamferDistance(source,target):
    chamferdist = ChamferDistance()
    dist_bidirectional = chamferdist(source/255, target/255, bidirectional=True)
    return dist_bidirectional.detach().cpu().item()


if __name__ == '__main__':
    anchor = torch.randn(2, 3, 128, 128, requires_grad=True)
    positive = torch.randn(2, 3, 128, 128, requires_grad=True)
    negative = torch.randn(2, 3, 128, 128, requires_grad=True)
    trdneigb = torch.randn(2, 3, 128, 128, requires_grad=True)

    #consistency = nn.TripletMarginLoss(margin=1)
    #output = consistency(anchor, positive, negative)
    #print('tested triplet loss with random inputs: ',output.item())

    criterion = TripletLoss()
    output = criterion(anchor, positive, negative)
    print('tested Triplet loss with random inputs: ',output.item())

    criterion = QuadrupletLoss()
    output = criterion(anchor, positive, negative, trdneigb)
    print('tested Quadplet loss with random inputs: ',output.item())