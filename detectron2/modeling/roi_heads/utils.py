"""
Script that implements the different loss funcitons and NMS algorithms
adopted in our study.
"""

import numpy as np
import torch
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
from torch.nn import functional as F
from torchvision.ops import box_iou, generalized_box_iou, clip_boxes_to_image

# Smoothing constant to avoid zero divison and reduction type constants
SMOOTH = 1e-07
REDUCTION = "mean"


def dice_loss(inp, target, reduction=REDUCTION):
    """
    Dice Loss (DL) implementation.

    Adapted from: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss
    :param inp: input tensor
    :param target: target tensor
    :param reduction: reduction type
    :return: DL value
    """

    # Perform sigmoid operation on the input to get the probabilities
    inp = torch.sigmoid(inp)

    # Flatten the input and target tensors
    inp = inp.view(-1)
    target = target.view(-1)

    # Calculate the intersection and the Dice coefficient
    intersection = (inp * target).sum()
    dice = (2. * intersection + SMOOTH) / (inp.sum() + target.sum() + SMOOTH)

    if reduction == "mean":
        return (1 - dice).mean()
    elif reduction == "sum":
        return (1 - dice).sum()
    else:
        return 1 - dice


def dice_focal(inp, target, alpha=.25, gamma=2., reduction=REDUCTION):
    """
    Focal DL (FDL) implementation which is the mean of the sum of DL and Focal Loss (FL).

    Adapted from: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss

    :param inp: input tensor
    :param target: target tensor
    :param alpha: alpha value
    :param gamma: gamma value
    :param reduction: reduction type
    :return: FDL value
    """

    # Calculate DL
    d_loss = dice_loss(inp, target, reduction=reduction)

    # Calculate FL
    f_loss = sigmoid_focal_loss_jit(inp, target, alpha=alpha, gamma=gamma,
                                    reduction=reduction)  # performs sigmoid internally
    return (d_loss + f_loss) / 2.


def dice_bce(inp, target, reduction=REDUCTION):
    """
    DiceBCE Loss (DBL) implementation which is the sum of DL and Binary Cross-Entropy (BCE).

    :param inp: input tensor
    :param target: target tensor
    :param reduction: reduction type
    :return: DBL value
    """

    # Calculate BCE
    bce = F.binary_cross_entropy_with_logits(inp, target, reduction=reduction)

    # Calculate DL
    d_loss = dice_loss(inp, target, reduction=reduction)
    return d_loss + bce


def tversky_loss(inp, target, alpha=.3, beta=.7, reduction=REDUCTION):
    """
    Tversky Loss (TL) implementation.

    :param inp: input tensor
    :param target: target tensor
    :param alpha: alpha value (applied to False Positives (FP))
    :param beta: beta value (applied to False Negatives (FN))
    :param reduction: reduction type
    :return: TL value
    """

    # Perform sigmoid operation on the input to get the probabilities
    inp = torch.sigmoid(inp)

    # Flatten the input and target tensors
    inp = inp.view(-1)
    target = target.view(-1)

    # Calculate TP, FP and FNs
    tp = (inp * target).sum()
    fp = ((1. - target) * inp).sum()
    fn = (target * (1. - inp)).sum()

    # Compute the Tverski index
    tversky = (tp + SMOOTH) / (tp + alpha * fp + beta * fn + SMOOTH)

    if reduction == "mean":
        return (1. - tversky).mean()


def explog_loss(inp, target, gamma=.3, reduction=REDUCTION, eps=SMOOTH):
    """
    Exponential Logarithmic Loss (ELL) implementation.

    :param inp: input tensor
    :param target: target tensor
    :param gamma: gamma value
    :param reduction: reduction type
    :param eps: epsilon value
    :return: ELL value
    """

    # Calculate DL
    d_loss = dice_loss(inp, target, reduction=reduction)  # weight=0.8

    # Compute BCE
    bce = F.binary_cross_entropy_with_logits(inp, target, reduction=reduction)  # weight=0.2

    # Compute ELL
    explog_loss = 0.8 * torch.pow(-torch.log(torch.clamp(d_loss, eps)), gamma) + \
                  0.2 * torch.pow(bce, gamma)

    return explog_loss


def logcosh_dice_loss(inp, target, reduction=REDUCTION):
    """
    Log-Cosh DL (LCDL) implementation.

    :param inp: input tensor
    :param target: target tensor
    :param reduction: reduction type
    :return: LCDL value
    """

    # Get the DL
    d_loss = dice_loss(inp, target, reduction=reduction)
    return torch.log((torch.exp(d_loss) + torch.exp(-d_loss)) / 2.)


def matrix_nms(masks, scores, method="gauss", sigma=.5):
    """
    Original mask-based Matrix NMS (MNMS) as presented in SOLOv2's paper.

    SOLOv2 paper: https://arxiv.org/pdf/2003.10152.pdf

    :param masks: instance masks
    :param scores: mask scores
    :param method: linear or gaussian
    :param sigma: sigma value
    :return: MNMS value
    """

    n_samples = len(masks)
    if n_samples == 0:
        return []

    masks = masks.reshape(n_samples, -1)

    intersection = torch.mm(masks, masks.T)
    areas = masks.sum(dim=1).expand(n_samples, n_samples)
    union = areas + areas.T - intersection
    ious = (intersection / union).triu(diagonal=1)

    ious_cmax, _ = ious.max(0)
    ious_cmax = ious_cmax.expand(n_samples, n_samples).T

    if method == "gauss":
        decay = torch.exp(-((ious ** 2) - (ious_cmax ** 2)) / sigma)
    else:
        decay = (1 - ious) / (1 - ious_cmax)

    decay, _ = decay.min(dim=0)
    return (scores * decay) >= .5


def matrix_bbox_nms(boxes, scores, idxs, iou_threshold, sigma=.5, kernel='gaussian'):
    """
    Our proposed Matrix Bounding Box Non-Maximum Suppression (MBBNMS).

    For more details about the implementation, refer to: https://arxiv.org/pdf/2003.10152.pdf

    :param boxes: bounding box coordinates
    :param scores: bounding box scores
    :param idxs: class indices
    :param iou_threshold: IoU threshold
    :param sigma: sigma value as used in the SOLOv2's Matrix NMS
    :param kernel: linear or gaussian
    :return: the bounding box indices to keep after MBBNMS
    """

    # Perform the coordinate trick which enables NMS to be performed
    # for each class independently by introducing bounding box offsets.
    # For details, refer to the "batch_nms" algorithm implementation of
    # Detectron2.
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes = boxes + offsets[:, None]

    # Sort the bounding boxes according to their scores in
    # descending order.
    scores, idxes = torch.sort(scores, descending=True)
    boxes = boxes[idxes]

    # Get the IoUs of the boxes.
    ious = box_iou(boxes, boxes).triu(diagonal=1)

    # Get the max IoU for each bounding box
    ious_cmax, _ = ious.max(0)
    n_samples = len(boxes)
    ious_cmax = ious_cmax.expand(n_samples, n_samples).T

    # Calculate the decay factor.
    if kernel == "gaussian":
        decay = torch.exp(-((ious ** 2) - (ious_cmax ** 2)) / sigma)
    else:  # linear
        decay = (1 - ious) / (1 - ious_cmax)
    decay, _ = decay.min(dim=0)  # Column-wise min

    # Update the bounding box scores using the decay factor.
    scores *= decay

    # Get the bounding box indices to keep by applying the IoU threshold
    # given the new bounding box scores.
    keep = idxes[scores > iou_threshold]
    return keep


def py_cpu_nms(boxes, scores, idxs, thresh):
    """
    Pure Python CPU NMS.

    Adapted from: https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/py_cpu_nms.py

    :param boxes: bounding box coordinates
    :param scores: score for each box
    :param idxs: class indexes
    :return: boxes to keep after SNMS
    """

    # Perform the coordinate trick which allows NMS to
    # be performed independently for each class
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes = boxes + offsets[:, None]

    # Get box coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Get areas and order by score
    areas = (x2 - x1) * (y2 - y1)
    order = torch.argsort(scores, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]  # Get max score index
        keep.append(i)  # Append the index

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        # Compute the width and height of the bbox
        w = torch.clamp(xx2 - xx1, min=0.0)  # clamp with min
        h = torch.clamp(yy2 - yy1, min=0.0)

        # Intersection
        inter = w * h

        # Calculate the IoU and determine which boxes to
        # keep according to their scores
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


def soft_nms(boxes, scores, idxs, thresh, method="gaussian", sigma=.5):
    """
    Soft NMS (SNMS) function implementation.

    Adapted from: https://github.com/OneDirection9/soft-nms/blob/master/py_nms.py

    :param boxes: bounding box coordinates
    :param scores: score for each box
    :param idxs: class indexes
    :param thresh: IoU threshold
    :param method: gaussian or linear
    :param sigma: sigma value
    :return: boxes to keep after SNMS
    """

    # Perform the coordinate trick which allows NMS to
    # be performed independently for each class
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes = boxes + offsets[:, None]

    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    N = boxes.shape[0]
    indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)

    # Tensor which contains the box coordinates, scores, areas and indexes
    dets = torch.cat([boxes, scores[:, None], areas[:, None], indexes], dim=1)

    # Boxes to keep
    keep = []

    # Loop while there are still boxes left
    while dets.numel() > 0:
        idx = torch.argmax(dets[:, 4], dim=0)
        dets[[0, idx], :] = dets[[idx, 0], :]
        keep.append(dets[0, -1].int())

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = torch.maximum(dets[0, 0], dets[1:, 0])
        yy1 = torch.maximum(dets[0, 1], dets[1:, 1])
        xx2 = torch.minimum(dets[0, 2], dets[1:, 2])
        yy2 = torch.minimum(dets[0, 3], dets[1:, 3])

        # Compute the width and height of the bbox
        w = torch.clamp(xx2 - xx1, min=0.0)  # clamp with min
        h = torch.clamp(yy2 - yy1, min=0.0)

        # Intersection
        inter = w * h

        # Calculate the IoU of boxes
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        # Calculate the weight to adjust the scores
        if method == "gaussian":
            weight = torch.exp(-(iou * iou) / sigma)

        # Adjust the box scores
        dets[1:, 4] *= weight

        # Indices of boxes to retain
        retained_idx = torch.where(dets[1:, 4] > thresh)[0]

        # Adjust the dets tensor
        dets = dets[retained_idx + 1, :]

    return torch.tensor(keep, device=scores.device, dtype=torch.long)


def soft_nms_2(boxes, scores, idxs, thresh, sigma=.5, cuda=1):
    """
    SNMSv2 function implementation.

    Adapted from: https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py

    :param boxes: bounding box coordinates
    :param scores: score for each box
    :param idxs: class indexes
    :param thresh: IoU threshold
    :param method: gaussian or linear
    :param sigma: sigma value
    :param cuda: a boolean flag to determine if GPU or CPU will be used
    :return: boxes to keep after SNMS
    """

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes = boxes + offsets[:, None]

    # Indexes concatenate boxes with the last column
    N = boxes.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((boxes, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = box_scores
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + pos] = dets[maxpos.item() + pos].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + pos] = scores[maxpos.item() + pos].clone(), scores[i].clone()
                areas[i], areas[maxpos + pos] = areas[maxpos + pos].clone(), areas[i].clone()

        # IoU calculate
        xx1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        yy1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] *= weight

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].long()

    return keep


def combo_loss(inp, target, alpha=.5, ce_ratio=.5):
    """
    Combo Loss (CL) function implementation.

    Adapted from: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss

    :param inp: input tensor
    :param target: target tensor
    :param alpha: alpha value
    :param ce_ratio: BCE to DL ratio
    :return: the CL value
    """

    # Perform sigmoid operation on the input to get the probabilities
    inp = torch.sigmoid(inp)

    # Flatten the input and target tensors
    inp = inp.view(-1)
    target = target.view(-1)

    # Calculate the intersection and the Dice coefficient
    intersection = (inp * target).sum()
    dice = (2. * intersection + SMOOTH) / (inp.sum() + target.sum() + SMOOTH)

    # Calculate the CL
    inp = torch.clamp(inp, SMOOTH, 1. - SMOOTH)
    out = - (alpha * (target * torch.log(inp))) + ((1. - alpha) * ((1. - target) * torch.log(1. - inp)))
    weighted_ce = out.mean(-1)
    combo = (ce_ratio * weighted_ce) - ((1. - ce_ratio) * dice)
    return combo
