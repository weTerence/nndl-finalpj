import torch
import numpy as np


def cutout(img, n_holes, length):
    """
    Randomly mask out one or more patches from an image.

    Args:
        img (Tensor): Tensor image of size (B, C, H, W).
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
    """
    h = img.size(2)
    w = img.size(3)

    mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)


        mask[y1: y2, x1: x2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask

    return img


def rand_bbox(size, lam):
    """
    Cutmix

    Args:
        size:
        lam: lambda

    Returns:
        bbx1:
        bby1:
        bbx2:
        bby2:
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Mixup

    Args:
        x:
        y:
        alpha:
        use_cuda:

    Returns:
        returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)