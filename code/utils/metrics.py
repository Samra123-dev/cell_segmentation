import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output: object, target: object) -> object:
    smooth = 1e-5
    output1 = torch.tensor([]).numpy()
    target1 = torch.tensor([]).numpy()
    if torch.is_tensor(output):
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output.data, dim=1).view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    for i in range(0, 6):
        if max(target[(i*65536):(65536*(i+1))]) == 1:
            output_ = (output[(i*65536):(65536*(i+1))] == 1)
            target_ = (target[(i*65536):(65536*(i+1))] == 1)
        elif max(target[(i*65536):(65536*(i+1))]) == 2:
            output_ = (output[(i*65536):(65536*(i+1))] == 2)
            target_ = (target[(i*65536):(65536*(i+1))] == 2)
        elif max(target[(i*65536):(65536*(i+1))]) == 3:
            output_ = (output[(i*65536):(65536*(i+1))] == 3)
            target_ = (target[(i*65536):(65536*(i+1))] == 3)
        elif max(target[(i*65536):(65536*(i+1))]) == 4:
            output_ = (output[(i*65536):(65536*(i+1))] == 4)
            target_ = (target[(i*65536):(65536*(i+1))] == 4)
        elif max(target[(i*65536):(65536*(i+1))]) == 5:
            output_ = (output[(i*65536):(65536*(i+1))] == 5)
            target_ = (target[(i*65536):(65536*(i+1))] == 5)
        elif max(target[(i*65536):(65536*(i+1))]) == 6:
            output_ = (output[(i*65536):(65536*(i+1))] == 6)
            target_ = (target[(i*65536):(65536*(i+1))] == 6)
        elif max(target[(i*65536):(65536*(i+1))]) == 7:
            output_ = (output[(i*65536):(65536*(i+1))] == 7)
            target_ = (target[(i*65536):(65536*(i+1))] == 7)
        else:
            output_ = np.zeros(65536).astype('bool')
            target_ = np.zeros(65536).astype('bool')
        output1 = np.hstack((output1, output_)).astype('bool')
        target1 = np.hstack((target1, target_)).astype('bool')
    intersection = (output1 & target1).sum()
    union = (output1 | target1).sum()
    return (intersection + smooth) / (union + smooth)


def f1_scorex(output, target):
    smooth = 1e-5
    output1 = torch.tensor([]).numpy()
    target1 = torch.tensor([]).numpy()
    if torch.is_tensor(output):
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output.data, dim=1).view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    for i in range(0, 6):
        if max(target[(i * 65536):(65536 * (i + 1))]) == 1:
            output_ = (output[(i * 65536):(65536 * (i + 1))] == 1)
            target_ = (target[(i * 65536):(65536 * (i + 1))] == 1)
        elif max(target[(i * 65536):(65536 * (i + 1))]) == 2:
            output_ = (output[(i * 65536):(65536 * (i + 1))] == 2)
            target_ = (target[(i * 65536):(65536 * (i + 1))] == 2)
        elif max(target[(i * 65536):(65536 * (i + 1))]) == 3:
            output_ = (output[(i * 65536):(65536 * (i + 1))] == 3)
            target_ = (target[(i * 65536):(65536 * (i + 1))] == 3)
        elif max(target[(i * 65536):(65536 * (i + 1))]) == 4:
            output_ = (output[(i * 65536):(65536 * (i + 1))] == 4)
            target_ = (target[(i * 65536):(65536 * (i + 1))] == 4)
        elif max(target[(i * 65536):(65536 * (i + 1))]) == 5:
            output_ = (output[(i * 65536):(65536 * (i + 1))] == 5)
            target_ = (target[(i * 65536):(65536 * (i + 1))] == 5)
        elif max(target[(i * 65536):(65536 * (i + 1))]) == 6:
            output_ = (output[(i * 65536):(65536 * (i + 1))] == 6)
            target_ = (target[(i * 65536):(65536 * (i + 1))] == 6)
        elif max(target[(i * 65536):(65536 * (i + 1))]) == 7:
            output_ = (output[(i * 65536):(65536 * (i + 1))] == 7)
            target_ = (target[(i * 65536):(65536 * (i + 1))] == 7)
        else:
            output_ = np.zeros(65536).astype('bool')
            target_ = np.zeros(65536).astype('bool')
        output1 = np.hstack((output1, output_)).astype('bool')
        target1 = np.hstack((target1, target_)).astype('bool')

    tp = (output1 & target1).sum()
    fp = (output1 & ~target1).sum()
    fn = (~output1 & target1).sum()

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    f1_score = 2 * (precision * recall) / (precision + recall + smooth)

    return f1_score


def dice_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output.data, dim=1).view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    intersection = (output & target).sum()
    dice_coefficient = (2 * intersection + smooth) / (output.sum() + target.sum() + smooth)
    return dice_coefficient


def iou_score2(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output.data, dim=1).view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    if max(target) == 1:
        output_ = (output == 1)
        target_ = (target == 1)
    elif max(target) == 2:
        output_ = (output == 2)
        target_ = (target == 2)
    elif max(target) == 3:
        output_ = (output == 3)
        target_ = (target == 3)
    elif max(target) == 4:
        output_ = (output == 4)
        target_ = (target == 4)
    elif max(target) == 5:
        output_ = (output == 5)
        target_ = (target == 5)
    elif max(target) == 6:
        output_ = (output == 6)
        target_ = (target == 6)
    elif max(target) == 7:
        output_ = (output == 7)
        target_ = (target == 7)

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return (intersection + smooth) / (union + smooth)



def f1_scorex2(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output.data, dim=1).view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    if max(target) == 1:
        output_ = (output == 1)
        target_ = (target == 1)
    elif max(target) == 2:
        output_ = (output == 2)
        target_ = (target == 2)
    elif max(target) == 3:
        output_ = (output == 3)
        target_ = (target == 3)
    elif max(target) == 4:
        output_ = (output == 4)
        target_ = (target == 4)
    elif max(target) == 5:
        output_ = (output == 5)
        target_ = (target == 5)
    elif max(target) == 6:
        output_ = (output == 6)
        target_ = (target == 6)
    elif max(target) == 7:
        output_ = (output == 7)
        target_ = (target == 7)

    tp = (output_ & target_).sum()
    fp = (output_ & ~target_).sum()
    fn = (~output_ & target_).sum()

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    f1_score = 2 * (precision * recall) / (precision + recall + smooth)
    return f1_score

