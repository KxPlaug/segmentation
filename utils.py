import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def intersect_and_union(pred, target, num_class):
    '''
    References https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
    '''
    pred = np.asarray(pred, dtype=np.uint8).copy()
    target = np.asarray(target, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    target += 1
    pred = pred * (target > 0)

    inter = pred * (pred == target)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_target, _) = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_target - area_inter

    return (area_inter, area_union)


def calculate_accuracy(loader, model, device, num_classes):
    model.eval()
    total_intersect = np.array([0] * num_classes)
    total_union = np.array([0] * num_classes)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(loader), total=len(loader)):
            data = data.to(device)
            if model.__class__.__name__ in ['LRASPP', 'FCN', 'DeepLabV3']:
                output = model(data)['out']
            else:
                output = model(data)
            output_predictions = output.argmax(1).detach().cpu()
            intersect, union = intersect_and_union(
                output_predictions, target, num_classes)
            total_intersect += intersect
            total_union += union

    total_union[total_union == 0] = 1
    iou = (total_intersect/total_union)
    avg_iou = np.mean(iou)
    return avg_iou
