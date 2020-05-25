import torch
import torch.nn.functional as func


def compute_dice_metrics(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Computes the dice overlay as accuracy and accuracy
    Args:
        inputs: The output of the model
        targets: The ground truth used to compute the metrics
    Returns:
        Tuple: The dice percentage.
    """

    eps = 0.000000001
    intersection = (func.softmax(inputs, dim=1) * targets).sum(0)
    union = (func.softmax(inputs, dim=1) + targets).sum(0)
    numerator = 2 * intersection
    denominator = union + eps
    dic = 100 * ((numerator / denominator).sum() / 32)
    return dic


def compute_acc_metrics(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Computes the dice overlay as accuracy and accuracy
    Args:
        inputs: The output of the model
        targets: The ground truth used to compute the metrics
    Returns:
        Output: The percentage accuracy.
    """
    pred = torch.max(inputs, 1)[1]
    gt = torch.max(targets, 1)[1]
    acc = 100 * pred.eq(gt).sum().double() / gt.numel()

    return acc
