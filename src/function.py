import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd


def unbalanced_weights(x: torch.tensor, sensitivity_weight: float = 0.5):
    """
    Returns the weight tensor for a 2 dimensional tensor:
    assuming first dimenion is batch, and second dimension is an array of zeros and ones
    provides weights for the loss function to address unbalanced binary classes.
    Works when the indices of the lowest dimension is randomly shuffled
    which is applied for each frame of the play (defensive and offensive players are shuffled)
    """
    mean_positive = x.mean()
    pos_weight = sensitivity_weight / (mean_positive * x.shape[1])
    neg_weight = (1 - sensitivity_weight) / ((1 - mean_positive) * x.shape[1])
    w_pos = torch.ones_like(x) * pos_weight * x
    w_neg = torch.ones_like(x) * neg_weight * (1 - x)
    weights = w_pos + w_neg
    weights /= weights.sum(1).reshape(x.shape[0], -1).repeat_interleave(x.shape[1], 1)
    return weights


def unbalanced_weight_binary(
    x: torch.tensor, positive_rate: float = 0.5, sensitivity_weight: float = 0.5
):
    negative_rate = 1 - positive_rate
    positive_weight = sensitivity_weight / positive_rate * x
    negative_weight = (1 - sensitivity_weight) / negative_rate * (1 - x)
    return positive_weight + negative_weight


def weighted_L1_loss(pred: torch.tensor, target: torch.tensor, x_weight: float = 0.5):
    """
    Errors in the x direction (yardline) should weigh more than errors in the y direction
    Due to missed tackles, we should expect some large "errors" - however we not want the model
    to overcorrect to these errors, hence use L1 norm - which is less sensitive to outliers
    """
    loss = nn.L1Loss()
    pred_x, pred_y = pred.permute(1, 0)
    target_x, target_y = target.permute(1, 0)
    return x_weight * loss(pred_x, target_x) + (1 - x_weight) * loss(pred_y, target_y)


def weighted_loss(class_loss, regression_loss, class_weight):
    return class_weight * class_loss + (1 - class_weight) * regression_loss


def relative_speed_2d(s1: float, dir1: float, s2: float, dir2: float) -> tuple[float, float]:
    if any(np.isnan([s1, dir1, s2, dir2])):
        return np.nan, np.nan
    vx1 = math.sin(math.radians(dir1)) * s1
    vy1 = math.cos(math.radians(dir1)) * s1
    vx2 = math.sin(math.radians(dir2)) * s2
    vy2 = math.cos(math.radians(dir2)) * s2

    return vx1 - vx2, vy1 - vy2


def orient_plays_to_right(df: pd.DataFrame) -> pd.DataFrame:
    df["flippedToRight"] = False
    df.loc[df.playDirection.eq("left"), "x"] = 120 - df.loc[df.playDirection.eq("left"), "x"]
    df.loc[df.playDirection.eq("left"), "y"] = (160 / 3) - df.loc[df.playDirection.eq("left"), "y"]
    df.loc[df.playDirection.eq("left"), "o"] = 180 + df.loc[df.playDirection.eq("left"), "o"]
    df.loc[df.o.gt(360), "o"] -= 360
    df.loc[df.playDirection.eq("left"), "dir"] = 180 + df.loc[df.playDirection.eq("left"), "dir"]
    df.loc[df.dir.gt(360), "dir"] -= 360
    df.loc[df.playDirection.eq("left"), "flippedToRight"] = True
    df.loc[df.playDirection.eq("left"), "playDirection"] = "right"
    return df


def flip_y_axis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = (160 / 3) - df["y"]
    df["o"] = df["o"].apply(lambda x: 180 - x + 360 * (int(x > 180)))
    df["dir"] = df["dir"].apply(lambda x: 180 - x + 360 * (int(x > 180)))
    return df


def get_smallest_distance(x: float, a: float, b: float) -> float:
    """
    Returns the number closer to x
    """
    if abs(x - a) < abs(x - b):
        return a
    else:
        return b
