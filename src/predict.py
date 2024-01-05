from model import tackleNetwork, touchdownNetwork
from features import play_data
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path, PurePath
import pickle
from itertools import chain
import numpy as np
from dataset import nfl_data, nfl_data_replacement
from utils import get_play_features_files, load_model_from_checkpoint, if_none, write_df_to_parquet
from local import (
    DAT,
    MODEL_LOGS,
    TACKLE_MODEL_NAME,
    TACKLE_MODEL_CHECKPOINT,
    TD_MODEL_NAME,
    TD_MODEL_CHECKPOINT,
)

DAT += "/kaggle/input"
FEATURES = "../features"
WEEKS = list(range(1, 10))


def cap_total_probabilty(
    df: pd.DataFrame, grouping_col: str, probabiliity_col: str, cap: float
) -> pd.DataFrame:
    """
    There are never more than 3 recorded tackles in a play
    Ensure total probability of tackles is capped at 3
    which can sometimes occur early in the play
    """
    temp_col = f"{probabiliity_col}_total"
    total_prob = df.groupby(grouping_col)[probabiliity_col].sum().rename(temp_col)
    df = df.merge(total_prob, on=grouping_col)
    df[probabiliity_col] = df[probabiliity_col] * cap / df[temp_col].apply(lambda x: max(cap, x))
    df = df.drop(columns=temp_col)
    return df


def maxSpeed(weeks=WEEKS) -> pd.DataFrame:
    dfList = []
    for week in weeks:
        df = pd.read_csv(f"{DAT}/tracking_week_{week}.csv")
        df = df[df.club.ne("football")]
        df["week"] = week
        dfList.append(df)
    tracking = pd.concat(dfList, ignore_index=True)
    maxSpeedData = (
        tracking[~tracking.nflId.isna()]
        .sort_values(by="s", ascending=False)
        .drop_duplicates(["nflId", "gameId", "playId"])
        .groupby(["nflId", "displayName"])["s"]
        .nlargest(2)
        .to_frame()
        .reset_index()
        .drop(columns="level_2")
        .drop_duplicates("nflId", keep="last")
        .sort_values(by="s", ascending=False)
        .rename(columns={"s": "maxSpeed"})
    )
    return maxSpeedData


def calc_max_speed_over_avg(maxSpeedData, playersData, weigtMaxSpeedScalar):
    avgMaxSpeed = (
        maxSpeedData.merge(playersData[["nflId", "position"]], on="nflId")
        .groupby("position")["maxSpeed"]
        .mean()
        .rename("positionAvgMaxSpeed")
    )
    maxSpeedOverAvg = maxSpeedData.merge(playersData[["nflId", "position"]], on="nflId").merge(
        avgMaxSpeed, on="position"
    )
    maxSpeedOverAvg["speedOverAvg"] = (
        maxSpeedOverAvg["maxSpeed"] - maxSpeedOverAvg["positionAvgMaxSpeed"]
    )
    maxSpeedOverAvg["speedOverAvgNorm"] = maxSpeedOverAvg["speedOverAvg"] / float(
        weigtMaxSpeedScalar[1]
    )
    return maxSpeedOverAvg


def calc_weight_over_avg(playersData, weigtMaxSpeedScalar):
    weightOverAvg = playersData[["nflId", "displayName", "position", "weight"]]
    weightOverAvg = weightOverAvg.merge(avgWeight, on="position")
    weightOverAvg["weightOverAvg"] = weightOverAvg["weight"] - weightOverAvg["positionAvgWeight"]
    weightOverAvg["weightOverAvgNorm"] = weightOverAvg["weightOverAvg"] / float(
        weigtMaxSpeedScalar[0]
    )
    return weightOverAvg


def predict(playFeatures, tackle_model, td_model, smooth=True):
    """
    Includes additional data used for visual assessment (e.g: x, y)
    """
    features = nfl_data(playFeatures)
    tackleIdList = if_none(playFeatures.assistTackleIdList, []) + if_none(
        playFeatures.tackleIdList, []
    )
    tackleCoordinates = if_none(playFeatures.tackleCoordinates, ())
    if playFeatures.flippedToRight and len(tackleCoordinates) > 0:
        x, y = tackleCoordinates
        x = 120 - x
        y = 160 / 3 - y
        tackleCoordinates = (x, y)
    if len(tackleCoordinates) == 0:
        tackleCoordinates = (np.nan, np.nan)
    td_result = playFeatures.touchdownResult
    data_loader = DataLoader(
        features, batch_size=200, shuffle=False, drop_last=False, num_workers=20
    )
    tackleFeatures, defenseIdList, touchdownResult, frameId_list = next(data_loader.__iter__())
    defenseIdList = [list(x) for x in torch.stack(defenseIdList).permute(1, 0).numpy()]
    frameId_list = list(frameId_list.numpy())
    td_pred_batch = td_model.predict(tackleFeatures)
    tackle_pred_batch, x_y_coord_pred_batch = tackle_model.predict(tackleFeatures)
    td_pred_batch = td_pred_batch.detach().numpy()
    tackle_pred_batch = tackle_pred_batch.detach().numpy()
    x_y_coord_pred_batch = x_y_coord_pred_batch.detach().numpy()
    IDdata = []
    tacklePred = []
    tackleList = []
    tdPred = []
    if smooth:
        td_pred_batch = pd.Series(td_pred_batch).ewm(halflife=2).mean().values

    for nflIdList, td_pred, tackle_pred in zip(defenseIdList, td_pred_batch, tackle_pred_batch):
        IDdata.extend(nflIdList)
        tacklePred.extend(tackle_pred)
        tdPred.extend([td_pred[0]] * 11)
        tackleList.extend([int(x in tackleIdList) for x in nflIdList])
    df = pd.DataFrame(
        {
            "frameId": list(chain.from_iterable([x] * 11 for x in frameId_list)),
            "nflId": IDdata,
            "tackleResult": tackleList,
            "tackle_x": tackleCoordinates[0],
            "tackle_y": tackleCoordinates[1],
            "td_result": td_result,
            "tacklePred": tacklePred,
            "tdPred": tdPred,
            "expected_x": list(
                chain.from_iterable([x] * 11 for x in x_y_coord_pred_batch.transpose()[0])
            ),
            "expected_y": list(
                chain.from_iterable([y] * 11 for y in x_y_coord_pred_batch.transpose()[1])
            ),
        }
    )
    # flip back to original orientation
    if playFeatures.flippedToRight:
        df["expected_x"] = 120 - df["expected_x"]
        df["expected_y"] = (160 / 3) - df["expected_y"]
    # Nullify predictions outside of active ball carrier frames
    df.loc[
        df.frameId.lt(playFeatures.ballCarrierStartFrame),
        ["tacklePred", "tdPred", "expected_x", "expected_y"],
    ] = np.nan
    df.loc[
        df.frameId.gt(playFeatures.playEndFrame),
        ["tacklePred", "tdPred", "expected_x", "expected_y"],
    ] = np.nan
    if smooth:
        df["tacklePred"] = np.log(df["tacklePred"])
        df["tacklePred"] = df.groupby("nflId")["tacklePred"].transform(
            lambda x: x.ewm(halflife=2).mean().values
        )
        df["tacklePred"] = np.exp(df["tacklePred"])
    df["tacklePred"] *= 1 - df["tdPred"]
    df = cap_total_probabilty(df, "frameId", "tacklePred", 2)
    return df


def predict_replacement(
    playFeatures, tackle_model, td_model, weightOverAvg, maxSpeedOverAvg, smooth=True
):
    """
    Includes additional data used for visual assessment (e.g: x, y)
    """
    df_list = []
    for nflId in playFeatures.defenseIdList:
        IDdata = []
        tacklePred = []
        tackleList = []
        tdPred = []
        features = nfl_data_replacement(playFeatures, nflId, weightOverAvg, maxSpeedOverAvg)
        data_loader = DataLoader(
            features, batch_size=200, shuffle=False, drop_last=False, num_workers=20
        )
        tackleFeatures, defenseIdList, touchdownResult, frameId_list = next(data_loader.__iter__())
        defenseIdList = [list(x) for x in torch.stack(defenseIdList).permute(1, 0).numpy()]
        frameId_list = list(frameId_list.numpy())
        td_pred_batch = td_model.predict(tackleFeatures)
        tackle_pred_batch, x_y_coord_pred_batch = tackle_model.predict(tackleFeatures)
        td_pred_batch = td_pred_batch.detach().numpy()
        tackle_pred_batch = tackle_pred_batch.detach().numpy()
        x_y_coord_pred_batch = x_y_coord_pred_batch.detach().numpy()
        for nflIdList, td_pred, tackle_pred in zip(defenseIdList, td_pred_batch, tackle_pred_batch):
            idx = nflIdList.index(nflId)
            IDdata.append(nflId)
            tacklePred.append(tackle_pred[idx])
            tdPred.append(td_pred[0])
        df = pd.DataFrame(
            {
                "frameId": frameId_list,
                "nflId": IDdata,
                "tacklePredAvg": tacklePred,
                "tdPredAvg": tdPred,
                "expected_xAvg": x_y_coord_pred_batch.transpose()[0],
                "expected_yAvg": x_y_coord_pred_batch.transpose()[1],
            }
        )
        df_list.append(df)
        # flip back to original orientation
    df = pd.concat(df_list, ignore_index=True)
    if playFeatures.flippedToRight:
        df["expected_xAvg"] = 120 - df["expected_xAvg"]
        df["expected_yAvg"] = (160 / 3) - df["expected_yAvg"]
    # Nullify predictions outside of active ball carrier frames
    df.loc[
        df.frameId.lt(playFeatures.ballCarrierStartFrame),
        ["tacklePredAvg", "tdPredAvg", "expected_xAvg", "expected_yAvg"],
    ] = np.nan
    df.loc[
        df.frameId.gt(playFeatures.playEndFrame),
        ["tacklePredAvg", "tdPredAvg", "expected_xAvg", "expected_yAvg"],
    ] = np.nan
    if smooth:
        df["tacklePredAvg"] = np.log(df["tacklePredAvg"])
        df["tacklePredAvg"] = df.groupby("nflId")["tacklePredAvg"].transform(
            lambda x: x.ewm(halflife=2).mean().values
        )
        df["tacklePredAvg"] = np.exp(df["tacklePredAvg"])
    df["tacklePredAvg"] *= 1 - df["tdPredAvg"]
    df = cap_total_probabilty(df, "frameId", "tacklePredAvg", 3)
    df.insert(0, "week", playFeatures.week)
    df.insert(0, "gameId", playFeatures.gameId)
    df.insert(0, "playId", playFeatures.playId)
    return df


# TODO: update class with correceted load method
def load_play_features(play: str):
    with open(play, "rb") as p:
        playFeatures = pickle.load(p)
    return playFeatures


if __name__ == "__main__":
    from tqdm import tqdm

    tackle_model = Path(MODEL_LOGS, TACKLE_MODEL_NAME, TACKLE_MODEL_CHECKPOINT)
    tackle_model = load_model_from_checkpoint(tackleNetwork(), tackle_model)
    td_model = Path(MODEL_LOGS, TD_MODEL_NAME, TD_MODEL_CHECKPOINT)  ## update frist arg
    td_model = load_model_from_checkpoint(touchdownNetwork(), td_model)

    play_FEATURESs = get_play_features_files(FEATURES_OUT, WEEKS)
    weigtMaxSpeedScalar = torch.load(f"{FEATURES_OUT}/normalization/offenseTensorScalar.pt")
    weigtMaxSpeedScalar = weigtMaxSpeedScalar.flatten().index_select(0, torch.tensor([0, 2]))
    playersData = pd.read_csv(f"{DAT}/players.csv")
    playersData.loc[playersData.position.eq("MLB"), "position"] = "ILB"
    playersData.loc[playersData.position.eq("NT"), "position"] = "DT"
    playersData.loc[playersData.nflId.eq(48226), "position"] = "TE"  # Matt Orzech, only listed LS
    playersData.loc[
        playersData.nflId.eq(52416), "position"
    ] = "ILB"  # Isaiah Simmons, only listed DB
    avgWeight = playersData.groupby("position")["weight"].mean().rename("positionAvgWeight")

    maxSpeedData = maxSpeed()
    maxSpeedOverAvg = calc_max_speed_over_avg(maxSpeedData, playersData, weigtMaxSpeedScalar)
    weightOverAvg = calc_weight_over_avg(playersData, weigtMaxSpeedScalar)

    def make_predictions(path):
        playFeatures = load_play_features(path)
        df = predict(playFeatures, tackle_model, td_model).merge(
            predict_replacement(
                playFeatures, tackle_model, td_model, weightOverAvg, maxSpeedOverAvg
            ),
            on=["frameId", "nflId"],
        )
        write_df_to_parquet(df, "predictions", ["week", "gameId", "playId"])

    for path in tqdm(play_FEATURESs):
        make_predictions(path)
