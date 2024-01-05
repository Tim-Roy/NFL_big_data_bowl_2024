import pandas as pd
import numpy as np
from pathlib import Path
from random import shuffle
from typing import Union, List
from sklearn.preprocessing import minmax_scale
import math
import torch
import pickle
from tqdm import tqdm
from local import DAT
from function import relative_speed_2d, orient_plays_to_right, flip_y_axis, get_smallest_distance
from utils import get_play_features_files

DAT += "/kaggle/input"
FEATURES = "../features"
WEEKS = list(range(1, 10))


class play_data:
    def __init__(self, week, gameId, playId):
        self.week = week
        self.gameId = gameId
        self.playId = playId
        self.tackleIdList = []
        self.assistTackleIdList = []
        self.forcedFumbleList = []
        self.missedTackleIdList = []
        self.tackleCoordinates = ()
        self.outOfBounds = False
        self.defenseIdList = []
        self.offenseIdList = []
        self.touchdownResult = False
        self.touchdownFrame = None
        self.touchdownCoordinates = ()
        self.penaltyList = []
        self.ballSnapFrame = None
        self.handoffFrame = None
        self.pass_forwardFrame = None
        self.pass_outcome_caughtFrame = None
        self.playNullifiedByPenalty = False
        self.ballCarrierFeaturesTensor = dict()
        self.offenseFeaturesTensor = dict()
        self.flippedToRight = False
        self.playEndFrame = None
        self.tackleFrame = None
        self.ballCarrierStartFrame = 1

    def shuffle_players(self):
        assert (self.defenseIdList) is not None, "defenseIdList must be entered to perform shuffle"
        assert (self.offenseIdList) is not None, "offenseIdList must be entered to perform shuffle"
        shuffle(self.defenseIdList)
        shuffle(self.offenseIdList)

    def update_week(self, week: str):
        self.week = week

    def update_ball_carrier(self, nflId: int):
        self.ballCarrierId = nflId

    def update_offensive_players(self, idList: list):
        """
        Excludes ball carrier - should only contain 10 players
        """
        assert (len(idList)) == 10, f"idList contains {(len(idList))} players; should contain 10"
        self.offenseIdList = idList

    def update_defensive_players(self, idList: list):
        assert (len(idList)) == 11, f"idList contains {(len(idList))} players; should contain 11"
        self.defenseIdList = idList

    def update_tackles(self, tackleData):
        tacklesData = tackleData[
            tackleData.gameId.eq(self.gameId) & tackleData.playId.eq(self.playId)
        ]
        if not tacklesData[tacklesData.tackle.eq(1)].empty:
            self.tackleIdList = list(tacklesData[tacklesData.tackle.eq(1)].nflId)
        if not tacklesData[tacklesData.assist.eq(1)].empty:
            self.assistTackleIdList = list(tacklesData[tacklesData.assist.eq(1)].nflId)
        if not tacklesData[tacklesData.forcedFumble.eq(1)].empty:
            self.forcedFumbleIdList = list(tacklesData[tacklesData.forcedFumble.eq(1)].nflId)
        if not tacklesData[tacklesData.pff_missedTackle.eq(1)].empty:
            self.missedTackleIdList = list(tacklesData[tacklesData.pff_missedTackle.eq(1)].nflId)

    def update_penalties(self, penaltyList: list):
        self.penaltyList = penaltyList

    def update_tackle_coordinates(self, x: float, y: float):
        self.tackleCoordinates = (x, y)

    def update_tackle_frame(self, frame: int):
        self.tackleFrame = frame

    def update_play_type(self, playType: str):
        """
        Either pass (p), rush (r), or QB scramble (q)
        """
        self.playType = playType

    def update_pass_event_frames(self, frameId: int, event: str):
        if event == "pass_forward":
            self.pass_forwardFrame = frameId
        if event == "pass_outcome_caught":
            self.pass_outcome_caughtFrame = frameId

    def update_touchdown_result(self, touchdownResult: bool, frame: int, x: float, y: float):
        self.touchdownResult = touchdownResult
        self.touchdownFrame = frame
        self.touchdownCoordinates = (x, y)

    def update_ball_snap(self, frame: int):
        self.ballSnapFrame = frame

    def update_handoff(self, frame: int):
        self.handoffFrame = frame

    def update_out_of_bounds(self, outOfBounds: bool):
        self.outOfBounds = outOfBounds

    def update_play_nullified_by_penalty(self, playNullifiedByPenalty: bool):
        self.playNullifiedByPenalty = playNullifiedByPenalty

    def update_play_result(self, playResult: float):
        self.playResult = playResult

    def update_abs_yard_line(self, absoluteYardline: float):
        self.absoluteYardline = absoluteYardline

    def update_flipped_to_right(self, flippedToRight: bool):
        self.flippedToRight = flippedToRight

    def update_play_end_frame(self, playEndFrame: int):
        self.playEndFrame = playEndFrame

    def update_ball_carrier_start_frame(self, ballCarrierStartFrame: int):
        self.ballCarrierStartFrame = ballCarrierStartFrame

    def update_weight_max_speed_tensor(self, weightData: pd.DataFrame, speedData: pd.DataFrame):
        """
        Update the 2 tensors storing delta weight & delta max speed

        ...

        Class attributes updated (or created):

        weightMaxSpeedBallCarrierTensor :
            dim = 11 X 2 X 2
            broadcasts the delta weight and delta max speed across 2D
            for each 11 defensive players vs. ball carrier

        weightMaxSpeedOffenseTensor :
            dim = 11 X 10 X 2 X 2
            broadcasts the delta weight and delta max speed across 2D
            for each 11 defensive players vs. each 10 offensive players (excluding the ball carrier)
        """
        weightBc = weightData[weightData.nflId.eq(self.ballCarrierId)].weight.to_numpy()
        weightOff = (
            weightData[weightData.nflId.isin(self.offenseIdList)]
            .sort_values(
                by="nflId", key=lambda column: column.map(lambda e: self.offenseIdList.index(e))
            )
            .weight.to_numpy()
        )
        weightDef = (
            weightData[weightData.nflId.isin(self.defenseIdList)]
            .sort_values(
                by="nflId", key=lambda column: column.map(lambda e: self.defenseIdList.index(e))
            )
            .weight.to_numpy()
        )
        weightBcMat = weightDef.reshape(-1, 1) - weightBc
        weightOffMat = weightDef.reshape(-1, 1) - weightOff

        speedBc = speedData[speedData.nflId.eq(self.ballCarrierId)].s.to_numpy()
        speedOff = (
            speedData[speedData.nflId.isin(self.offenseIdList)]
            .sort_values(
                by="nflId", key=lambda column: column.map(lambda e: self.offenseIdList.index(e))
            )
            .s.to_numpy()
        )
        speedDef = (
            speedData[speedData.nflId.isin(self.defenseIdList)]
            .sort_values(
                by="nflId", key=lambda column: column.map(lambda e: self.defenseIdList.index(e))
            )
            .s.to_numpy()
        )
        speedBcMat = speedDef.reshape(-1, 1) - speedBc
        speedOffMat = speedDef.reshape(-1, 1) - speedOff

        # Repeat scalars for compatability with 2D tensors
        weightBcMat = np.concatenate((weightBcMat, weightBcMat), axis=1)
        speedBcMat = np.concatenate((speedBcMat, speedBcMat), axis=1)
        self.weightMaxSpeedBallCarrierTensor = torch.as_tensor(
            np.concatenate((weightBcMat, speedBcMat), axis=1).reshape((11, 2, -1))
        )

        weightOffMat = np.concatenate(
            (weightOffMat.reshape((-1, 1, 1)), weightOffMat.reshape((-1, 1, 1))), axis=2
        ).reshape(11, 10, -1)
        speedOffMat = np.concatenate(
            (speedOffMat.reshape((-1, 1, 1)), speedOffMat.reshape((-1, 1, 1))), axis=2
        ).reshape(11, 10, -1)
        self.weightMaxSpeedOffenseTensor = torch.as_tensor(
            np.concatenate((weightOffMat, speedOffMat), axis=2).reshape(11, 10, 2, -1)
        )

    def update_frame_tensor(self, trackingData: pd.DataFrame, frame: int):
        """
        Update the 2 dicts containing the two tensors used for the model

        ...

        Class attributes updated (or created):

        ballCarrierFeaturesTensor :
            dim = 11 X 5 X 2
            The 5 attributes included are:
            - delta max weight
            - delta max speed
            - x, y of ball carrier
            - delta x, y of defensive player vs. ball carrier
            - delta speed (projected to x, y) of defensive player vs. ball carrier
            for each 11 defensive players vs. ball carrier
            The dict also stores the order of the defensive players

        offenseFeaturesTensor :
            dim = 11 X 10 X 4 X 2
            The 5 attributes included are:
            - delta max weight
            - delta max speed
            - delta x, y of defensive player vs. offensive player
            - delta speed (projected to x, y) of defensive player vs. offensive player
            for each 11 defensive players vs. each 10 offensive players (excluding the ball carrier)
            The dict also stores the order of the defensive and offensive players
        """
        bcData = trackingData[trackingData.nflId.eq(self.ballCarrierId)]
        defData = trackingData[trackingData.nflId.isin(self.defenseIdList)].sort_values(
            by="nflId", key=lambda column: column.map(lambda e: self.defenseIdList.index(e))
        )
        offData = trackingData[trackingData.nflId.isin(self.offenseIdList)].sort_values(
            by="nflId", key=lambda column: column.map(lambda e: self.offenseIdList.index(e))
        )

        xBc = bcData.x.values[0]
        yBc = bcData.y.values[0]
        speedBc = bcData.s.values[0]
        dirBc = bcData.dir.values[0]

        xDef = defData.x.to_numpy()
        yDef = defData.y.to_numpy()
        speedDef = defData.s.to_numpy()
        dirDef = defData.dir.to_numpy()

        xOff = offData.x.to_numpy()
        yOff = offData.y.to_numpy()
        speedOff = offData.s.to_numpy()
        dirOff = offData.dir.to_numpy()

        deltaSpeedBc = list(
            map(relative_speed_2d, speedDef, dirDef, np.repeat(speedBc, 11), np.repeat(dirBc, 11))
        )

        deltaSpeedOff = [
            list(
                map(
                    relative_speed_2d,
                    speedOff,
                    dirOff,
                    np.repeat(speedDef[i], speedOff.size),
                    np.repeat(dirDef[i], speedOff.size),
                )
            )
            for i in range(speedDef.size)
        ]

        # dim = 2 X 5 X 11
        self.ballCarrierFeaturesTensor.update(
            {
                frame: (
                    torch.cat(
                        (
                            self.weightMaxSpeedBallCarrierTensor,
                            torch.tile(torch.tensor([xBc, yBc]), (11, 1)).reshape(11, 1, -1),
                            torch.cat(
                                (
                                    (torch.tensor(xDef) - torch.tensor(xBc).repeat(11)).reshape(
                                        -1, 1
                                    ),
                                    (torch.tensor(yDef) - torch.tensor(yBc).repeat(11)).reshape(
                                        -1, 1
                                    ),
                                ),
                                axis=1,
                            ).reshape(11, 1, -1),
                            torch.tensor(deltaSpeedBc).reshape((11, 1, -1)),
                        ),
                        axis=1,
                    ).type(torch.float32),
                    self.defenseIdList.copy(),
                )
            }
        )

        # dim = 2 X 4 X 10 X 11
        self.offenseFeaturesTensor.update(
            {
                frame: (
                    torch.cat(
                        (
                            self.weightMaxSpeedOffenseTensor,
                            torch.tensor(
                                np.concatenate(
                                    (
                                        (xDef.reshape(-1, 1) - xOff).reshape(11, 10, 1, 1),
                                        (yDef.reshape(-1, 1) - yOff).reshape(11, 10, 1, 1),
                                    ),
                                    axis=2,
                                )
                            ).reshape(11, 10, -1, 2),
                            torch.tensor(deltaSpeedOff).reshape(11, 10, -1, 2) * -1,  # reverse sign
                        ),
                        axis=2,
                    ).type(torch.float32),
                    self.defenseIdList.copy(),
                    self.offenseIdList.copy(),
                )
            }
        )

    def pickle(self):
        out = f"{FEATURES_OUT}/w{self.week}/g{self.gameId}"
        Path(out).mkdir(parents=True, exist_ok=True)
        fname = f"{out}/p{self.playId}.pkl"
        with open(fname, "wb") as pfile:
            pickle.dump(self, pfile)

    @staticmethod
    def load(play: str):
        with open(play, "rb") as p:
            playFeatures = pickle.load(p)
        return playFeatures


if __name__ == "__main__":
    gamesData = pd.read_csv(f"{DAT}/games.csv")
    playersData = pd.read_csv(f"{DAT}/players.csv")
    playsData = pd.read_csv(f"{DAT}/plays.csv")
    tacklesData = pd.read_csv(f"{DAT}/tackles.csv")

    dfList = []
    for week in WEEKS:
        df = pd.read_csv(f"{DAT}/tracking_week_{week}.csv")
        df = df[df.club.ne("football")]
        df["week"] = week
        dfList.append(df)
    tracking = pd.concat(dfList, ignore_index=True)
    tracking = orient_plays_to_right(tracking)

    playsData["playType"] = "rush"
    playsData.loc[playsData.passResult.eq("C"), "playType"] = "pass"
    playsData.loc[playsData.passResult.eq("R"), "playType"] = "qb_scramble"

    # Use play with second highest speed for max speed to control for outliers
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
    )

    nflGames = list(gamesData.gameId)
    for game in tqdm(nflGames):
        for row in playsData[playsData.gameId.eq(game)].itertuples():
            playId = row.playId
            ballCarrierId = row.ballCarrierId
            possessionTeam = row.possessionTeam
            playType = row.playType
            foulName1 = row.foulName1
            foulName2 = row.foulName2
            playNullifiedByPenalty = False
            if row.playNullifiedByPenalty == "Y":
                playNullifiedByPenalty = True
            playResult = row.playResult
            absoluteYardline = row.absoluteYardlineNumber

            frames = list(
                tracking[tracking.gameId.eq(game) & tracking.playId.eq(playId)].frameId.unique()
            )

            # at least 1 play where the player tracking data is empty
            if len(frames) == 0:
                continue

            tackleEvenetData = tracking[
                tracking.gameId.eq(game)
                & tracking.playId.eq(playId)
                & tracking.event.eq("tackle")
                & tracking.nflId.eq(ballCarrierId)
            ]
            outOfBounds = False

            if tackleEvenetData.empty:
                tackleEvenetData = tracking[
                    tracking.gameId.eq(game)
                    & tracking.playId.eq(playId)
                    & tracking.event.eq("out_of_bounds")
                    & tracking.nflId.eq(ballCarrierId)
                ].copy()
                if not tackleEvenetData.empty:
                    outOfBounds = True
                    tackleEvenetData["y"] = get_smallest_distance(
                        # force ball carrier exactly at sidelines
                        tackleEvenetData["y"].values[0],
                        0,
                        160 / 3,
                    )

            playEndFrame = max(frames)
            if not tackleEvenetData.empty:
                playEndFrame = tackleEvenetData["frameId"].values[0]

            for frame in frames:
                df = tracking[
                    tracking.gameId.eq(game)
                    & tracking.playId.eq(playId)
                    & tracking.frameId.eq(frame)
                ]
                if frame == 1:
                    week = df["week"].values[0]
                    playDataNow = play_data(week, game, playId)
                    playDataNow.update_week(week)
                    playDataNow.update_ball_carrier(ballCarrierId)
                    playDataNow.update_out_of_bounds(outOfBounds)
                    playDataNow.update_play_nullified_by_penalty(playNullifiedByPenalty)
                    playDataNow.update_play_result(playResult)
                    playDataNow.update_abs_yard_line(absoluteYardline)
                    offenseIdList = list(
                        df[df.club.eq(possessionTeam) & df.nflId.ne(ballCarrierId)].nflId
                    )
                    defenseIdList = list(df[df.club.ne(possessionTeam)].nflId)
                    playDataNow.update_offensive_players(offenseIdList)
                    playDataNow.update_defensive_players(defenseIdList)
                    playDataNow.update_tackles(tacklesData)
                    playDataNow.update_play_type(playType)
                    playDataNow.update_flipped_to_right(df.flippedToRight.values[0])
                    playDataNow.update_play_end_frame(playEndFrame)
                    playDataNow.update_ball_carrier_start_frame(1)
                    if not pd.isna(foulName1):
                        penaltyList = [foulName1]
                        if not pd.isna(foulName2):
                            penaltyList.append(foulName2)
                        playDataNow.update_penalties(penaltyList)
                    if not tackleEvenetData.empty:
                        playDataNow.update_tackle_coordinates(
                            tackleEvenetData.x.values[0], tackleEvenetData.y.values[0]
                        )
                        playDataNow.update_tackle_frame(tackleEvenetData.frameId.values[0])

                if "pass_forward" in df.event.unique():
                    playDataNow.update_pass_event_frames(frame, "pass_forward")
                if "pass_outcome_caught" in df.event.unique():
                    playDataNow.update_pass_event_frames(frame, "pass_outcome_caught")
                    playDataNow.update_ball_carrier_start_frame(frame)
                if "run" in df.event.unique():
                    playDataNow.update_ball_carrier_start_frame(frame)
                if "touchdown" in df.event.unique():
                    playDataNow.update_touchdown_result(
                        True,
                        frame,
                        110,  # force ball carrier coordinate exactly at goal line
                        df[df.nflId.eq(ballCarrierId)]["y"].values[0],
                    )
                    playDataNow.update_play_end_frame(frame)
                if "ball_snap" in df.event.unique():
                    playDataNow.update_ball_snap(frame)
                    playDataNow.update_ball_carrier_start_frame(frame)
                if "handoff" in df.event.unique():
                    playDataNow.update_handoff(frame)
                    playDataNow.update_ball_carrier_start_frame(frame)
                playDataNow.shuffle_players()
                playDataNow.update_weight_max_speed_tensor(playersData, maxSpeedData)
                playDataNow.update_frame_tensor(df, frame)

            # Enrich data by giving more weight to the final ten frames and flipping those frames along the y-axis
            for frame in range(max(playDataNow.playEndFrame - 9, 1), playDataNow.playEndFrame + 1):
                df = tracking[
                    tracking.gameId.eq(game)
                    & tracking.playId.eq(playId)
                    & tracking.frameId.eq(frame)
                ]
                df = flip_y_axis(df)
                playDataNow.shuffle_players()
                playDataNow.update_weight_max_speed_tensor(playersData, maxSpeedData)
                playDataNow.update_frame_tensor(df, -frame)
            playDataNow.pickle()

    # Final step -- apply normalization after all tensors are calculated
    # Calculate the largest absolute value for x and y axis for both ball carrier and offense tensors
    # Except use 120 for x, y
    bc_tensors = []
    off_tensors = []
    play_FEATURESs = get_play_features_files(FEATURES_OUT, WEEKS)
    for play in play_FEATURESs:
        playFeatures = play_data.load(play)
        for i in list(playFeatures.ballCarrierFeaturesTensor.keys()):
            bc_tensors.append(playFeatures.ballCarrierFeaturesTensor.get(i)[0])
            off_tensors.append(playFeatures.offenseFeaturesTensor.get(i)[0])

    # TODO - refactor this - take abs first and cut steps in half
    off_max_val = (
        torch.stack(
            (
                torch.stack(off_tensors)  # max
                .max(0)
                .values.abs()
                .max(0)
                .values.max(0)
                .values.max(1)
                .values,
                torch.stack(off_tensors)
                .min(0)
                .values.abs()
                .max(0)
                .values.max(0)
                .values.max(1)
                .values,
            )
        )
        .max(0)
        .values
    )

    # TODO - refactor this - take abs first and cut steps in half
    bc_max_val = (
        torch.stack(
            (
                torch.stack(bc_tensors).max(0).values.abs().max(0).values.max(1).values,
                torch.stack(bc_tensors).min(0).values.abs().max(0).values.max(1).values,
            )
        )
        .max(0)
        .values
    )

    bc_scalar = torch.tensor(
        [
            max(bc_max_val[0], off_max_val[0]),
            max(bc_max_val[1], off_max_val[1]),
            120,
            max(bc_max_val[3], off_max_val[2]),
            max(bc_max_val[4], off_max_val[3]),
        ]
    )
    bc_scalar = torch.stack((bc_scalar, bc_scalar)).transpose(0, 1)

    off_scalar = torch.tensor(
        [
            max(bc_max_val[0], off_max_val[0]),
            max(bc_max_val[1], off_max_val[1]),
            max(bc_max_val[3], off_max_val[2]),
            max(bc_max_val[4], off_max_val[3]),
        ]
    )
    off_scalar = torch.stack((off_scalar, off_scalar)).transpose(0, 1)

    for play in play_FEATURESs:
        playFeatures = play_data.load(play)
        playFeatures.ballCarrierFeaturesTensorNorm = dict()
        playFeatures.offenseFeaturesTensorNorm = dict()
        for i in list(playFeatures.ballCarrierFeaturesTensor.keys()):
            vals, idx = playFeatures.ballCarrierFeaturesTensor.get(i)
            vals = vals / bc_scalar.expand_as(vals)
            playFeatures.ballCarrierFeaturesTensorNorm[i] = (vals, idx)
            vals_off, idx_def, idx_off = playFeatures.offenseFeaturesTensor.get(i)
            vals_off = vals_off / off_scalar.expand_as(vals_off)
            playFeatures.offenseFeaturesTensorNorm[i] = (vals_off, idx_def, idx_off)
            playFeatures.pickle()

    torch.save(bc_scalar, Path(FEATURES_OUT, "normalization", "ballCarrierTensorScalar.pt"))
    torch.save(off_scalar, Path(FEATURES_OUT, "normalization", "offenseTensorScalar.pt"))
