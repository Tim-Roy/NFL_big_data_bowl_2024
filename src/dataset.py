import pandas as pd
from pathlib import Path
import csv
import torch
from torch.utils.data import Dataset


class nfl_touchdown_data(Dataset):
    def __init__(self, data_dir: str, master_file: str):
        """
        Master file contains the location frame tensors and labels
        """
        self.directory = data_dir
        self.master = pd.read_csv(Path(self.directory, master_file))
        self.len = self.master.shape[0]

    def __len__(self):
        return len(self.master)

    def __getitem__(self, idx: int):
        tensorBc = torch.load(Path(self.directory, self.master.loc[idx, "ballCarrierTensorFname"]))
        tensorOff = torch.load(Path(self.directory, self.master.loc[idx, "offenseTensorFname"]))
        touchdownResult = torch.tensor([self.master.loc[idx, "touchdownResult"]]).type(
            torch.float32
        )

        # Decided to flatten the 2d (x, y) dimensions to reduce the number of dimensions
        # and eliminate the duplication (broadcasting) of scalar values (max speed and weight)
        # hence the slightly awkward shuffling in the next few lines

        tensorBc = (
            tensorBc.flatten(1, 2)
            .index_select(1, torch.tensor([4, 5, 0, 2, 6, 7, 8, 9]))
            .expand(10, 11, 8)
            .permute(1, 0, 2)
        )
        tensorOff = tensorOff.flatten(2, 3).index_select(2, torch.tensor([0, 2, 4, 5, 6, 7]))
        playFeatures = torch.cat([tensorBc, tensorOff], dim=2).permute(2, 0, 1)

        return playFeatures, touchdownResult


class nfl_tackle_data(Dataset):
    def __init__(self, data_dir: str, master_file: str):
        """
        Master file contains the location frame tensors and labels
        """
        self.directory = data_dir
        master = pd.read_csv(Path(self.directory, master_file))
        # self.master = master[master["touchdownResult"].eq(0)].reset_index(drop=True)
        self.master = master[
            master["touchdownResult"].eq(0) & master["augmented"].eq(0)
        ].reset_index(drop=True)
        self.len = self.master.shape[0]

    def __len__(self):
        return len(self.master)

    def __getitem__(self, idx: int):
        tensorBc = torch.load(Path(self.directory, self.master.loc[idx, "ballCarrierTensorFname"]))
        tensorOff = torch.load(Path(self.directory, self.master.loc[idx, "offenseTensorFname"]))
        labelTackle_fname = Path(self.directory, self.master.loc[idx, "tackleLabelFname"])
        with open(labelTackle_fname, "r") as f:
            labelTackle = torch.tensor(
                [int(x) for x in list(csv.reader(f, delimiter=","))[0]]
            ).type(torch.float32)
        labelCoord_fname = Path(self.directory, self.master.loc[idx, "tackleCoordinateFname"])
        with open(labelCoord_fname, "r") as f:
            labelCoord = torch.tensor(
                [float(x) for x in list(csv.reader(f, delimiter=","))[0]]
            ).type(torch.float32)

        # Decided to flatten the 2d (x, y) dimensions to reduce the number of dimensions
        # and eliminate the duplication (broadcasting) of scalar values (max speed and weight)
        # hence the slightly awkward shuffling in the next few lines

        tensorBc = (
            tensorBc.flatten(1, 2)
            .index_select(1, torch.tensor([4, 5, 0, 2, 6, 7, 8, 9]))
            .expand(10, 11, 8)
            .permute(1, 0, 2)
        )
        tensorOff = tensorOff.flatten(2, 3).index_select(2, torch.tensor([0, 2, 4, 5, 6, 7]))
        tackleFeatures = torch.cat([tensorBc, tensorOff], dim=2).permute(2, 0, 1)

        return tackleFeatures, labelTackle, labelCoord


class nfl_data(Dataset):
    def __init__(self, playFeatures):
        """
        For predictions
        Remove augmented data (indicated with a negative index)
        """
        self.ballCarrierFeaturesTensor = {
            k: playFeatures.ballCarrierFeaturesTensorNorm[k]
            for k in playFeatures.ballCarrierFeaturesTensorNorm.keys()
            if k > 0
        }
        self.offenseFeaturesTensor = {
            k: playFeatures.offenseFeaturesTensorNorm[k][0]
            for k in playFeatures.offenseFeaturesTensorNorm.keys()
            if k > 0
        }
        self.touchdownResult = int(playFeatures.touchdownResult)

    def __len__(self):
        return len(self.ballCarrierFeaturesTensor)

    def __getitem__(self, idx: int):
        tensorBc, defenseIdList = self.ballCarrierFeaturesTensor.get(idx + 1)
        tensorOff = self.offenseFeaturesTensor.get(idx + 1)
        tensorBc = (
            tensorBc.flatten(1, 2)
            .index_select(1, torch.tensor([4, 5, 0, 2, 6, 7, 8, 9]))
            .expand(10, 11, 8)
            .permute(1, 0, 2)
        )
        tensorOff = tensorOff.flatten(2, 3).index_select(2, torch.tensor([0, 2, 4, 5, 6, 7]))
        playFeatures = torch.cat([tensorBc, tensorOff], dim=2).permute(2, 0, 1)

        frameId = idx + 1

        return playFeatures, defenseIdList, self.touchdownResult, frameId


class nfl_data_replacement(Dataset):
    def __init__(self, playFeatures, nflId, weightOverAvg, maxSpeedOverAvg):
        """
        For predictions when replacing one player with an average player at that position
        Remove augmented data (indicated with a negative index)
        """
        self.ballCarrierFeaturesTensor = {
            k: playFeatures.ballCarrierFeaturesTensorNorm[k]
            for k in playFeatures.ballCarrierFeaturesTensorNorm.keys()
            if k > 0
        }
        self.offenseFeaturesTensor = {
            k: playFeatures.offenseFeaturesTensorNorm[k]
            for k in playFeatures.offenseFeaturesTensorNorm.keys()
            if k > 0
        }
        self.touchdownResult = int(playFeatures.touchdownResult)
        self.nflId = nflId
        self.avgWeightNormDelta = torch.tensor(
            weightOverAvg[weightOverAvg.nflId.eq(nflId)]["weightOverAvgNorm"].values[0]
        )
        self.avgSpeedNormDelta = torch.tensor(
            maxSpeedOverAvg[maxSpeedOverAvg.nflId.eq(nflId)]["speedOverAvgNorm"].values[0]
        )

    def __len__(self):
        return len(self.ballCarrierFeaturesTensor)

    def __getitem__(self, idx: int):
        tensorBc, defenseIdList = self.ballCarrierFeaturesTensor.get(idx + 1)
        playerIdx = defenseIdList.index(self.nflId)
        tensorBc[playerIdx][0] -= self.avgWeightNormDelta
        tensorBc[playerIdx][1] -= self.avgSpeedNormDelta
        tensorBc = (
            tensorBc.flatten(1, 2)
            .index_select(1, torch.tensor([4, 5, 0, 2, 6, 7, 8, 9]))
            .expand(10, 11, 8)
            .permute(1, 0, 2)
        )

        tensorOff = self.offenseFeaturesTensor.get(idx + 1)[0]
        defenseIdList = self.offenseFeaturesTensor.get(idx + 1)[1]
        playerIdx = defenseIdList.index(self.nflId)
        tensorOff[playerIdx] -= (
            torch.cat(
                (
                    self.avgWeightNormDelta.reshape(
                        1,
                    ).repeat(2),
                    self.avgSpeedNormDelta.reshape(
                        1,
                    ).repeat(2),
                    torch.zeros(4),
                )
            )
            .reshape(4, -1)
            .expand_as(tensorOff[playerIdx])
        )
        tensorOff = tensorOff.flatten(2, 3).index_select(2, torch.tensor([0, 2, 4, 5, 6, 7]))
        playFeatures = torch.cat([tensorBc, tensorOff], dim=2).permute(2, 0, 1)

        frameId = idx + 1

        return playFeatures, defenseIdList, self.touchdownResult, frameId
