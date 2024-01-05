import torch
import random
import pickle
from features import play_data
from pathlib import Path, PurePosixPath
import os
from shutil import rmtree
from utils import if_none

FEATURES = "features"
TRAIN = "../train"
TEST = "../test"
VAL = "../validate"

train_percent = 0.7
train_test_weeks = ["w1", "w2", "w3", "w4", "w5", "w6", "w7"]
val_weeks = ["w8", "w9"]

MASTER_FNAME = "master.csv"


def get_train_test_val_splits(parent_dir: str = FEATURES) -> dict:
    """
    Manually assign games into train and val while holding out the last 2 weeks
    A production model should consider using a more robust approach such as k-folds,
    grouping on gameId
    """
    trainTestValAssignments = {"train": [], "val": [], "test": []}
    for child in Path(parent_dir).iterdir():
        if PurePosixPath(child).name in train_test_weeks:
            for gamePath in Path(child).iterdir():
                if random.random() < train_percent:
                    trainTestValAssignments["train"].append(gamePath)
                else:
                    trainTestValAssignments["val"].append(gamePath)
        elif PurePosixPath(child).name in val_weeks:
            for gamePath in Path(child).iterdir():
                trainTestValAssignments["test"].append(gamePath)
    return trainTestValAssignments


def create_data_space():
    """
    Create an empty master file and a master file with only the final 10 frames
    leading up to the tackle to load the location of each frame data
    """
    # fmt: off
    if (
        input(
            "This will delete all existing files in TRAIN, TEST, and VAL."
            "\nDo you want to continue? [y/n]") == "y"):
        # fmt: on
        pass
    else:
        return
    Path.unlink(MASTER_FNAME, missing_ok=True)
    with open(Path(FEATURES, MASTER_FNAME), 'w') as f:
        f.write('ballCarrierTensorFname,')
        f.write('offenseTensorFname,')
        f.write('tackleLabelFname,')
        f.write('tackleCoordinateFname,')
        f.write('touchdownResult,')
        f.write('week,')
        f.write('gameId,')
        f.write('playId,')
        f.write('framesFromStart,')
        f.write('framesFromEnd,')
        f.write('augmented,')
        f.write('frameId\n')
    for dir in [TRAIN, TEST, VAL]:
        dir = Path(FEATURES, dir)
        if os.path.isdir(dir):
            rmtree(dir)
        Path(dir).mkdir(parents=True, exist_ok=True)
        Path.unlink(Path(dir, MASTER_FNAME), missing_ok=True)
        with open(Path(dir, MASTER_FNAME), 'w') as f:
            f.write('ballCarrierTensorFname,')
            f.write('offenseTensorFname,')
            f.write('tackleLabelFname,')
            f.write('tackleCoordinateFname,')
            f.write('touchdownResult,')
            f.write('week,')
            f.write('gameId,')
            f.write('playId,')
            f.write('framesFromStart,')
            f.write('framesFromEnd,')
            f.write('augmented,')
            f.write('frameId\n')


def create_model_data(parent_dir: str, out_dir: str):
    """
    For this iteration, only include plays resulting in tackle event. Ignore touchdowns
    Future iterations, update with touchdowns
    """
    for playFname in parent_dir.iterdir():
        if playFname.is_file():
            with open(playFname, "rb") as f:
                playFeatures = pickle.load(f)
            week = playFeatures.week
            gameId = playFeatures.gameId
            playId = playFeatures.playId
            baseFname = f"w{week}_g{gameId}_p{playId}"

            startFrame = playFeatures.ballCarrierStartFrame
            endFrame = playFeatures.playEndFrame

            # Discard plays nullified due to penalty
            if playFeatures.playNullifiedByPenalty:
                continue

            # Discard plays with less than 1.3 seconds of ball carrier activity from training data
            # Trial and error determined this to be a good threshold for
            # eliminating plays with mistakes (fumbled snap/handoff, immediately blow block, trip)
            # and kneel-downs
            if endFrame < 13:
                continue

            tacklesIdList = if_none(playFeatures.assistTackleIdList, []) + if_none(
                playFeatures.tackleIdList, []
            )
            touchdownResult = playFeatures.touchdownResult
            if len(tacklesIdList) == 0 and not touchdownResult:
                continue
            tackleCoordinates = ()
            if not touchdownResult:
                tackleCoordinates = if_none(playFeatures.tackleCoordinates, ())
            if len(tackleCoordinates) == 0 and not touchdownResult:
                continue
            touchdownResult = int(touchdownResult)

            # labels = [0] * 11
            labelsFname = ""
            labelsFnameAugmented = ""
            coordFname = ""
            coordFnameAugmented = ""
            for i in range(startFrame, endFrame + 1):
                framesFromStart = i - startFrame
                framesFromEnd = endFrame - i
                bcTensor = playFeatures.ballCarrierFeaturesTensorNorm[i][0]
                bcFname = f"{baseFname}_f{i}_bc.pt"
                torch.save(bcTensor, Path(FEATURES, out_dir, bcFname))

                offTensor = playFeatures.offenseFeaturesTensorNorm[i][0]
                offFname = f"{baseFname}_f{i}_off.pt"
                torch.save(offTensor, Path(FEATURES, out_dir, offFname))

                if not touchdownResult:
                    defIdList = playFeatures.ballCarrierFeaturesTensorNorm[i][1]
                    labels = [int(x in tacklesIdList) for x in defIdList]
                    labelsFname = f"{baseFname}_f{i}_labels.csv"
                    with open(Path(FEATURES, out_dir, labelsFname), "w+") as f:
                        f.write(",".join(str(x) for x in labels))

                if i == startFrame:
                    if not touchdownResult:
                        coordFname = f"{baseFname}_coord.csv"
                        with open(Path(FEATURES, out_dir, coordFname), "w+") as f:
                            f.write(",".join(str(x) for x in tackleCoordinates))

                        coordFnameAugmented = f"{baseFname}_coord_augmented.csv"
                        x, y = tackleCoordinates
                        y = (160 / 3) - y
                        tackleCoordinatesAugmented = (x, y)
                        with open(Path(FEATURES, out_dir, coordFnameAugmented), "w+") as f:
                            f.write(",".join(str(x) for x in tackleCoordinatesAugmented))

                with open(Path(FEATURES, out_dir, MASTER_FNAME), "a+") as f:
                    f.write(bcFname + ",")
                    f.write(offFname + ",")
                    f.write(labelsFname + ",")
                    f.write(coordFname + ",")
                    f.write(str(touchdownResult) + ",")
                    f.write(str(week) + ",")
                    f.write(str(gameId) + ",")
                    f.write(str(playId) + ",")
                    f.write(str(framesFromStart) + ",")
                    f.write(str(framesFromEnd) + ",")
                    f.write(str(0) + ",")  # augmented data = False
                    f.write(str(i) + "\n")
                with open(Path(FEATURES, MASTER_FNAME), "a+") as f:
                    f.write(f"{out_dir}/{bcFname}" + ",")
                    f.write(f"{out_dir}/{offFname}" + ",")
                    f.write(f"{out_dir}/{labelsFname}" + ",")
                    f.write(f"{out_dir}/{coordFname}" + ",")
                    f.write(str(touchdownResult) + ",")
                    f.write(str(week) + ",")
                    f.write(str(gameId) + ",")
                    f.write(str(playId) + ",")
                    f.write(str(framesFromStart) + ",")
                    f.write(str(framesFromEnd) + ",")
                    f.write(str(0) + ",")  # augmented data = False
                    f.write(str(i) + "\n")
                if framesFromEnd < 10:
                    # Write augmented data - using final ten frames flipped on y-axis
                    # To give frames with more predictability higher weight
                    # And supplement existing data
                    bcTensorAugmented = playFeatures.ballCarrierFeaturesTensorNorm[-i][0]
                    bcFnameAugmented = f"{baseFname}_f{i}_bc_augmented.pt"
                    torch.save(bcTensorAugmented, Path(FEATURES, out_dir, bcFnameAugmented))

                    offTensorAugmented = playFeatures.offenseFeaturesTensorNorm[-i][0]
                    offFnameAugmented = f"{baseFname}_f{i}_off_augmented.pt"
                    torch.save(offTensorAugmented, Path(FEATURES, out_dir, offFnameAugmented))

                    if not touchdownResult:
                        defIdListAugmented = playFeatures.ballCarrierFeaturesTensorNorm[-i][1]
                        labelsAugmented = [int(x in tacklesIdList) for x in defIdListAugmented]
                        labelsFnameAugmented = f"{baseFname}_f{i}_labels_augmented.csv"
                        with open(Path(FEATURES, out_dir, labelsFnameAugmented), "w+") as f:
                            f.write(",".join(str(x) for x in labelsAugmented))

                    with open(Path(FEATURES, out_dir, MASTER_FNAME), "a+") as f:
                        f.write(bcFnameAugmented + ",")
                        f.write(offFnameAugmented + ",")
                        f.write(labelsFnameAugmented + ",")
                        f.write(coordFnameAugmented + ",")
                        f.write(str(touchdownResult) + ",")
                        f.write(str(week) + ",")
                        f.write(str(gameId) + ",")
                        f.write(str(playId) + ",")
                        f.write(str(framesFromStart) + ",")
                        f.write(str(framesFromEnd) + ",")
                        f.write(str(1) + ",")  # augmented data = True
                        f.write(str(i) + "\n")
                    with open(Path(FEATURES, MASTER_FNAME), "a+") as f:
                        f.write(f"{out_dir}/{bcFnameAugmented}" + ",")
                        f.write(f"{out_dir}/{offFnameAugmented}" + ",")
                        f.write(f"{out_dir}/{labelsFnameAugmented}" + ",")
                        f.write(f"{out_dir}/{coordFnameAugmented}" + ",")
                        f.write(str(touchdownResult) + ",")
                        f.write(str(week) + ",")
                        f.write(str(gameId) + ",")
                        f.write(str(playId) + ",")
                        f.write(str(framesFromStart) + ",")
                        f.write(str(framesFromEnd) + ",")
                        f.write(str(1) + ",")  # augmented data = True
                        f.write(str(i) + "\n")


if __name__ == "__main__":
    create_data_space()
    trainTestValAssignments = get_train_test_val_splits()

    for gamePath in trainTestValAssignments.get("train"):
        create_model_data(gamePath, TRAIN)

    for gamePath in trainTestValAssignments.get("val"):
        create_model_data(gamePath, VAL)

    for gamePath in trainTestValAssignments.get("test"):
        create_model_data(gamePath, TEST)
