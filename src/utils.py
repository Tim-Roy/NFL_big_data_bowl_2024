from typing import Union
from pathlib import Path
import torch.nn as nn
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_play_features_files(base: str, weeks: list[int]) -> list:
    play_FEATURESs = []
    for week in weeks:
        data = f"{base}/w{week}"
        for game in Path(data).iterdir():
            for play in Path(game).iterdir():
                play_FEATURESs.append(Path(play))
    return play_FEATURESs


def load_model_from_checkpoint(model: nn.Module, model: Union[Path, str]):
    checkpoint = torch.load(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def if_none(x, noneReturnValue):
    if x is None:
        return noneReturnValue
    else:
        return x


def write_df_to_parquet(df: pd.DataFrame, out: str, partition_cols: list):
    pa_dat = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        pa_dat,
        out,
        partition_cols,
        existing_data_behavior="overwrite_or_ignore",
    )
