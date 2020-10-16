import pandas as pd
from tqdm import tqdm

from imc.utils import z_score


def z_score_by_column(
    df, column, exclude=["roi", "sample"], clip=(-2.5, 2.5),
):
    _zdf = list()
    for roi in tqdm(df[column].unique()):
        sel = df[column] == roi
        _zdf.append(z_score(df.loc[sel].drop(exclude, axis=1, errors="ignore")))

    zdf = pd.concat(_zdf)
    if clip is not None:
        zdf = zdf.clip(*clip)
    return pd.concat([zdf, df[exclude]], axis=1)
