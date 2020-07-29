#!/usr/bin/env python

"""
A module to provide the boilerplate needed for all the analysis.
"""

import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from imc import Project
from imc.types import Path

# constants
channels_exclude_strings = ["<EMPTY>", "SARSCoV2S1", "CD45"]  # "DNA", "CD11b"]
roi_exclude_strings = [
    "20200701_COVID_11_LATE-01",
    "20200701_COVID_11_LATE-09",
    "20200701_COVID_11_LATE-10",
]

attributes = ["disease", "phenotypes", "acquisition_id", "acquisition_date"]

figkws = dict(dpi=300, bbox_inches="tight")

# directories
metadata_dir = Path("metadata")
results_dir = Path("results")
qc_dir = results_dir / "qc"


# lists of channels
panel_markers = pd.read_csv("metadata/panel_markers.COVID19-2.csv", index_col=0)
illustration_channel_list = json.load(
    open(metadata_dir / "illustration_markers.json")
)

# Initialize project
prj = Project("metadata/samples.csv", name="COVID19-2")

# Filter channels and ROIs
channels = (
    prj.channel_labels.stack().drop_duplicates().reset_index(level=1, drop=True)
)
channels_exclude = channels.loc[
    channels.str.contains(r"^\d")
    | channels.str.contains("|".join(channels_exclude_strings))
].tolist()
channels_include = channels[~channels.isin(channels_exclude)]
cell_type_channels = panel_markers.query("cell_type == 1").index.tolist()

for s in prj:
    s.rois = [r for r in s if r.name not in roi_exclude_strings]

roi_names = [x.name for x in prj.rois]

roi_attributes = pd.DataFrame(
    np.asarray(
        [getattr(r.sample, attr) for r in prj.rois for attr in attributes]
    ).reshape((-1, len(attributes))),
    index=roi_names,
    columns=attributes,
).rename_axis(index="roi")

cat_order = {
    "disease": ["FLU", "ARDS", "Healthy", "COVID19"],
    "phenotypes": [
        "Flu",
        "ARDS",
        "Healthy",
        "COVID19_late",
        "COVID19_early",
        "Pneumonia",
    ],
    # "acquisition_date": sorted(list(set([str(r.sample.acquisition_date) for r in prj.rois]))),
}
for cat, order in cat_order.items():
    roi_attributes[cat] = pd.Categorical(
        roi_attributes[cat], categories=order, ordered=True
    )

roi_attributes["acquisition_date"] = np.log10(
    roi_attributes["acquisition_date"].astype(int)
)


sample_attributes = roi_attributes.copy()
sample_attributes.index = (
    roi_attributes.index.str.split("-").to_series().apply(pd.Series)[0]
)
sample_attributes = sample_attributes.drop_duplicates()
