#!/usr/bin/env python

"""
A module to provide the boilerplate needed for all the analysis.
"""

import json
import re
from functools import partial

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import matplotlib  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import seaborn as sns  # type: ignore[import]

from imc import Project
from imc.types import Path

from seaborn_extensions import swarmboxenplot  # , activate_annotated_clustermap

# activate_annotated_clustermap()
swarmboxenplot = partial(swarmboxenplot, test_kws=dict(parametric=False))


def set_prj_clusters(
    prefix="roi_zscored.filtered.", cluster_str="cluster_1.0", aggregated=True
):
    # remove any existing assignment
    # and revert to what's on disk (clusters labeled by integers)
    prj._clusters = None
    new_labels = json.load(open("metadata/cluster_names.json"))[
        f"{prefix};{cluster_str}"
    ]
    new_labels = {int(k): v for k, v in new_labels.items()}
    for k in prj.clusters.unique():
        if k not in new_labels:
            new_labels[k] = "999 - ?()"
    new_labels_agg = {
        k: "".join(re.findall(r"\d+ - (.*) \(", v))
        for k, v in new_labels.items()
    }
    # add numbering for consistent plotting across ROIs
    ll = pd.Series(sorted(np.unique(list(new_labels_agg.values()))))
    lll = ll.index.astype(str).str.zfill(2) + " - " + ll
    lll.index = ll

    new_labels_agg = pd.Series(new_labels_agg).replace(lll.to_dict()).to_dict()

    # prj.clusters
    prj.set_clusters(
        prj.clusters.replace(new_labels_agg if aggregated else new_labels),
        write_to_disk=False,
    )


# constants
channels_exclude_strings = [
    "<EMPTY>",
    "190BCKG",
    "80ArAr",
    "129Xe",
]  # "DNA", "CD11b"]
roi_exclude_strings = [
    "20200701_COVID_11_LATE-01",
    "20200701_COVID_11_LATE-09",
    "20200701_COVID_11_LATE-10",
]

attributes = [
    "name",
    "disease",
    "phenotypes",
    "acquisition_id",
    "acquisition_date",
]

figkws = dict(dpi=300, bbox_inches="tight", pad_inches=0, transparent=False)

# directories
metadata_dir = Path("metadata")
data_dir = Path("data")
processed_dir = Path("processed")
results_dir = Path("results")
qc_dir = results_dir / "qc"


# lists of channels
panel_markers = pd.read_csv("metadata/panel_markers.COVID19-2.csv", index_col=0)
illustration_channel_list = json.load(
    open(metadata_dir / "illustration_markers.json")
)

# Initialize project
prj = Project(metadata_dir / "samples.csv", name="COVID19-2")

# Filter channels and ROIs
channels = (
    prj.channel_labels.stack().drop_duplicates().reset_index(level=1, drop=True)
)
channels_exclude = channels.loc[
    channels.str.contains(r"^\d")
    | channels.str.contains("|".join(channels_exclude_strings))
].tolist() + ["<EMPTY>(Sm152-In115)"]
channels_include = channels[~channels.isin(channels_exclude)]
cell_type_channels = panel_markers.query("cell_type == 1").index.tolist()

for roi in prj.rois:
    roi.set_channel_exclude(channels_exclude)

for s in prj:
    s.rois = [r for r in s if r.name not in roi_exclude_strings]


# # ROIs
roi_names = [x.name for x in prj.rois]
roi_attributes = (
    pd.DataFrame(
        np.asarray(
            [getattr(r.sample, attr) for r in prj.rois for attr in attributes]
        ).reshape((-1, len(attributes))),
        index=roi_names,
        columns=attributes,
    )
    .rename_axis(index="roi")
    .rename(columns={"name": "sample"})
)


# # Samples
sample_names = [x.name for x in prj.samples]
sample_attributes = (
    pd.DataFrame(
        np.asarray(
            [getattr(s, attr) for s in prj.samples for attr in attributes]
        ).reshape((-1, len(attributes))),
        index=sample_names,
        columns=attributes,
    )
    .rename_axis(index="sample")
    .drop(["name"], axis=1)
)

cat_order = {
    "disease": ["Healthy", "FLU", "ARDS", "COVID19"],
    "phenotypes": [
        "Healthy",
        "Flu",
        "ARDS",
        "Pneumonia",
        "COVID19_early",
        "COVID19_late",
    ],
    # "acquisition_date": sorted(list(set([str(r.sample.acquisition_date) for r in prj.rois]))),
}

for df in [roi_attributes, sample_attributes]:
    for cat, order in cat_order.items():
        df[cat] = pd.Categorical(df[cat], categories=order, ordered=True)

    df["acquisition_date"] = df["acquisition_date"].astype(int)
    df["acquisition_date"] -= df["acquisition_date"].min()


# Color codes
colors = dict()
# # Diseases
colors["disease"] = np.asarray(sns.color_palette())[[2, 0, 1, 3]]
["Healthy", "Flu", "ARDS", "COVID"]

# # Phenotypes
colors["phenotypes"] = np.asarray(sns.color_palette())[[2, 0, 1, 5, 4, 3]]


# Output files
metadata_file = metadata_dir / "clinical_annotation.pq"
quantification_file = results_dir / "cell_type" / "quantification.pq"
quantification_file_sum = results_dir / "cell_type" / "quantification.sum.pq"
gating_file = results_dir / "cell_type" / "gating.pq"
positive_file = results_dir / "cell_type" / "gating.positive.pq"
positive_count_file = results_dir / "cell_type" / "gating.positive.count.pq"
quantification_file = results_dir / "cell_type" / "quantification.pq"
h5ad_file = results_dir / "cell_type" / "anndata.all_cells.processed.h5ad"
counts_file = results_dir / "cell_type" / "cell_type_counts.pq"
counts_agg_file = results_dir / "cell_type" / "cell_type_counts.aggregated_pq"
roi_areas_file = results_dir / "roi_areas.csv"
sample_areas_file = results_dir / "sample_areas.csv"
