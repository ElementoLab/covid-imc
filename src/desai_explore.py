# coding: utf-8

import sys
from typing import Tuple
from functools import wraps
import io

import requests

import parmap
import numpy as np
import pandas as pd
import imageio
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc
import gseapy as gp

from ngs_toolkit.general import enrichr, query_biomart

from seaborn_extensions import activate_annotated_clustermap, swarmboxenplot

from imc.types import Path, DataFrame, Series
from imc.graphics import get_grid_dims


activate_annotated_clustermap()

PIL.Image.MAX_IMAGE_PIXELS = 933120000

figkws = dict(dpi=300, bbox_inches="tight")
data_dir = Path("data") / "rna-seq"
output_dir = Path("results") / "rna-seq"
output_dir.mkdir(exist_ok=True)

database_dir = Path("data") / "gene_set_libraries"
scrnaseq_dir = Path("data") / "krasnow_scrna-seq"

PROTEIN_MATRIX_URL = (
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series"
    "/GSE159nnn"
    "/GSE159785"
    "/suppl"
    "/GSE159785_analyzed_count_matrix.txt.gz"
)
RNA_MATRIX_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159787/suppl/GSE159787_analyzed_counts.txt.gz"

SERIES_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159788/matrix/GSE159788-GPL29228_series_matrix.txt.gz"

DECONVOLUTION_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159787/suppl/GSE159787_cell_deconvolution.txt.gz"


SOURCE_DATA_URL = (
    "https://static-content.springer.com"
    "/esm"
    "/art%3A10.1038%2Fs41467-020-20139-7"
    "/MediaObjects"
    "/41467_2020_20139_MOESM8_ESM.xlsx"
)

CLINVARS = [
    "disease",
    "phenotypes",
    "pcr_spike_positive",
    # "days_of_disease",
    # "gender",
]

CELL_GROUPS = {
    "Basal": "Basal|Goblet",
    "Endothelial": "Artery|Vein|Vessel|Capilary",
    "Vascular": "Fibroblast|Pericyte|Smooth|Fibromyocyte|Myofibroblast",
    "Epithelial": "Epithelial",
    "Myeloid": "Monocyte|Macrophage|Dendritic",
    "Ciliated": "Ciliated|Ionocyte",
    "Lymphoid": "CD4|CD8|Plasma|B",
}


def main() -> int:
    # load data and plot sample correlations
    x, y = load_X_Y_data()

    return 0


def close_plots(func):
    """
    Decorator to close all plots on function exit.
    """

    @wraps(func)
    def close(*args, **kwargs):
        func(*args, **kwargs)
        plt.close("all")

    return close


def load_X_Y_data() -> Tuple[DataFrame, DataFrame]:
    """
    Read up expression data (X), metadata (Y) and match them by index.
    """

    def fix_axis(df):
        df.columns = df.columns.str.strip()
        df.index = (
            df.index.str.strip().str.replace("cse", "case").str.capitalize()
        )
        return df.sort_index()

    def stack(df):
        cols = range(len(df.columns.levels[0]))
        return pd.concat(
            [
                df.loc[:, df.columns.levels[0][i]]
                .stack()
                .reset_index(level=1, drop=True)
                for i in cols
            ],
            axis=1,
        ).rename(columns=dict(zip(cols, df.columns.levels[0])))

    # X: expression data

    u = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159787/suppl/GSE159787_collapsed_target_counts.txt.gz"
    x = pd.read_csv(u, sep="\t", index_col=0)
    x_rna = pd.read_csv(RNA_MATRIX_URL, sep="\t", index_col=0)
    x_pro = pd.read_csv(PROTEIN_MATRIX_URL, sep="\t", index_col=0)

    # Y: metadata

    content = io.BytesIO()
    with requests.get(SOURCE_DATA_URL) as req:
        content.write(req.content)

    # metadata and data
    viral_load = fix_axis(
        pd.read_excel(content, index_col=0, sheet_name="figure 1b")
    )
    time_since_symptoms = fix_axis(
        pd.read_excel(content, index_col=0, sheet_name="figure 1 d")
    )
    struc_cell_content = pd.read_excel(
        content, index_col=0, sheet_name="figure 1e", header=[0, 1]
    )
    struc_cell_content = fix_axis(stack(struc_cell_content))

    lymph_content = pd.read_excel(
        content, index_col=0, sheet_name="figure 4 b", header=[0, 1]
    )
    lymph_content = fix_axis(stack(lymph_content))

    myelo_content = pd.read_excel(content, index_col=0, sheet_name="figure 4 d")
    myelo_content.columns = myelo_content.columns.str.replace(
        "Macrophages", "Macrophage"
    )
    myelo_content = myelo_content.loc[~myelo_content.isnull().all(1)]
    myelo_content.columns = pd.MultiIndex.from_frame(
        myelo_content.columns.str.extract(r"^(.*) ?.* (.*)$")
    )
    myelo_content = fix_axis(stack(myelo_content))
    myelo_content.columns = myelo_content.columns + " / %"
    myelo_content.index = (
        myelo_content.index.str.replace("Case ", "Case")
        .str.replace("Case", "Case ")
        .str.replace(" ", ".")
        .str.replace("-", ".")
        .str.replace(r"\.\.", ".")
        .str.extract(r"(Case\..*?\.).*")[0]
        .str.replace(r"\.", " ")
        .str.strip()
    )
    myelo_content_red = myelo_content.groupby(level=0).mean()

    deconv = pd.read_excel(
        content, index_col=0, sheet_name="Supplementary figure 5"
    ).T
    deconv = deconv.loc[~deconv.isnull().all(1)]
    deconv_red = deconv.copy()
    deconv_red.index = (
        (
            (deconv_red.index.str.replace("Case ", "Case")).str.extract(
                r"(.*)?-"
            )[0]
        )
        .fillna(pd.Series(deconv_red.index))
        .str.replace("Case", "Case ")
        .str.replace("NegControl", "NegControl ")
        .str.strip()
    )
    deconv_red = fix_axis(deconv_red)
    deconv_red = deconv_red.groupby(level=0).mean()

    deconv = pd.read_table(DECONVOLUTION_URL, index_col=0)
    deconv = deconv.loc[~deconv.isnull().all(1), :].T

    diffs = pd.read_excel(content, sheet_name="figure 7 a")
    effectsize = diffs.pivot_table(index="gene", columns="tiss", values="est")
    signif = diffs.pivot_table(index="gene", columns="tiss", values="fdr")
    signif = -np.log10(signif)

    prjmeta, samplemeta = series_matrix2csv(SERIES_URL)
    samplemeta["roi_id"] = samplemeta["description_2"].str.replace(".dcc", "")
    samplemeta["case_id"] = samplemeta["characteristics_ch1"].str.replace(
        "case number: ", ""
    )

    roi2case = samplemeta.set_index("roi_id")["case_id"]
    case2roi = samplemeta.set_index("case_id")["roi_id"].sort_index()
    case2roi.index = case2roi.index.str.extract(r"(.* \d+) ?")[0]

    y = pd.concat(
        [
            time_since_symptoms,
            viral_load,
            struc_cell_content,
            lymph_content,
            myelo_content_red,
            deconv_red,
        ],
        axis=1,
    )
    y.index.name = "case_id"
    y.index = y.index.str.replace("Negcontrol", "Control")

    y = roi2case.reset_index().merge(y.reset_index()).set_index("roi_id")
    y = y.reindex(x.columns)

    y["phenotypes"] = pd.Categorical(
        y["phenotypes"], ordered=True, categories=["Control", "Early", "Late"],
    )

    # align indeces of data and metadata
    x = x.reindex(columns=y.index)
    return x, y


def plot_time_and_viral_load(y):
    color = ["red" if o == "High" else "blue" for o in y["Virus High/Low"]]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(y["Duration (days)"], y["Viral load% by RNA ISH"], c=color)
    ax.set(xlabel="Time since symptoms (days)", ylabel="Viral load% by RNA ISH")


def plot_time_and_cell_type_abundance(y, order: int = 1):
    ct_cols = y.columns.where(y.columns.str.contains(" / mm2")).dropna()
    n, m = get_grid_dims(len(ct_cols))
    fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), sharex=True)
    for i, ct in enumerate(ct_cols):
        ax = axes.flatten()[i]
        ax.scatter(y["Duration (days)"], y[ct])
        sns.regplot(y["Duration (days)"], y[ct], ax=ax, order=order)
        ax.set(title=ct, ylabel=None, xlim=(-1, None))
    for ax in axes[:, 0]:
        ax.set_ylabel("Cells per mm2 (IHC)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time since symptoms (days)")

    ct_cols = y.columns.where(y.columns.str.contains(" / %")).dropna()
    n, m = get_grid_dims(len(ct_cols))
    fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), sharex=True)
    for i, ct in enumerate(ct_cols):
        ax = axes.flatten()[i]
        ax.scatter(y["Duration (days)"], y[ct])
        sns.regplot(y["Duration (days)"], y[ct], ax=ax, order=order)
        ax.set(title=ct, ylabel=None, xlim=(-1, None))
    for ax in axes[:, 0]:
        ax.set_ylabel("% abundance (CYBERSORT)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time since symptoms (days)")


def unsupervised():
    grid = sns.clustermap(
        deconv.reindex(y.index).dropna(), row_colors=y["case_id"]
    )


def series_matrix2csv(matrix_url, prefix=None):
    """
    matrix_url: gziped URL with GEO series matrix.
    """
    from collections import Counter
    import pandas as pd
    import gzip
    import subprocess

    subprocess.call("wget {}".format(matrix_url).split(" "))
    filename = matrix_url.split("/")[-1]

    with gzip.open(filename, "rb") as f:
        file_content = f.read()

    # separate lines with only one field (project-related)
    # from lines with >2 fields (sample-related)
    prj_lines = dict()
    sample_lines = dict()
    idx_counts = Counter()
    col_counts = Counter()

    for line in file_content.decode("utf-8").strip().split("\n"):
        line = line.strip().split("\t")
        key = line[0].replace('"', "")
        if len(line) == 2:
            if key in idx_counts:
                key = f"{key}_{idx_counts[key] + 1}"
            idx_counts[key] += 1
            prj_lines[key] = line[1].replace('"', "")
        elif len(line) > 2:
            if key in col_counts:
                key = f"{key}_{col_counts[key] + 1}"
            col_counts[key] += 1
            sample_lines[key] = [x.replace('"', "") for x in line[1:]]

    prj = pd.Series(prj_lines)
    prj.index = prj.index.str.replace("!Series_", "")

    samples = pd.DataFrame(sample_lines)
    samples.columns = samples.columns.str.replace("!Sample_", "")

    if prefix is not None:
        prj.to_csv(os.path.join(prefix + ".project_annotation.csv"), index=True)
        samples.to_csv(
            os.path.join(prefix + ".sample_annotation.csv"), index=False
        )

    return prj, samples


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\t - Exiting due to user interruption.")
        sys.exit(1)
