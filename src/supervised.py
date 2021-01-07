#!/usr/bin/env python

"""
Investigation of factors conditioning lung pathology.
"""

import sys

from typing import Union

import parmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndi
import skimage as ski
import skimage.feature
from skimage.exposure import equalize_hist as eq
import tifffile
import pingouin as pg
import numpy_groupies as npg
from anndata import AnnData
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scanpy as sc

from seaborn_extensions import swarmboxenplot

from imc.types import Path, Array

from src.config import (
    prj,
    roi_attributes,
    metadata_dir,
    results_dir,
    figkws,
)


output_dir = results_dir / "supervised"
output_dir.mkdir()


def main():
    comorbidity_regression()


def comorbidity_regression() -> None:
    ann = sc.read(output_dir / "cell_type_abundance.h5ad")

    # add comorbidities
    meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq")
    cont_vars = pd.read_csv(
        output_dir / "pvals.csv", index_col=0
    ).index.tolist()

    # one-hot encode comorbidities
    coms = (
        meta["comorbidities_text"]
        .str.replace(" Type 2", "")
        .str.split(", ")
        .apply(pd.Series)
        .stack()
        .reset_index(level=1, drop=True)
        .rename("comorbidities")
        .rename_axis("sample")
        .to_frame()
        .assign(count=1)
        .pivot_table(
            index="sample",
            columns="comorbidities",
            values="count",
        )
        .reindex(meta.index)
        .fillna(0)
        # .astype(int)
        .assign(sample=meta["sample_name"])
    )
    coms.columns = [x.replace(" ", "_") for x in coms.columns]

    coms_summary = coms.sum(0)[:-1].sort_values(ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    c = sns.color_palette()[0]
    sns.barplot(
        coms_summary, coms_summary.index, orient="horiz", ax=ax, color=c
    )
    ax.axvline(1.8, linestyle="--", color="grey")
    ax.set(xlabel="Number of patients")
    fig.savefig(output_dir / "regression.comorbidity_frequency.svg", **figkws)

    # add commorbidities to anndata
    ann.obs = ann.obs.reset_index().merge(coms, on="sample").set_index("roi")

    # add other clinical vars
    # # categorical
    cats = ["sex", "race", "smoker", "hospitalization", "intubated"]
    # # continuous
    ann.obs = (
        ann.obs.reset_index()
        .merge(
            meta[["sample_name"] + cats + cont_vars],
            left_on="sample",
            right_on="sample_name",
        )
        .set_index("roi")
    )

    # Generate X and Y matrices
    X = pd.DataFrame(ann.X, index=ann.obs.index)
    # # make statsmodels compatible names and keep track of original variable names
    X.columns = "CT" + (X.columns + 1).astype(str)
    ct_names = pd.Series(ann.var.index, index=X.columns)

    to_use = coms.columns[:-1][coms.drop("sample", 1).sum() > 1].tolist()
    Y = ann.obs[cont_vars + cats + ["phenotypes"] + to_use]  # .astype(float)

    # Fix data types for statsmodels compatibility
    ints = [c for c in Y.columns if Y[c].dtype.name == "Int64"]
    Y = Y.drop(ints, 1).join(Y[ints].astype(float))

    bools = [c for c in Y.columns if Y[c].dtype.name == "boolean"]
    Y = Y.drop(bools, 1).join(Y[bools].astype(float))

    dat = X.join(Y)
    com_str = " + ".join(to_use)
    dem_str = "sex + race + smoker + age + phenotypes + lung_weight_grams"  #  + hospitalization + intubated
    attributes = [
        "fvalue",
        "f_pvalue",
        "rsquared",
        "rsquared_adj",
        "aic",
        "bic",
        "llf",
        "mse_model",
        "mse_resid",
    ]
    _res = list()
    _coefs = dict()
    for ct in ct_names.index:
        com_model = smf.ols(f"{ct} ~ {com_str}", data=dat).fit()
        dem_model = smf.ols(f"{ct} ~ {dem_str}", data=dat).fit()
        bot_model = smf.ols(f"{ct} ~ {com_str} + {dem_str}", data=dat).fit()

        r = pd.DataFrame(
            {
                "dem": [getattr(dem_model, a) for a in attributes],
                "com": [getattr(com_model, a) for a in attributes],
                "bot": [getattr(bot_model, a) for a in attributes],
            },
            index=attributes,
        ).T.assign(ct=ct_names.loc[ct])
        _res.append(r)
        _coefs[ct] = bot_model.params / bot_model.bse
    res = pd.concat(_res).set_index("ct", append=True)
    res.index.names = ["model", "ct"]
    res.to_csv(
        output_dir
        / "regression.demographics-comorbidity_comparison.results.csv"
    )
    res["f_pvalue"] = -np.log10(res["f_pvalue"])

    ncols = len(attributes)
    fig, axes = plt.subplots(
        3, ncols, figsize=(ncols * 5, 3 * 5), sharex="col", sharey="col"
    )
    for axs, var in zip(axes.T, attributes):
        axs[0].set_title(var)
        vmin, vmax = res[var].min(), res[var].max()
        vmin += vmin * 0.1
        vmax += vmax * 0.1
        for ax, (x, y) in zip(
            axs, [("dem", "com"), ("dem", "bot"), ("com", "bot")]
        ):
            ax.set(xlim=(vmin, vmax), ylim=(vmin, vmax), xlabel=x, ylabel=y)
            ax.plot((vmin, vmax), (vmin, vmax), linestyle="--", color="grey")
            ax.scatter(res.loc[x, var], res.loc[y, var], rasterized=True)
            for ct in ct_names:
                ax.text(res.loc[(x, ct), var], res.loc[(y, ct), var], s=ct)
    fig.savefig(
        output_dir
        / "regression.demographics-comorbidity_comparison.scatterplot.svg",
        **figkws,
    )

    ncols = len(attributes)
    fig, axes = plt.subplots(
        1, ncols, figsize=(ncols * 5, 1 * 5), sharex="col", sharey="col"
    )
    for ax, var in zip(axes.T, attributes):
        ax.set_title(var)
        for x in res.index.levels[0]:
            sns.distplot(res.loc[x, var], label=x, ax=ax)
        ax.legend()
    fig.savefig(
        output_dir
        / "regression.demographics-comorbidity_comparison.distplot.svg",
        **figkws,
    )

    melt_res = res.reset_index().melt(id_vars=["model", "ct"])
    grid = sns.catplot(
        data=melt_res,
        col="variable",
        y="value",
        x="model",
        sharey=False,
        kind="bar",
    )
    grid.fig.savefig(
        output_dir
        / "regression.demographics-comorbidity_comparison.barplot.svg",
        **figkws,
    )

    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 1 * 5))
    for attr, ax in zip(attributes, axes):
        stats = swarmboxenplot(data=res.reset_index(), x="model", y=attr, ax=ax)
    fig.savefig(
        output_dir
        / "regression.demographics-comorbidity_comparison.swarmboxenplot.svg",
        **figkws,
    )

    # To get mean values
    res.groupby(level=0).mean()

    # To get a sense for which covariates are most important across cell types
    pd.DataFrame(_coefs).abs().sum(1).sort_values()


def old():
    """
    Exploratory and never included in the paper
    """

    prj.sample_comparisons(
        sample_attributes=roi_attributes.columns[:2].tolist(),
        output_prefix=output_dir / "comparisons.",
        channel_exclude=channels_exclude,
    )

    df = prj.clusters.reset_index()
    c = df["cluster"].value_counts()
    c = c[c > 500].index.tolist()

    df = df.loc[df["cluster"].isin(c)]
    df = df.loc[~df["cluster"].isin(["?", "<EMPTY>"])]

    df = df.merge(roi_attributes, left_on="roi", right_index=True)

    perc = (
        df.groupby("roi")
        .apply(lambda x: (x["cluster"].value_counts() / x.shape[0]) * 100)
        .rename("percentage")
    )
    perc = roi_attributes.join(perc)
    perc.index.names = ["roi", "cluster"]

    # grid = sns.catplot(
    #     data=perc.reset_index(),
    #     x="cluster",
    #     y="percentage",
    #     hue=attr,
    #     kind="boxen",
    # )

    _test_res = list()
    for attr in roi_attributes.columns:
        # Test for differences
        aov = pd.concat(
            [
                pg.anova(
                    data=perc.loc[perc.index.get_level_values(1) == val],
                    dv="percentage",
                    between=attr,
                ).assign(variable=val)
                for val in perc.index.levels[1]
            ]
        ).set_index("variable")
        _test_res.append(aov)

        kws = dict(
            data=perc.reset_index(),
            x=attr,
            y="percentage",
            palette="tab10",
        )
        grid = sns.FacetGrid(
            data=perc.reset_index(),
            col="cluster",
            height=3,
            col_wrap=4,
            sharey=False,
        )
        grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=False, **kws)
        for ax in grid.axes.flat:
            [
                x.set_alpha(0.25)
                for x in ax.get_children()
                if isinstance(
                    x,
                    (
                        matplotlib.collections.PatchCollection,
                        matplotlib.collections.PathCollection,
                    ),
                )
            ]
        grid.map_dataframe(sns.swarmplot, **kws)
        for ax in grid.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # for ax in grid.axes.flat:
        #     if ax.get_title().endswith("_number"):
        #         ax.set_yscale("log")
        for ax in grid.axes.flat:
            var = ax.get_title().replace("cluster = ", "")
            f = aov.loc[var, "F"]
            p = aov.loc[var, "p-unc"]
            stats = f"\nF = {f:.3f}; p = {p:.3e}"
            ax.set_title(var + stats)

        grid.savefig(
            output_dir / f"differential_cell_types.{attr}.boxen_swarm_plot.svg",
            **figkws,
        )
        plt.close(grid.fig)

    test_res = pd.concat(_test_res)
    test_res.to_csv(
        output_dir / "differential_cell_types.anova_test_results.csv"
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
