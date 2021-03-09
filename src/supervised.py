#!/usr/bin/env python

"""
Investigation of factors conditioning lung pathology.
"""

import sys
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scanpy as sc

from seaborn_extensions import swarmboxenplot

from imc.types import Path

from src.config import (
    prj,
    roi_attributes,
    sample_attributes,
    metadata_dir,
    results_dir,
    figkws,
    colors,
)


output_dir = results_dir / "supervised"
output_dir.mkdir()

swarmboxenplot = partial(swarmboxenplot, test_kws=dict(parametric=False))


def main():
    comorbidity_regression()


def comorbidity_regression() -> None:
    """
    We will fit linear models that have either 1) demographic and clinical covariates,
    2) comorbidity covariates or 3) both demographi and clinical, and comorbidity covariates.

    These will be used to predict three sets of variables:
    a) cell type abundance (each cell type independently)
    b) fibrosis scores (combined score and each component individually)
    c) lacunarity scores (for each of the metrics - should be highly correlated anyway)

    Performance and variance explained by models 1 and 2 will be compared to each other
    and to model 3.
    """
    ann = sc.read(results_dir / "unsupervised" / "cell_type_abundance.h5ad")

    # Get dataframe with cell type composition
    cta = pd.DataFrame(ann.X, index=ann.obs.index)
    # # make statsmodels compatible names and keep track of original variable names
    cta.columns = "CT" + (cta.columns + 1).astype(str)
    ct_names = pd.Series(ann.var.index, index=cta.columns)
    cta_dep_vars = cta.columns.tolist()
    # Add in sample identifiers
    cta = cta.join(roi_attributes)

    # Read in fibrosis or lacunarity data
    fib = pd.read_parquet(
        results_dir
        / "pathology"
        / "fibrosis.extent_and_intensity.quantification.pq"
    )
    fib.index.name = "roi"
    fib["sample"] = fib.index.to_series().str.split("-").apply(lambda x: x[0])

    lacunae_quantification_file = (
        results_dir / "pathology" / "lacunae.quantification_per_image.csv"
    )
    lac = pd.read_csv(lacunae_quantification_file, index_col=0)
    lac_dep_vars = lac.columns.tolist()
    lac = lac.join(roi_attributes)

    # This is what is going to be run
    params = dict(
        cell_type_abundance=dict(dataframe=cta, dep_vars=cta_dep_vars),
        fibrosis=dict(
            dataframe=fib, dep_vars=["intensity", "fraction", "score"]
        ),
        lacunarity=dict(dataframe=lac, dep_vars=lac_dep_vars),
    )

    # add comorbidities
    meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq")
    cont_vars = pd.read_csv(
        results_dir / "unsupervised" / "pca_associations.pvals.csv", index_col=0
    ).index.tolist()
    cat_vars = ["sex", "race", "smoker"]  # , "hospitalization", "intubated"]

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

    # Describe comorbidities
    coms_summary = (
        coms.set_index("sample")
        .join(sample_attributes["phenotypes"])
        .sort_values("phenotypes")
        .drop("phenotypes", 1)
    )

    p1 = coms_summary.sum(0).sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(4, 4))
    c = sns.color_palette()[0]
    sns.barplot(p1, p1.index, orient="horiz", ax=axes[0], color=c)
    axes[0].axvline(1.8, linestyle="--", color="grey")
    axes[0].set(xlabel="Number of patients")
    p2 = coms_summary.sum(1)
    sns.barplot(p2, p2.index, orient="horiz", ax=axes[1], color=c)
    axes[1].axvline(1.8, linestyle="--", color="grey")
    axes[1].set(xlabel="Number of patients")
    fig.savefig(output_dir / "regression.comorbidity_frequency.svg", **figkws)

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(coms_summary[p1.index], cmap="binary", vmin=-0.2, square=True)
    fig.savefig(output_dir / "regression.comorbidities.heatmap.svg", **figkws)

    x = p2.to_frame("conditions").join(sample_attributes["phenotypes"])
    fig, stats = swarmboxenplot(
        data=x,
        x="phenotypes",
        y="conditions",
        plot_kws=dict(palette=colors["phenotypes"]),
    )
    fig.axes[0].set_ylabel("Conditions per patient")
    fig.savefig(
        output_dir
        / "regression.comorbidities.number_per_patient.swarmboxenplot.svg",
        **figkws,
    )

    # Now run regression for the various datasets
    for param, vals in params.items():
        df = vals["dataframe"]
        dep_vars = vals["dep_vars"]

        # add commorbidities to anndata
        df = df.reset_index().merge(coms, on="sample").set_index("roi")

        # add other clinical vars
        df = (
            df.reset_index()
            .merge(
                meta[["sample_name"] + cat_vars + cont_vars],
                left_on="sample",
                right_on="sample_name",
            )
            .set_index("roi")
        )

        # Generate X and Y matrices
        X = df[dep_vars]
        # # make statsmodels compatible names and keep track of original variable names

        to_use = coms.columns[:-1][coms.drop("sample", 1).sum() > 1].tolist()
        Y = df[cont_vars + cat_vars + ["phenotypes"] + to_use]  # .astype(float)

        # Fix data types for statsmodels compatibility
        ints = [c for c in Y.columns if Y[c].dtype.name == "Int64"]
        Y = Y.drop(ints, 1).join(Y[ints].astype(float))

        bools = [c for c in Y.columns if Y[c].dtype.name == "boolean"]
        Y = Y.drop(bools, 1).join(Y[bools].astype(float))

        dat = X.join(Y)
        com_str = " + ".join(to_use)
        dem_str = "sex + race + smoker + age + phenotypes + lung_weight_grams"  #  + hospitalization + intubated
        output_prefix = (
            output_dir
            / f"regression.{param}.demographics-comorbidity_comparison"
        )
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
        for var in dep_vars:
            com_model = smf.ols(f"{var} ~ {com_str}", data=dat).fit()
            dem_model = smf.ols(f"{var} ~ {dem_str}", data=dat).fit()
            bot_model = smf.ols(
                f"{var} ~ {com_str} + {dem_str}", data=dat
            ).fit()

            r = pd.DataFrame(
                {
                    "dem": [getattr(dem_model, a) for a in attributes],
                    "com": [getattr(com_model, a) for a in attributes],
                    "bot": [getattr(bot_model, a) for a in attributes],
                },
                index=attributes,
            ).T.assign(var=var)
            _res.append(r)
            _coefs[var] = bot_model.params / bot_model.bse
        res = pd.concat(_res).set_index("var", append=True)
        res.index.names = ["model", "var"]
        res.to_csv(output_prefix + ".results.csv")
        res["f_pvalue"] = -np.log10(res["f_pvalue"])

        if param == "cell_type_abundance":
            res.index = pd.MultiIndex.from_arrays(
                [
                    res.index.get_level_values(0),
                    res.index.get_level_values(1).to_series().replace(ct_names),
                ],
                names=["model", "var"],
            )
            dep_vars = ct_names.tolist()

        ncols = len(attributes)
        fig, axes = plt.subplots(
            3, ncols, figsize=(ncols * 5, 3 * 5), sharex="col", sharey="col"
        )
        for axs, metric in zip(axes.T, attributes):
            axs[0].set_title(metric)
            vmin, vmax = res[metric].min(), res[metric].max()
            vmin += vmin * 0.1
            vmax += vmax * 0.1
            for ax, (x, y) in zip(
                axs, [("dem", "com"), ("dem", "bot"), ("com", "bot")]
            ):
                # ax.set(xlim=(vmin, vmax), ylim=(vmin, vmax), xlabel=x, ylabel=y)
                ax.set(xlabel=x, ylabel=y)
                ax.plot(
                    (vmin, vmax), (vmin, vmax), linestyle="--", color="grey"
                )
                ax.scatter(
                    res.loc[x, metric], res.loc[y, metric], rasterized=True
                )
                for var in dep_vars:
                    ax.text(
                        res.loc[(x, var), metric],
                        res.loc[(y, var), metric],
                        s=var,
                    )
        fig.savefig(
            output_prefix + ".scatterplot.svg",
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
            output_prefix + ".distplot.svg",
            **figkws,
        )

        melt_res = res.reset_index().melt(id_vars=["model", "var"])
        grid = sns.catplot(
            data=melt_res,
            col="variable",
            y="value",
            x="model",
            sharey=False,
            kind="bar",
            aspect=1,
            height=3,
        )
        grid.fig.savefig(
            output_prefix + ".barplot.svg",
            **figkws,
        )

        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 1 * 5))
        for attr, ax in zip(attributes, axes):
            stats = swarmboxenplot(
                data=res.reset_index(), x="model", y=attr, ax=ax
            )
        fig.savefig(
            output_prefix + ".swarmboxenplot.svg",
            **figkws,
        )

        # To get mean values
        res.groupby(level=0).mean()

        # To get a sense for which covariates are most important across cell types
        pd.DataFrame(_coefs).abs().sum(1).sort_values()

    attrs = [
        "rsquared",
        "aic",
        "bic",
        "mse_resid",
        "llf",
    ]
    order = ["com", "dem", "bot"]
    cols = ["#7034c784", "#cf5b8b70", "#80640070"]
    opts = [
        ("fibrosis", "score"),
        ("lacunarity", "lacunae_area"),
        ("cell_type_abundance", "mean"),
    ]
    fig, axes = plt.subplots(3, len(attrs), figsize=(len(attrs) * 3, 3 * 3))
    for axs, var in zip(axes.T, attrs):
        for ax, (param, key_metric) in zip(
            axs,
            opts,
        ):
            output_prefix = (
                output_dir
                / f"regression.{param}.demographics-comorbidity_comparison"
            )
            res = pd.read_csv(output_prefix + ".results.csv", index_col=[0, 1])
            if param == "cell_type_abundance":
                p = res.reset_index(level=1, drop=True)[var].loc[order]
            else:
                p = res.loc[:, key_metric, :][var].loc[order]
            sns.barplot(p.index, p, ax=ax, palette=cols, alpha=0.6)
            for i, t in enumerate(p.index.unique()):
                v = p.loc[[t]].mean()
                ax.text(i, v, s=f"{v:.3f}", ha="center")
            ax.set(xticklabels=[], xlabel="", ylabel="")
    for ax, attr in zip(axes[0, :], attrs):
        ax.set_title(attr)
    for ax, attr in zip(axes[:, 0], opts):
        ax.set_ylabel(attr[0])
    fig.savefig(output_prefix + f".key_metric.barplot.svg", **figkws)


def treatment_regression() -> None:
    """
    We will fit linear models that have either 1) demographic and clinical covariates,
    2) comorbidity covariates or 3) both demographi and clinical, and comorbidity covariates.

    These will be used to predict three sets of variables:
    a) cell type abundance (each cell type independently)
    b) fibrosis scores (combined score and each component individually)
    c) lacunarity scores (for each of the metrics - should be highly correlated anyway)

    Performance and variance explained by models 1 and 2 will be compared to each other
    and to model 3.
    """
    ann = sc.read(results_dir / "unsupervised" / "cell_type_abundance.h5ad")

    # Get dataframe with cell type composition
    cta = pd.DataFrame(ann.X, index=ann.obs.index)
    # # make statsmodels compatible names and keep track of original variable names
    cta.columns = "CT" + (cta.columns + 1).astype(str)
    ct_names = pd.Series(ann.var.index, index=cta.columns)
    cta_dep_vars = cta.columns.tolist()
    # Add in sample identifiers
    cta = cta.join(roi_attributes)

    # Read in fibrosis or lacunarity data
    fib = pd.read_parquet(
        results_dir
        / "pathology"
        / "fibrosis.extent_and_intensity.quantification.pq"
    )
    fib.index.name = "roi"
    fib["sample"] = fib.index.to_series().str.split("-").apply(lambda x: x[0])

    lacunae_quantification_file = (
        results_dir / "pathology" / "lacunae.quantification_per_image.csv"
    )
    lac = pd.read_csv(lacunae_quantification_file, index_col=0)
    lac_dep_vars = lac.columns.tolist()
    lac = lac.join(roi_attributes)

    # This is what is going to be run
    params = dict(
        cell_type_abundance=dict(dataframe=cta, dep_vars=cta_dep_vars),
        fibrosis=dict(
            dataframe=fib, dep_vars=["intensity", "fraction", "score"]
        ),
        lacunarity=dict(dataframe=lac, dep_vars=lac_dep_vars),
    )

    # add treatments
    meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq")
    meta = meta.loc[lambda x: x["sample_name"].str.contains("COVID")]
    meta.columns = meta.columns.str.replace(":", "_")
    cont_vars = ["age", "lung_weight_grams"]
    cat_vars = ["gender", "race", "smoker"]  # , "hospitalization", "intubated"]

    for var in ["treatments_", "cause_of_death_"]:
        # get one-hot encode treatments
        coms = meta.loc[
            :,
            (meta.columns.str.startswith(var))
            & (~meta.columns.str.endswith("_text")),
        ].join(meta["sample_name"])

        # Describe treatments
        coms_summary = (
            coms.set_index("sample_name")
            .join(sample_attributes["phenotypes"])
            .sort_values("phenotypes")
            .drop("phenotypes", 1)
        )

        p1 = coms_summary.sum(0).sort_values(ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(4, 4))
        c = sns.color_palette()[0]
        sns.barplot(p1, p1.index, orient="horiz", ax=axes[0], color=c)
        axes[0].axvline(1.8, linestyle="--", color="grey")
        axes[0].set(xlabel="Number of patients")
        p2 = coms_summary.sum(1)
        sns.barplot(p2, p2.index, orient="horiz", ax=axes[1], color=c)
        axes[1].axvline(1.8, linestyle="--", color="grey")
        axes[1].set(xlabel=f"Number of {var}")
        fig.savefig(output_dir / f"regression.{var}.svg", **figkws)

        fig, ax = plt.subplots(1, 1)
        sns.heatmap(
            coms_summary.loc[p2.sort_values(ascending=False).index, p1.index],
            cmap="binary",
            vmin=-0.2,
            square=True,
        )
        fig.savefig(output_dir / f"regression.{var}.heatmap.svg", **figkws)

        x = p2.to_frame("conditions").join(sample_attributes["phenotypes"])
        fig, stats = swarmboxenplot(
            data=x,
            x="phenotypes",
            y="conditions",
            plot_kws=dict(palette=colors["phenotypes"]),
        )
        fig.axes[0].set_ylabel("Conditions per patient")
        fig.savefig(
            output_dir
            / f"regression.{var}.number_per_patient.swarmboxenplot.svg",
            **figkws,
        )

    # get one-hot encoded treatments
    coms = meta.loc[
        :,
        (
            meta.columns.str.startswith("treatments_")
            | meta.columns.str.startswith("cause_of_death_")
        )
        & (~meta.columns.str.endswith("_text")),
    ].join(meta["sample_name"])

    # Now run regression for the various datasets
    for param, vals in params.items():
        df = vals["dataframe"]
        dep_vars = vals["dep_vars"]

        # add commorbidities to anndata
        df = (
            df.reset_index()
            .merge(coms, left_on="sample", right_on="sample_name")
            .set_index("roi")
        )

        # add other clinical vars
        df = (
            df[dep_vars + ["sample"]]
            .reset_index()
            .merge(
                meta,  # [["sample_name"] + cat_vars + cont_vars],
                left_on="sample",
                right_on="sample_name",
            )
            .set_index("roi")
        )

        # Generate X and Y matrices
        X = df[dep_vars]
        # # make statsmodels compatible names and keep track of original variable names

        to_use = coms.columns[:-1][
            coms.drop("sample_name", 1).sum() > 1
        ].tolist()
        Y = df[cont_vars + cat_vars + ["phenotypes"] + to_use]  # .astype(float)

        # Fix data types for statsmodels compatibility
        ints = [c for c in Y.columns if Y[c].dtype.name == "Int64"]
        Y = Y.drop(ints, 1).join(Y[ints].astype(float))

        bools = [c for c in Y.columns if Y[c].dtype.name == "boolean"]
        Y = Y.drop(bools, 1).join(Y[bools].astype(float))

        dat = X.join(Y)
        com_str = " + ".join(to_use)
        dem_str = "gender + race + smoker + age + phenotypes + lung_weight_grams"  #  + hospitalization + intubated
        output_prefix = (
            output_dir
            / f"regression.{param}.demographics-treatments_comparison"
        )
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
        for var in dep_vars:
            com_model = smf.ols(f"{var} ~ {com_str}", data=dat).fit()
            dem_model = smf.ols(f"{var} ~ {dem_str}", data=dat).fit()
            bot_model = smf.ols(
                f"{var} ~ {com_str} + {dem_str}", data=dat
            ).fit()

            r = pd.DataFrame(
                {
                    "dem": [getattr(dem_model, a) for a in attributes],
                    "com": [getattr(com_model, a) for a in attributes],
                    "bot": [getattr(bot_model, a) for a in attributes],
                },
                index=attributes,
            ).T.assign(var=var)
            _res.append(r)
            _coefs[var] = bot_model.params / bot_model.bse
        res = pd.concat(_res).set_index("var", append=True)
        res.index.names = ["model", "var"]
        res.to_csv(output_prefix + ".results.csv")
        res["f_pvalue"] = -np.log10(res["f_pvalue"])

        if param == "cell_type_abundance":
            res.index = pd.MultiIndex.from_arrays(
                [
                    res.index.get_level_values(0),
                    res.index.get_level_values(1).to_series().replace(ct_names),
                ],
                names=["model", "var"],
            )
            dep_vars = ct_names.tolist()

        ncols = len(attributes)
        fig, axes = plt.subplots(
            3, ncols, figsize=(ncols * 5, 3 * 5), sharex="col", sharey="col"
        )
        for axs, metric in zip(axes.T, attributes):
            axs[0].set_title(metric)
            vmin, vmax = res[metric].min(), res[metric].max()
            vmin += vmin * 0.1
            vmax += vmax * 0.1
            for ax, (x, y) in zip(
                axs, [("dem", "com"), ("dem", "bot"), ("com", "bot")]
            ):
                # ax.set(xlim=(vmin, vmax), ylim=(vmin, vmax), xlabel=x, ylabel=y)
                ax.set(xlabel=x, ylabel=y)
                ax.plot(
                    (vmin, vmax), (vmin, vmax), linestyle="--", color="grey"
                )
                ax.scatter(
                    res.loc[x, metric], res.loc[y, metric], rasterized=True
                )
                for var in dep_vars:
                    ax.text(
                        res.loc[(x, var), metric],
                        res.loc[(y, var), metric],
                        s=var,
                    )
        fig.savefig(
            output_prefix + ".scatterplot.svg",
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
            output_prefix + ".distplot.svg",
            **figkws,
        )

        melt_res = res.reset_index().melt(id_vars=["model", "var"])
        grid = sns.catplot(
            data=melt_res,
            col="variable",
            y="value",
            x="model",
            sharey=False,
            kind="bar",
            aspect=1,
            height=3,
        )
        grid.fig.savefig(
            output_prefix + ".barplot.svg",
            **figkws,
        )

        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 1 * 5))
        for attr, ax in zip(attributes, axes):
            stats = swarmboxenplot(
                data=res.reset_index(), x="model", y=attr, ax=ax
            )
        fig.savefig(
            output_prefix + ".swarmboxenplot.svg",
            **figkws,
        )

        # To get mean values
        res.groupby(level=0).mean()

        # To get a sense for which covariates are most important across cell types
        pd.DataFrame(_coefs).abs().sum(1).sort_values()

    attrs = [
        "rsquared",
        "aic",
        "bic",
        "mse_resid",
        "llf",
    ]
    order = ["com", "dem", "bot"]
    cols = ["#7034c784", "#cf5b8b70", "#80640070"]
    opts = [
        ("fibrosis", "score"),
        ("lacunarity", "lacunae_area"),
        ("cell_type_abundance", "mean"),
    ]
    fig, axes = plt.subplots(3, len(attrs), figsize=(len(attrs) * 3, 3 * 3))
    for axs, var in zip(axes.T, attrs):
        for ax, (param, key_metric) in zip(
            axs,
            opts,
        ):
            output_prefix = (
                output_dir
                / f"regression.{param}.demographics-treatments_comparison"
            )
            res = pd.read_csv(output_prefix + ".results.csv", index_col=[0, 1])
            if param == "cell_type_abundance":
                p = res.reset_index(level=1, drop=True)[var].loc[order]
            else:
                p = res.loc[:, key_metric, :][var].loc[order]
            sns.barplot(p.index, p, ax=ax, palette=cols, alpha=0.6)
            for i, t in enumerate(p.index.unique()):
                v = p.loc[[t]].mean()
                ax.text(i, v, s=f"{v:.3f}", ha="center")
            ax.set(xticklabels=[], xlabel="", ylabel="")
    for ax, attr in zip(axes[0, :], attrs):
        ax.set_title(attr)
    for ax, attr in zip(axes[:, 0], opts):
        ax.set_ylabel(attr[0])
    fig.savefig(output_prefix + f".key_metric.barplot.svg", **figkws)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
