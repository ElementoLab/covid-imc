#!/usr/bin/env python


"""
This script does univariate gating of cells for each marker.

Also addresses questions that depend on whether a cell is positive for one
specific marker such as SARS-CoV-2.
"""

import sys, itertools, re
from functools import partial

from matplotlib.colors import LogNorm
import parmap
import networkx as nx
import pingouin as pg

from imc.operations import (
    get_population,
    fit_gaussian_mixture,
    get_best_mixture_number,
    get_threshold_from_gaussian_mixture,
)
from imc.graphics import get_grid_dims
from imc.types import Series, DataFrame
from seaborn_extensions import swarmboxenplot

from src.config import *


swarmboxenplot = partial(swarmboxenplot, test_kws=dict(parametric=False))


output_dir = results_dir / "gating"


quantification_file = results_dir / "cell_type" / "quantification.pq"
gating_file = results_dir / "cell_type" / "gating.pq"
positive_file = results_dir / "cell_type" / "gating.positive.pq"
positive_count_file = results_dir / "cell_type" / "gating.positive.count.pq"
ids = ["sample", "roi"]


def main() -> int:

    threshold()

    plot_infection()

    infection_interactions()

    infected_macs()

    return 0


def threshold() -> None:
    # load quantification
    quant = pd.read_parquet(quantification_file)
    quant.loc[:, "DNA"] = quant[["DNA1(Ir191)", "DNA2(Ir193)"]].mean(1)
    quant = pd.concat([np.log1p(quant.drop(ids, axis=1)), quant[ids]], axis=1)

    # remove excluded channels
    quant = quant.drop(channels_exclude, axis=1, errors="ignore")

    # Univariate gating of each channel
    thresholds_file = output_dir / "thresholds.json"
    mixes_file = output_dir / "mixes.json"
    if not (thresholds_file.exists() and thresholds_file.exists()):
        mixes = dict()
        thresholds = dict()
        for m in quant.columns:
            if m not in thresholds:
                mixes[m] = get_best_mixture_number(quant[m], 2, 8)
                thresholds[m] = get_threshold_from_gaussian_mixture(
                    quant[m], None, mixes[m]
                ).to_dict()
        json.dump(thresholds, open(thresholds_file, "w"))
        json.dump(mixes, open(mixes_file, "w"))
    thresholds = json.load(open(output_dir / "thresholds.json"))
    mixes = json.load(open(output_dir / "mixes.json"))

    # Make dataframe with population for each marker
    if not gating_file.exists():
        pos = pd.DataFrame(index=quant.index, columns=quant.columns.drop(ids))
        for m in pos.columns:
            name = m.split("(")[0]
            o = sorted(thresholds[m])
            if mixes[m] == 2:
                pos[m] = (quant[m] > thresholds[m][o[0]]).replace(
                    {False: name + "-", True: name + "+"}
                )
            else:
                pos[m] = (quant[m] > thresholds[m][o[-1]]).replace(
                    {True: name + "+"}
                )
                sel = pos[m] == False
                pos.loc[sel, m] = (
                    quant.loc[sel, m] > thresholds[m][o[-2]]
                ).replace({True: name + "dim", False: name + "-"})
        pos = pd.concat([pos, quant[ids]], axis=1)
        pos.to_parquet(gating_file)
    pos = pd.read_parquet(gating_file)
    pos.index.name = "obj_id"

    # Compare with clustering
    set_prj_clusters(aggregated=True)
    posc = pos.reset_index().merge(
        prj.clusters.reset_index(), on=["sample", "roi", "obj_id"]
    )
    posc = posc.query("cluster != ''")

    # # number of positive cells, per ROI, per cell type
    poscc = pd.DataFrame(
        [
            posc.groupby(["roi", "cluster"])[m].apply(
                lambda x: x.str.contains(r"\+|dim").sum()
            )
            for m in quant.columns.drop(ids)
        ]
    ).T

    # # sum by cluster
    posccg = poscc.groupby(level="cluster").sum()
    grid = sns.clustermap(
        posccg / posccg.sum(),
        center=0,
        z_score=0,
        cmap="RdBu_r",
        robust=True,
        xticklabels=True,
        yticklabels=True,
    )
    grid.fig.savefig(
        output_dir / "gating_vs_clustering_comparison.clustermap.svg", **figkws
    )

    # # sum by sample
    s = poscc.index.get_level_values("roi").str.extract(r"(.*)-")[0].values
    posccs = poscc.groupby(by=s).sum()

    grid = sns.clustermap(
        posccs / posccs.sum(),
        center=0,
        z_score=0,
        cmap="RdBu_r",
        robust=True,
        xticklabels=True,
        yticklabels=True,
    )

    # # sum by sample, per cell type
    q = (
        posc.drop(ids + ["obj_id", "cluster"], axis=1)
        .apply(lambda x: x.str.endswith("+"))
        .join(posc[ids + ["obj_id", "cluster"]])
    )
    q.to_parquet(positive_file)

    poscc = q.drop("obj_id", axis=1).groupby(["roi", "cluster"]).sum()
    poscc.to_parquet(positive_count_file)


def plot_infection() -> None:
    poscc = pd.read_parquet(positive_count_file)

    # Let's look at infection
    sample_area = (
        pd.Series(
            [sum([r.area for r in s]) for s in prj], [s.name for s in prj]
        )
        / 1e6  # square microns to square milimeters
    )
    roi_area = (
        pd.Series([r.area for r in prj.rois], [r.name for r in prj.rois])
        / 1e6  # square microns to square milimeters
    )
    roi_cell_count = prj.clusters.groupby(level="roi").size()
    roi_cluster_count = (
        prj.clusters.to_frame()
        .assign(count=1)
        .pivot_table(
            index="roi",
            columns="cluster",
            aggfunc=sum,
            values="count",
            fill_value=0,
        )
    ).drop("", 1)

    _stats = list()
    phenotypes = sample_attributes["phenotypes"].unique()
    for marker, state in [
        ("SARSCoV2S1(Eu153)", "infected"),
        ("C5bC9(Gd155)", "complement"),
        ("CleavedCaspase3(Yb172)", "apoptotic"),
        ("IL6(Gd160)", "inflamatory_IL6"),
        ("IL1beta(Er166)", "inflamatory_IL1beta"),
        ("Ki67(Er168)", "proliferative"),
        ("pCREB(Ho165)", "pCREB"),
        ("cKIT(Nd143)", "cKIT"),
        ("iNOS(Nd142)", "iNOS"),
        ("MPO(Yb173)", "MPO"),
        ("Arginase1(Dy164)", "Arginase1"),
    ]:
        p = (
            poscc[marker]
            .to_frame()
            .pivot_table(
                index="roi", columns="cluster", values=marker, fill_value=0
            )
        )
        p_area = (p.T / roi_area).T
        p_perc = (p / (roi_cluster_count + 1)) * 100

        for df, label, unit in [
            (p_area, "area", "per mm2"),
            (p_perc, "percentage", "(%)"),
        ]:
            # # add jitter so it can be tested
            df += np.random.random(df.shape) * 1e-5

            # Clustermaps
            grid = sns.clustermap(
                df, cbar_kws=dict(label=f"{state} cells {unit}")
            )
            grid.savefig(
                output_dir / f"{state}_cells.per_{label}.clustermap.svg"
            )
            plt.close(grid.fig)
            grid = sns.clustermap(
                df,
                cbar_kws=dict(label=f"{state} cells {unit}"),
                norm=LogNorm(),
            )
            grid.savefig(
                output_dir / f"{state}_cells.per_{label}.clustermap.log.svg"
            )
            plt.close(grid.fig)

            grid = sns.clustermap(
                df.join(roi_attributes[["phenotypes"]])
                .groupby("phenotypes")
                .mean(),
                cbar_kws=dict(label=f"{state} cells {unit}"),
                figsize=(6, 3),
            )
            grid.savefig(
                output_dir
                / f"{state}_cells.per_{label}.clustermap.by_phenotypes.svg"
            )
            plt.close(grid.fig)

            # Swarmboxemplots
            n, m = get_grid_dims(df.shape[1])
            fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4))
            fig.suptitle(f"{state.capitalize()} cells ({marker}+)")
            for ax, ct in zip(axes.flat, df.columns):
                stats = swarmboxenplot(
                    data=df[[ct]].join(roi_attributes["phenotypes"]),
                    x="phenotypes",
                    y=ct,
                    ax=ax,
                    plot_kws=dict(palette=colors["phenotypes"]),
                )
                _stats.append(
                    stats.assign(state=state, kind=label, cell_type=ct)
                )
            fig.savefig(
                output_dir / f"{state}_cells.{label}.swarmboxenplot.svg"
            )
            plt.close(fig)

            # Let's also plot for each phenotype separately, which cell types are most affected
            n, m = get_grid_dims(len(phenotypes))
            fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4))
            fig.suptitle(f"{state.capitalize()} cells ({marker}+)")
            for phenotype in phenotypes:
                v = (
                    df.join(roi_attributes[["phenotypes"]])
                    .query(f"phenotypes == '{phenotype}'")
                    .drop("phenotypes", axis=1)
                )
                vm = v.mean().sort_values(ascending=False)
                vmelt = v.melt(var_name="cell_type", value_name=f"cells {unit}")
                # order
                vmelt["cell_type"] = pd.Categorical(
                    vmelt["cell_type"], categories=vm.index, ordered=True
                )
                vmelt = vmelt.sort_values("cell_type")

                # # for each roi
                fig, stats = swarmboxenplot(
                    data=vmelt, x="cell_type", y=f"cells {unit}"
                )
                fig.axes[0].set(
                    title=f"{state.capitalize()} cells ({marker}+)",
                    xlabel=f"Cells {unit}",
                )
                fig.savefig(
                    output_dir
                    / f"{state}_cells.{label}.cell_types_affected.{phenotype}_only.swarmboxenplot.svg",
                    **figkws,
                )
                plt.close(fig)
                _stats.append(
                    stats.assign(state=state, kind=label, phenotype=phenotype)
                )

                # # reduced by mean
                fig, ax = plt.subplots(1, 1, figsize=(4, 2))
                ax.set(
                    title=f"{state.capitalize()} cells ({marker}+)",
                    xlabel=f"Cells {unit}",
                )
                sns.barplot(vm, vm.index, ax=ax)
                fig.savefig(
                    output_dir
                    / f"{state}_cells.{label}.cell_types_affected.{phenotype}_only.mean.barplot.svg",
                    **figkws,
                )
                plt.close(fig)
    stats = pd.concat(_stats)
    stats.to_csv(
        output_dir / "functional_state_comparison.statistics.csv", index=False
    )
    # Save for supplement
    stats.to_excel("manuscript/Supplementary Table 4.xlsx", index=False)

    # get mean/max values to be reported
    roi_attributes["phenotypes"] = roi_attributes[
        "phenotypes"
    ].cat.add_categories(["COVID19"])
    roi_cell_count = (
        prj.clusters.reset_index().groupby(["roi", "cluster"]).size()
    )
    mean_values = (poscc.T / roi_cell_count).T.join(
        roi_attributes[["phenotypes"]]
    ).groupby(["phenotypes", "cluster"]).mean().dropna() * 100
    mean_covid = (poscc.T / roi_cell_count).T.join(
        roi_attributes[["disease"]]
    ).groupby(["disease", "cluster"]).mean().dropna().loc[["COVID19"]] * 100
    mean_covid.index.names = ["phenotypes", "cluster"]
    mean_covid.index = pd.MultiIndex.from_tuples(mean_covid.index.to_list())
    mean_values = pd.concat([mean_values, mean_covid])
    mean_values.to_csv(
        output_dir / "functional_state_comparison.mean_values.csv"
    )

    max_values = (poscc.T / roi_cell_count).T.join(
        roi_attributes[["phenotypes"]]
    ).groupby(["phenotypes", "cluster"]).max().dropna() * 100
    max_covid = (poscc.T / roi_cell_count).T.join(
        roi_attributes[["disease"]]
    ).groupby(["disease", "cluster"]).max().dropna().loc[["COVID19"]] * 100
    max_covid.index.names = ["phenotypes", "cluster"]
    max_covid.index = pd.MultiIndex.from_tuples(max_covid.index.to_list())
    max_values = pd.concat([max_values, max_covid])
    max_values.to_csv(output_dir / "functional_state_comparison.max_values.csv")

    # #
    # # Calculate fisher p-values for double-expression
    # # # e.g. SARS and C5bC9

    # # quickly get boolean dataframe
    # ts = dict()
    # for ct, t in mixes.items():
    #     if mixes[ct] == 2:
    #         ts[ct] = thresholds[ct][list(thresholds[ct].keys())[-1]]
    #     else:
    #         ts[ct] = thresholds[ct][list(thresholds[ct].keys())[-2]]
    # ts = pd.Series(ts, name="thresholds")
    # p = pd.concat(
    #     [quant[m] > ts[m] for m in ts.index[ts.index.isin(quant.columns)]]
    #     + [quant[x] for x in ids],
    #     1,
    # )
    # p.index.name = "obj_id"
    # # annotate with clusters
    # pc = p.reset_index().merge(
    #     prj.clusters.reset_index(), on=["sample", "roi", "obj_id"]
    # )
    # pc = pc.query("cluster != ''")

    # pc["covid"] = pc["roi"].str.contains("COVID")
    # q = pc.query("covid == True")
    # qe = q.query("cluster == 'Epithelial cells'")
    # markers = [
    #     "SARSCoV2S1(Eu153)",
    #     "C5bC9(Gd155)",
    #     # "area",
    #     "CleavedCaspase3(Yb172)",
    #     "pCREB(Ho165)",
    #     "IL1beta(Er166)",
    #     "pSTAT3(Gd158)",
    #     "IL6(Gd160)",
    # ]

    # # # get numbers of cells co-expressing markers progressively
    # df = qe.copy()
    # res1 = list()
    # for marker in markers:
    #     v1 = df[marker].sum()
    #     v0 = df.shape[0] - v1
    #     res1.append((marker, v1, v0))
    #     df = df.loc[df[marker] == True]
    # df = qe.copy()
    # res2 = list()
    # for i, marker in enumerate(markers):
    #     v1 = df[marker].sum()
    #     v0 = df.shape[0] - v1
    #     res2.append((marker, v1, v0))
    #     df = df.loc[df[marker] == (False if i == 0 else True)]

    # # # plot as kind-of sankei plots
    # fig, axes = plt.subplots(2, 1, figsize=(6, 3 * 2))
    # for marker, v1, v0 in res1:
    #     total = v0 + v1
    #     axes[0].bar(marker, v0 / total, color="grey")
    #     axes[0].bar(marker, v1 / total, bottom=v0 / total, color="green")
    # for marker, v1, v0 in res2:
    #     total = v0 + v1
    #     axes[1].bar(marker, v0 / total, color="grey")
    #     axes[1].bar(marker, v1 / total, bottom=v0 / total, color="blue")
    # fig.savefig(
    #     output_dir
    #     / "infected_cells_coexpression_positivity.serial.COVID.epithelial.svg",
    #     **figkws,
    # )

    # Check cKIT expression in Epithelial cells
    pos = pd.read_parquet(positive_file)
    total = pos.groupby(["roi", "cluster"]).size()
    posc = pd.read_parquet(positive_count_file)

    perc = (posc.T / total).T * 100

    nin = pos.loc[pos["SARSCoV2S1(Eu153)"] == False]
    inf = pos.loc[pos["SARSCoV2S1(Eu153)"] == True]

    posc_nin = nin.drop("obj_id", 1).groupby(["roi", "cluster"]).sum()
    posc_inf = inf.drop("obj_id", 1).groupby(["roi", "cluster"]).sum()

    total_nin = nin.groupby(["roi", "cluster"]).size()
    total_inf = inf.groupby(["roi", "cluster"]).size()

    perc_nin = (posc_nin.T / total_nin).T * 100
    perc_inf = (posc_inf.T / total_inf).T * 100

    fig, axes = plt.subplots(1, 3, figsize=(4 * 3, 4))
    swarmboxenplot(
        data=perc.loc[:, "08 - Epithelial cells", :]
        .join(roi_attributes[["phenotypes"]])
        .reset_index(),
        x="phenotypes",
        y="cKIT(Nd143)",
        ax=axes[0],
    )
    swarmboxenplot(
        data=perc_nin.loc[:, "08 - Epithelial cells", :]
        .join(roi_attributes[["phenotypes"]])
        .reset_index(),
        x="phenotypes",
        y="cKIT(Nd143)",
        ax=axes[1],
    )
    swarmboxenplot(
        data=perc_inf.loc[:, "08 - Epithelial cells", :]
        .join(roi_attributes[["phenotypes"]])
        .reset_index(),
        x="phenotypes",
        y="cKIT(Nd143)",
        ax=axes[2],
    )
    for ax, (lab, n) in zip(
        axes,
        [
            ("All cells", pos.shape[0]),
            ("Uninfected", nin.shape[0]),
            ("Infected", inf.shape[0]),
        ],
    ):
        ax.set(title=f"{lab}, n = {n}", ylabel="% cKIT+ cells")
    fig.savefig("cKIT_positivity.pdf", **figkws)

    #

    #


def infection_interactions() -> None:
    """
    Cell type-cell type interactions dependent on infection of the cells.
    """

    # # measure interactions dependent on state
    pos = pd.read_parquet(positive_file)
    posc = pd.read_parquet(positive_count_file)
    state_vector = pos.set_index(["roi", "obj_id"])["SARSCoV2S1(Eu153)"]
    # # start with infection
    interac_file = output_dir / "interactions.dependent_on_state.infection.csv"
    if not interac_file.exists():
        covid_rois = [r for r in prj.rois if "COVID" in r.name]
        # fracs = list()
        # for roi in covid_rois:
        #     state = (
        #         pos["SARSCoV2S1(Eu153)"]
        #         .query(f'roi == "{roi.name}"')
        #         .str.endswith("+")
        #     )
        #     fracs.append(get_interaction_by_state(roi, state))
        fracs = parmap.map(
            get_interaction_by_state,
            covid_rois,
            state_vector=state_vector,
            pm_pbar=True,
        )

        all_frac = pd.concat(fracs)
        all_frac.to_csv(interac_file)
    all_frac = (
        pd.read_csv(interac_file, index_col=[0, 1])
        .sort_index(0, level=[0, 1])
        .sort_index(1)
    )

    state_counts = (
        prj.clusters.reset_index(level=0, drop=True)
        .sort_index()
        .to_frame()
        .join(state_vector)
        .groupby(["roi", "cluster"])["SARSCoV2S1(Eu153)"]
        .sum()
    )
    state_counts.index = pd.MultiIndex.from_arrays(
        [
            state_counts.index.get_level_values("roi"),
            state_counts.index.get_level_values("cluster").str.extract(
                r"\d+ - (.*)"
            )[0],
        ],
        names=["roi", "cluster"],
    )

    all_frac = all_frac.loc[
        ~all_frac.index.get_level_values(1).str.startswith(" - "),
        ~all_frac.columns.str.startswith(" - "),
    ]

    el = all_frac.index.get_level_values(0).str.contains("EARLY")
    mean_ea = (
        all_frac.loc[el].groupby(level=1).mean().sort_index(0).sort_index(1)
    )
    mean_la = (
        all_frac.loc[~el].groupby(level=1).mean().sort_index(0).sort_index(1)
    )

    mean_ea = mean_ea.loc[
        mean_ea.index.str.endswith("False"),
        mean_ea.columns.str.endswith("True"),
    ]
    mean_la = mean_la.loc[
        mean_la.index.str.endswith("False"),
        mean_la.columns.str.endswith("True"),
    ]
    kws = dict(center=0, cmap="RdBu_r", xticklabels=True, yticklabels=True)
    fig, axes = plt.subplots(2, 2)
    sns.heatmap(mean_ea - mean_ea.values.mean(), ax=axes[0][0], **kws)
    axes[0][0].set_title("COVID19_early")
    sns.heatmap(mean_la - mean_la.values.mean(), ax=axes[0][1], **kws)
    axes[0][1].set_title("COVID19_late")

    fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4 * 1))
    sns.heatmap(
        all_frac.loc[
            all_frac.index.get_level_values(1).str.endswith(" - False"),
            all_frac.columns.str.endswith(" - False"),
        ]
        .groupby(level=1)
        .mean(),
        ax=axes[0],
        square=True,
        vmin=-2,
        vmax=2,
        **kws,
    )
    sns.heatmap(
        all_frac.loc[
            all_frac.index.get_level_values(1).str.endswith(" - True"),
            all_frac.columns.str.endswith(" - True"),
        ]
        .groupby(level=1)
        .mean(),
        ax=axes[1],
        square=True,
        vmin=-2,
        vmax=2,
        **kws,
    )
    fig.savefig(
        output_dir
        / f"infected_cells.interaction.COVID.infected-uninfected.heatmap.svg",
        **figkws,
    )

    kws = dict(
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        cbar_kws=dict(label="Interaction strength"),
    )
    el_all = np.asarray([True] * el.shape[0])
    for label1, sel in [("all", el_all), ("early", el), ("late", ~el)]:
        for label2, (a, b) in [
            ("infected-infected", ("True", "True")),
            ("infected-uninfected", ("True", "False")),
            ("uninfected-uninfected", ("False", "False")),
        ]:
            ss = all_frac.loc[sel].groupby(level=1).mean()
            p = ss.loc[
                ss.index.str.endswith(f" - {a}"),
                ss.columns.str.endswith(f" - {b}"),
            ]

            roi_names = all_frac.loc[sel].index.get_level_values("roi").unique()
            total = prj.clusters.loc[:, roi_names, :].value_counts()
            total.index = total.index.str.extract(r"\d+ - (.*)")[0]
            curstate = (
                state_counts.loc[roi_names].groupby(level="cluster").sum()
                / total
            ) * 100
            if a == "False":
                astate = 100 - curstate
                astate.index = astate.index + " - False"
            else:
                astate = curstate.copy()
                astate.index = astate.index + " - True"
            if b == "False":
                bstate = 100 - curstate
                bstate.index = bstate.index + " - False"
            else:
                bstate = curstate.copy()
                bstate.index = bstate.index + " - True"

            atotal = total.copy()
            atotal.index = atotal.index + f" - {a}"
            btotal = total.copy()
            btotal.index = btotal.index + f" - {b}"

            al, bl = label2.split("-")
            grid = sns.clustermap(
                p.fillna(0),
                mask=p.isnull(),
                row_colors=astate.rename(f"% {al}")
                .to_frame()
                .join(atotal.rename("cells")),
                col_colors=bstate.rename(f"% {bl}")
                .to_frame()
                .join(btotal.rename("cells")),
                row_cluster=False,
                col_cluster=False,
                # norm=LogNorm(),
                **kws,
            )
            grid.fig.suptitle(f"{label1} - {a}; {b}")
            grid.fig.savefig(
                output_dir
                / f"infected_cells.interaction.COVID.{label1}.{label2}.clustermap.ordered.svg",
                **figkws,
            )
            plt.close(grid.fig)

            grid = sns.clustermap(
                p.fillna(0),
                mask=p.isnull(),
                row_colors=astate.rename(f"% {al}")
                .to_frame()
                .join(atotal.rename("cells")),
                col_colors=bstate.rename(f"% {bl}")
                .to_frame()
                .join(btotal.rename("cells")),
                # norm=LogNorm(),
                **kws,
            )
            grid.fig.suptitle(f"{label1} - {a}; {b}")
            grid.fig.savefig(
                output_dir
                / f"infected_cells.interaction.COVID.{label1}.{label2}.clustermap.clustered.svg",
                **figkws,
            )
            plt.close(grid.fig)

    early = all_frac.loc[el].groupby(level=1).mean()["Epithelial cells - True"]
    late = all_frac.loc[~el].groupby(level=1).mean()["Epithelial cells - True"]

    roi_names = all_frac.loc[el].index.get_level_values("roi").unique()
    total = prj.clusters.loc[:, roi_names, :].value_counts()

    df = early.to_frame(name="early").join(late.rename("late"))
    df = df.groupby(df.index.str.extract(r"(.*) - .*")[0].values).mean()
    mean = df.mean(1)
    fc = df["late"] - df["early"]

    fig, axes = plt.subplots(1, 2, figsize=(6 * 2, 4))
    axes[0].scatter(df["early"], df["late"])
    for i, row in df.iterrows():
        axes[0].text(row["early"], row["late"], s=i)
    v = df.abs().max().max()
    axes[0].plot((-v, v), (-v, v), linestyle="--", color="grey")
    axes[0].set(xlabel="Early")
    axes[0].set(ylabel="Late")

    axes[1].scatter(mean, fc)
    axes[1].axhline(0, linestyle="--", color="grey")
    for i, row in df.iterrows():
        axes[1].text(mean.loc[i], fc.loc[i], s=i)

    # Test

    # # tests are done comparing the interaction of uninfected cells from cell type A with cell type/state B
    # # to the interaction of infected cells from cell type A with cell type/state B
    to_test = all_frac.reset_index(level=1).melt(id_vars=["level_1"]).dropna()
    to_test["idx"] = to_test["level_1"].str.extract("(.*) - ")[0]
    to_test["col"] = to_test["variable"].str.extract("(.*) - ")[0]
    to_test["idx_infected"] = to_test["level_1"].str.endswith(" - True")
    to_test["col_infected"] = to_test["variable"].str.endswith(" - True")
    to_test["label"] = to_test["level_1"] + " <-> " + to_test["variable"]
    _res = list()
    for ct1 in to_test["idx"].unique():
        for ct2 in to_test["variable"].unique():
            print(ct1, ct2)
            neg = to_test.query(
                f"(level_1 == '{ct1} - False') & (variable == '{ct2}')"
            )["value"]
            pos = to_test.query(
                f"(level_1 == '{ct1} - True') & (variable == '{ct2}')"
            )["value"]
            if pos.empty or neg.empty:
                continue
            pos_mean = pos.mean()
            neg_mean = neg.mean()
            _res.append(
                pg.mwu(pos, neg, tail="two-sided").assign(
                    ct1=ct1,
                    ct2=ct2,
                    pos_mean=pos_mean,
                    neg_mean=neg_mean,
                    diff=pos_mean - neg_mean,
                )
            )
    res = pd.concat(_res)
    res["p-cor"] = pg.multicomp(res["p-val"].tolist(), method="fdr_bh")[1]
    res["mean"] = res[["pos_mean", "neg_mean"]].mean(1)
    res = res.reset_index(drop=True)
    res.to_csv(
        output_dir / f"infected_cells.interaction.COVID.tests.csv", index=False
    )

    cts = to_test["idx"].unique()

    fig, axes = plt.subplots(
        2, 1 + len(cts), figsize=(3 * (1 + len(cts)), 3 * 2)
    )
    pv = (-np.log10(res["p-cor"])).max()
    pv += pv / 10
    dv = res["diff"].abs().max()
    dv += dv / 10
    mv = (res["mean"].min(), res["mean"].max())
    mv = (mv[0] + mv[0] / 10, mv[1] + mv[1] / 10)

    for ax in axes[0, :]:
        ax.axvline(0, linestyle="--", color="grey")
        ax.set_xlim((-dv, dv))
        ax.set_ylim((0, pv))
    for ax in axes[1, :]:
        ax.axhline(0, linestyle="--", color="grey")
        ax.set_xlim(mv)
        ax.set_ylim((-dv, dv))
    axes[0][0].scatter(res["diff"], -np.log10(res["p-cor"]), c=res["CLES"])
    axes[1][0].scatter(res["mean"], res["diff"], c=-np.log10(res["p-cor"]))
    for i, ct in enumerate(cts):
        axes[0][1 + i].set(title=ct)
        res2 = res.query(f"ct1 == '{ct}'")
        axes[0][1 + i].scatter(
            res2["diff"], -np.log10(res2["p-cor"]), c=res2["CLES"]
        )
        for _, row in res2.sort_values("p-val").head(3).iterrows():
            axes[0][1 + i].text(
                row["diff"], -np.log10(row["p-cor"]), s=row["ct2"]
            )
        axes[1][1 + i].scatter(
            res2["mean"], res2["diff"], c=-np.log10(res2["p-cor"])
        )
    fig.savefig(
        output_dir / f"infected_cells.interaction.COVID.tests.svg",
        transparent=True,
        **{"dpi": 300, "bbox_inches": "tight", "pad_inches": 0},
    )

    for label, ending in [("uninfected", " - False"), ("infected", " - True")]:
        fig, axes = plt.subplots(
            2, 1 + len(cts), figsize=(3 * (1 + len(cts)), 3 * 2)
        )
        res3 = res.loc[res["ct2"].str.endswith(ending)]

        pv = (-np.log10(res3["p-cor"])).max()
        pv += pv / 10
        dv = res3["diff"].abs().max()
        dv += dv / 10
        mv = (res3["mean"].min(), res3["mean"].max())
        mv = (mv[0] + mv[0] / 10, mv[1] + mv[1] / 10)

        for ax in axes[0, :]:
            ax.axvline(0, linestyle="--", color="grey")
            ax.set_xlim((-dv, dv))
            ax.set_ylim((0, pv))
        for ax in axes[1, :]:
            ax.axhline(0, linestyle="--", color="grey")
            ax.set_xlim(mv)
            ax.set_ylim((-dv, dv))
        axes[0][0].scatter(
            res3["diff"], -np.log10(res3["p-cor"]), c=res3["CLES"]
        )
        axes[1][0].scatter(
            res3["mean"], res3["diff"], c=-np.log10(res3["p-cor"])
        )
        for i, ct in enumerate(cts):
            axes[0][1 + i].set(title=ct)
            res2 = res3.query(f"ct1 == '{ct}'")
            axes[0][1 + i].scatter(
                res2["diff"], -np.log10(res2["p-cor"]), c=res2["CLES"]
            )
            for _, row in res2.sort_values("p-val").head(3).iterrows():
                axes[0][1 + i].text(
                    row["diff"], -np.log10(row["p-cor"]), s=row["ct2"]
                )
            axes[1][1 + i].scatter(
                res2["mean"], res2["diff"], c=-np.log10(res2["p-cor"])
            )
        fig.savefig(
            output_dir
            / f"infected_cells.interaction.COVID.tests.with_{label}.svg",
            transparent=True,
            **{"dpi": 300, "bbox_inches": "tight", "pad_inches": 0},
        )


def get_interaction_by_state(
    roi: "ROI",
    state_vector: Series,
    state_name: str = None,
    correct: bool = True,
) -> DataFrame:
    from imc.operations import (
        # correct_interaction_background_pharmacoscopy,
        correct_interaction_background_random,
    )

    def label(x):
        return (
            x.assign(roi=roi.name)
            .set_index("roi", append=True)
            .reorder_levels([1, 0])
        )

    if state_name is None:
        state_name = "state"

    state = state_vector.loc[roi.name].str.endswith("+")

    # Align state with graph nodes
    # # in case cells quanitfied were filtered out or don't have a cluster
    objs = roi.adjacency_graph.nodes.keys()
    state = state.reindex(objs)
    clusters = roi.clusters.reindex(objs)

    name = f"cluster_{state_name}"
    new_clusters = clusters + " - " + state.astype(str)
    nx.set_node_attributes(roi.adjacency_graph, new_clusters, name=name)
    rf, rl = nx.linalg.attrmatrix.attr_matrix(
        roi.adjacency_graph,
        node_attr=name,
    )
    rl = pd.Series(rl, dtype=roi.clusters.dtype)
    freqs = pd.DataFrame(rf, index=rl, columns=rl).sort_index(0).sort_index(1)
    if not correct:
        return label(freqs)
    # freqs_norm = correct_interaction_background_pharmacoscopy(
    #     freqs, new_clusters.value_counts(), new_clusters.shape[0]
    # )
    freqs_norm = correct_interaction_background_random(
        roi, freqs, name, 100, False, ""
    )
    return label(freqs_norm)


def infected_macs() -> None:
    """
    See if there is evidence infected macrophages are infact just phagocyting
    infected epithelial cells but observing the amount of epithelial-specific
    markers in non-infected and infected COVID-19 macrophages.
    """
    q = pd.read_parquet(positive_file).drop("obj_id", axis=1)

    macs = q["cluster"].str.contains("Macrophage")
    infected = q["SARSCoV2S1(Eu153)"] == True

    mpos = q.loc[macs].groupby(["roi"]).sum()
    mtotal = q.loc[macs].groupby(["roi"]).size()

    mperc = (mpos.T / mtotal).T * 100

    mipos = q.loc[macs & infected].groupby(["roi"]).sum()
    mitotal = q.loc[macs & infected].groupby(["roi"]).size()

    miperc = (mipos.T / mitotal).T * 100

    nipos = q.loc[macs & ~infected].groupby(["roi"]).sum()
    nitotal = q.loc[macs & ~infected].groupby(["roi"]).size()

    niperc = (nipos.T / nitotal).T * 100

    # f = "Keratin818(Yb174)"
    # p = (
    #     mperc[f]
    #     .to_frame("non-infected")
    #     .join(miperc[f].rename("infected"))
    #     .loc[lambda x: x.index.str.contains("COVID")]
    #     .dropna()
    # )
    # fig, stats = swarmboxenplot(
    #     data=p.join(roi_attributes), x="phenotypes", y=p.columns
    # )

    p = (
        niperc.assign(group="non-infected")
        .append(miperc.assign(group="infected"))
        .loc[lambda x: x.index.str.contains("COVID")]
        .dropna()
    )
    p["group"] = pd.Categorical(
        p["group"], ordered=True, categories=["non-infected", "infected"]
    )

    markers = [
        "SARSCoV2S1(Eu153)",  # pos control
        "CD31(Eu151)",  # neg control
        "CD206(Nd144)",  # pos control
        "CD163(Sm147)",  # pos control
        "TTF1(Nd145)",  # test
        "Keratin818(Yb174)",  # test
    ]

    fig, stats = swarmboxenplot(
        data=p.join(roi_attributes),
        x="group",
        y=markers,
        test_kws=dict(parametric=False),
    )
    fig.savefig(
        results_dir
        / "gating"
        / "epithelial_markers_in_infected_macrophages.svg",
        **figkws,
    )
    stats.to_csv(
        results_dir
        / "gating"
        / "epithelial_markers_in_infected_macrophages.csv"
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
