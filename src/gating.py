#!/usr/bin/env python

import itertools
import re
from functools import partial

from matplotlib.colors import LogNorm
import parmap
import networkx as nx

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
        roi.adjacency_graph, node_attr=name,
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


output_dir = results_dir / "gating"


quantification_file = results_dir / "cell_type" / "quantification.pq"
gating_file = results_dir / "cell_type" / "gating.pq"
positive_file = results_dir / "cell_type" / "gating.positive.pq"
positive_count_file = results_dir / "cell_type" / "gating.positive.count.pq"
ids = ["sample", "roi"]

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

# Let's look at infection
sample_area = (
    pd.Series([sum([r.area for r in s]) for s in prj], [s.name for s in prj])
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
        grid = sns.clustermap(df, cbar_kws=dict(label=f"{state} cells {unit}"))
        grid.savefig(output_dir / f"{state}_cells.per_{label}.clustermap.svg")
        plt.close(grid.fig)
        grid = sns.clustermap(
            df, cbar_kws=dict(label=f"{state} cells {unit}"), norm=LogNorm(),
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
            _stats.append(stats.assign(state=state, kind=label, cell_type=ct))
        fig.savefig(output_dir / f"{state}_cells.{label}.swarmboxenplot.svg")
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


# get mean/max values to be reported
roi_attributes["phenotypes"] = roi_attributes["phenotypes"].cat.add_categories(
    ["COVID19"]
)
roi_cell_count = prj.clusters.reset_index().groupby(["roi", "cluster"]).size()
mean_values = (poscc.T / roi_cell_count).T.join(
    roi_attributes[["phenotypes"]]
).groupby(["phenotypes", "cluster"]).mean().dropna() * 100
mean_covid = (poscc.T / roi_cell_count).T.join(
    roi_attributes[["disease"]]
).groupby(["disease", "cluster"]).mean().dropna().loc[["COVID19"]] * 100
mean_covid.index.names = ["phenotypes", "cluster"]
mean_covid.index = pd.MultiIndex.from_tuples(mean_covid.index.to_list())
mean_values = pd.concat([mean_values, mean_covid])
mean_values.to_csv(output_dir / "functional_state_comparison.mean_values.csv")

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


#


#

# # measure interactions dependent on state
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

    state_vector = pos.set_index("roi", append=True).reorder_levels([1, 0])[
        "SARSCoV2S1(Eu153)"
    ]
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
    .to_frame()
    .join(state_vector.str.endswith("+"))
    .groupby(["roi", "cluster"])["SARSCoV2S1(Eu153)"]
    .sum()
)

all_frac = all_frac.loc[
    ~all_frac.index.get_level_values(1).str.startswith(" - "),
    ~all_frac.columns.str.startswith(" - "),
]

el = all_frac.index.get_level_values(0).str.contains("EARLY")
mean_ea = all_frac.loc[el].groupby(level=1).mean().sort_index(0).sort_index(1)
mean_la = all_frac.loc[~el].groupby(level=1).mean().sort_index(0).sort_index(1)

mean_ea = mean_ea.loc[
    mean_ea.index.str.endswith("False"), mean_ea.columns.str.endswith("True")
]
mean_la = mean_la.loc[
    mean_la.index.str.endswith("False"), mean_la.columns.str.endswith("True")
]
kws = dict(center=0, cmap="RdBu_r", xticklabels=True, yticklabels=True)
fig, axes = plt.subplots(2, 2)
sns.heatmap(mean_ea - mean_ea.values.mean(), ax=axes[0][0], **kws)
axes[0][0].set_title("COVID19_early")
sns.heatmap(mean_la - mean_la.values.mean(), ax=axes[0][1], **kws)
axes[0][1].set_title("COVID19_late")


fig, axes = plt.subplots(1, 2)
sns.heatmap(
    all_frac.loc[
        all_frac.index.get_level_values(1).str.endswith(" - False"),
        all_frac.columns.str.endswith(" - False"),
    ]
    .groupby(level=1)
    .mean(),
    ax=axes[0],
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
    **kws,
)


kws = dict(
    cmap="RdBu_r", center=-0.5, cbar_kws=dict(label="Interaction strength")
)
for label1, sel in [("early", el), ("late", ~el)]:
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
        curstate = (
            state_counts.loc[roi_names].groupby(level="cluster").sum() / total
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

# # # aggregate by phenotype (early/late)


# #
# from patsy import dmatrices
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# ori = pd.DataFrame(
#     [
#         posc.groupby(["roi", "cluster"])[m].apply(
#             lambda x: x.str.contains(r"\+").sum()
#         )
#         for m in quant.columns.drop(ids)
#     ]
# ).T
# ori["sample"] = pd.Categorical(
#     ori.index.get_level_values("roi").str.extract(r"(.*)-(\d+)")[0].values
# )
# ori.columns = ori.columns.str.extract(r"(.*)\(")[0].fillna(
#     ori.columns.to_series().reset_index(drop=True)
# )
# to_test = ori.join(roi_attributes).reset_index(level=1)
# to_test["phenotypes"] = to_test["phenotypes"].cat.reorder_categories(
#     ["Healthy", "Flu", "ARDS", "COVID19_late", "COVID19_early", "Pneumonia"],
#     ordered=True,
# )

# _coefs = list()
# _pvals = list()
# for cluster in to_test["cluster"].unique():
#     data = to_test.query(f"cluster == '{cluster}'")
#     __coefs = list()
#     __pvals = list()
#     for col in ori.columns[:-1]:
#         if data[col].sum() == 0:
#             continue
#         formula = f"""{col} ~ sample + phenotypes"""
#         response, predictors = dmatrices(formula, data, return_type="dataframe")
#         results = sm.GLM(
#             response, predictors.astype(int), family=sm.families.Poisson()
#         ).fit()
#         __coefs.append(results.params.rename(col))
#         __pvals.append(results.pvalues.rename(col))
#     _coefs.append(pd.DataFrame(__coefs).assign(cluster=cluster))
#     _pvals.append(pd.DataFrame(__pvals).assign(cluster=cluster))
# coefs = pd.concat(_coefs)
# pvals = pd.concat(_pvals)


# for cluster in to_test["cluster"].unique():
#     c = coefs.query(f"cluster == '{cluster}'").drop(
#         ["Intercept", "cluster"], axis=1
#     )
#     grid = sns.clustermap(
#         c.T, xticklabels=True, center=0, cmap="RdBu_r", robust=True
#     )

# #


# #


# #


# #


# #


# #


# #


# # Old stuff:


# # Cell death
# # # CleavedCaspase3, C5bC9

# # Inflamation
# # # IL1B, IL6

# #


# dna = "DNA"
# cd45 = "CD45(Sm152)"
# sp = "SARSCoV2S1(Eu153)"
# ker = "Keratin818(Yb174)"

# # structural
# structural = [
#     "CD31(Eu151)",
#     "Keratin818(Yb174)",
#     "Vimentin(Sm154)",
#     "AlphaSMA(Pr141)",
#     "CollagenTypeI(Tm169)",
#     "TTF1(Nd145)",
# ]

# # Immune
# immune = [
#     "CD45(Sm152)",
#     "MastCellTryptase(Lu175)",
#     "cKIT(Nd143)",
#     "CD11c(Yb176)",
#     "CD15(Dy163)",
#     "CD16(Nd146)",
#     "CD68(Nd150)",
#     "CD206(Nd144)",
#     "CD163(Sm147)",
#     "CD14(Nd148)",
#     "CD11b(Sm149)",
#     "CD56(Tb159)",
#     "CD57(Yb171)",
#     "CD20(Dy161)",
#     "CD3(Er170)",
#     "CD8a(Dy162)",
#     "CD4(Gd156)",
#     "CD123(Er167)",
# ]

# # Functional
# "SARSCoV2S1(Eu153)"
# "C5bC9(Gd155)"
# "iNOS(Nd142)"
# "cKIT(Nd143)"
# "Arginase1(Dy164)"
# "pCREB(Ho165)"
# "IL1beta(Er166)"
# "Ki67(Er168)"
# "CleavedCaspase3(Yb172)"
# "MPO(Yb173)"
# "IL6(Gd160)"
# "pSTAT3(Gd158)"


# dna_n = get_best_mixture_number(quant[dna], 2, 8)
# dna_thresh = get_threshold_from_gaussian_mixture(quant[dna], None, dna_n)

# cd45_n = get_best_mixture_number(quant[cd45], 2, 8)
# cd45_thresh = get_threshold_from_gaussian_mixture(quant[cd45], None, cd45_n)

# sp_n = get_best_mixture_number(quant[sp], 2, 8)
# sp_thresh = get_threshold_from_gaussian_mixture(quant[sp], None, sp_n)
# sp_pos = quant[sp] > sp_thresh.loc[0]

# ker_n = get_best_mixture_number(quant[ker], 2, 8)
# ker_thresh = get_threshold_from_gaussian_mixture(quant[ker], None, ker_n)
# ker_pos = quant[ker] > ker_thresh.loc[0]


# # from sklearn.mixture import GaussianMixture
# # mix = GaussianMixture(2)
# # mix.fit(quant[[dna, sp]])
# # color = mix.predict(quant[[dna, sp]])


# covid = quant["sample"].str.contains("COVID")
# n_quant = quant.loc[~covid, :]
# c_quant = quant.loc[covid, :]

# fig, axes = plt.subplots(
#     2,
#     5,
#     figsize=(4, 2),
#     gridspec_kw=dict(width_ratios=[0.4, 0.2, 0.4, 0.2, 0.4]),
#     sharey=False,
# )
# for axes, df in zip(axes, [n_quant, c_quant]):
#     axes[0].scatter(df[dna], df[sp], s=0.5, alpha=0.01, rasterized=True)
#     for t in dna_thresh:
#         axes[0].axvline(t, linestyle="--", color="grey")
#     for t in sp_thresh:
#         axes[0].axhline(t, linestyle="--", color="grey")
#         axes[1].axhline(t, linestyle="--", color="grey")
#     axes[0].set(
#         xlabel=dna,
#         ylabel=sp,
#         xlim=(0, quant[dna].quantile(0.999)),
#         ylim=(0, quant[sp].quantile(0.999)),
#     )
#     sns.distplot(df[sp], vertical=True, hist=True, ax=axes[1])
#     axes[1].set_yticklabels([])
#     axes[1].set_ylabel("")
#     axes[1].set_ylim(axes[0].get_ylim())
#     # axes[1].set_xscale("log")

#     axes[2].scatter(df[dna], df[ker], s=0.5, alpha=0.01, rasterized=True)
#     for t in dna_thresh:
#         axes[2].axvline(t, linestyle="--", color="grey")
#     for t in ker_thresh:
#         axes[2].axhline(t, linestyle="--", color="grey")
#         axes[3].axhline(t, linestyle="--", color="grey")
#     axes[2].set(
#         xlabel=dna,
#         ylabel=ker,
#         xlim=(0, quant[dna].quantile(0.999)),
#         ylim=(0, quant[ker].quantile(0.999)),
#     )
#     sns.distplot(df[ker], vertical=True, hist=True, ax=axes[3])
#     axes[3].set_yticklabels([])
#     axes[3].set_ylabel("")
#     axes[3].set_ylim(axes[2].get_ylim())
#     # axes[3].set_xscale("log")

#     infected = df.loc[df[sp] > sp_thresh[0], :]

#     axes[4].scatter(
#         infected[ker], infected[sp], s=0.5, alpha=0.01, rasterized=True
#     )
#     for t in ker_thresh:
#         axes[4].axvline(t, linestyle="--", color="grey")
#     for t in sp_thresh:
#         axes[4].axhline(t, linestyle="--", color="grey")
#     axes[4].set(
#         xlabel=ker,
#         ylabel=sp,
#         xlim=(0, quant[ker].quantile(0.999)),
#         ylim=(0, quant[sp].quantile(0.999)),
#     )
# fig.savefig(output_dir / f"{dna}-{ker}-{sp}.scatter_distplot.svg", **figkws)


# # Pairwise for a group of markers
# pos = pd.DataFrame(index=quant.index, columns=quant.columns)
# for m in quant.columns:
#     name = m.split("(")[0]
#     if mixes[m] == 2:
#         pos[m] = (quant[m] > thresholds[m][0]).replace(
#             {False: name + "-", True: name + "+"}
#         )
#     else:
#         pos[m] = (quant[m] > thresholds[m].iloc[-1]).replace({True: name + "+"})
#         sel = pos[m] == False
#         pos.loc[sel, m] = (quant.loc[sel, m] > thresholds[m].iloc[-2]).replace(
#             {True: name + "dim", False: name + "-"}
#         )

# # get all negative cells
# boo = pd.concat([pos[m].str.contains(r"\+|dim") for m in markers], 1)
# quant[cd45].groupby(boo.sum(1).values).mean()


# combs = list(itertools.combinations(structural, 2))
# n = len(combs)
# x, y = get_grid_dims(n)
# fig, axes = plt.subplots(x, y, figsize=(y * 4, x * 4), tight_layout=True)
# for ax, (m1, m2) in zip(axes.flat, combs):
#     ax.scatter(quant[m1], quant[m2], s=0.5, alpha=0.01, rasterized=True)
#     for t in thresholds[m1]:
#         ax.axvline(t, linestyle="--", color="grey")
#     for t in thresholds[m2]:
#         ax.axhline(t, linestyle="--", color="grey")
#     ax.set(
#         xlabel=m1,
#         ylabel=m2,
#         xlim=(0, quant[m1].quantile(0.999)),
#         ylim=(0, quant[m2].quantile(0.999)),
#     )

# fig.savefig(output_dir / "structural_markers.svg", dpi=100)


# #

# #

# #

# # Cellular states for all cells
# sars = "SARSCoV2S1(Eu153)"
# cc3 = "CleavedCaspase3(Yb172)"
# c5bc9 = "C5bC9(Gd155)"
# ki67 = "Ki67(Er168)"
# arg1 = "Arginase1(Dy164)"
# mpo = "MPO(Yb173)"
# inos = "iNOS(Nd142)"
# il6 = "IL6(Gd160)"
# il1b = "IL1beta(Er166)"

# states = [sars, cc3, c5bc9, ki67, arg1, mpo, inos, il6, il1b]
# infection = quant[sars]
# apoptosis = quant[cc3]
# complement = quant[c5bc9]
# inflamatory = quant[il6]


# #

# #

# #

# #

# # Correlate fever with IL1b levels
# meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq").set_index(
#     "sample_name"
# )

# # across all cells
# gx = (
#     quant.loc[quant["sample"].str.contains("COVID")]
#     .groupby("sample")
#     .mean()
#     .join(meta)
# )


# xf = gx.loc[
#     :, gx.columns[gx.dtypes.apply(lambda x: x.name).isin(["float64", "int64"])]
# ]
# xf = xf.loc[:, ~xf.isnull().all()]

# corr = xf.corr(method="spearman")
# corr = corr.loc[corr.index.isin(quant.columns), corr.columns.isin(meta.columns)]
# grid = sns.clustermap(
#     corr,
#     cmap="RdBu_r",
#     center=0,
#     xticklabels=True,
#     yticklabels=True,
#     cbar_kws=dict(label="Spearman correlation"),
# )
# grid.fig.savefig(
#     results_dir
#     / "clinical"
#     / "channel_intensity_in_cells_correlation_with_clinical.clustermap.svg",
#     **figkws,
# )

# factors = corr.columns
# fig, axes = plt.subplots(
#     1 + 6, len(factors), figsize=(len(factors) * 4, 1 + 6 * 4)
# )
# for var, ax in zip(factors, axes.T):
#     # rank vs corr
#     rank = corr[var].rank()
#     ax[0].scatter(rank / rank.max(), corr[var], alpha=0.5)
#     ax[0].set(title=var, ylim=(-1, 1))

#     # plot top/bottom 3
#     s = corr[var].sort_values()
#     for f, a in zip(s.head(3).index, ax[1:]):
#         a.scatter(xf[f], xf[var], alpha=0.5)
#         a.set(title=f)
#     for f, a in zip(s.tail(3).index, ax[4:]):
#         a.scatter(xf[f], xf[var], alpha=0.5)
#         a.set(title=f)
# fig.savefig(
#     results_dir
#     / "clinical"
#     / "channel_intensity_in_cells_correlation_with_clinical.scatter_demo.svg",
#     **figkws,
# )

# # plt.scatter(gx[il6], gx["fever_temperature_celsius"])


# # per cell type
# prefix = "roi_zscored.filtered."
# cluster_str = "cluster_1.0"
# new_labels = json.load(open("metadata/cluster_names.json"))[
#     f"{prefix};{cluster_str}"
# ]
# new_labels = {int(k): v for k, v in new_labels.items()}
# for k in prj.clusters.unique():
#     if k not in new_labels:
#         new_labels[k] = "999 - ?()"
# new_labels_agg = {
#     k: "".join(re.findall(r"\d+ - (.*) \(", v)) for k, v in new_labels.items()
# }
# prj._clusters = None
# prj.set_clusters(
#     prj.clusters.replace(new_labels_agg), write_to_disk=False,
# )

# #
# gx = (
#     (
#         quant.rename_axis("obj_id")
#         .reset_index()
#         .merge(prj.clusters.reset_index())
#         .drop("obj_id", 1)
#         .query("sample.str.contains('COVID').values")
#         .groupby(["sample", "cluster"])
#         .mean()
#         .reset_index(level=1)
#         .join(meta)
#     )
#     .set_index("cluster", append=True)
#     .reorder_levels([1, 0])
# )
# xf = gx.loc[
#     :, gx.columns[gx.dtypes.apply(lambda x: x.name).isin(["float64", "int64"])]
# ]
# xf = xf.loc[:, ~xf.isnull().all()]

# _corrs = list()
# for cell_type in xf.index.levels[0]:
#     corr = xf.loc[cell_type].corr(method="spearman")
#     corr = corr.loc[
#         corr.index.isin(quant.columns), corr.columns.isin(meta.columns)
#     ]
#     corr.index = cell_type + " - " + corr.index
#     _corrs.append(corr)

# corrs = pd.concat(_corrs)

# colors = corrs.index.to_series().str.extract(
#     r"(?P<cell_type>.*) - (?P<channel>.*)"
# )
# grid = sns.clustermap(
#     corrs,
#     cmap="RdBu_r",
#     center=0,
#     xticklabels=True,
#     row_colors=colors["cell_type"],
#     cbar_kws=dict(label="Spearman correlation"),
# )
# grid.fig.savefig(
#     results_dir
#     / "clinical"
#     / "channel_intensity_in_cells_correlation_with_clinical.per_cell_type.clustermap.svg",
#     **figkws,
# )


# factors = corrs.columns
# n = 10
# half = n // 2
# fig, axes = plt.subplots(
#     1 + n, len(factors), figsize=(len(factors) * 4, 1 + n * 4)
# )
# for var, ax in zip(factors, axes.T):
#     # rank vs corrs
#     rank = corrs[var].rank()
#     ax[0].scatter(rank / rank.max(), corrs[var], alpha=0.5)
#     ax[0].set(title=var, ylim=(-1, 1))

#     # plot top/bottom
#     s = corrs[var].sort_values()
#     for f, a in zip(s.head(half).index, ax[1:]):
#         (ct, ch) = f.split(" - ")
#         a.scatter(xf.loc[ct, ch], xf.loc[ct, var], alpha=0.5)
#         a.set(title=ct + " - " + ch)
#     for f, a in zip(s.tail(half).index, ax[half + 1 :]):
#         (ct, ch) = f.split(" - ")
#         a.scatter(xf.loc[ct, ch], xf.loc[ct, var], alpha=0.5)
#         a.set(title=ct + " - " + ch)
# fig.savefig(
#     results_dir
#     / "clinical"
#     / "channel_intensity_in_cells_correlation_with_clinical.per_cell_type.scatter_demo.svg",
#     **figkws,
# )

# #

# #

# #
