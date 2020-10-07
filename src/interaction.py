#!/usr/bin/env python

"""
Spatial analysis of lung tissue
"""

import re

import parmap
from tqdm import tqdm
import pingouin as pg

from imc.types import Path, Array
from imc.operations import get_adjacency_graph
from imc.graphics import swarmboxenplot, get_grid_dims
from imc.utils import align_channels_by_name, z_score
from imc.operations import measure_cell_type_adjacency

from src.config import *


output_dir = results_dir / "interaction"
output_dir.mkdir()

prefix = "roi_zscored.filtered."
cluster_str = "cluster_1.0"

# interaction_file = (
#     output_dir
#     / f"pairwise_cell_type_interaction.{prefix}.{cluster_str}.per_roi.no_correction.pq"
# )
interaction_file = (
    output_dir
    / f"pairwise_cell_type_interaction.{prefix}.{cluster_str}.per_roi.random1000.pq"
)


# Full:
# prj.measure_adjacency()

# Step by step:
if not interaction_file.exists():
    norm_freqs = parmap.map(
        measure_cell_type_adjacency,
        prj.rois,
        method="random",
        inf_replace_method=None,
        n_iterations=1000,
        plot=False,
        save=False,
        pm_pbar=True,
    )

    # norm_freqs = list()
    # for r in prj.rois:
    #     freqs = pd.read_csv(
    #         Path(
    #             r.sample.root_dir / "single_cell" / r.name
    #             + ".cluster_adjacency_graph.frequencies.csv"
    #         ),
    #         index_col=0,
    #     )
    #     freqs.index = freqs.index.astype(int)
    #     freqs.columns = freqs.columns.astype(int)

    #     shuffled_freq = pd.read_csv(
    #         Path(
    #             r.sample.root_dir / "single_cell" / r.name
    #             + ".cluster_adjacency_graph.random_frequencies.csv"
    #         ),
    #         index_col=0,
    #     )
    #     shuffled_freq.index = shuffled_freq.index.astype(int)
    #     shuffled_freq.columns = shuffled_freq.columns.astype(int)

    #     fl = np.log1p((freqs / freqs.values.sum()) * 1e6)
    #     sl = np.log1p((shuffled_freq / shuffled_freq.values.sum()) * 1e6)
    #     # make sure both contain all edges/nodes
    #     fl = fl.reindex(sl.index, axis=0).reindex(sl.index, axis=1).fillna(0)
    #     sl = sl.reindex(fl.index, axis=0).reindex(fl.index, axis=1).fillna(0)
    #     norm_freqs.append(fl - sl)

    norm_freqs = pd.concat(
        [
            x.rename_axis(index="A", columns="B")
            .reset_index()
            .melt(id_vars=["A"])
            .assign(roi=r.name)
            for x, r in zip(norm_freqs, prj.rois)
        ]
    )
    norm_freqs["value"] = norm_freqs["value"].replace(-np.inf, np.nan)

    # add attributes
    norm_freqs = norm_freqs.merge(roi_attributes.reset_index())
    norm_freqs.to_parquet(interaction_file)

norm_freqs = pd.read_parquet(interaction_file)


# set_prj_clusters(aggregated=False)
# suffix = "resolved.random"
set_prj_clusters(aggregated=True)
suffix = "aggregated.random"

# Aggregate across all
cluster_counts = prj.clusters.value_counts()
counts = {
    "disease": prj.clusters.to_frame()
    .join(sample_attributes["disease"])
    .assign(count=1)
    .groupby(["disease", "cluster"])["count"]
    .sum(),
    "phenotypes": prj.clusters.to_frame()
    .join(sample_attributes["phenotypes"])
    .assign(count=1)
    .groupby(["phenotypes", "cluster"])["count"]
    .sum(),
}
clusters = cluster_counts.loc[cluster_counts > 50].index
repl = (
    clusters.to_series()
    .str.split(" - ")
    .apply(pd.Series)[0]
    .astype(int)
    .reset_index()
    .set_index(0)["index"]
    .to_dict()
)

# select only "good" clusters
clusters_int = (
    clusters[~clusters.str.contains(r"\?")]
    .str.extract(r"(\d+) - .*")[0]
    .astype(int)
)
norm_freqs["value"] -= norm_freqs["value"].abs().min()
norm_freqs = norm_freqs.loc[
    norm_freqs["A"].isin(clusters_int) & norm_freqs["B"].isin(clusters_int)
]
norm_freqs["A"] = norm_freqs["A"].replace(repl)
norm_freqs["B"] = norm_freqs["B"].replace(repl)
mx = (
    norm_freqs.groupby(["A", "B"])["value"]
    .apply(np.nanmean)
    .reset_index()
    .pivot_table(index="A", columns="B", values="value", fill_value=0)
)
f = mx.index.astype(str).str.contains(" - ") & (
    ~mx.index.astype(str).str.contains(r"\?")
)

for df, label in [(mx, "all"), (mx.loc[f, f], "filtered")]:
    c = counts["phenotypes"].groupby(level=1).sum()
    grid = sns.clustermap(
        df,
        cmap="RdBu_r",
        center=0,
        # robust=True,
        vmin=-1,
        vmax=1,
        cbar_kws=dict(label="Strength of interaction"),
        rasterized=True,
        xticklabels=True,
        yticklabels=True,
        col_colors=c,
        row_cluster=False,
        col_cluster=False,
    )
    grid.ax_heatmap.set_title("All disease groups")
    grid.fig.savefig(
        results_dir
        / "interaction"
        / f"interactions.mean_across_all_rois.heatmap.{label}.{suffix}.svg",
        **figkws,
    )

    grid = sns.clustermap(
        df,
        cmap="RdBu_r",
        center=0,
        # robust=True,
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        col_colors=c,
        metric="correlation",
        figsize=np.asarray(df.shape) * 0.55,
    )
    grid.fig.savefig(
        results_dir
        / "interaction"
        / f"interactions.mean_across_all_rois.clustermap.{label}.{suffix}.svg",
        **figkws,
    )


# Aggregate across disease
interactions = dict()
for attr_class in ["disease", "phenotypes"]:
    m = (
        norm_freqs.groupby(["A", "B", attr_class])["value"]
        .apply(np.nanmean)
        .reset_index()
        .pivot_table(index=[attr_class, "A"], columns="B", values="value")
    )
    attrs = sample_attributes[attr_class].unique()
    for attr in attrs:
        interactions[attr_class + " - " + attr] = m.loc[attr]


total = (
    norm_freqs.groupby(["A", "B"])["value"]
    .apply(np.nanmean)
    .reset_index()
    .pivot_table(index=["A"], columns="B", values="value")
).dropna()
total = total.loc[:, total.index]

n, m = get_grid_dims(len(interactions))
fig, axes = plt.subplots(n, m, figsize=(8 * m, 8 * n))
for phenotype, ax in zip(interactions, axes.flat):
    g, v = phenotype.split(" - ")
    sns.heatmap(
        interactions[phenotype].loc[total.index, total.columns],
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        rasterized=True,
        # col_colors=counts[g].loc[v],
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(phenotype)
    ax.set_aspect("equal")
fig.savefig(
    output_dir
    / f"differential_interactions.by_disease_phenotypes.{suffix}.svg",
    **figkws,
)


n, m = get_grid_dims(len(interactions))
fig, axes = plt.subplots(n, m, figsize=(12 * m, 12 * n))
for phenotype, ax in zip(interactions, axes.flat):
    g, v = phenotype.split(" - ")
    healthy = interactions[g + " - " + "Healthy"]
    sns.heatmap(
        interactions[phenotype].loc[total.index, total.columns] - healthy,
        cmap="RdBu_r",
        ax=ax,
        vmin=-1,
        vmax=1,
        rasterized=True,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(phenotype)
    ax.set_aspect("equal")
fig.savefig(
    output_dir
    / f"differential_interactions.by_disease_phenotypes.over_healthy.{suffix}.svg",
    **figkws,
)


# Now just covid

for label, (a, b) in [
    # # COVID over healthy
    ("COVID_vs_Healthy", ("disease - Healthy", "disease - COVID19")),
    # # late over early
    (
        "COVIDlate_vs_early",
        ("phenotypes - COVID19_early", "phenotypes - COVID19_late"),
    ),
]:

    diff = (
        np.log(
            interactions[b].loc[total.index, total.columns]
            / interactions[a].loc[total.index, total.columns]
        )
        .replace(np.inf, np.nan)
        .replace(-np.inf, np.nan)
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.heatmap(
        diff,
        cmap="RdBu_r",
        center=0,
        cbar_kws=dict(label="Difference in interaction"),
        ax=ax,
        rasterized=True,
        mask=diff.isnull(),
    )
    ax.set_title(f"'{b}' over '{a}'")
    ax.set_aspect("equal")
    fig.savefig(
        output_dir
        / f"differential_interactions.by_disease_phenotypes.{label}.{suffix}.svg",
        **figkws,
    )
    grid = sns.clustermap(
        diff.fillna(0),
        cmap="RdBu_r",
        center=0,
        cbar_kws=dict(label="Difference in interaction"),
        rasterized=True,
        mask=diff.isnull(),
        metric="correlation",
    )
    grid.ax_heatmap.set_title(f"'{b}' over '{a}'")
    grid.fig.savefig(
        output_dir
        / f"differential_interactions.by_disease_phenotypes.{label}.{suffix}.clustered.svg",
        **figkws,
    )

# dv = (
#     diff.rename_axis("B")
#     .reset_index()
#     .melt(id_vars=["B"])
#     .sort_values("value")
#     .dropna()
# )


# # MA plots
def maplot(cond, background, ax, phenotype):
    merge = cond[["A", "B", "value"]].merge(
        background[["A", "B", "value"]], on=["A", "B"]
    )
    merge["mean"] = merge[["value_x", "value_y"]].mean(1)
    merge["lfc"] = np.log(merge["value_x"] / merge["value_y"])
    merge["lfc"] = merge["value_x"] - merge["value_y"]

    merge["set"] = merge.apply(
        lambda x: tuple(sorted([x["A"], x["B"]])), axis=1
    )
    merge = merge.drop_duplicates("set").drop("set", axis=1)

    ax.scatter(merge["value_x"], merge["lfc"], alpha=0.2, s=5)
    for y in [-1, 1]:
        ax.axhline(y, linestyle="--", color="grey")
    # for x in [-0.25, 0.25]:
    #     ax.axvline(x, linestyle="--", color="grey")
    ax.set(title=phenotype)

    res = merge.loc[
        (merge["value_x"].abs() > 0.0) & (merge["lfc"].abs() > 1)
    ].sort_values("lfc")

    for i in res.head(n_top).index:
        ax.text(
            merge.loc[i, "value_x"],
            merge.loc[i, "lfc"],
            s=merge.loc[i, "A"] + " <->" + merge.loc[i, "B"],
        )
    for i in res.tail(n_top).index:
        ax.text(
            merge.loc[i, "value_x"],
            merge.loc[i, "lfc"],
            s=merge.loc[i, "A"] + " <->" + merge.loc[i, "B"],
        )


n_top = 20
phenotypes = norm_freqs["phenotypes"].unique()
n, m = get_grid_dims(len(phenotypes) + 1)
fig, axes = plt.subplots(n, m, figsize=(8 * m, 6 * n), sharex=True, sharey=True)
for phenotype, ax in zip(phenotypes, axes.flat):
    cond = (
        norm_freqs.query(f"phenotypes == '{phenotype}'")
        .groupby(["A", "B"])["value"]
        .mean()
        .reset_index()
    )
    healthy = (
        norm_freqs.query(f"phenotypes == 'Healthy'")
        .groupby(["A", "B"])["value"]
        .mean()
        .reset_index()
    )
    maplot(cond, healthy, ax, f"'{phenotype}' over healthy")
late = (
    norm_freqs.query(f"phenotypes == 'COVID19_late'")
    .groupby(["A", "B"])["value"]
    .mean()
    .reset_index()
)
early = (
    norm_freqs.query(f"phenotypes == 'COVID19_early'")
    .groupby(["A", "B"])["value"]
    .mean()
    .reset_index()
)
maplot(late, early, axes.flatten()[-1], "Late over early")

fig.savefig(
    output_dir
    / f"differential_interactions.by_disease_phenotypes.over_healthy.{suffix}.svg",
    **figkws,
)


# test
p = norm_freqs.assign(interaction=norm_freqs["A"] + " <-> " + norm_freqs["B"])
p["label"] = p.apply(lambda x: tuple(sorted([x["A"], x["B"]])), axis=1)
p = p.drop_duplicates(["label", "roi"]).drop("label", axis=1)
_res = list()
test_kws = dict(parametric=False)
for interaction in p["interaction"].drop_duplicates():
    try:
        r = pg.pairwise_ttests(
            data=p.query(f"interaction == '{interaction}'"),
            dv="value",
            between="disease",
            **test_kws,
        ).assign(between="disease", interaction=interaction)
        r["p-cor"] = pg.multicomp(r["p-unc"].values, method="fdr_bh")[1]
    except:
        print(f"error {interaction}")
        pass
    _res.append(r)
    try:
        r = pg.pairwise_ttests(
            data=p.query(f"interaction == '{interaction}'"),
            dv="value",
            between="phenotypes",
            **test_kws,
        ).assign(between="phenotypes", interaction=interaction)
        r["p-cor"] = pg.multicomp(r["p-unc"].values, method="fdr_bh")[1]
    except:
        print(f"error {interaction}")
        continue
    _res.append(r)
test_results = pd.concat(_res)
test_results.to_csv(
    output_dir
    / f"differential_interactions.by_disease_phenotypes.tests.{suffix}.csv",
    index=False,
)
test_results = pd.read_csv(
    results_dir
    / "interaction"
    / f"differential_interactions.by_disease_phenotypes.tests.{suffix}.csv",
)


# # volcano plot
n, m = get_grid_dims(len(test_results["B"].unique()) + 2)
fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), sharex=True, sharey=True)
axes = axes.flatten()
i = 0
for realm in test_results["between"].unique():
    res = test_results.query(f"between == '{realm}' & A == 'Healthy'")
    res = res.loc[res["B"] != "Healthy"]
    for group in res["B"].drop_duplicates():
        p = res.query(f"B == '{group}'")

        axes[i].set(
            title=f"{group} over Healthy", xlabel="Hedges g", ylabel="-log10(p)"
        )
        axes[i].scatter(
            p["hedges"],
            -np.log10(p["p-unc"]),
            c=p["hedges"],
            s=5 + (10 * (1 - p["p-unc"])),
            cmap="coolwarm",
            alpha=0.5,
        )

        sig = p.query("abs(hedges) > 0.2 & `p-cor` < 1e-2")
        for j, r in sig.iterrows():
            axes[i].text(r["hedges"], -np.log10(r["p-unc"]), s=r["interaction"])
        i += 1
for ax in axes[i:]:
    ax.axis("off")
fig.savefig(
    output_dir
    / f"differential_interactions.by_disease_phenotypes.tests.{suffix}.volcano_plots.svg",
    **figkws,
)

g1 = "COVID19_early"
g2 = "COVID19_late"
p = test_results.query(f"B == '{g2}' & A == '{g1}'")
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set(title=f"{g2} over {g1}", xlabel="Hedges g", ylabel="-log10(p)")
ax.scatter(
    p["hedges"],
    -np.log10(p["p-unc"]),
    c=p["hedges"],
    s=5 + (10 * (1 - p["p-unc"])),
    cmap="coolwarm",
    alpha=0.5,
)
sig = p.query("abs(hedges) > 0.2 & `p-cor` < 1e-2")
for j, r in sig.iterrows():
    ax.text(r["hedges"], -np.log10(r["p-unc"]), s=r["interaction"])
fig.savefig(
    output_dir
    / f"differential_interactions.by_disease_phenotypes.COVIDlate_over_early.tests.{suffix}.volcano_plots.svg",
    **figkws,
)


res = test_results.query(f"between == 'phenotypes' & A == 'Healthy'")
res["smlogp"] = res["p-unc"] * (res["hedges"] > 0).astype(int).replace(0, -1)

to_ = res.pivot_table(index="interaction", columns="B", values="smlogp")
cl = sns.clustermap(to_, metric="correlation", cmap="RdBu_r", center=0)

x = to_.iloc[cl.dendrogram_row.reordered_ind, cl.dendrogram_col.reordered_ind]
g = sns.relplot(
    data=res,
    row_order=x.index,
    col_order=x.columns,
    x="interaction",
    y="B",
    hue="hedges",
    size="mlogp",
    palette="vlag",
    hue_norm=(-1, 1),
    edgecolor=".7",
    height=2,
    aspect=6,
    sizes=(50, 250),
    size_norm=(-0.2, 0.8),
)
g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=90)
g.fig.savefig("a.svg", bbox_inches="tight")


# # illustrate

# plot interaction between two cell types
from imc.graphics import cell_labels_to_mask

(output_dir / "illustrations").mkdir(exist_ok=True)

markers = json.load(
    open(metadata_dir / "broad_markers_per_aggregated_cell_type.json")
)

res = test_results.query("between == 'phenotypes'")
for _, row in res.query("`p-cor` < 1e-3").iterrows():

    ct1, ct2 = row["interaction"].split(" <-> ")

    output_fig = (
        output_dir
        / "illustrations"
        / f"differential_interactions.{row['A']}-{row['B']}.{ct1}-{ct2}.new.svg"
    )
    # if output_fig.exists():
    #     continue

    sign = 1 if row["hedges"] > 0 else -1

    a = norm_freqs.query(f"phenotypes == '{row['A']}'")
    amost = (
        a.loc[(a["A"] == ct1) & (a["B"] == ct2)]
        .sort_values("value")
        .dropna()
        .set_index("roi")["value"]
    ) * sign

    b = norm_freqs.query(f"phenotypes == '{row['B']}'")
    bmost = (
        b.loc[(b["A"] == ct1) & (b["B"] == ct2)]
        .sort_values("value")
        .dropna()
        .set_index("roi")["value"]
    ) * sign

    # Get ROI with least interaction but while still having cells from cell types
    i = 1
    found = False
    while not found:
        aroi = [
            r for r in prj.rois if r.name == amost.head(i).tail(1).index[0]
        ][0]
        ae1 = (prj.clusters.loc[:, aroi.name] == ct1).sum()
        ae2 = (prj.clusters.loc[:, aroi.name] == ct2).sum()
        if (ae1 > 10) and (ae2 > 10):
            found = True
        i += 1

    # Get ROI with least interaction but while still having cells from cell types
    i = 1
    found = False
    while not found:
        broi = [
            r for r in prj.rois if r.name == bmost.tail(i).head(1).index[0]
        ][0]
        be1 = (prj.clusters.loc[:, broi.name] == ct1).sum()
        be2 = (prj.clusters.loc[:, broi.name] == ct2).sum()
        if (be1 > 10) and (be2 > 10):
            found = True
        i += 1

    ctm1 = cell_labels_to_mask(aroi.cell_mask, aroi.clusters == ct1)
    ctm2 = cell_labels_to_mask(aroi.cell_mask, aroi.clusters == ct2)
    amask = ctm1 + (ctm2 * 2) if ct1 != ct2 else ctm1

    ctm1 = cell_labels_to_mask(broi.cell_mask, broi.clusters == ct1)
    ctm2 = cell_labels_to_mask(broi.cell_mask, broi.clusters == ct2)
    bmask = ctm1 + (ctm2 * 2) if ct1 != ct2 else ctm1

    if "aggregated" not in suffix:
        amarker = (
            ct1.replace("+", "")
            .replace("dim", "")
            .split(" (")[-1]
            .replace(")", "")
            .split(", ")
        )[0] + "("
        bmarker = (
            ct2.replace("+", "")
            .replace("dim", "")
            .split(" (")[-1]
            .replace(")", "")
            .split(", ")
        )[0] + "("
    else:
        amarker = markers[ct1][0] + "("
        bmarker = markers[ct2][0] + "("

    fig, axes = plt.subplots(
        2, 5, figsize=(4 * 5, 4 * 2), squeeze=False, sharex="row", sharey="row"
    )
    fig.suptitle(
        f"{ct1} <-> {ct2}\n"
        f"hedges: {row['hedges']:.2f}; p-val: {row['p-cor']:.2e}"
    )
    ccolors = sns.color_palette("tab10")

    axes[0][0].imshow(amask)
    axes[0][0].set_title(f"{aroi.name} - {ae1}, {ae2}")
    axes[0][0].axis("off")
    axes[0][0].set_title(aroi.name)
    aroi.plot_channel(amarker, ax=axes[0][1])
    aroi.plot_channel(bmarker, ax=axes[0][2])
    aroi.plot_channels(
        [amarker, bmarker, "DNA1"], axes=[axes[0][3]], merged=True
    )
    aroi.plot_channels(
        [amarker, bmarker, "DNA1"], axes=[axes[0][4]], merged=True
    )
    c = axes[0][4].contour(
        amask == 1, levels=1, colors=ccolors[-4], linewidths=0.35
    )
    [o.remove() for o in c.collections[1:]]

    if ct1 != ct2:
        c = axes[0][4].contour(
            amask == 2, levels=1, colors=ccolors[-2], linewidths=0.35
        )
        [o.remove() for o in c.collections[1:]]

    axes[1][0].imshow(bmask)
    axes[1][0].set_title(f"{broi.name} - {be1}, {be2}")
    axes[1][0].axis("off")
    axes[1][0].set_title(broi.name)
    broi.plot_channel(amarker, ax=axes[1][1])
    broi.plot_channel(bmarker, ax=axes[1][2])
    broi.plot_channels(
        [amarker, bmarker, "DNA1"], axes=[axes[1][3]], merged=True
    )
    broi.plot_channels(
        [amarker, bmarker, "DNA1"], axes=[axes[1][4]], merged=True
    )
    c = axes[1][4].contour(
        bmask == 1, levels=1, colors=ccolors[-4], linewidths=0.35
    )
    [o.remove() for o in c.collections[1:]]
    if ct1 != ct2:
        c = axes[1][4].contour(
            bmask == 2, levels=1, colors=ccolors[-2], linewidths=0.35
        )
        [o.remove() for o in c.collections[1:]]

    fig.savefig(output_fig, **figkws)
    plt.close(fig)

#


#


#


#


#


# # Plot with aggregated clusters
new_labels = json.load(open("metadata/cluster_names.json"))[
    f"{prefix};{cluster_str}"
]
new_labels = {int(k): v for k, v in new_labels.items()}
for k in prj.clusters.unique():
    if k not in new_labels:
        new_labels[k] = "999 - ?()"
new_labels_agg = {
    k: "".join(re.findall(r"\d+ - (.*) \(", v)) for k, v in new_labels.items()
}

interactions_agg = dict()
for k, v in interactions.items():
    v.index = v.index.to_series().replace(new_labels_agg)
    v.columns = v.columns.to_series().replace(new_labels_agg)
    v2 = v.groupby(level=0).mean().T.groupby(level=0).mean().T
    interactions_agg[k] = v2

n, m = get_grid_dims(len(interactions_agg))
fig, axes = plt.subplots(n, m, figsize=(8 * m, 8 * n))
for phenotype, ax in zip(interactions_agg, axes.flat):
    sns.heatmap(
        interactions_agg[phenotype],
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        rasterized=True,
        square=True,
    )
    ax.set_title(phenotype)
fig.savefig(
    output_dir
    / "differential_interactions.by_disease_phenotypes.aggregated.svg",
    **figkws,
)

# Demonstrate how methods compare to each other

roi = prj.rois[0]
demos = list()
demos.append(measure_cell_type_adjacency(roi, method="random", n_iterations=10))
demos.append(
    measure_cell_type_adjacency(roi, method="random", n_iterations=500)
)
demos.append(
    measure_cell_type_adjacency(
        roi, method="pharmacoscopy", inf_replace_method="min"
    )
)
demos.append(
    measure_cell_type_adjacency(
        roi, method="pharmacoscopy", inf_replace_method="min_symmetric"
    )
)
demos.append(
    measure_cell_type_adjacency(
        roi, method="pharmacoscopy", inf_replace_method="max"
    )
)
demos.append(
    measure_cell_type_adjacency(
        roi, method="pharmacoscopy", inf_replace_method="max_symmetric"
    )
)

labels = [
    "Shuffling, n = 10",
    "Shuffling, n = 500",
    "Pharmacoscopy, min",
    "Pharmacoscopy, min_symmetric",
    "Pharmacoscopy, max",
    "Pharmacoscopy, max_symmetric",
]
fig, axes = plt.subplots(1, 6, figsize=(6 * 6, 4), sharex=True, sharey=True)
for ax, d, l in zip(axes, demos, labels):
    sns.heatmap(d, center=0, cmap="RdBu_r", ax=ax, rasterized=True)
    ax.set_title(l)
fig.savefig(output_dir / "method_comparison.svg")


import networkx as nx

G = nx.from_pandas_adjacency(total)

nx.drawing.layout.circular_layout(G)
nx.draw_circular(G)


elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0]

pos = nx.circular_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=70)

# edges
cmap = plt.get_cmap("RdBu_r")
nx.draw_networkx_edges(G, pos, width=2, alpha=0.25, edge_cmap=plt.cm.RdBu_r)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=1, alpha=0.25, edge_color="blue",
)

# labels
nx.draw_networkx_labels(G, pos, font_size=4, font_family="sans-serif")

plt.axis("off")
