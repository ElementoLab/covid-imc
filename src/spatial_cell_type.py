import re

from src.config import *

"""
Seeing cell types and their expression in the macroscopic context of the lung.
"""


def plot_distances(df, axes=None):
    mpos = df.groupby("cluster")["pos_lacunae"].mean()
    mneg = df.groupby("cluster")["neg_lacunae"].mean()

    clust_dist = pd.DataFrame([mpos, mneg]).T

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 4))
    fig = axes[0].figure
    # # rank vs distance for vessel
    p = clust_dist.sort_values("pos_lacunae")
    if p.index.str.contains("-").all():
        p.index = p.index.str.extract(r"(\d+ - .*)\(?")[0]
    axes[0].scatter(p.index, p["pos_lacunae"])
    axes[0].set_xticklabels(p.index, rotation=90)

    # # rank vs distance for alveoli
    p = clust_dist.sort_values("neg_lacunae")
    if p.index.str.contains("-").all():
        p.index = p.index.str.extract(r"(\d+ - .*)\(?")[0]
    axes[1].scatter(p.index, p["neg_lacunae"])
    axes[1].set_xticklabels(p.index, rotation=90)

    # # scatter of vessel vs alveoli
    axes[2].scatter(clust_dist["pos_lacunae"], clust_dist["neg_lacunae"])
    for t in mneg.index:
        axes[2].text(*clust_dist.loc[t], s=t)

    axes[0].set(
        xlabel="Clusters", ylabel="Distance to vessel\n" + r"(mean, $\mu$m)"
    )
    axes[1].set(
        xlabel="Clusters", ylabel="Distance to alveoli\n" + r"(mean, $\mu$m)"
    )
    axes[2].set(
        xlabel="Distance to vessel\n" + r"(mean, $\mu$m)",
        ylabel="Distance to alveoli\n" + r"(mean, $\mu$m)",
    )
    return fig


output_dir = results_dir / "pathology"
prefix = "roi_zscored.filtered."
cluster_str = "cluster_1.0"

quantification_file = results_dir / "cell_type" / "quantification.pq"

# read in distances to structures
dists = pd.read_parquet(
    results_dir / "pathology" / "cell_distance_to_lacunae.pq"
)
dists.index.name = "obj_id"

dists_pheno = dists.merge(
    sample_attributes[["phenotypes"]], left_on="sample", right_index=True
)
phenos = sample_attributes["phenotypes"].cat.categories

# See if cell types have preference for position relative to lacunae
# # add cluster identity (nice labels)
set_prj_clusters(aggregated=False)
clusters = prj.clusters.loc[~prj.clusters.str.contains(r"\?"), :].reset_index()

d = dists.reset_index().merge(clusters, on=["roi", "sample", "obj_id"])
fig = plot_distances(d)
fig.savefig(
    output_dir / f"cell_type_distance_to_lacunae.{prefix}.{cluster_str}.svg",
    **figkws,
)

fig, axes = plt.subplots(len(phenos), 3, figsize=(3 * 3, 3 * len(phenos)))
for i, pheno in enumerate(phenos):
    d = (
        dists_pheno.query(f"phenotypes == '{pheno}'")
        .reset_index()
        .merge(clusters, on=["roi", "sample", "obj_id"])
    )
    plot_distances(d, axes=axes[i])
    axes[i][-1].set_title(pheno)
fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.{prefix}.{cluster_str}.phenotype_comparison.svg",
    **figkws,
)


# # aggregate clusters
set_prj_clusters(aggregated=True)
clusters = prj.clusters.loc[
    ~prj.clusters.str.startswith("00 - "), :
].reset_index()

dagg = dists.reset_index().merge(clusters, on=["roi", "sample", "obj_id"])
fig = plot_distances(dagg)
fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.{prefix}.{cluster_str}.reduced_clusters.svg",
    **figkws,
)

dists_pheno = dists.merge(
    sample_attributes[["phenotypes"]], left_on="sample", right_index=True
)
phenos = sample_attributes["phenotypes"].cat.categories
fig, axes = plt.subplots(len(phenos), 3, figsize=(3 * 3, 3 * len(phenos)))
for i, pheno in enumerate(phenos):
    dagg = (
        dists_pheno.query(f"phenotypes == '{pheno}'")
        .reset_index()
        .merge(clusters, on=["roi", "sample", "obj_id"])
    )
    plot_distances(dagg, axes=axes[i])
    axes[i][-1].set_title(pheno)
fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.{prefix}.{cluster_str}.reduced_clusters.phenotype_comparison.svg",
    **figkws,
)


# Use vessel/airway/alveoli annotations to further identify resolved clusters
# read in distances to structures
dists = pd.read_parquet(
    results_dir
    / "pathology"
    / "cell_distance_to_lacunae.vessel_airway_alveoli.pq"
)
dists.index.name = "obj_id"

dists_pheno = dists.merge(
    sample_attributes[["phenotypes"]], left_on="sample", right_index=True
)
phenos = sample_attributes["phenotypes"].cat.categories

# See if cell types have preference for position relative to lacunae
set_prj_clusters(aggregated=True)
clusters = prj.clusters.loc[~prj.clusters.str.contains(r"\?"), :].reset_index()

d = dists.reset_index().merge(clusters, on=["roi", "sample", "obj_id"])
dstruc = d.groupby("cluster")[["vessel", "airway", "alveoli"]].mean().dropna()
dstruc_norm = ((dstruc - dstruc.mean()) / dstruc.std()) * -1
order = (
    open(metadata_dir / "cluster_ordering.aggregated.txt")
    .read()
    .strip()
    .split("\n")
)
order = list(
    map(
        lambda x: x[0],
        filter(
            lambda x: len(x),
            [
                dstruc_norm.index[dstruc_norm.index.str.contains(x)]
                for x in order
            ],
        ),
    )
)
dstruc_norm = dstruc_norm.reindex(order)
grid = sns.clustermap(
    dstruc_norm.fillna(0),
    cmap="inferno",
    center=0,
    robust=True,
    figsize=(2, 4),
    yticklabels=True,
)
grid.fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.aggregated.clustermap.svg",
    **figkws,
)
grid = sns.clustermap(
    dstruc_norm.fillna(0),
    cmap="inferno",
    center=0,
    robust=True,
    row_cluster=False,
    col_cluster=False,
    figsize=(2, 4),
    yticklabels=True,
)
grid.fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.aggregated.clustermap.ordered.svg",
    **figkws,
)

# # add cluster identity (nice labels)
set_prj_clusters(aggregated=False)
clusters = prj.clusters.loc[~prj.clusters.str.contains(r"\?"), :].reset_index()

d = dists.reset_index().merge(clusters, on=["roi", "sample", "obj_id"])
dstruc = d.groupby("cluster")[["vessel", "airway", "alveoli"]].mean().dropna()
dstruc_norm = ((dstruc - dstruc.mean()) / dstruc.std()) * -1
dstruc_mac = dstruc_norm.loc[
    dstruc_norm.index.str.contains("Macrophages|Monocytes"), :
]
order = (
    open(metadata_dir / "cluster_ordering.resolved.txt")
    .read()
    .strip()
    .split("\n")
)
order = list(
    map(
        lambda x: x[0],
        filter(
            lambda x: len(x),
            [
                dstruc_norm.index[dstruc_norm.index.str.startswith(x)]
                for x in order
            ],
        ),
    )
)
dstruc_norm = dstruc_norm.reindex(order)

grid = sns.clustermap(dstruc_norm, cmap="inferno", center=0, robust=True)
grid.fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.clustermap.svg",
    **figkws,
)
grid = sns.clustermap(
    dstruc_norm,
    cmap="inferno",
    center=0,
    robust=True,
    row_cluster=False,
    col_cluster=False,
)
grid.fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.clustermap.ordered.svg",
    **figkws,
)
grid = sns.clustermap(
    dstruc_norm,
    cmap="Reds",
    z_score=0,
    center=0,
    robust=True,
    row_cluster=False,
    col_cluster=False,
)
grid.fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.clustermap.binary_rowzscore.svg",
    **figkws,
)
grid = sns.clustermap(dstruc_mac, cmap="inferno", center=0, figsize=(2, 3))
grid.fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.clustermap.macrophages_only.svg",
    **figkws,
)

from math import pi

vmin = -2
vmax = 2

colors = dict(
    zip(
        dstruc.index.str.extract(r"\d+ - (.*) \(")[0].sort_values().unique(),
        sns.color_palette("tab20"),
    )
)

N = dstruc.shape[1]
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

n, m = get_grid_dims(dstruc.shape[0])
fig, axes = plt.subplots(
    n, m, figsize=(m * 2, n * 2), subplot_kw=dict(polar=True)
)
for ax, ct in zip(axes.flat, dstruc.index):
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dstruc.columns)

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_yticks([vmin, 0, vmax])
    ax.set_yticklabels([vmin, 0, vmax])
    ax.set_ylim((vmin, vmax))

    # Ind1
    values = dstruc_norm.loc[ct].values.flatten().tolist()
    values += values[:1]
    c = re.findall(r"\d+ - (.*) \(", ct)[0]
    ax.plot(
        angles,
        values,
        linewidth=1,
        linestyle="solid",
        label=ct,
        color=colors[c],
    )
    ax.fill(angles, values, alpha=0.1, color=colors[c])
    ax.set_title(ct)
fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.radar.svg",
    **figkws,
)

# # plot only macrophages
vmin = -1
vmax = 1

N = dstruc_mac.shape[1]
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

n, m = get_grid_dims(dstruc_mac.shape[0])
fig, axes = plt.subplots(
    n, m, figsize=(m * 2, n * 2), subplot_kw=dict(polar=True)
)
figs, axss = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
axss.set_theta_offset(pi / 2)
axss.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet
axss.set_xticks(angles[:-1])
axss.set_xticklabels(dstruc_mac.columns)

# Draw ylabels
axss.set_rlabel_position(0)
axss.set_yticks([vmin, 0, vmax])
axss.set_yticklabels([vmin, 0, vmax])
axss.set_ylim((vmin, vmax))

for ax, ct in zip(axes.flat, dstruc_mac.index):
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dstruc_mac.columns)

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_yticks([vmin, 0, vmax])
    ax.set_yticklabels([vmin, 0, vmax])
    ax.set_ylim((vmin, vmax))

    # Ind1
    values = dstruc_norm.loc[ct].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle="solid", label="group A")
    ax.fill(angles, values, alpha=0.1)
    axss.plot(angles, values, linewidth=1, linestyle="solid", label=ct)
    axss.fill(angles, values, alpha=0.1)
    ax.set_title(ct)
fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.radar.macrophages_only.svg",
    **figkws,
)
figs.legend()
figs.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.vessel_airway_alveoli.{prefix}{cluster_str}.radar.macrophages_only.overlay.svg",
    **figkws,
)


#

#

#


# See w
from imc.graphics import get_grid_dims

# # normalize by both roi size and lacunae number in roi
roi_areas = pd.read_csv(roi_areas_file, index_col=0, squeeze=True) / 1e6
lacunae_quantification_file = (
    results_dir / "pathology" / "lacunae.quantification_per_image.csv"
)
lac = pd.read_csv(lacunae_quantification_file, index_col=0)["lacunae_number"]

dists_norm = pd.concat(
    [
        np.log1p(
            dists.loc[dists["roi"] == r, "parenchyma_lacunae"]
            / roi_areas.loc[r]
            / lac.loc[r]
        )
        .to_frame()
        .assign(roi=r)
        for r in dists["roi"].unique()
    ]
)

p = (
    dists_norm.reset_index()
    .merge(clusters, on=["roi", "obj_id"])
    .groupby(["roi", "cluster"])
    .mean()
    .merge(roi_attributes[["phenotypes"]], left_on="roi", right_index=True)
)

cts = p.index.levels[1]
n, m = get_grid_dims(len(cts))
fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n))
for ct, ax in zip(cts, axes.flat):
    stats = swarmboxenplot(
        data=p.query(f"cluster == '{ct}'"),
        x="phenotypes",
        y="parenchyma_lacunae",
        ax=ax,
    )
    ax.set_title(ct)
fig.savefig(
    output_dir
    / f"cell_type_distance_to_lacunae.{prefix}.{cluster_str}.reduced_clusters.distance_parenchyma_per_cluster_per_phenotype.svg",
    **figkws,
)


# See if cell types have preference for position relative to lacunae

# WIP!
quant = pd.read_parquet(quantification_file)
quant.index.name = "obj_id"
set_prj_clusters(aggregated=False)
d = dists.reset_index().merge(
    quant.reset_index(), on=["roi", "sample", "obj_id"]
)

c = d.drop(["obj_id"], axis=1).corr()
c = c.loc[~c.index.str.contains("lacunae"), c.columns.str.contains("lacunae")]

grid = sns.clustermap(c, cmap="RdBu_r", center=0, yticklabels=True)
grid.fig.show()


chs = ["DNA1(Ir191)", "AlphaSMA(Pr141)", "CD31(Eu151)", "SARSCoV2S1(Eu153)"]

fig, axes = plt.subplots(
    len(c.columns), len(chs), figsize=(len(chs) * 3, len(c.columns) * 3)
)
for i, lac in enumerate(c.columns):
    for j, ch in enumerate(chs):
        axes[i][j].scatter(
            1 + d[lac], d[ch] + 1, s=0.5, alpha=0.01, rasterized=True
        )
        axes[i][j].loglog()
for ax, t in zip(axes[0, :], chs):
    ax.set_title(t)
for ax, t in zip(axes[:, 0], c.columns):
    ax.set_ylabel(t)


pm = d.groupby(pd.cut(d["pos_lacunae"], 20))["SARSCoV2S1(Eu153)"].mean()
nm = d.groupby(pd.cut(d["neg_lacunae"], 20))["SARSCoV2S1(Eu153)"].mean()
plt.scatter(pm.index.astype(str), pm)
plt.scatter(nm.index.astype(str), nm)
