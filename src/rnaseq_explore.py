# coding: utf-8

from pathlib import Path

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
from imc.graphics import get_grid_dims


activate_annotated_clustermap()

PIL.Image.MAX_IMAGE_PIXELS = 933120000

figkws = dict(dpi=300, bbox_inches="tight")
data_dir = Path("data") / "rna-seq"
output_dir = Path("results") / "rna-seq"
output_dir.mkdir(exist_ok=True)


def ssgsea(sample, database, x):
    res = gp.ssgsea(x.loc[:, sample], database)
    return res.resultsOnSamples["sample1"]


# metadata and data
x = pd.read_table(data_dir / "RNAseq_counts_forIMC.txt", index_col=0)
y = (
    pd.read_table(data_dir / "RNAseq_metadata_forIMC.txt")
    .set_index("SeqID")
    .sort_index()
)
y["sample_id"] = y["SampleID"]
y["disease"] = y["STATUS"]

# # load extra metadata from patient files
y_extra = pd.read_excel(
    "metadata/original/Hyperion samples.original.20200820.xlsx",
    na_values=["na"],
)
y_extra["Autopsy Code"] = y_extra["Autopsy Code"].str.capitalize()
y_extra["pcr_spike_positive"] = y_extra["Sample Classification"].str.endswith(
    "Positive"
)
y_extra["phenotypes"] = (
    y_extra["Days of disease"].replace("na", np.nan).convert_dtypes() < 30
).replace({True: "Early", False: "Late"})

y_extra = y_extra.rename(
    columns={
        "Days of disease": "days_of_disease",
        "Days Intubated": "days_intubated",
        "lung WEIGHT g": "lung_weight_grams",
        "AGE (years)": "age_years",
        "GENDER (M/F)": "gender",
        "RACE": "race",
        "SMOKE (Y/N)": "smoker",
        "Fever (Tmax)": "fever",
        "Cough": "cough",
        "Shortness of breath": "shortness_of_breath",
        "COMORBIDITY (Y/N; spec)": "comorbidities",
        "TREATMENT": "treatment",
    }
)

cols = [
    "Autopsy Code",
    "phenotypes",
    "pcr_spike_positive",
    "days_of_disease",
    "days_intubated",
    "days_intubated",
    "lung_weight_grams",
    "age_years",
    "gender",
    "race",
    "smoker",
    "fever",
    "cough",
    "shortness_of_breath",
    "comorbidities",
    "treatment",
    "PLT/mL",
    "D-dimer (mg/L)",
    "WBC",
    "LY%",
    "PMN %",
]

# merge metadata
y = (
    y.reset_index()
    .merge(
        y_extra[cols],
        left_on="AutopsyCode",
        right_on="Autopsy Code",
        how="left",
    )
    .set_index("SeqID")
)
y.loc[y["disease"] == "Control", "phenotypes"] = "Control"

y["phenotypes"] = pd.Categorical(
    y["phenotypes"], ordered=True, categories=["Control", "Early", "Late"],
)

# align indeces of data and metadata
x = x.reindex(columns=y.index)

# map to gene symbols by max isoform expression
xx = x.groupby(x.index.str.extract(r"(.*)\.\d+")[0].values).max()
gene_map = query_biomart()
g = (
    xx.join(gene_map.set_index("ensembl_gene_id")["external_gene_name"])
    .groupby("external_gene_name")
    .max()
)


clinvars = [
    "disease",
    "phenotypes",
    "pcr_spike_positive",
    "days_of_disease",
    "gender",
]

a = AnnData(x.T, obs=y.assign(nreads=x.sum()))
sc.pp.log1p(a)
sc.pp.normalize_total(a)
sc.pp.scale(a)
sc.tl.pca(a)
sc.pp.neighbors(a)
sc.tl.umap(a)

fig = sc.pl.pca(a, color=["nreads"] + clinvars, s=150, show=False)[0].figure
fig.savefig(output_dir / "gene_expression.pca.svg")

fig = sc.pl.umap(a, color=["nreads"] + clinvars, s=150, show=False)[0].figure
fig.savefig(output_dir / "gene_expression.umap.svg")


output_file = output_dir / "rna-seq.ssGSEA_enrichment.h.all.v7.2.csv"
if not output_file.exists():
    res = parmap.map(ssgsea, x.columns, database="h.all.v7.2.symbols.gmt", x=g)
    enr = pd.concat(res, axis=1)
    enr.columns = x.columns
    enr.to_csv(output_file)
enr = pd.read_csv(output_file, index_col=0)

output_file = output_dir / "rna-seq.ssGSEA_enrichment.c8.all.v7.2.csv"
if not output_file.exists():
    res2 = parmap.map(
        ssgsea, x.columns, database="c8.all.v7.2.symbols.gmt", x=g
    )
    ct = pd.concat(res2, axis=1)
    ct.columns = x.columns
    ct.to_csv(output_file)
ct = pd.read_csv(output_file, index_col=0)

output_file = output_dir / "rna-seq.ssGSEA_enrichment.scsig.all.v1.0.1.csv"
if not output_file.exists():
    res3 = parmap.map(
        ssgsea, x.columns, database="scsig.all.v1.0.1.symbols.gmt", x=g
    )
    ct2 = pd.concat(res3, axis=1)
    ct2.columns = x.columns
    ct2.to_csv(output_file)
ct2 = pd.read_csv(output_file, index_col=0)

# sns.histplot(x.mean())


df2 = np.log1p(x)
df3 = (df2 / df2.sum()) * 1e4

# Intra patient, inter location correlation

means = (
    df3.T.join(y[["disease", "SampleID"]])
    .groupby(["disease", "SampleID"])
    .mean()
    .T
)

corrs = means.corr()

v = corrs.values.min()
v -= v * 0.1
grid = sns.clustermap(corrs, vmin=v, cmap="RdBu_r",)


grid = sns.clustermap(
    enr,
    center=0,
    cmap="RdBu_r",
    # col_colors=y[clinvars],
    z_score=0,
    metric="correlation",
)
grid.savefig(
    output_dir / "ssGSEA_enrichment.h.all.v7.2.all_ROIs.clustermap.svg",
    **figkws,
)


fig, stats = swarmboxenplot(
    data=enr.T.join(y),
    x="phenotypes",
    y="HALLMARK_IL6_JAK_STAT3_SIGNALING",
    # hue='phenotypes',
    test=True,
    test_kws=dict(parametric=False),
)
fig.savefig(
    output_dir / "ssGSEA_enrichment.IL6_signature.swarmboxenplot.svg", **figkws,
)


grid = sns.clustermap(
    ct,
    center=0,
    cmap="RdBu_r",
    # col_colors=y[clinvars],
    cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
    rasterized=True,
    z_score=0,
)
grid.savefig(
    output_dir / "ssGSEA_enrichment.c8.all.v7.2.all_ROIs.clustermap.svg",
    **figkws,
)


grid = sns.clustermap(
    ct2,
    center=0,
    cmap="RdBu_r",
    # col_colors=y[clinvars],
    cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
    rasterized=True,
    z_score=0,
)
grid.savefig(
    output_dir / "ssGSEA_enrichment.scsig.all.v1.0.1.all_ROIs.clustermap.svg",
    **figkws,
)


c = ct.append(ct2)
grid = sns.clustermap(
    c,
    center=0,
    cmap="RdBu_r",
    # col_colors=y[clinvars],
    cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
    rasterized=True,
    z_score=0,
)
grid.savefig(
    output_dir / "ssGSEA_enrichment.cell_type.all_ROIs.clustermap.svg", **figkws
)

c = ((c.T - c.mean(1)) / c.std(1)).T
grid = sns.clustermap(
    c,
    center=0,
    cmap="RdBu_r",
    # col_colors=y[clinvars],
    cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
    rasterized=True,
)
grid.savefig(
    output_dir / "ssGSEA_enrichment.cell_type.all_ROIs.clustermap.z_score.svg",
    **figkws,
)


cells = [
    "Epithelial",
    "Mesenchymal",
    "Fibroblast",
    "Smooth_muscle",
    "Club",
    "CD4_T",
    "CD8_T",
    "NK_cell",
    "Macrophage",
    "Monocyte",
    "Neutrophil",
    "B_cell",
    "Mast",
    "Dendritic",
]

_res = dict()
for cell in cells:
    _res[cell] = c.loc[c.index.str.contains(cell, case=False)].mean()

res = pd.DataFrame(_res)

grid = sns.clustermap(
    res.T.dropna(),
    metric="correlation",
    center=0,
    cmap="RdBu_r",
    # col_colors=y[clinvars],
    cbar_kws=dict(label="ssGSEA enrichment\n(Z-score)"),
    figsize=(5, 4),
    xticklabels=False,
)
grid.savefig(
    output_dir
    / "ssGSEA_enrichment.cell_type_aggregated.all_ROIs.clustermap.svg",
    **figkws,
)


n, m = get_grid_dims(res.shape[1])
fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))
for i, c in enumerate(res.columns):
    swarmboxenplot(
        data=res[[c]].join(y), x="phenotypes", y=c, ax=axes.flatten()[i]
    )
    axes.flatten()[i].set(title=c)
for ax in axes.flat[i + 1 :]:
    ax.axis("off")

fig.savefig(
    output_dir
    / "ssGSEA_enrichment.cell_type_aggregated.all_ROIs.swarmboxenplot.svg",
    **figkws,
)


n, m = get_grid_dims(res.shape[1])
fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))
for i, c in enumerate(res.columns):
    ax = axes.flatten()[i]
    p = res.join(y[["disease", "days_of_disease"]])
    p.loc[p["disease"] == "Control", "days_of_disease"] = 0
    ax.scatter(p["days_of_disease"], p[c])
    ax.set(title=c)


grid = sns.clustermap(
    res.T.dropna(),
    metric="correlation",
    center=0,
    cmap="RdBu_r",
    # col_colors=y[clinvars],
    cbar_kws=dict(label="ssGSEA enrichment\nZ-score)"),
    # figsize=(5, 4),
)

grid.savefig(
    output_dir
    / "ssGSEA_enrichment.cell_type_aggregated.sample_reduced.all_ROIs.clustermap.svg",
    **figkws,
)


# Unsupervised dimres

# # signature space
a = AnnData(res.T.dropna().T, obs=res.join(y))
sc.tl.pca(a)

fig = sc.pl.pca(
    a,
    color=clinvars,
    components=["1,2", "2,3", "3,4", "4,5"],
    show=False,
    s=150,
)[0].figure
fig.savefig(output_dir / "signature_space.pca.svg")
sc.pp.neighbors(a)
sc.tl.umap(a)
fig = sc.pl.umap(a, color=clinvars, s=150, show=False)[0].figure
fig.savefig(output_dir / "signature_space.umap.svg")


# # cell type space
enrz = (enr.T - enr.mean(1)) / enr.std(1)

a = AnnData(enrz.T.dropna().T, obs=enrz.join(y))
sc.tl.pca(a)
fig = sc.pl.pca(
    a,
    color=clinvars,
    components=["1,2", "2,3", "3,4", "4,5"],
    show=False,
    s=150,
)[0].figure
fig.savefig(output_dir / "cell_type_space.pca.svg")
sc.pp.neighbors(a)
sc.tl.umap(a)
fig = sc.pl.umap(a, color=clinvars, s=150, show=False)[0].figure
fig.savefig(output_dir / "cell_type_space.umap.svg")


#


#


#


#


#


#

ct_means = ct.T.join(y["location"]).groupby("location").mean().T
(ct_means["Large Airway"] - ct_means["Alveolar"]).sort_values()
(ct_means["Vascular"] - ct_means["Alveolar"]).sort_values()


((ct_means - ct_means.mean(1)) / ct_means.std(1))
((ct_means["Alveolar"] - ct_means.mean(1)) / ct_means.std(1)).sort_values()


# mouse genes :/
res3 = parmap.map(
    ssgsea, x.columns, database="10.1186|1471-2164-15-726.gmt", x=x
)
ct2 = pd.concat(res3, axis=1)
ct2.columns = x.columns


a = sc.read("droplet_normal_lung_blood_scanpy.20200205.RC4.h5ad")
obs = pd.read_csv("krasnow_hlca_10x_metadata.csv", index_col=0)
mean = a.to_df().join(obs).groupby("free_annotation").mean().T
mean.iloc[:-10, :].to_csv("krasnow_hlca_10x.average.expression.csv")
