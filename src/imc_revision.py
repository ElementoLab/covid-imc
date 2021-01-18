#! /usr/bin/env python

"""
Analysis of revision data looking in more depth at the immune compartment.
"""

import sys
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
from joblib import parallel_backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc

from imc import Project
from imc.types import Path, DataFrame, Array
from imc.graphics import close_plots
from imc.graphics import rasterize_scanpy, add_centroids

from seaborn_extensions import clustermap


figkws = dict(bbox_inches="tight", dpi=300)

metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")
output_dir = results_dir / "imc_revision"
output_dir.mkdir()


exclude_channels = [
    "80ArAr(ArAr80)",
    "129Xe(Xe129)",
    "190BCKG(BCKG190)",
    "<EMPTY>(Pb204)",
]


global args
args: Namespace
cli = None


def main(cli=None) -> int:
    args = get_parser().parse_args(cli)
    prj = Project(name="imc_revision")
    prj.samples = [s for s in prj if "58_Panel2" in s.name]
    for s in prj:
        s.rois = [r for r in s if r._get_input_filename("cell_mask").exists()]

    illustrations(prj)

    qc(prj)

    phenotyping(prj)

    return 0


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resolutions", default=[0.5, 1.0, 1.5, 2.0], nargs="+"
    )
    parser.add_argument("--algos", default=["umap"], nargs="+")
    return parser


@close_plots
def illustrations(prj: Project) -> None:
    (output_dir / "full_stacks").mkdir()
    (output_dir / "illustration").mkdir()

    for r in tqdm(prj.rois):
        output_f = output_dir / "full_stacks" / r.name + ".pdf"
        fig = r.plot_channels()
        fig.savefig(output_f, **figkws)

    for r in tqdm(prj.rois):
        output_f = output_dir / "illustration" / r.name + ".svg"
        fig = r.plot_probabilities_and_segmentation()
        fig.savefig(output_f, **figkws)


def qc(prj: Project) -> None:
    (output_dir / "qc").mkdir()
    output_prefix = output_dir / "qc" / "channel_summary."

    c = prj.rois[0].channel_labels
    exc = [x for x in c if x in exclude_channels]

    prj.channel_summary(output_prefix=output_prefix, channel_exclude=exc)


def phenotyping(prj: Project) -> None:
    (output_dir / "phenotyping").mkdir()
    output_prefix = output_dir / "phenotyping" / prj.name + "."

    quant_f = output_prefix + "quantification.pq"
    if not quant_f.exists():
        prj.quantify_cells()
        prj.quantification.to_parquet(quant_f)
    quant = pd.read_parquet(quant_f).reset_index()

    # If Tbet is spelled differently
    # quant.loc[quant["TBet(Sm149)"].isnull(), "TBet(Sm149)"] = quant[
    #     "Tbet(Sm149)"
    # ]
    # quant = quant.drop("Tbet(Sm149)", axis=1)

    quant = quant.drop(exclude_channels, axis=1)
    quant["DNA"] = quant.loc[:, quant.columns.str.contains("DNA")].mean(1)
    quant["Ki67"] = quant.loc[:, quant.columns.str.contains("Ki67")].mean(1)
    quant = quant.drop(
        quant.columns[
            quant.columns.str.contains(r"DNA\d\(")
            | quant.columns.str.contains(r"Ki67\(")
        ],
        axis=1,
    )

    # filter out cells
    quant_ff = quant_f.replace_(".pq", ".filtered.pq")
    if not quant_ff.exists():
        exclude = filter_out_cells(
            quant, plot=True, output_prefix=output_prefix
        )
        tqdm.write(
            f"Filtering out {exclude.sum()} cells ({(exclude.sum() / exclude.shape[0]) * 100:.2f} %)"
        )

        quant = quant.loc[~exclude]
        quant.to_parquet(quant_ff)
    quant = pd.read_parquet(quant_ff)

    # Drop unwanted channels and redundant morphological features
    h5ad_f = output_prefix + prj.name + ".h5ad"
    if not h5ad_f.exists() or args.overwrite:
        q = quant.drop(["perimeter", "major_axis_length"], axis=1).reset_index(
            drop=True
        )
        id_cols = ["sample", "roi", "obj_id"]

        a = AnnData(
            q.drop(id_cols + ["area"], axis=1), obs=q[id_cols + ["area"]]
        )
        a.raw = a

        # notes:
        ###
        ## Scaling the features prior to PCA is important
        ###
        ## if not using log scale, decrease umap gamma to 5.
        ## if using log scale, use high gamma values in umap e.g. 25
        ###
        ## regresing out area works and it's fast, my feeling is that it does not add too much

        sc.pp.log1p(a)
        # sc.pp.regress_out(a, "area", n_jobs=12)
        # sc.pp.regress_out(a, "sample", n_jobs=12)
        sc.pp.scale(a)
        sc.pp.pca(a)
        with parallel_backend("threading", n_jobs=12):
            sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
        with parallel_backend("threading", n_jobs=12):
            sc.tl.umap(a, gamma=25)

        for res in args.resolutions:
            sc.tl.leiden(a, resolution=res, key_added=f"cluster_{res}")
            a.obs[f"cluster_{res}"] = pd.Categorical(
                a.obs[f"cluster_{res}"].astype(int) + 1
            )
        sc.write(h5ad_f, a)

    a = sc.read(h5ad_f)
    a = a[a.obs.sample(frac=1).index, :]

    # Plot projections
    # vmin = [min(x, 0.0) for x in np.percentile(a.raw.X, 1, axis=0).tolist()]
    # vmax = [max(x, 1.0) for x in np.percentile(a.raw.X, 99, axis=0).tolist()]
    # vmin = np.percentile(a.raw.X, 1, axis=0).tolist()
    vmin = None
    vmax = np.percentile(a.raw.X, 99, axis=0).tolist()
    # notes:
    ## if scaling values clip both ends to percentiles
    ## if using log or raw original values clip top to percentiles

    color = (
        a.var.index.tolist()
        + ["area", "sample"]
        + [f"cluster_{res}" for res in args.resolutions]
    )
    for algo in args.algos:
        f = output_prefix + f"{algo}.pdf"
        projf = getattr(sc.pl, algo)
        axes = projf(
            a,
            color=color,
            show=False,
            vmin=vmin,  # + [None] * len(args.resolutions),
            vmax=vmax
            + [np.percentile(a.obs["area"], 99)]
            + [None]
            + [None] * (len(args.resolutions)),
            use_raw=True,
        )
        fig = axes[0].figure
        for ax, res in zip(axes[-len(args.resolutions) :], args.resolutions):
            add_centroids(a, res=res, ax=ax)

        rasterize_scanpy(fig)

        fig.savefig(f, **figkws)

    #

    # Cluster phenotypes
    for res in args.resolutions:
        m = a.to_df().groupby(a.obs[f"cluster_{res}"]).mean()
        mr = (
            AnnData(a.raw.X, var=a.var, obs=a.obs)
            .to_df()
            .groupby(a.obs[f"cluster_{res}"])
            .mean()
        )
        for conf in ["abs", "z"]:
            grid = clustermap(m, config=conf, figsize=(8, 5))
            grid.fig.savefig(
                output_prefix
                + f"phenotypes.cluster_{res}.clustermap.{conf}.svg",
                **figkws,
            )


def filter_out_cells(
    quant: DataFrame, plot=True, output_prefix: Path = None
) -> Array:
    from imc.operations import get_population
    from imc.utils import minmax_scale
    from mpl_toolkits.mplot3d import Axes3D

    # create combined score for artifact likelihood
    score = minmax_scale(
        (minmax_scale(quant["solidity"]) * 2)
        * (1 - minmax_scale(quant["area"]))
        * (1 - minmax_scale(quant["DNA"]))
    )

    # get population with highest score
    ## KMeans with k == 3 also works well but we have to assume smallest cluster is to remove
    # from sklearn.cluster import KMeans
    # al = KMeans(3)
    # al.fit(score.values.reshape((-1, 1)))
    # c = al.predict(score.values.reshape((-1, 1)))
    # to_filter = c == pd.Series(c).value_counts().idxmin()

    ## Mixture of gaussians
    to_filter = get_population(score)

    if plot:
        assert output_prefix is not None
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for sel, edgecolor in [(to_filter, "red"), (~to_filter, "black")]:
            s = ax.scatter(
                quant.loc[sel]["solidity"],
                np.sqrt(quant.loc[sel]["area"]),
                np.log1p(quant.loc[sel]["DNA"]),
                s=2,
                alpha=0.25,
                c=score[sel],
                edgecolors=edgecolor,
                linewidths=0.25,
                rasterized=True,
            )
        fig.colorbar(s, ax=ax, label="score")
        ax.set(xlabel="solidity", ylabel="area", zlabel="DNA")
        fig.savefig(output_prefix + "3d_scatter.svg", **figkws)

        fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 1 * 4), sharey=True)
        for ax, var in zip(axes, ["area", "DNA", "solidity"]):
            sns.distplot(
                quant[var][~to_filter], label="To keep", kde=False, ax=ax
            )
            sns.distplot(
                quant[var][to_filter], label="To remove", kde=False, ax=ax
            )
            ax.set_xlabel(var)
            ax.legend()
        axes[0].set_ylabel("Cell number")
        fig.savefig(output_prefix + "per_variable_histogram.svg", **figkws)

    return to_filter


cluster_idents = {
    1.0: {
        1: "Stroma ()",
        2: "CD39 cells (CD39+)",
        3: "Stroma (CD31+)",
        4: "Neutrophils (CD15+, CD11c+)",
        5: "Stroma ()",
        6: "Stroma ()",
        7: "Macrophages ()",
        8: "Epithelial (Keratin8/18+)",
        9: "CD38, CD44 cells (CD38+, CD44+)",
        10: "T cells (CD3+, CD8a+, CD57+)",
        11: "Macrophages (CD14+, CD16+, CD163+, CD86+, CD123+, VISTA+, S100A9+, PDL1+, CD11c+)",
        12: "Neutrophils (CD15+, CD11c+, GranzymeB+, S100A9+, VISTA+)",
        13: "Stroma (CD31+, Vimentin+)",
    },
    1.5: {
        1: "CD39+ cells (CD39+, CD20+, CD11cdim)",
        2: "No marker ()",
        3: "? (TIM3+, CD31+, Tbet+, CD123+, HLADR+)",
        4: "No marker ()",
        5: "? (CD45RO+, TIM3+, CD31+)",
        6: "Neutrophils",
        7: "Stroma (CD31+)",
        8: "Epithelial (Keratin8/18+)",
        9: "Proliferating (Ki67+, CD39+)",
        10: "Macrophages",
        11: "T cells (CD3+, CD8a+, CD57+)",
        12: "Macrophages",
        13: "Neutrophils",
        14: "? (CD45ROdim, TIM3dim, CD31dim)",
        15: "? (CD44dim, Ki67dim)",
        16: "? (CD206+, CD163+)",
        17: "? (CD4dim, pNFKbp64dim)",
        18: "? No marker ()",
        19: "Neutrophils (CD15+, CD11c+, GranzymeB+, VISTA+, S100A9+, PDL1+, CleavedCaspase3+)",
        20: "CD38 cells (CD38+)",
        21: "Stroma (Vimentin+, CD31+)",
    },
}

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
