#!/usr/bin/env python

"""
Spatial analysis of lung tissue
"""

from typing import Union, List
from functools import partial

import parmap  # type: ignore[import]
import scipy.ndimage as ndi  # type: ignore[import]
import skimage as ski  # type: ignore[import]
import skimage.feature  # type: ignore[import]
from skimage.exposure import equalize_hist as eq  # type: ignore[import]
import tifffile  # type: ignore[import]
import pingouin as pg  # type: ignore[import]
import numpy_groupies as npg  # type: ignore[import]

from imc import ROI
from seaborn_extensions import swarmboxenplot
from imc.graphics import get_grid_dims
from imc.types import Path, Array, DataFrame, Axis, Figure


from src.config import *


def get_lacunae(
    image: Union[Path, Array],
    selem_diam: int = 5,
    min_diam: int = 25,
    max_area_percentage: float = 50,
) -> Array:
    if isinstance(image, Path):
        image = tifffile.imread(image.as_posix())

    # threshold, close
    img = image > ski.filters.threshold_otsu(image)
    img = ski.morphology.binary_dilation(
        img, selem=ski.morphology.disk(selem_diam)
    )
    img = ski.morphology.closing(img, ski.morphology.disk(5))

    # clean up small objects inside
    img = ~ndi.binary_fill_holes(~img)
    img = ~ski.morphology.remove_small_objects(~img, min_size=min_diam ** 2)

    lac = ndi.label(~img)[0]

    # remove objects too large
    remove = [
        i
        for i in np.unique(lac)
        if ((lac == i).sum() / img.size) * 100 > max_area_percentage
    ]
    if remove:
        for i in remove:
            lac[lac == i] = 0
    return lac


def quantify_lacunae(lac: Array, stack: Array, selem_diam: int = 15):
    n_objs = lac.max()
    if isinstance(n_objs, np.ma.core.MaskedConstant):
        return np.empty((0, len(stack)))
    objs = range(1, n_objs + 1)
    if selem_diam > 0:
        selem = ski.morphology.disk(selem_diam)
        if np.asarray(stack.shape).argmin() == 0:
            stack = np.moveaxis(stack, 0, -1)
        res = list()
        for i in objs:  # skip background 0
            dil = ski.morphology.binary_dilation(lac == i, selem=selem)
            res.append(stack[dil].mean(0))
        return np.asarray(res)
    return np.asarray(
        [stack[j][lac == i].mean() for j in range(stack.shape[0]) for i in objs]
    ).reshape((n_objs, len(stack)))


def add_object_labels(label, ax=None):  # cmap=None,
    if ax is None:
        ax = plt.gca()
    index = np.unique(label)
    centroids = ndi.center_of_mass(
        np.ones_like(label), labels=label, index=index
    )
    # cmap = plt.get_cmap(cmap)
    for i, (y, x) in zip(index, centroids):
        ax.text(x, y, s=i)  # , color=cmap(i))


def get_lung_lacunae(
    roi: ROI,
    pos_chs: List[str] = ["AlphaSMA(Pr141)"],
    # important! sort channels
    neg_chs: List[str] = ["CD31(Eu151)", "Keratin818(Yb174)"],
    boundary: float = 0.1,
    plot: bool = True,
    return_masks: bool = True,
    output_prefix: Path = None,
    overwrite: bool = False,
):
    if output_prefix is None:
        output_prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    output_figure = output_prefix + "lacunae.svg"
    output_figure.parent.mkdir()

    parenc_file = output_prefix + "parenchyma.tiff"
    pos_file = output_prefix + "pos_lacunae.tiff"
    neg_file = output_prefix + "neg_lacunae.tiff"

    if neg_file.exists() and not overwrite:
        return

    # Get lacunae based on channel mean
    # # one could use any single channel too, just make sure to max normalize
    stack = roi.stack[roi.channel_labels.isin(channels_include)]
    stack = np.asarray([eq(c) for c in stack])
    labels = roi.channel_labels[roi.channel_labels.isin(channels_include)]
    mean = stack.mean(0)
    mean_lac = get_lacunae(mean, min_diam=25)
    mean_lac = np.ma.masked_array(mean_lac, mask=mean_lac == 0)

    # Quantify lacunae in relevant channels
    chs = pos_chs + neg_chs
    stack_mini = stack[labels.isin(chs)]
    labels_mini = labels[labels.isin(chs)]
    quant = pd.DataFrame(
        quantify_lacunae(mean_lac, stack_mini, selem_diam=10), columns=chs
    )
    quant = quant / quant.mean()
    quant.index += 1

    # Get ratio between AlphaSMA and Keratin to distinguish
    ratio = np.log(quant[pos_chs].mean(1) / quant[neg_chs].mean(1)) * 10

    # plt.scatter(ratio.rank(), ratio)

    # # select Alpha lacunae
    pos_lac = ratio[ratio > boundary].index.values
    mean_pos_lac = mean_lac.copy()
    mean_pos_lac[~np.isin(mean_pos_lac, pos_lac)] = 0
    mean_pos_lac = np.ma.masked_array(mean_pos_lac, mask=mean_pos_lac == 0)

    # # select Keratin lacunae
    neg_lac = ratio[ratio < boundary].index.values
    mean_neg_lac = mean_lac.copy()
    mean_neg_lac[~np.isin(mean_neg_lac, neg_lac)] = 0
    mean_neg_lac = np.ma.masked_array(mean_neg_lac, mask=mean_neg_lac == 0)

    # Get parenchyma as the remaining
    parenc = np.ma.masked_array(np.asarray(mean_lac), mask=~mean_lac.mask)

    tifffile.imwrite(parenc_file, parenc)
    tifffile.imwrite(pos_file, mean_pos_lac)
    tifffile.imwrite(neg_file, mean_neg_lac)
    ret = (parenc, mean_pos_lac, mean_neg_lac) if return_masks else None
    if not plot:
        return ret

    # Plot
    # # Demo plot with channel and lacunae images
    vmax = mean_lac.max()
    fig, axes = plt.subplots(2, 4, figsize=(4 * 4, 4 * 2))
    axes = axes.flatten()
    axes[0].imshow(mean, rasterized=True)
    axes[0].set(title="Channel Mean")
    axes[0].axis("off")
    roi.plot_channel("DNA", ax=axes[1])
    axes[2].imshow(stack_mini[labels_mini.isin(pos_chs)].mean(0))
    axes[2].set(title=", ".join(pos_chs))
    axes[2].axis("off")
    axes[3].imshow(stack_mini[labels_mini.isin(neg_chs)].mean(0))
    axes[3].set(title=", ".join(neg_chs))
    axes[3].axis("off")
    axes[4].imshow(parenc, cmap="coolwarm", rasterized=True)
    axes[4].set(title="Parenchyma")
    axes[4].axis("off")
    kwargs = dict(cmap="coolwarm", vmax=vmax, rasterized=True)
    axes[5].imshow(mean_lac, **kwargs)
    axes[5].set(title="All lacunae")
    axes[5].axis("off")
    add_object_labels(mean_lac, ax=axes[5])
    axes[6].imshow(mean_pos_lac, **kwargs)
    axes[6].set(title=f"{', '.join(pos_chs)} lacunae")
    axes[6].axis("off")
    add_object_labels(mean_pos_lac, ax=axes[6])
    axes[7].imshow(mean_neg_lac, **kwargs)
    axes[7].set(title=f"{', '.join(neg_chs)} lacunae")
    axes[7].axis("off")
    add_object_labels(mean_neg_lac, ax=axes[7])
    fig.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ret

    # grid = sns.clustermap(np.log1p(quant), standard_scale=1)
    # grid = sns.clustermap(quant, z_score=1, cmap="RdBu_r", center=0)
    # grid = sns.clustermap(
    #     quant.loc[:, quant.columns.str.contains("|".join(chs))],
    #     metric="correlation",
    #     standard_scale=1,
    # )


def summarize_lung_lacunae(roi, prefix: Path = None):
    def summarize(lac):
        objs = np.unique(lac)[1:]
        n = len(objs)
        if n == 0:
            return [0.0] * 5
        areas = np.asarray([((lac == i).sum() / lac.size) * 100 for i in objs])
        return [
            n,
            areas.sum(),
            areas.mean(),
            areas.max(),
            # mean ratio of largest to other lacunae
            (areas.max() / np.delete(areas, areas.argmax())).mean()
            if n > 1
            else np.nan,
        ]

    if prefix is None:
        prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    # parenc_file = prefix + "parenchyma.tiff"
    pos_file = prefix + "pos_lacunae.tiff"
    neg_file = prefix + "neg_lacunae.tiff"

    pos = tifffile.imread(pos_file)
    neg = tifffile.imread(neg_file)
    index = pd.Series(
        ["number", "area", "mean_area", "max_area", "mean_maxratio"]
    )
    return pd.Series(
        summarize(pos + neg) + summarize(pos) + summarize(neg),
        index=("lacunae_" + index).tolist()
        + ("pos_lacunae_" + index).tolist()
        + ("neg_lacunae_" + index).tolist(),
        name=roi.name,
    )


def get_cell_distance_to_lung_lacunae(
    roi: ROI, prefix: Path = None,  # plot: bool = False
) -> "DataFrame":
    def get_cell_distance_to_mask(cells, mask):
        if mask.sum() == 0:
            return (
                pd.Series(dtype=float, index=np.unique(cells)[1:])
                .rename(roi.name)
                .rename_axis("cell")
            )
        # fill in the gaps between the cells
        max_i = 2 ** 16 - 1
        cells[cells == 0] = max_i
        # set parenchyma to zero
        cells[mask > 0] = 0
        # get distance to parenchyma (zero)
        dist = ndi.distance_transform_edt(cells > 0)
        # reduce per cell
        return (
            (
                pd.Series(
                    npg.aggregate(
                        cells.ravel(),
                        dist.ravel(),
                        func="mean",
                        fill_value=np.nan,
                    )
                )
                .dropna()
                .drop(max_i)
            )
            .rename(roi.name)
            .rename_axis("cell")
        )

    if prefix is None:
        prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    parenc_file = prefix + "parenchyma.tiff"
    pos_file = prefix + "pos_lacunae.tiff"
    neg_file = prefix + "neg_lacunae.tiff"

    cells = roi.mask
    roi._cell_mask = None
    parenc = tifffile.imread(parenc_file)
    pos = tifffile.imread(pos_file)
    neg = tifffile.imread(neg_file)

    r = pd.concat(
        [
            get_cell_distance_to_mask(cells.copy(), x)
            for x in [parenc, pos, neg]
        ],
        axis=1,
    )
    r = r.drop(0, errors="ignore")  # background
    r.columns = ["parenchyma_lacunae", "pos_lacunae", "neg_lacunae"]
    r["sample"] = roi.sample.name
    r["roi"] = roi.name

    # if not plot:
    #     return r
    # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    # axes[0].imshow(cells, rasterized=True)
    # axes[1].imshow(parenc, rasterized=True)
    # axes[2].imshow(dist, rasterized=True)
    # for ax in axes:
    #     ax.axis("off")
    # fig.savefig(prefix + "cell_distance_to_lacunae.svg", **figkws)

    return r


def get_fraction_with_marker(roi: ROI, marker: str) -> float:
    x = np.log1p(roi._get_channel(marker)[1].squeeze())
    area = np.multiply(*roi.shape[1:])
    return (x > ski.filters.threshold_otsu(x)).sum() / area


# WIP:
# def get_microthrombi(roi):
#     roi = prj.rois[61]

#     prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
#     parenc_file = prefix + "parenchyma.tiff"
#     pos_file = prefix + "pos_lacunae.tiff"
#     neg_file = prefix + "neg_lacunae.tiff"

#     lac = tifffile.imread(pos_file)


#     col = roi._get_channel("Collagen")[1].squeeze()
#     sma = roi._get_channel("AlphaSMA")[1].squeeze()

#     selem_diam = 20
#     dil = ski.morphology.binary_dilation(
#         sma > ski.filters.threshold_otsu(sma),
#         selem=ski.morphology.disk(selem_diam),
#     )
#     dil[0, :] = True
#     dil[-1, :] = True
#     dil[:, 0] = True
#     dil[:, -1] = True
#     img = ndi.label(dil)[0]
#     img = ~ndi.binary_fill_holes(~dil)

#     img = ski.morphology.closing(dil, ski.morphology.disk(5))

#     # clean up small objects inside
#     img = ~ski.morphology.remove_small_objects(~img, min_size=10 ** 2)
#     img = ski.morphology.binary_dilation(
#         img, ski.morphology.disk(selem_diam / 2)
#     )

#     img = ~ndi.binary_fill_holes(~img)
#     img = ~ski.morphology.remove_small_objects(~img, min_size=min_diam ** 2)

#     lac = ndi.label(~img)[0]


#     lac = ndi.label(~img)[0]
#     lac[lac == 1] = 0

#     fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 4))
#     axes[0].imshow(np.log1p(sma))
#     axes[1].imshow(dil)
#     axes[2].imshow(lac)

#     # remove objects too large
#     remove = [
#         i
#         for i in np.unique(lac)
#         if ((lac == i).sum() / img.size) * 100 > max_area_percentage
#     ]


output_dir = results_dir / "pathology"
output_dir.mkdir()


# Get lacunae
rois = [
    r
    for r in prj.rois
    if not (
        r.prj.results_dir / "qc" / "lacunae" / r.name + ".parenchyma.tiff"
    ).exists()
]
if rois:
    parmap.map(
        get_lung_lacunae,
        rois,
        plot=True,
        return_masks=False,
        overwrite=True,
        pm_pbar=True,
    )

# measurements per image

# # number of lacunae, lacunae average area,
# # number of lacunae of a given type, area of a given type
# #

lacunae_quantification_file = (
    output_dir / "lacunae.quantification_per_image.csv"
)

if not lacunae_quantification_file.exists():
    res = pd.concat(
        parmap.map(summarize_lung_lacunae, prj.rois, pm_pbar=True), axis=1
    ).T
    res.index.name = "roi"
    res.to_csv(lacunae_quantification_file)
res = pd.read_csv(lacunae_quantification_file, index_col=0)

# add jitter
res2 = res.fillna(0) + np.random.random(res.shape) * 1e-25

# Visualize all data as heatmap
grid = sns.clustermap(
    res2, metric="euclidean", standard_scale=1, row_colors=roi_attributes,
)
grid.savefig(output_dir / "lacunae.clustermap.euclidean.svg", **figkws)
plt.close(grid.fig)

grid = sns.clustermap(
    res2, metric="correlation", standard_scale=1, row_colors=roi_attributes,
)
grid.savefig(output_dir / "lacunae.clustermap.correlation.svg", **figkws)
plt.close(grid.fig)

# Boxen-swarm plots
res2 = res.reset_index().melt(id_vars="roi").set_index("roi")

_test_res = list()
variables = res2["variable"].unique()
for col in ["disease", "phenotypes"]:
    n, m = get_grid_dims(len(variables))
    fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4))
    for ax, val in zip(axes.flat, variables):
        data = res2.query(f"variable == '{val}'").join(roi_attributes[col])
        stats = swarmboxenplot(
            data=data,
            x=col,
            y="value",
            ax=ax,
            test_kws=dict(parametric=False),
            plot_kws=dict(palette=colors.get(col)),
        )
        _test_res.append(stats.assign(grouping=col, variable=val))
        ax.set_title(val)
    fig.savefig(output_dir / f"lacunae.{col}.swarmboxenplot.svg", **figkws)
    plt.close(fig)

test_res = pd.concat(_test_res)
test_res.to_csv(output_dir / "lacunae.test_results.csv", index=False)


# Get the distance of each cell to these structures
for roi in prj.rois:
    get_cell_distance_to_lung_lacunae(roi)

dists = pd.concat(
    parmap.map(get_cell_distance_to_lung_lacunae, prj.rois, pm_pbar=True),
    axis=0,
)
dists.to_parquet(output_dir / "cell_distance_to_lacunae.pq")


# Fibrosis
collagen_channel = "CollagenTypeI(Tm169)"

fibrosis_score_file = (
    output_dir / "fibrosis.extent_and_intensity.quantification.csv"
)

if not fibrosis_score_file.exists():
    summary = pd.read_csv(
        qc_dir / prj.name + ".channel_summary.csv", index_col=0
    )
    fib = summary.loc[collagen_channel].to_frame().join(roi_attributes)
    colarea = parmap.map(
        get_fraction_with_marker, prj.rois, marker=collagen_channel
    )
    colarea = pd.Series(
        colarea, index=[roi.name for roi in prj.rois], name="fraction"
    )
    fib = fib.join(colarea).rename(columns={collagen_channel: "intensity"})
    score = (
        pd.DataFrame(
            [
                # (fib[var] - fib[var].min()) / (fib[var].max() - fib[var].min())
                (fib[var] - fib[var].mean()) / fib[var].std()
                for var in ["fraction", "intensity"]
            ]
        )
        .mean()
        .rename("score")
    )
    fib = fib.join(score)
    # Use parquet to keep categorical order
    fib.to_csv(fibrosis_score_file)
    fib.to_parquet(fibrosis_score_file.as_posix().replace(".csv", ".pq"))
fib = pd.read_parquet(fibrosis_score_file.as_posix().replace(".csv", ".pq"))

# get mean per sample
fib_sample = (
    fib.groupby(fib.index.str.extract("(.*)-")[0].values)[
        ["fraction", "intensity", "score"]
    ]
    .mean()
    .join(sample_attributes)
)


_fib_stats = list()
for label, data in [("roi", fib), ("sample", fib_sample)]:
    for grouping in ["disease", "phenotypes"]:
        for var in ["intensity", "fraction", "score"]:
            fig, stats = swarmboxenplot(
                data=data,
                x=grouping,
                y=var,
                plot_kws=dict(palette=colors.get(grouping)),
            )
            fig.savefig(
                output_dir / f"fibrosis.by_{label}.{var}.by_{grouping}.svg",
                **figkws,
            )
            plt.close(fig)
            _fib_stats.append(stats.assign(group=label, variable=var))
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        sns.scatterplot(
            data=data,
            x="intensity",
            y="fraction",
            hue=grouping,
            hue_order=fib[grouping].cat.categories,
            palette=colors.get(grouping),
            alpha=0.75,
            s=20,
            ax=ax,
        )
        ax.set(
            xlabel="Collagen intensity",
            ylabel="Collagen extent\n(Fraction of image)",
        )
        fig.savefig(
            output_dir
            / f"fibrosis.extent_vs_intensity.by_{label}.by_{grouping}.svg",
            **figkws,
        )
        plt.close(fig)

fib_stats = pd.concat(_fib_stats, axis=0)
fib_stats.to_csv(
    output_dir / "fibrosis.extent_and_intensity.differential_tests.csv",
    index=False,
)

# Microthrombi
