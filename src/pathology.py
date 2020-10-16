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
    # img = image > ski.filters.threshold_multiotsu(image)[1]
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


def get_filled_lacunae_based_on_borders(
    image: Union[Path, Array], min_diam: int = 25,
) -> Array:
    if isinstance(image, Path):
        image = tifffile.imread(image.as_posix())

    # threshold, close
    img = image > ski.filters.threshold_multiotsu(image)[1]
    img = ndi.binary_fill_holes(img)
    img = ski.morphology.closing(img, ski.morphology.disk(5))
    img = ski.morphology.binary_dilation(img, selem=ski.morphology.disk(5))
    img = ndi.binary_fill_holes(img)
    # clean up small objects inside
    img = ski.morphology.remove_small_objects(img, min_size=min_diam ** 2)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image)
    axes[1].imshow(img)
    lac = ndi.label(img)[0]
    axes[2].imshow(lac)

    return img

    img = image > ski.filters.threshold_multiotsu(image)[1]
    img = ski.morphology.binary_dilation(img, selem=ski.morphology.disk(5))
    img = ndi.label(img)[0]
    max_area_percentage = 50

    remove = [
        i
        for i in np.unique(lac)
        if ((lac == i).sum() / img.size) * 100 > max_area_percentage
    ]
    if remove:
        for i in remove:
            lac[lac == i] = 0

    img = ~ski.morphology.remove_small_holes(~img, 100)
    img = ndi.binary_fill_holes(~img)


def quantify_lacunae(
    lac: Array, stack: Array, selem_diam: int = 15, only_border: bool = True
):
    objs = np.unique(lac.data)[1:]
    n_objs = len(objs)
    if n_objs == 0:
        return np.empty((0, len(stack)))
    if isinstance(n_objs, np.ma.core.MaskedConstant):
        return np.empty((0, len(stack)))
    if selem_diam > 0:
        selem = ski.morphology.disk(selem_diam)
        if np.asarray(stack.shape).argmin() == 0:
            stack = np.moveaxis(stack, 0, -1)
        res = list()
        for i in objs:
            dil = ski.morphology.binary_dilation(lac == i, selem=selem)
            if only_border:
                dil = dil.astype(int) + (lac.data == i).astype(int) == 1
            res.append(stack[dil].mean(0))
        return np.asarray(res)
    else:
        assert only_border == False
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
    # pos_mean_int = stack[labels.isin(pos_chs)].mean()
    # pos_quant = quant[pos_chs].mean(1) / pos_mean_int
    # neg_mean_int = stack[labels.isin(neg_chs)].mean()
    # neg_quant = quant[neg_chs].mean(1) / neg_mean_int
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


def get_markers_per_lacunae(
    roi: ROI, output_prefix: Path = None, overwrite: bool = False,
):
    from skimage.measure import regionprops_table

    if output_prefix is None:
        output_prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."

    comb = tifffile.imread(output_prefix + "parenchyma2.tiff")
    comb = np.ma.masked_array(comb, mask=comb == 0)
    quant_parenc_file = (
        output_prefix + "lacunae.mean_rings_combined.quantification.pq"
    )
    laoi = ["Keratin818", "AlphaSMA", "Vimentin"]
    if (not quant_parenc_file.exists()) or overwrite:
        ministack = roi._get_channels(laoi, equalize=True, minmax=True)[1]
        quant = pd.DataFrame(
            quantify_lacunae(comb, ministack, selem_diam=5, only_border=True),
            columns=laoi,
        )
        quant.index += 1
        areas = pd.Series(
            regionprops_table(comb, properties=["area"])["area"],
            index=quant.index,
            name="area",
        )
        quant = quant.join(areas)
        quant.to_parquet(quant_parenc_file)
    else:
        quant = pd.read_parquet(quant_parenc_file)
    return quant


def cluster_lacunae():
    quants = parmap.map(
        get_markers_per_lacunae, prj.rois, overwrite=False, pm_pbar=True
    )
    quants = pd.concat([q.assign(roi=r.name) for q, r in zip(quants, prj.rois)])
    quants.index.name = "obj_id"
    quants = quants.reset_index()
    quants_norm = (
        ((quants - quants.mean()) / quants.std())
        .drop(["roi", "obj_id"], axis=1)
        .join(quants[["roi", "obj_id"]])
    )

    from src.utils import z_score_by_column

    q = (
        z_score_by_column(quants[laoi + ["roi"]], "roi", ["roi"])
        .drop(["roi"], axis=1)
        .dropna()
    )
    q.index = q.index.astype(str)
    o = (
        quants[["roi", "obj_id"]]
        .join(quants_norm[["area"]])
        .merge(roi_attributes, left_on="roi", right_index=True)
    )
    o.index = o.index.astype(str)
    o = o.reindex(q.index)
    ann2 = AnnData(q, obs=o)
    sc.pp.pca(ann2)
    sc.pp.neighbors(ann2, n_neighbors=15)
    sc.tl.umap(ann2)
    sc.tl.leiden(ann2, resolution=0.1, key_added="leiden_0.1")
    sc.tl.leiden(ann2, resolution=0.5, key_added="leiden_0.5")
    sc.tl.leiden(ann2, resolution=1.0, key_added="leiden_1.0")
    clusters = ["leiden_0.5", "leiden_1.0", "leiden_0.1"]
    sc.pl.pca(ann2, color=ann2.var.index.tolist() + clusters, vmin=0, vmax=1)
    sc.pl.umap(ann2, color=ann2.var.index.tolist() + clusters, vmin=0, vmax=4)

    ann = AnnData(
        quants[laoi],
        obs=quants[["roi", "obj_id"]]
        .join(quants_norm[["area"]])
        .merge(roi_attributes, left_on="roi", right_index=True),
    )
    sc.pp.log1p(ann)
    sc.pp.scale(ann)
    sc.pp.pca(ann)
    sc.pp.neighbors(ann, n_neighbors=15)
    sc.tl.umap(ann)
    sc.tl.leiden(ann, resolution=0.5, key_added="leiden_0.5")
    sc.tl.leiden(ann, resolution=1.0, key_added="leiden_1.0")
    clusters = ["leiden_0.5", "leiden_1.0"]
    sc.pl.pca(ann, color=ann.var.index.tolist() + clusters, vmin=0, vmax=1)
    sc.pl.umap(ann, color=ann.var.index.tolist() + clusters, vmin=0, vmax=4)

    mean = ann2.to_df().groupby(ann2.obs["leiden_0.1"].values).mean()
    grid = sns.clustermap(mean)

    cs = np.unique(ann2.obs["leiden_0.1"])
    n_examples = 8

    fig1, axes1 = plt.subplots(
        len(cs), n_examples, figsize=(4 * n_examples, 4 * len(cs))
    )
    fig2, axes2 = plt.subplots(
        len(cs), n_examples, figsize=(4 * n_examples, 4 * len(cs))
    )
    for i, c in enumerate(cs):
        sel = ann2.obs.loc[
            ann2.obs["leiden_0.1"] == c, ["roi", "obj_id"]
        ].sample(n=n_examples)
        for j, (roi_name, obj_id) in enumerate(sel.values):
            roi = [r for r in prj.rois if r.name == roi_name][0]
            output_prefix = (
                roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
            )
            comb = tifffile.imread(output_prefix + "parenchyma2.tiff")
            axes1[i, j].imshow(comb == obj_id, rasterized=True)
            axes1[i, j].set(title=f"Cluster: {c}\nROI: {roi_name}")
            roi.plot_channel("mean", ax=axes2[i, j])
    for ax in axes1.flat:
        ax.axis("off")
    for ax in axes2.flat:
        ax.axis("off")
    fig1.savefig("Classes.svgz", dpi=200)
    fig2.savefig("Means.svgz", dpi=200)

    fig3, axes3 = plt.subplots(
        len(cs), n_examples, figsize=(4 * n_examples, 4 * len(cs))
    )
    for i, c in enumerate(cs):
        sel = ann2.obs.loc[
            ann2.obs["leiden_0.1"] == c, ["roi", "obj_id"]
        ].sample(n=n_examples)
        for j, (roi_name, obj_id) in enumerate(sel.values):
            print(i, j)
            roi = [r for r in prj.rois if r.name == roi_name][0]
            output_prefix = (
                roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
            )
            comb = tifffile.imread(output_prefix + "parenchyma2.tiff")
            roi.plot_channel("mean", ax=axes3[i, j])
            axes3[i, j].set(title=f"Cluster: {c}\nROI: {roi_name}")
            axes3[i, j].contour(comb == obj_id, cmap="Reds", rasterized=True)
    for ax in axes3.flat:
        ax.axis("off")
    fig3.savefig("Combined.svgz", dpi=200)


def get_rings():
    output_prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    parenc_file = output_prefix + "parenchyma2.tiff"
    comb_parenc_file = output_prefix + "lacunae.mean_rings_combined.tiff"

    mean = roi._get_channel("mean", equalize=True)[1].squeeze()
    asma = eq(roi._get_channel("AlphaSMA")[1].squeeze())
    # vim = eq(roi._get_channel("Vimentin")[1].squeeze())

    # if not parenc_file.exists():
    #     parenc = get_lacunae(mean)
    #     tifffile.imwrite(output_prefix + "parenchyma2.tiff", parenc)
    # else:
    #     parenc = tifffile.imread(output_prefix + "parenchyma2.tiff")
    # parenc = np.ma.masked_array(parenc, mask=parenc == 0)

    # if (not comb_parenc_file.exists()) or overwrite:
    #     # parenc = tifffile.imread(parenc_file)

    #     # complemet the "mean" lacunae with AlphaSMA rings
    #     ringsp = get_filled_lacunae_based_on_borders(asma, min_diam=100)
    #     ringsp = ndi.label(ringsp)[0]
    #     ringsp = np.ma.masked_array(ringsp, mask=ringsp == 0)

    #     # comb = (
    #     #     (parenc.data > 0).astype(int) + ((ringsp.data > 0)).astype(int)
    #     # ) > 0

    #     comb = parenc.copy()
    #     rs = np.unique(ringsp.data)[1:]
    #     for r in rs:
    #         # check if already existing in original parenchyma
    #         rr = ringsp.data == r
    #         if parenc.data[rr].sum() / rr.sum() < 0.1:
    #             comb[rr] = comb.max() + 1
    #     comb = np.ma.masked_array(comb, mask=comb == 0)

    #     # # Probably no AlphaSMA rings in image, use previous parenchima as total
    #     # frac = (comb > 0).sum() / np.multiply(*comb.shape)
    #     # if frac > 0.8:
    #     #     comb = parenc
    #     #     rings = ndi.label(ringsp.data < 0)[0]
    #     #     rings = np.ma.masked_array(rings, mask=rings == 0)
    #     # else:
    #     #     comb = ndi.label(comb)[0]
    #     #     rings = ringsp

    #     fig, axes = plt.subplots(1, 6, figsize=(6 * 3, 3))
    #     axes[0].imshow(mean, rasterized=True)
    #     axes[1].imshow(parenc, rasterized=True)
    #     axes[2].imshow(asma, rasterized=True)
    #     axes[3].imshow(ringsp, rasterized=True)
    #     # axes[4].imshow(rings, rasterized=True)
    #     axes[5].imshow(comb, rasterized=True)
    #     axes[0].set_title("Mean")
    #     axes[1].set_title("Parenchyma")
    #     axes[2].set_title("AlphaSMA")
    #     axes[3].set_title("Rings preliminary")
    #     axes[4].set_title("Rings")
    #     axes[5].set_title(f"Combined. f = {frac:.2f}")
    #     for ax in axes:
    #         ax.axis("off")
    #     fig.savefig(output_prefix + "lacunae.mean_rings_combined.demo.svg")

    #     # tifffile.imwrite(comb_parenc_file, comb)

    # else:
    #     comb = tifffile.imread(comb_parenc_file)
    # comb = np.ma.masked_array(comb, mask=comb == 0)


def get_cell_distance_to_vessel_airway_alveoli(
    roi, prefix: Path = None,  # plot: bool = False
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
        cells[mask == True] = 0
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

    lacunae_classification_file = (
        prefix + "lacunae.mean_rings_combined.lacunae_classification.tiff"
    )
    clas = tifffile.imread(lacunae_classification_file)

    roi._cell_mask = None
    cells = roi.mask
    lac = clas > 0
    vessel = clas == 2
    airway = clas == 3
    alveoli = clas == 1

    _r = list()
    for x in [lac, vessel, airway, alveoli]:
        roi._cell_mask = None
        _r.append(get_cell_distance_to_mask(roi.mask.copy(), x))
    r = pd.concat(_r, axis=1)
    r = r.drop(0, errors="ignore")  # background
    r.columns = ["lacunae", "vessel", "airway", "alveoli"]
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


overwrite = False
savefig = True
choi = ["Keratin818(Yb174)", "AlphaSMA(Pr141)", "Vimentin(Sm154)"]
laoi = ["Keratin818", "AlphaSMA", "Vimentin"]

# roi = [r for r in prj.rois if r.name == "20200716_COVID_1_EARLY-03"][0]
# roi = [r for r in prj.rois if r.name == "20200702_NL_ARCHOI114-05"][0]
# roi = prj.rois[2]

for roi in tqdm(prj[0].rois):
    output_prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    parenc_file = output_prefix + "parenchyma2.tiff"
    comb_parenc_file = output_prefix + "lacunae.mean_rings_combined.tiff"
    quant_parenc_file = (
        output_prefix + "lacunae.mean_rings_combined.quantification.pq"
    )
    lacunae_classification_file = (
        output_prefix
        + "lacunae.mean_rings_combined.lacunae_classification.tiff"
    )

    mean = roi._get_channel("mean", equalize=True)[1].squeeze()
    asma = eq(roi._get_channel("AlphaSMA")[1].squeeze())
    comb = tifffile.imread(parenc_file)
    comb = np.ma.masked_array(comb, mask=comb == 0)

    # Quantify markers and classify based on function
    quant = get_markers_per_lacunae(roi)
    quant_norm = (quant - quant.mean()) / quant.std()

    # Plot Rules
    fig, axes = plt.subplots(1, 2, figsize=(2 * 5, 4))
    # for i in [0, 2]:
    #     ax.axvline(i, linestyle="--", color="grey")
    # for i in [0, 1]:
    #     ax.axhline(i, linestyle="--", color="grey")
    for ax, df in zip(axes, [quant, quant_norm]):
        pts = ax.scatter(
            df["Keratin818"],
            df["AlphaSMA"],
            c=df["Vimentin"],
            s=10 + (4 ** quant_norm["area"]),
        )
        [
            ax.text(df.loc[i, "Keratin818"], df.loc[i, "AlphaSMA"], s=i,)
            for i in df.index
        ]
        ax.set(xlabel="Keratin 8/18", ylabel="Alpha SMA")
        fig.colorbar(pts, ax=ax, label="Vimentin")
    if savefig:
        fig.savefig(
            output_prefix
            + "lacunae.mean_rings_combined.lacunae_classification.quantification.scatterplot.svg"
        )
    plt.close(fig)

    # Rules
    scores = pd.DataFrame(
        index=quant.index, columns=["alveoli", "vessel", "airway"]
    ).fillna(0)

    # # absolute
    vessel_i = quant.loc[
        (quant["AlphaSMA"] > 0.1)
        & (quant["Keratin818"] < 0.1)  #  & (quant["area"] > 75000)
    ].index
    scores.loc[vessel_i, "vessel"] += 1
    airway_i = quant.loc[
        (quant["Keratin818"] > 0.1)  #  & (quant["area"] > 75000)
    ].index
    scores.loc[airway_i, "airway"] += 1
    alveol_i = quant.index[
        ~quant.index.isin(airway_i.tolist() + vessel_i.tolist())
    ]
    scores.loc[alveol_i, "alveoli"] += 1

    # # z-score
    vessel_i = quant_norm.loc[
        (quant_norm["AlphaSMA"] > 1)
        & (quant_norm["Keratin818"] < 0)  #  & (quant_norm["area"] > 0.5)
    ].index
    scores.loc[vessel_i, "vessel"] += 1
    airway_i = quant_norm.loc[
        (quant_norm["Keratin818"] > 1)  #  & (quant_norm["area"] > 0.5)
    ].index
    scores.loc[airway_i, "airway"] += 1
    alveol_i = quant_norm.index[
        ~quant_norm.index.isin(airway_i.tolist() + vessel_i.tolist())
    ]
    scores.loc[alveol_i, "alveoli"] += 1

    # # ratios
    r = quant_norm["AlphaSMA"] - quant_norm["Keratin818"]
    vessel_i = r[r >= 0.5].index
    scores.loc[vessel_i, "vessel"] += 1
    airway_i = r[r <= -0.5].index
    scores.loc[airway_i, "airway"] += 1
    alveol_i = r.index[~r.index.isin(airway_i.tolist() + vessel_i.tolist())]
    scores.loc[alveol_i, "alveoli"] += 1

    # handle draws
    for i in scores.index[(scores // 2).sum(1) == 0]:
        # # if neither, alveoli
        if (quant.loc[i, "AlphaSMA"] < 0.1) & (
            quant.loc[i, "Keratin818"] < 0.1
        ):
            scores.loc[i, "alveoli"] += 1
        elif (quant.loc[i, "AlphaSMA"] > 0.1) & (
            quant.loc[i, "Keratin818"] < 0.1
        ):
            scores.loc[i, "vessel"] += 1
        elif (quant.loc[i, "AlphaSMA"] < 0.1) & (
            quant.loc[i, "Keratin818"] > 0.1
        ):
            scores.loc[i, "airway"] += 1

    # Separate
    ass = scores.idxmax(1)
    clas = comb.copy()
    clas[np.isin(comb, ass[ass == "alveoli"].index)] = 1
    clas[np.isin(comb, ass[ass == "vessel"].index)] = 2
    clas[np.isin(comb, ass[ass == "airway"].index)] = 3
    # tifffile.imwrite(lacunae_classification_file, clas)

    # tifffile.imwrite(lacunae_classification_file, clas)

    ass = pd.Series(
        pd.Categorical(
            ass, categories=["alveoli", "vessel", "airway"], ordered=True
        ),
        index=ass.index,
        name="class",
    )
    grid = sns.clustermap(
        quant_norm,
        row_colors=scores.join(ass),
        colors_ratio=(0.05, 0),
        z_score=1,
        metric="euclidean",
        yticklabels=True,
        cbar_kws=dict(label="Z-score"),
        rasterized=True,
        figsize=(3, 8),
    )
    grid.ax_heatmap.set_yticklabels(
        grid.ax_heatmap.get_yticklabels(), fontsize=6
    )
    if savefig:
        grid.fig.savefig(
            output_prefix
            + "lacunae.mean_rings_combined.quantification.clustermap.svg",
            **figkws,
        )
    plt.close(grid.fig)

    # Plot
    vim = roi._get_channel("Vimentin", equalize=True)[
        1
    ].squeeze()  # just for plotting
    ker = roi._get_channel("Keratin", equalize=True)[
        1
    ].squeeze()  # just for plotting

    # # Plot overlay of final classification
    cmap = matplotlib.colors.ListedColormap(
        np.asarray(sns.color_palette("tab10"))[:3]
    )

    # make sure all classes are represented in one pixel (fix colors if one class is absent)
    clas[0, 0] = 1
    clas[0, -1] = 2
    clas[-1, 0] = 3
    fig, axes = plt.subplots(1, 4, figsize=(3 * 4, 3 * 1))
    axes[0].imshow(asma, rasterized=True)
    axes[1].imshow(vim, rasterized=True)
    axes[2].imshow(ker, rasterized=True)
    axes[3].imshow(clas, cmap=cmap)
    add_object_labels(comb, axes[3])
    for ax in axes:
        ax.axis("off")
    for ax, t in zip(axes, ["AlphaSMA", "Vimentin", "Keratin"]):
        ax.set_title(t)
    if savefig:
        fig.savefig(
            output_prefix
            + "lacunae.mean_rings_combined.lacunae_classification.proposal.svg",
            **figkws,
        )
    plt.close(fig)

    # # # Plot more extensive explanation
    # mean = roi._get_channel("mean", equalize=True)[
    #     1
    # ].squeeze()  # just for plotting
    # fig, axes = plt.subplots(3, 4, figsize=(3 * 4, 3 * 3))
    # axes[0][0].imshow(mean, rasterized=True)
    # axes[0][1].imshow(asma, rasterized=True)
    # axes[0][2].imshow(vim, rasterized=True)
    # axes[0][3].imshow(ker, rasterized=True)
    # axes[1][0].imshow(parenc, cmap="coolwarm", rasterized=True)
    # axes[1][1].imshow(rings, cmap="coolwarm", rasterized=True)
    # axes[1][2].imshow(comb, cmap="coolwarm", rasterized=True)
    # add_object_labels(comb, axes[1][2])
    # axes[2][0].imshow(vessels, cmap="tab20", rasterized=True)
    # add_object_labels(vessels, axes[2][0])
    # axes[2][1].imshow(airways, cmap="tab20", rasterized=True)
    # add_object_labels(airways, axes[2][1])
    # axes[2][2].imshow(alveoli, cmap="tab20", rasterized=True)
    # add_object_labels(alveoli, axes[2][2])
    # for ax in axes.ravel():
    #     ax.axis("off")
    # names = [
    #     "Mean",
    #     "AlphaSMA",
    #     "Vimentin",
    #     "Keratin 8/18",
    #     "Lacunae (from mean)",
    #     "Lacunae (from AlphaSMA rings)",
    #     "All lacunae",
    #     "",
    #     "Vessels",
    #     "Airways",
    #     "Alveoli",
    # ]
    # for ax, name in zip(axes.flat, names):
    #     ax.set_title(name)
    # fig.savefig(
    #     output_prefix
    #     + "lacunae.mean_rings_combined.lacunae_classification.explanation.svg",
    #     **figkws,
    # )


manual_annot = pd.read_csv(
    metadata_dir / "vessel_airway.manual_annotation.csv", index_col=0
)
manual_annot = manual_annot.query("include == True")
for roi in tqdm([r for r in prj.rois if r.name in manual_annot.index]):
    output_prefix = roi.prj.results_dir / "qc" / "lacunae" / roi.name + "."
    parenc_file = output_prefix + "parenchyma2.tiff"
    comb_parenc_file = output_prefix + "lacunae.mean_rings_combined.tiff"
    quant_parenc_file = (
        output_prefix + "lacunae.mean_rings_combined.quantification.pq"
    )
    lacunae_classification_file = (
        output_prefix
        + "lacunae.mean_rings_combined.lacunae_classification.tiff"
    )
    comb = tifffile.imread(parenc_file)
    comb = np.ma.masked_array(comb, mask=comb == 0)

    # Assign classes
    clas = comb.copy()
    clas[clas > 0] = 1
    v = manual_annot.loc[roi.name, ["vessel"]]
    if not pd.isnull(v).any():
        v = (
            pd.Series(v.astype(str).str.split(",").squeeze())
            .astype(float)
            .astype(int)
            .tolist()
        )
        clas[np.isin(comb, v)] = 2
    v = manual_annot.loc[roi.name, ["airway"]]
    if not pd.isnull(v).any():
        v = (
            pd.Series(v.astype(str).str.split(",").squeeze())
            .astype(float)
            .astype(int)
            .tolist()
        )
        clas[np.isin(comb, v)] = 3

    tifffile.imwrite(lacunae_classification_file, clas)

    # Plot
    mean = roi._get_channel("mean", equalize=True)[1].squeeze()
    asma = eq(roi._get_channel("AlphaSMA")[1].squeeze())
    vim = roi._get_channel("Vimentin", equalize=True)[
        1
    ].squeeze()  # just for plotting
    ker = roi._get_channel("Keratin", equalize=True)[
        1
    ].squeeze()  # just for plotting

    # # Plot overlay of final classification
    cmap = matplotlib.colors.ListedColormap(
        np.asarray(sns.color_palette("tab10"))[:3]
    )

    # make sure all classes are represented in one pixel (fix colors if one class is absent)
    clas[0, 0] = 1
    clas[0, -1] = 2
    clas[-1, 0] = 3
    fig, axes = plt.subplots(1, 5, figsize=(3 * 5, 3 * 1))
    axes[0].imshow(mean, rasterized=True)
    axes[1].imshow(asma, rasterized=True)
    axes[2].imshow(vim, rasterized=True)
    axes[3].imshow(ker, rasterized=True)
    axes[4].imshow(clas, cmap=cmap)
    add_object_labels(comb, axes[4])
    for ax in axes:
        ax.axis("off")
    for ax, t in zip(
        axes,
        ["Channel mean", "AlphaSMA", "Vimentin", "Keratin", "Classification"],
    ):
        ax.set_title(t)
    if savefig:
        fig.savefig(
            output_prefix
            + "lacunae.mean_rings_combined.lacunae_classification.final.svg",
            **figkws,
        )
    plt.close(fig)


# Measure distance to structures for each cell
_dists = list()
for roi in tqdm([r for r in prj.rois if r.name in manual_annot.index]):
    _dists.append(get_cell_distance_to_vessel_airway_alveoli(roi))
dists = pd.concat(_dists)
dists.to_parquet(
    results_dir
    / "pathology"
    / "cell_distance_to_lacunae.vessel_airway_alveoli.pq"
)


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
# for roi in prj.rois:
#     get_cell_distance_to_lung_lacunae(roi)

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
