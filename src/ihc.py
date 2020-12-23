# coding: utf-8

"""
This script loads H-DAB images, segments
and quantifies positive cells per image.
"""

import io
import sys
import json
from typing import Tuple, Dict
from functools import lru_cache as cache

from tqdm import tqdm
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.color import rgb2hed
from skimage import filters
import tifffile

from imc.types import DataFrame, Path, Series
from imc.operations import get_population
from imc.segmentation import normalize
from imc.utils import minmax_scale

from seaborn_extensions import swarmboxenplot

from boxsdk import OAuth2, Client, BoxOAuthException  # OAuth2, JWTAuth

SECRETS_FILE = Path("~/.imctransfer.auth.json").expanduser().absolute()
figkws = dict(dpi=300, bbox_inches="tight")

metadata_dir = Path("metadata")
metadata_dir.mkdir(exist_ok=True)
data_dir = Path("data") / "ihc"
data_dir.mkdir(exist_ok=True)
results_dir = Path("results") / "ihc"
results_dir.mkdir(exist_ok=True)


def main() -> int:
    # Get file list
    files_json = metadata_dir / "ihc_files.box_dir.json"
    if not files_json.exists():
        files = get_urls()
        json.dump(files, open(files_json, "w"), indent=4)
    files = json.load(open(files_json, "r"))

    # get, parse and assemble data from "source data" files
    quant: Dict[str, Dict[str, float]] = dict()
    for sf in tqdm(files, desc="subfolder"):
        if sf not in quant:
            quant[sf] = dict()
        for file, url in tqdm(files[sf].items(), desc="image"):
            if file not in quant[sf]:
                img = get_image(url)
                quant[sf][file] = simple_quantify_hed_colors(img)
    res = pd.concat(
        [pd.DataFrame(v).T.assign(folder=k) for k, v in quant.items()]
    )
    res.to_csv(results_dir / "simple_quantify_hd_color_quantification.csv")

    res = pd.read_csv(
        results_dir / "simple_quantify_hd_color_quantification.csv", index_col=0
    )

    name = res["diaminobenzidine"].sort_values().tail(1).index[0]
    img = get_image(files["MPO"][name])

    plot_res(res)

    # Segment with stardist
    # # the "he_heavy_augment" model is really good with H&E
    # # but is not available in the Python API, so I use ImageJ
    # # However, for some reason the ImageJ plugin does not return
    # # the output image correctly when run as a macro through the CLI,
    # # so I resort to running commands in ImageJ editor - don't know
    # # why but that works
    segment_stardist_imagej(files)
    qp = quantify_segmentation(files)
    plot_segmentation(qp)
    return 0


# def segmentation(files):
#     # Segment
#     import scipy.ndimage as ndi
#     from stardist.models import StarDist2D

#     model = StarDist2D.from_pretrained("2D_versatile_he")

#     # get, parse and assemble data from "source data" files
#     for sf in tqdm(files, desc="subfolder"):
#         for file, url in tqdm(files[sf].items(), desc="image"):
#             f = data_dir / sf / file
#             mask_file = f.replace_(".tif", ".mask.tiff")
#             try:
#                 img = tifffile.imread(f)
#             except:
#                 img = get_image(url)
#             if not mask_file.exists():
#                 mask, _ = model.predict(img)
#                 t = filters.threshold_otsu(mask, 21)
#                 mask = ndi.label(mask > t)[0]
#                 tifffile.imwrite(mask_file, mask)
#             else:
#                 mask = tifffile.imread(mask_file)
#             fig = seg(img, mask)
#             fig.savefig(
#                 mask_file.replace_(".tiff", ".svg"),
#                 **figkws,
#             )
#             plt.close("all")


def segmentation(files):
    # get, parse and assemble data from "source data" files
    for sf in tqdm(files, desc="subfolder"):
        for file, url in tqdm(files[sf].items(), desc="image"):
            f = data_dir / sf / file
            mask_file = f.replace_(".tif", ".stardist_mask.tiff")
            try:
                img = tifffile.imread(f)
            except:
                img = get_image(url)
            mask = tifffile.imread(mask_file)
            fig = seg(img, mask)
            fig.savefig(
                mask_file.replace_(".tiff", ".svg"),
                **figkws,
            )
            plt.close("all")


def quantify_cell_intensity(stack, mask, red_func="mean"):
    cells = np.unique(mask)
    # the minus one here is to skip the background "0" label which is also
    # ignored by `skimage.measure.regionprops`.
    n_cells = len(cells) - 1
    n_channels = stack.shape[0]

    res = np.zeros(
        (n_cells, n_channels), dtype=int if red_func == "sum" else float
    )
    for channel in np.arange(stack.shape[0]):
        res[:, channel] = [
            getattr(x.intensity_image, red_func)()
            for x in skimage.measure.regionprops(mask, stack[channel])
        ]
    return pd.DataFrame(res, index=cells[1:])


def seg(img, mask):
    from imc.graphics import get_random_label_cmap

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2 * 4, 2 * 4),
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
        sharex="col",
        sharey="col",
    )
    axes[0].imshow(img, rasterized=True)
    axes[1].imshow(mask, rasterized=True, cmap=get_random_label_cmap())
    for ax in axes:
        ax.axis("off")
    return fig


def quantify_segmentation(files):
    # get, parse and assemble data from "source data" files
    labels = ["hematoxilyn", "diaminobenzidine"]
    _quants = list()
    for sf in tqdm(files, desc="subfolder"):
        for file, url in tqdm(files[sf].items(), desc="image"):
            f = data_dir / sf / file
            mask_file = f.replace_(".tif", ".stardist_mask.tiff")
            mask = tifffile.imread(mask_file)
            # mask = ndi.label(ndi.zoom(mask > 0, (2, 2)))[0]
            try:
                img = tifffile.imread(f)
            except:
                img = get_image(url)
            ihc = np.moveaxis(rgb2hed(img), -1, 0)
            ihc = np.stack([minmax_scale(ihc[0]), minmax_scale(ihc[2])])
            q = quantify_cell_intensity(ihc, mask)
            q.columns = labels
            _quants.append(q.assign(marker=sf, image=file))
    quants = pd.concat(_quants)
    quants.to_csv(results_dir / "ihc.stardist.quantification.csv")

    # threshold positivity
    t = get_population(quants["diaminobenzidine"])
    quants["dab_pos"] = t
    total = quants.groupby(["marker", "image"]).size()
    pos = quants.groupby(["marker", "image"])["dab_pos"].sum()
    perc = (pos / total) * 100

    mean_quant = quants.groupby(["marker", "image"]).mean().loc["MPO"]

    meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq")
    annot = pd.DataFrame(
        map(pd.Series, quants["image"].str.replace("x -", "x-").str.split(" ")),
    )
    annot.index = quants["image"].rename("image")
    annot.columns = ["disease", "wcmc_code", "location", "magnification"]
    annot["wcmc_code"] = "WCMC" + annot["wcmc_code"].str.zfill(3)

    annot = annot.reset_index().merge(meta, on="wcmc_code")

    for c in annot:
        if annot[c].dtype.name == "category":
            annot[c] = annot[c].cat.remove_unused_categories()

    q = mean_quant.reset_index().merge(
        annot[["image", "sample_name", "phenotypes"]].drop_duplicates(),
        on="image",
    )

    qp = (
        perc.rename("diaminobenzidine")
        .reset_index()
        .merge(
            annot[["image", "sample_name", "phenotypes"]].drop_duplicates(),
            on="image",
        )
    )
    fig, stats = swarmboxenplot(
        data=q,
        x="phenotypes",
        y="diaminobenzidine",
    )
    fig.axes[0].set(ylabel="MPO intensity")
    fig.savefig(
        results_dir / "ihc.stardist_segmentation.MPO_cells.intensity.svg",
        **figkws,
    )
    fig, stats = swarmboxenplot(
        data=qp,
        x="phenotypes",
        y="diaminobenzidine",
    )
    fig.axes[0].set(ylabel="MPO positive cells (%)")
    fig.savefig(
        results_dir / "ihc.stardist_segmentation.MPO_cells.percentage.svg",
        **figkws,
    )

    return qp


def plot_segmentation(qp):
    from imc.graphics import get_random_label_cmap

    s = qp.sort_values("diaminobenzidine")
    images = s.head(3)["image"].tolist() + s.tail(3)["image"].tolist()
    n_top = len(images)

    fig, axes = plt.subplots(
        2,
        n_top,
        figsize=(n_top * 4, 2 * 4),
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
        sharex="col",
        sharey="col",
    )
    for axs, name in zip(axes.T, images):
        idx = qp["diaminobenzidine"].sort_values().tail(n_top).index
        f = data_dir / sf / name
        img = tifffile.imread(f)
        mask_file = f.replace_(".tif", ".stardist_mask.tiff")
        mask = tifffile.imread(mask_file)
        axs[0].imshow(img, rasterized=True)
        axs[1].imshow(mask, rasterized=True, cmap=get_random_label_cmap())
        for ax in axs:
            ax.axis("off")
    fig.savefig(
        results_dir / "he_dab.stardist_segmentation.illustration.svg",
        **figkws,
    )


def download_all_files(files):
    # Download
    for direct in files:
        (data_dir / direct).mkdir()
        for file, url in files[direct].items():
            img = get_image(url)
            tifffile.imwrite(data_dir / direct / file, img)


urls = [
    "https://wcm.box.com/shared/static/p7456xiws8mtqt2in09ju3459k4lvdpb.tif",
    "https://wcm.box.com/shared/static/p1o8ytt2c3zn2gkkvsxgg14iviirt0ov.tif",
    "https://wcm.box.com/shared/static/8ql0u7ki7wbiyo0uh8sjdyv56r0ud80i.tif",  #
]


def get_urls(query_string="", file_type="tif"):
    secret_params = json.load(open(SECRETS_FILE, "r"))
    oauth = OAuth2(**secret_params)
    client = Client(oauth)

    folder = client.folder("128411248991")
    subfolders = folder.get_items()

    files = dict()
    for sf in subfolders:
        files[sf.name] = dict()
        for image in sf.get_items():
            files[sf.name][image.name] = image.get_shared_link_download_url()

    return files


@cache
def get_image(url=None):
    if url is None:
        url = urls[0]
    with requests.get(url) as req:
        return tifffile.imread(io.BytesIO(req.content))


def simple_quantify_hed_colors(img):
    """"""
    size = np.multiply(*img.shape[:2])
    ihc = np.moveaxis(rgb2hed(img), -1, 0)

    labels = ["hematoxilyn", "eosin", "diaminobenzidine"]
    res = dict()
    for im, label in zip(ihc, labels):
        im = normalize(im)
        # Filter illumination
        t = im > filters.threshold_local(im, 21)
        res[label] = t.sum() / size
    return res


def plot_res(res):

    meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq")
    annot = pd.DataFrame(
        map(pd.Series, res.index.str.replace("x -", "x-").str.split(" ")),
    )
    annot.index = res.index.rename("image")
    annot.columns = ["disease", "wcmc_code", "location", "magnification"]
    annot["wcmc_code"] = "WCMC" + annot["wcmc_code"].str.zfill(3)

    annot = annot.reset_index().merge(meta, on="wcmc_code").set_index("image")

    for c in annot:
        if annot[c].dtype.name == "category":
            annot[c] = annot[c].cat.remove_unused_categories()

    # grid = clustermap(res.drop(["folder", "eosin"], 1), z_score=1)

    res["ratio"] = res["diaminobenzidine"] / res["hematoxilyn"]
    res["fraction"] = res["diaminobenzidine"] / res[
        ["hematoxilyn", "diaminobenzidine", "eosin"]
    ].sum(1)

    fig, stats = swarmboxenplot(
        data=res.join(annot),
        x="phenotypes",
        y="diaminobenzidine",
    )
    fig.savefig(
        results_dir / "he_dab.color_quantification.swarmboxenplot.svg", **figkws
    )

    fig, stats = swarmboxenplot(
        data=res.join(annot),
        x="wcmc_code",
        y="diaminobenzidine",
        plot_kws=dict(palette="Set2"),
    )
    fig.savefig(
        results_dir
        / "he_dab.color_quantification.by_patient.swarmboxenplot.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=res.join(annot),
        x="phenotypes",
        y="diaminobenzidine",
        hue="location",
    )
    fig.savefig(
        results_dir
        / "he_dab.color_quantification.by_location.swarmboxenplot.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=res.join(annot),
        x="wcmc_code",
        y="diaminobenzidine",
        hue="location",
    )
    fig.savefig(
        results_dir
        / "he_dab.color_quantification.by_patient_location.swarmboxenplot.svg",
        **figkws,
    )

    # Plot a few examples
    n_top = 5
    desc = {pd.Series.tail: "most", pd.Series.head: "least"}
    fig, axes = plt.subplots(
        2,
        n_top,
        figsize=(n_top * 4, 2 * 4),
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for axs, fn in zip(axes, [pd.Series.tail, pd.Series.head]):
        idx = fn(res["diaminobenzidine"].sort_values(), n_top).index
        for i, name in enumerate(idx):
            img = get_image(files["MPO"][name])
            axs[i].imshow(img, rasterized=True)
            axs[i].set(
                title=name, xticks=[], yticks=[], xticklabels=[], yticklabels=[]
            )
            # axs[i].axis("off")
        axs[0].set_ylabel(f"Top {n_top} images with {desc[fn]} DAB.")
    fig.savefig(
        results_dir / "he_dab.color_quantification.top_illustration.svg",
        **figkws,
    )

    # Plot example of color separation
    from matplotlib.colors import LinearSegmentedColormap

    cmap_hema = LinearSegmentedColormap.from_list("mycmap", ["white", "navy"])
    cmap_dab = LinearSegmentedColormap.from_list(
        "mycmap", ["white", "saddlebrown"]
    )
    cmap_eosin = LinearSegmentedColormap.from_list(
        "mycmap", ["darkviolet", "white"]
    )

    name = res["diaminobenzidine"].sort_values().tail(1).index[0]
    img = get_image(files["MPO"][name])

    ihc = minmax_scale(np.moveaxis(rgb2hed(img), -1, 0))
    hema = minmax_scale(ihc[0] / ihc.sum(0))
    dab = minmax_scale(ihc[2] / ihc.sum(0))

    x = slice(600, 1200, 1)
    y = slice(1200, 1500, 1)
    width = x.stop - x.start
    height = y.stop - y.start

    fig, axes = plt.subplots(
        2, 3, figsize=(3 * 4, 2 * 4), sharex="row", sharey="row"
    )
    axes[0][0].imshow(img)
    axes[0][1].imshow(hema, cmap=cmap_hema, vmin=0.35)
    axes[0][2].imshow(dab, cmap=cmap_dab, vmin=0.4)
    for ax in axes[0]:
        art = plt.Rectangle(
            (x.start, y.start), width, height, fill=None, linestyle="--"
        )
        ax.add_artist(art)
    axes[1][0].imshow(img[y, x])
    axes[1][1].imshow(hema[y, x], cmap=cmap_hema, vmin=0.35)
    axes[1][2].imshow(dab[y, x], cmap=cmap_dab, vmin=0.4)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(
        results_dir / "he_dab.color_separation.top_illustration.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=res.join(annot),
        x="phenotypes",
        y="PMN %",
    )

    # Compare with IMC
    from src.config import roi_attributes

    meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq").set_index(
        "sample_name"
    )

    roi_areas = pd.Series(
        json.load(open(results_dir.parent / "cell_type" / "roi_areas.pq", "r"))
    )
    ctc = pd.read_parquet(
        results_dir.parent / "cell_type" / "cell_type_counts.pq"
    )
    cells_per_roi = ctc.sum(1)
    neutrophils_per_roi = ctc.loc[
        :, ctc.columns.str.contains("Neutrophil")
    ].sum(1)

    neutrophils_per_roi_perc = (neutrophils_per_roi / cells_per_roi) * 100

    gat = pd.read_parquet(
        results_dir.parent / "cell_type" / "gating.positive.count.pq"
    )
    cells_per_roi = gat.groupby(level=0).sum(1).sum(1)
    cells_per_cluster = gat.sum(1)
    gat["MPO(Yb173)"]

    mpo_pos = gat["MPO(Yb173)"].groupby(level=0).sum()
    perc = (
        ((mpo_pos / cells_per_roi) * 100)
        .to_frame("% MPO positive")
        .join(roi_attributes)
    )
    perc = perc.query(
        "phenotypes.str.contains('COVID').values", engine="python"
    )

    ppa = (
        ((mpo_pos / roi_areas) * 1e6)
        .to_frame("MPO positive (mm2)")
        .join(roi_attributes)
    )
    ppa = ppa.query("phenotypes.str.contains('COVID').values", engine="python")

    for c in perc:
        if perc[c].dtype.name == "category":
            perc[c] = perc[c].cat.remove_unused_categories()
    for c in ppa:
        if ppa[c].dtype.name == "category":
            ppa[c] = ppa[c].cat.remove_unused_categories()

    fig, stats = swarmboxenplot(
        data=perc,
        x="phenotypes",
        y="% MPO positive",
    )
    fig.savefig(
        results_dir / "imc_MPO_positive_cells.percentage.svg",
        **figkws,
    )
    fig, stats = swarmboxenplot(
        data=ppa,
        x="phenotypes",
        y="MPO positive (mm2)",
    )
    # fig.axes[0].set_ylim(bottom=10)
    # fig.axes[0].set_yscale("symlog")
    fig.savefig(
        results_dir / "imc_MPO_positive_cells.area.svg",
        **figkws,
    )


def segment_stardist_imagej(files):
    import subprocess

    macro_name = "Stardist 2D"
    macro_content = [
        """open("{input_file}");""",
        """run("Command From Macro", "command=[de.csbdresden.stardist.StarDist2D], args=['input':'{input_file_name}', 'modelChoice':'Model (.zip) from File', 'normalizeInput':'true', 'percentileBottom':'1.0', 'percentileTop':'99.8', 'probThresh':'0.5', 'nmsThresh':'0.4', 'outputType':'Label Image', 'modelFile':'/home/afr/Downloads/he_heavy_augment.zip', 'nTiles':'1', 'excludeBoundary':'2', 'roiPosition':'Automatic', 'verbose':'false', 'showCsbdeepProgress':'false', 'showProbAndDist':'false'], process=[false]");""",
        """selectImage("Label Image");""",
        """saveAs("Tiff", "{output_file}");""",
        """while (nImages>0) {{
        selectImage(nImages); 
        close(); 
    }}""",
    ]

    ijmacro = f"""run("{macro_name}", "{' '.join(macro_content)}");"""

    ijpath = (
        Path("~/Downloads/fiji/Fiji.app/ImageJ-linux64").expanduser().absolute()
    )

    macro_file = "macro.ijm"
    open(macro_file, "w")

    for sf in files:
        for file, url in files[sf].items():
            with open(macro_file, "a") as handle:
                f = (data_dir / sf / file).resolve()
                output_file = f.replace_(".tif", ".stardist_mask.tiff")
                handle.write(
                    ("\n".join(macro_content)).format(
                        input_file=f,
                        input_file_name=f.stem + ".tif",
                        output_file=output_file,
                    )
                    + "\n"
                )
    cmd = f"{ijpath} --ij2 --headless --console --run {macro_file}"
    o = subprocess.call(cmd.split(" "))
    assert o == 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\t - Exiting due to user interruption.")
        sys.exit(1)
