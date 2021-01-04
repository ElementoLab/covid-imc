# coding: utf-8

"""
This script loads H-DAB images, segments
and quantifies positive cells per image.
"""

import io
import sys
import json
import tempfile
from typing import Tuple, Dict, List, Optional
from functools import lru_cache as cache

from tqdm import tqdm
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from skimage.color import rgb2hed
from skimage import filters
import tifffile

from imc.types import DataFrame, Path, Series, Array
from imc.operations import get_population
from imc.segmentation import normalize
from imc.utils import minmax_scale
from imc.graphics import get_random_label_cmap, get_grid_dims

from seaborn_extensions import swarmboxenplot

from boxsdk import OAuth2, Client, BoxOAuthException, BoxAPIException

ROOT_BOX_FOLDER = "128411248991"
SECRETS_FILE = Path("~/.imctransfer.auth.json").expanduser().absolute()
STARDIST_MODEL_URL = "https://github.com/stardist/stardist-imagej/blob/master/src/main/resources/models/2D/he_heavy_augment.zip?raw=true"
STARDIST_MODEL_NAME = "he_heavy_augment"
IMAGE_J_PATH = (
    Path("~/Downloads/fiji/Fiji.app/ImageJ-linux64").expanduser().absolute()
)
figkws = dict(dpi=300, bbox_inches="tight")

metadata_dir = Path("metadata")
metadata_dir.mkdir(exist_ok=True)
data_dir = Path("data") / "ihc"
data_dir.mkdir(exist_ok=True)
results_dir = Path("results") / "ihc"
results_dir.mkdir(exist_ok=True)


cmap_hema = LinearSegmentedColormap.from_list("mycmap", ["white", "navy"])
cmap_dab = LinearSegmentedColormap.from_list("mycmap", ["white", "saddlebrown"])
cmap_eosin = LinearSegmentedColormap.from_list(
    "mycmap", ["darkviolet", "white"]
)


def main():
    col = ImageCollection()
    col.get_files(force_refresh=True, exclude_keys=["annotated svs files"])
    col.download_images()
    col.download_masks()
    col.segment()
    col.upload_masks()
    col.quantify()

    # quants = col.quantification
    # markers = quants["marker"].unique()
    # n, m = get_grid_dims(len(markers))
    # fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4))
    # for ax, marker in zip(axes.flat, markers):
    #     q = quants.query(f"marker == '{marker}'")
    #     ax.scatter(q["hematoxilyn"], q["diaminobenzidine"], alpha=0.1, s=2)
    #     ax.set(title=marker)

    # index = pd.Series([i.name for i in col.images], name="image")
    # annot = pd.DataFrame(
    #     map(
    #         pd.Series,
    #         pd.Series(
    #             [n.replace("x -", "x-").replace("nl", "nl ") for n in index]
    #         ).str.split(" "),
    #     ),
    #     index=index,
    # )
    # annot.columns = ["disease", "patient_id", "location", "replicate"]

    # # separate into early/late COVID
    # early = [12, 10, 3, 1]
    # late = [28, 23, 11, 21, 30, 24]
    # annot["phenotypes"] = np.nan
    # annot.loc[
    #     annot["patient_id"].astype(int).isin(early), "phenotypes"
    # ] = "COVID19_early"
    # annot.loc[
    #     annot["patient_id"].astype(int).isin(late), "phenotypes"
    # ] = "COVID19_late"
    # annot.loc[annot["disease"] != "covid", "phenotypes"] = "Normal"

    # # TODO: threshold, quantify
    # quants = col.quantification.merge(
    #     annot, left_on="image", right_index=True
    # ).dropna()
    # markers = quants["marker"].unique()
    # phenos = quants["phenotypes"].unique()
    # n, m = len(markers), len(phenos)
    # fig, axes = plt.subplots(
    #     n, m, figsize=(m * 4, n * 4), sharex=True, sharey=True
    # )
    # for axs, marker in zip(axes, markers):
    #     for ax, pheno in zip(axs, phenos):
    #         p = quants.query(f"marker == '{marker}' & phenotypes == '{pheno}'")
    #         if p.empty:
    #             continue
    #         x, y = p["hematoxilyn"], p["diaminobenzidine"]
    #         ax.scatter(x, y, alpha=0.1, s=1)
    #         # threshold, quantify
    #         m, s = y.apply([np.mean, np.std])
    #         t = 3 * s
    #         t = 0.3
    #         perc = ((y > t).sum() / x.shape[0]) * 100
    #         ax.text(0.5, 0.5, s=f"{perc:.2f}")
    #         ax.axhline(t, linestyle="--", color="grey")

    # for ax, dis in zip(axes[0, :], phenos):
    #     ax.set(title=dis)
    # for ax, marker in zip(axes[:, 0], markers):
    #     ax.set(ylabel=marker)

    # TODO:
    # Check for balance in n. images per patient
    # COVID11 lots of T cells in IHC


class Image:
    def __init__(
        self,
        marker: str,
        image_file_name: Path,
        image_url: Optional[str] = None,
        mask_file_name: Optional[Path] = None,
        mask_url: Optional[str] = None,
    ):
        self.marker = marker
        self.image_file_name = image_file_name.absolute()
        self.image_url = image_url
        self.mask_file_name = (
            mask_file_name
            or self.image_file_name.replace_(".tif", ".stardist_mask.tiff")
        ).absolute()
        self.mask_url = mask_url
        self.col: Optional["ImageCollection"] = None

    def __repr__(self):
        return f"Image of '{self.marker}': '{self.name}'"

    @property
    def name(self):
        return self.image_file_name.stem

    @property
    def image(self):
        try:
            return tifffile.imread(self.image_file_name)
        except (FileNotFoundError, ValueError):
            return get_image_from_url(self.image_url)

    @property
    def mask(self):
        try:
            return tifffile.imread(self.mask_file_name)
        except (FileNotFoundError, ValueError):
            return get_image_from_url(self.mask_url)

    @property
    def has_image(self):
        return self.image_file_name.exists()

    @property
    def has_mask(self):
        return self.mask_file_name.exists()

    def download(self, image_type: str = "image"):
        if image_type == "image":
            url = self.image_url
            file = self.image_file_name
        elif image_type == "mask":
            url = self.mask_url
            file = self.mask_file_name
        img = get_image_from_url(url)
        tifffile.imwrite(file, img)

    def upload(self, image_type: str = "mask"):
        assert image_type == "mask", NotImplementedError(
            f"Uploading {image_type} is not yet implemented"
        )
        img_dict = self.col.files[self.marker][self.image_file_name.parts[-1]]
        uploaded = image_type in img_dict
        if self.has_mask and not uploaded:
            upload_image(
                self.mask,
                self.mask_file_name.parts[-1],
                subfolder_name=self.marker,
                subfolder_suffix="_masks" if image_type == "mask" else "",
            )

    def decompose_hdab(self):
        ihc = np.moveaxis(rgb2hed(self.image), -1, 0)
        return np.stack([minmax_scale(ihc[0]), minmax_scale(ihc[2])])

    def quantify(self):
        quant = quantify_cell_intensity(self.decompose_hdab(), self.mask)
        quant.columns = ["hematoxilyn", "diaminobenzidine"]
        quant.index.name = "cell_id"
        return quant.assign(image=self.name, marker=self.marker)


class ImageCollection:
    def __init__(
        self,
        files: Dict[str, Dict[str, Dict[str, str]]] = {},
        images: List[Image] = [],
    ):
        self.files = files
        self.images = images
        # self.files_json = metadata_dir / "ihc_files.box_dir.json"

        self.files_json = metadata_dir / "ihc_files.image_mask_urls.json"
        self.quant_file = data_dir / "quantification_hdab.csv"

        self.get_files(regenerate=False)
        self.generate_image_objs()

    def __repr__(self):
        return f"Image collection with {len(self.images)} images."

    def get_files(
        self,
        force_refresh: bool = False,
        exclude_keys: List[str] = None,
        regenerate: bool = True,
    ):
        if exclude_keys is None:
            exclude_keys = []
        if force_refresh or not self.files_json.exists():
            files = get_urls()
            for key in exclude_keys:
                files.pop(key, None)
            json.dump(files, open(self.files_json, "w"), indent=4)
        self.files = json.load(open(self.files_json, "r"))

        if regenerate:
            return ImageCollection(files=self.files)

        # f = [pd.DataFrame(v).T.assign(marker=k) for k, v in files.items()]
        # file_df = pd.concat(f).reset_index()
        # file_df = file_df.set_index(["marker", "index"])
        # file_df.to_csv(metadata_dir / "ihc_images.csv")

        # file_df = pd.read_csv(metadata_dir / "ihc_images.csv")

    def generate_image_objs(self, force_refresh: bool = False):
        images = list()

        if self.files is None:
            print("Getting file URLs")
            self.files = self.get_files()
        for sf in self.files:
            for name, urls in self.files[sf].items():
                image = Image(
                    marker=sf,
                    image_file_name=data_dir / sf / name,
                    image_url=urls["image"],
                    mask_url=urls.get("mask"),
                )
                image.col = self
                images.append(image)
        self.images = images

    def download_images(self, overwrite: bool = False):
        for image in tqdm(self.images):
            if overwrite or not image.has_image:
                image.download("image")

    def download_masks(self, overwrite: bool = False):
        for image in tqdm(self.images):
            if overwrite or not image.has_mask:
                image.download("mask")

    def upload_images(self):
        raise NotImplementedError
        for image in tqdm(self.images):
            ...

    def upload_masks(self, refresh_files: bool = True):
        for image in tqdm(self.images):
            image.upload("mask")

    def remove_images(self):
        for image in tqdm(self.images):
            image.image_file_name.unlink()

    def remove_masks(self):
        for image in tqdm(self.images):
            image.mask_file_name.unlink()

    def segment(self):
        segment_stardist_imagej(self.files)

    @property
    def quantification(self):
        if self.quant_file.exists():
            quants = pd.read_csv(self.quant_file, index_col=0)
        else:
            quants = pd.DataFrame(
                index=pd.Series(name="cell_id", dtype=int),
                columns=["hematoxilyn", "diaminobenzidine", "image", "marker"],
            )
        return quants

    def quantify(self, force_refresh: bool = False, save: bool = True):
        quants = self.quantification
        _quants = list()
        for image in tqdm(self.images):
            e = quants.query(
                f"marker == '{image.marker}' & image == '{image.name}'"
            )
            if e.empty or force_refresh:
                tqdm.write(image.name)
                q = image.quantify()
                _quants.append(q)
        quants = pd.concat([quants] + _quants)
        if save:
            quants.to_csv(self.quant_file)
        return quants


def get_box_folder():
    """
    Get the root Box.com folder with a new connection.
    """
    secret_params = json.load(open(SECRETS_FILE, "r"))
    oauth = OAuth2(**secret_params)
    client = Client(oauth)

    return client.folder(ROOT_BOX_FOLDER)


@cache
def get_image_from_url(url=None):
    if url is None:
        url = urls[0]
    with requests.get(url) as req:
        return tifffile.imread(io.BytesIO(req.content))


def download_all_files(files, exclude_subfolders=None):
    if exclude_subfolders is None:
        exclude_subfolders = []
    # Download
    for sf in tqdm(files, desc="subfolder"):
        if sf in exclude_subfolders:
            continue
        (data_dir / sf).mkdir()
        for file, url in tqdm(files[sf].items(), desc="image"):
            f = data_dir / sf / file
            if not f.exists():
                img = get_image_from_url(url)
                tifffile.imwrite(f, img)


urls = [
    "https://wcm.box.com/shared/static/p7456xiws8mtqt2in09ju3459k4lvdpb.tif",
    "https://wcm.box.com/shared/static/p1o8ytt2c3zn2gkkvsxgg14iviirt0ov.tif",
    "https://wcm.box.com/shared/static/8ql0u7ki7wbiyo0uh8sjdyv56r0ud80i.tif",  #
]


def get_urls(query_string="", file_type="tif"):
    folder = get_box_folder()
    subfolders = list(folder.get_items())

    image_folders = [sf for sf in subfolders if not sf.name.endswith("_masks")]
    mask_folders = [sf for sf in subfolders if sf.name.endswith("_masks")]

    # pair iamge and mask directories
    subfolders = list()
    for sf in image_folders:
        two = [m for m in mask_folders if m.name.startswith(sf.name)]
        two = (two or [None])[0]
        subfolders.append((sf, two))

    files = dict()
    for sf, sfmask in tqdm(subfolders, desc="marker"):
        files[sf.name] = dict()
        fss = list(sf.get_items())
        if sfmask is not None:
            masks = list(sfmask.get_items())
        for image in tqdm(fss, desc="image"):
            add = {}
            if sfmask is not None:
                mask = [
                    m
                    for m in masks
                    if m.name.replace(".stardist_mask.tiff", ".tif")
                    == image.name
                ]
                if mask:
                    mask = mask[0]
                    add = {"mask": mask.get_shared_link_download_url()}
                else:
                    print(
                        f"Image still does not have mask: '{sf}/{image.name}'"
                    )
            files[sf.name][image.name] = {
                "image": image.get_shared_link_download_url(),
                **add,
            }

    return files


def segment_stardist_imagej(
    images: List[Image],
    exclude_markers: List[str] = None,
    overwrite: bool = False,
):
    import subprocess

    if exclude_markers is None:
        exclude_markers = []

    # exclude_markers = [
    #     "cd163",
    #     # "CD8",
    #     "Cleaved caspase 3",
    #     "MPO",
    # ]

    stardist_model_zip = (
        Path("_models").absolute() / STARDIST_MODEL_NAME + ".zip"
    )

    macro_name = "Stardist 2D"
    args = {
        "input": "{input_file_name}",
        "modelChoice": "Model (.zip) from File",
        "normalizeInput": "true",
        "percentileBottom": "1.0",
        "percentileTop": "99.8",
        "probThresh": "0.5",
        "nmsThresh": "0.4",
        "outputType": "Label Image",
        "modelFile": "{stardist_model_zip}",
        "nTiles": "1",
        "excludeBoundary": "2",
        "roiPosition": "Automatic",
        "verbose": "false",
        "showCsbdeepProgress": "false",
        "showProbAndDist": "false",
    }
    arg_str = ",".join([f"'{k}':'{v}'" for k, v in args.items()])
    macro_content = [
        """open("{input_file}");""",
        """run("Command From Macro", "command=[de.csbdresden.stardist.StarDist2D],"""
        # TODO: fix this
        f"""args=[{arg_str}], process=[false]");""",
        """selectImage("Label Image");""",
        """saveAs("Tiff", "{output_file}");""",
        """while (nImages>0) {{
        selectImage(nImages); 
        close(); 
    }}""",
    ]

    ijmacro = f"""run("{macro_name}", "{' '.join(macro_content)}");"""

    macro_file = data_dir / "macro.ijm"
    open(macro_file, "w")  # make sure it's empty before appending

    for image in images:
        if image.marker in exclude_markers:
            continue
        if overwrite or not image.mask_file_name.exists():
            print(image)

            macro = ("\n".join(macro_content)).format(
                input_file=image.image_file_name,
                input_file_name=image.image_file_name.parts[-1],
                output_file=image.mask_file_name,
                stardist_model_zip=stardist_model_zip,
            )
            with open(macro_file, "a") as handle:
                handle.write(macro + "\n")

    # ! subl $macro_file
    cmd = f"{IMAGE_J_PATH} --ij2 --headless --console --run {macro_file}"
    o = subprocess.call(cmd.split(" "))
    assert o == 0


def upload_image(
    img: Array,
    file_name: str,
    subfolder_name: str,
    subfolder_suffix: str = "_masks",
):
    root_folder = get_box_folder()
    subfolders = root_folder.get_items()
    subfolder = [
        f for f in subfolders if f.name == subfolder_name + subfolder_suffix
    ]
    if subfolder:
        subfolder = subfolder[0]
    else:
        subfolder = root_folder.create_subfolder(
            subfolder_name + subfolder_suffix
        )
    tmp_file = tempfile.NamedTemporaryFile()
    tifffile.imwrite(tmp_file, img)

    return subfolder.upload(tmp_file.name, file_name=file_name)


# def _get_stardist_model():
#     model_dir = Path("_models")
#     model_dir.mkdir()
#     model_file = model_dir / Path(STARDIST_MODEL_URL).stem + ".zip"
#     if not model_file.exists():
#         with requests.get(STARDIST_MODEL_URL) as req:
#             with open(model_file, "wb") as handle:
#                 handle.write(req.content)


# def upload_masks(files, file_df, exclude_subfolders=None):
#     if exclude_subfolders is None:
#         exclude_subfolders = []
#     # get, parse and assemble data from "source data" files
#     for sf in tqdm(files, desc="subfolder"):
#         if sf in exclude_subfolders:
#             continue
#         for file, url in tqdm(files[sf].items(), desc="image"):
#             f = data_dir / sf / file
#             q = file_df.query(f"marker == '{sf}' & index == '{file}'").squeeze()
#             if not pd.isnull(q["mask"]):
#                 continue
#             print(sf, file)
#             mask_file = f.replace_(".tif", ".stardist_mask.tiff")
#             try:
#                 img = tifffile.imread(f)
#             except:
#                 img = get_image_from_url(url)
#             mask = tifffile.imread(mask_file)
#             try:
#                 u = upload_image(mask, mask_file.name, sf)
#                 file_df.loc[
#                     (file_df["marker"] == sf) & (file_df["index"] == file),
#                     "mask",
#                 ] = u.get_shared_link_download_url()
#             except BoxAPIException:
#                 pass

#     return file_df


def quantify_cell_intensity(stack, mask, red_func="mean"):
    cells = np.unique(mask)[1:]
    # the minus one here is to skip the background "0" label which is also
    # ignored by `skimage.measure.regionprops`.
    n_cells = len(cells)
    n_channels = stack.shape[0]

    res = np.zeros(
        (n_cells, n_channels), dtype=int if red_func == "sum" else float
    )
    for channel in np.arange(stack.shape[0]):
        res[:, channel] = [
            getattr(x.intensity_image, red_func)()
            for x in skimage.measure.regionprops(mask, stack[channel])
        ]
    return pd.DataFrame(res, index=cells)


def quantify_segmentation(files):
    # get, parse and assemble data from "source data" files

    quant_csv = results_dir / "ihc.stardist.quantification.csv"
    try:
        quants = pd.read_csv(quant_csv, index_col=0)
    except:
        quants = pd.DataFrame(
            index=pd.Series(name="cell_id", dtype=int),
            columns=["hematoxilyn", "diaminobenzidine", "marker", "image"],
        )

    labels = ["hematoxilyn", "diaminobenzidine"]
    _quants = list()
    for sf in tqdm(files, desc="subfolder"):
        for file, url in tqdm(files[sf].items(), desc="image"):
            q = quants.query(f"marker == '{sf}' & image == '{file}'")
            if not q.empty:
                continue
            f = data_dir / sf / file
            mask_file = f.replace_(".tif", ".stardist_mask.tiff")
            mask = tifffile.imread(mask_file)
            try:
                img = tifffile.imread(f)
            except:
                img = get_image_from_url(url)
            ihc = np.moveaxis(rgb2hed(img), -1, 0)
            ihc = np.stack([minmax_scale(ihc[0]), minmax_scale(ihc[2])])
            q = quantify_cell_intensity(ihc, mask)
            q.columns = labels
            _quants.append(q.assign(marker=sf, image=file))
    quants = pd.concat(_quants)
    quants.index.name = "cell_id"
    quants.to_csv(results_dir / "ihc.stardist.quantification.csv")

    quants = pd.read_csv(
        results_dir / "ihc.stardist.quantification.csv", index_col=0
    )

    # threshold positivity
    t = get_population(quants["diaminobenzidine"])
    # m = quants["diaminobenzidine"].mean()
    # s = quants["diaminobenzidine"].std()
    # t = quants['diaminobenzidine'] > m + 1 * s
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

    annot_uniq = annot[
        ["image", "sample_name", "wcmc_code", "location", "phenotypes"]
    ].drop_duplicates()
    q = mean_quant.reset_index().merge(
        annot_uniq,
        on="image",
    )

    qp = (
        perc.rename("diaminobenzidine")
        .reset_index()
        .merge(
            annot_uniq,
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

    quants_annot = quants.reset_index().merge(
        annot[["image", "phenotypes"]].drop_duplicates(), on="image"
    )

    a = quants_annot.query("phenotypes == 'COVID19_early'")["diaminobenzidine"]
    b = quants_annot.query("phenotypes == 'COVID19_late'")["diaminobenzidine"]
    la, lb = a.shape[0], b.shape[0]

    x = np.linspace(*quants["diaminobenzidine"].agg([min, max]))[:-10]
    ratio = [np.log(((a > xi).sum() / la) / ((b > xi).sum() / lb)) for xi in x]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.distplot(
        a,
        ax=ax,
        label="COVID19_early",
        hist=False,
    )
    sns.distplot(
        b,
        ax=ax,
        label="COVID19_late",
        hist=False,
    )
    ax2 = ax.twinx()
    ax2.plot(x, ratio, color=sns.color_palette()[2])
    ax2.axhline(0, linestyle="--", color="grey")
    ax2.set_ylabel("Log (Early / Late)", color=sns.color_palette()[2])
    for i in range(4):
        x = m + i * s
        t = (a > x).sum() + (b > x).sum()
        pa = ((a > x).sum() / a.shape[0]) * 100
        pb = ((b > x).sum() / b.shape[0]) * 100
        ax.axvline(x, linestyle="--", color="grey")
        ax.text(x, 2, s=f"SD= {i}; t= {t}; {pa:.2f} / {pb:.2f}", rotation=90)
    ax.set(
        title=f"{quants.shape[0]} cells over {len(files['MPO'])} images",
        xlabel="DAB/MPO staining intensity",
        ylabel="Cell density",
    )
    fig.savefig(
        results_dir / "ihc.stardist_segmentation.MPO_distribution.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=qp,
        x="phenotypes",
        y="diaminobenzidine",
    )
    fig.savefig(
        results_dir
        / "he_dab.stardist_segmentation.MPO_positive.swarmboxenplot.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=qp,
        x="wcmc_code",
        y="diaminobenzidine",
        plot_kws=dict(palette="Set2"),
    )
    fig.savefig(
        results_dir
        / "he_dab.stardist_segmentation.MPO_positive.by_patient.swarmboxenplot.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=qp,
        x="phenotypes",
        y="diaminobenzidine",
        hue="location",
    )
    fig.savefig(
        results_dir
        / "he_dab.stardist_segmentation.MPO_positive.by_location.swarmboxenplot.svg",
        **figkws,
    )

    fig, stats = swarmboxenplot(
        data=qp,
        x="wcmc_code",
        y="diaminobenzidine",
        hue="location",
    )
    fig.savefig(
        results_dir
        / "he_dab.stardist_segmentation.MPO_positive.by_patient_location.swarmboxenplot.svg",
        **figkws,
    )

    return qp


# def plot_segmentation(qp, n=6):
#     # s = qp.sort_values("diaminobenzidine")
#     # images = s.head(3)["image"].tolist() + s.tail(3)["image"].tolist()
#     images = qp.sample(n=n)["image"]
#     n_top = len(images)
#     labels = [
#         "Original image",
#         "Hematoxilyn",
#         "Diaminobenzidine",
#         "Segmentation masks",
#         "Segmentation masks (overlay)",
#     ]

#     fig, axes = plt.subplots(
#         n_top,
#         5,
#         figsize=(5 * 4, n_top * 4),
#         gridspec_kw=dict(hspace=0, wspace=0.1),
#         sharex="col",
#         sharey="col",
#     )
#     for axs, name in zip(axes, images):
#         idx = qp["diaminobenzidine"].sort_values().tail(n_top).index
#         f = data_dir / sf / name
#         img = tifffile.imread(f)

#         ihc = minmax_scale(np.moveaxis(rgb2hed(img), -1, 0))
#         hema = minmax_scale(ihc[0] / ihc.sum(0))
#         dab = minmax_scale(ihc[2] / ihc.sum(0))

#         mask_file = f.replace_(".tif", ".stardist_mask.tiff")
#         mask = tifffile.imread(mask_file)

#         axs[0].set_ylabel(name)
#         axs[0].imshow(img, rasterized=True)
#         axs[1].imshow(hema, cmap=cmap_hema, vmin=0.35)
#         axs[2].imshow(dab, cmap=cmap_dab, vmin=0.4)
#         axs[3].imshow(mask, rasterized=True, cmap=get_random_label_cmap())
#         axs[4].imshow(img, rasterized=True)
#         axs[4].contour(
#             mask, levels=2, cmap="Reds", vmin=-0.2, vmax=0, linewidths=0.5
#         )
#         for c in axs[4].get_children():
#             if isinstance(c, matplotlib.collections.LineCollection):
#                 c.set_rasterized(True)
#     for ax in axes.flat:
#         ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#     for ax, lab in zip(axes[0], labels):
#         ax.set_title(lab)
#     fig.savefig(
#         results_dir / "he_dab.stardist_segmentation.illustration.svg",
#         **figkws,
#     )


# def simple_quantify_hed_colors(img):
#     """"""
#     size = np.multiply(*img.shape[:2])
#     ihc = np.moveaxis(rgb2hed(img), -1, 0)

#     labels = ["hematoxilyn", "eosin", "diaminobenzidine"]
#     res = dict()
#     for im, label in zip(ihc, labels):
#         im = normalize(im)
#         # Filter illumination
#         t = im > filters.threshold_local(im, 21)
#         res[label] = t.sum() / size
#     return res


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
            img = get_image_from_url(files["MPO"][name])
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
    name = res["diaminobenzidine"].sort_values().tail(1).index[0]
    # 'covid 12 alveolar 20x-2.tif'
    img = get_image_from_url(files["MPO"][name])

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

    img = get_image_from_url(files["MPO"][name])
    mask_file = data_dir / "MPO" / name.replace(".tif", ".stardist_mask.tiff")
    mask = tifffile.imread(mask_file)
    mask = np.ma.masked_array(mask, mask == 0)
    fig, axes = plt.subplots(
        2, 4, figsize=(4 * 4, 2 * 4), sharex="row", sharey="row"
    )
    axes[0][0].imshow(img)
    axes[0][1].imshow(hema, cmap=cmap_hema, vmin=0.35)
    axes[0][2].imshow(dab, cmap=cmap_dab, vmin=0.4)
    axes[0][3].imshow(mask, cmap=get_random_label_cmap())
    for ax in axes[0]:
        art = plt.Rectangle(
            (x.start, y.start), width, height, fill=None, linestyle="--"
        )
        ax.add_artist(art)
    axes[1][0].imshow(img[y, x])
    axes[1][1].imshow(hema[y, x], cmap=cmap_hema, vmin=0.35)
    axes[1][2].imshow(dab[y, x], cmap=cmap_dab, vmin=0.4)
    axes[1][3].imshow(mask[y, x], cmap=get_random_label_cmap())
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(
        results_dir
        / "he_dab.color_separation.segmentation.top_illustration.svg",
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


def _move_box_folders():
    """
    Restructure box directory by merging folders added later with original ones.
    The ones added later end with "other conditions".

    Was only run once, no need to run further.
    """
    folder = get_box_folder()
    subfolders = list(folder.get_items())

    for sf in subfolders:
        if "other conditions" not in sf.name or "mask" in sf.name:
            continue

        # find original folder
        t = sf.name.split(" ")[0]
        name = t.upper() if t in ["cd8", "mpo"] else sf.name
        dest = [
            s
            for s in subfolders
            if s.name == name.replace(" other conditions", "")
        ][0]
        print(f"{sf.name} -> {dest.name}")

        for f in sf.get_items():
            print(f.name)
            f.move(dest)

    # delete empty folders
    for sf in subfolders:
        if len(list(sf.get_items())) == 0:
            print(sf.name)
            sf.delete(recursive=False)


def _remove_wrong_masks(col):
    """
    at some point the image got copied to the mask file.
    """
    # find images
    remove = list()
    for i in col.images:
        try:
            a = i.image_file_name.stat().st_size
            b = i.mask_file_name.stat().st_size
        except FileNotFoundError:
            continue
        if b >= a:
            remove.append(i)

    for r in remove:
        r.mask_file_name.unlink()


"MPO", "covid 12 airway 40x-1"


def _remove_wrong_masks2(col):
    """
    at some point the image got copied to the mask file.
    """
    # find images
    for i in col.images:
        if "mask" in i.image_file_name.as_posix():
            i.mask_file_name.unlink(missing_ok=True)
            i.image_file_name.unlink(missing_ok=True)
            col.images.remove(i)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\t - Exiting due to user interruption.")
        sys.exit(1)
