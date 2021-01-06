# coding: utf-8

"""
This script loads H-DAB images, segments
and quantifies positive cells per image.
"""

import io, sys, json, tempfile
from typing import Tuple, Dict, List, Optional, Callable
from functools import lru_cache as cache, partial

from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile
import skimage
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from boxsdk import OAuth2, Client, BoxOAuthException, BoxAPIException
from boxsdk.object.folder import Folder as BoxFolder

from imc.types import DataFrame, Path, Array
from imc.operations import get_population
from imc.utils import minmax_scale
from imc.graphics import get_random_label_cmap

from seaborn_extensions import swarmboxenplot

swarmboxenplot = partial(swarmboxenplot, test_kws=dict(parametric=False))

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

cmap_hema = LinearSegmentedColormap.from_list("", ["white", "navy"])
cmap_dab = LinearSegmentedColormap.from_list("", ["white", "saddlebrown"])
cmap_eosin = LinearSegmentedColormap.from_list("", ["darkviolet", "white"])
phenotype_order = [
    "Healthy",
    "Flu",
    "ARDS",
    "Pneumonia",
    "COVID19_early",
    "COVID19_late",
]
p_palette = np.asarray(sns.color_palette("tab10"))[[2, 0, 1, 5, 4, 3]]
m_palette = np.asarray(sns.color_palette("Dark2"))


def main():
    col = ImageCollection()
    # query box.com for uploaded images
    col.get_files(force_refresh=True, exclude_keys=["annotated svs files"])
    col.download_images()
    col.download_masks()  # if existing

    # Segment and save masks in box
    col.segment()
    col.upload_masks()

    # Quantify intensity
    col.quantify()

    # # # quantify also without transformations
    # rquant = col.quantify(force_refresh=True, save=False, transform_func=None)
    # rquant.to_csv(col.quant_file.replace_(".csv", ".raw.csv"))

    # # # # quantify also with image-wise z-score
    # quantz = col.quantify(
    #     force_refresh=True, save=False, transform_func=z_score
    # )
    # quantz.to_csv(col.quant_file.replace_(".csv", ".z_score.csv"))

    # Get metadata
    file_df = files_to_dataframe(col.files)
    meta = join_metadata(file_df)

    # # add ordered categorical
    meta["phenotypes"] = pd.Categorical(
        meta["phenotypes"], categories=phenotype_order, ordered=True
    )

    # Gate
    quant = col.quantification
    quant = Analysis.gate_with_gmm_by_marker(quant)

    # Aggregate quantifications per image across cells
    means = quant.groupby(["marker", "image"]).mean()

    # Join with metadata (just disese group for now)
    group_var = "phenotypes"
    q_var = "diaminobenzidine"
    means = means.join(meta[["sample_id", group_var]])

    # Work only with samples where a disease group is assigned
    means = means.dropna(subset=["phenotypes"])
    # means = means.dropna(subset=["phenotypes", "imc_sample_id"])

    # # quantify percent positive
    pos = quant.groupby(["marker", "image"])["pos"].sum()
    total = quant.groupby(["marker", "image"])["pos"].size()
    perc = ((pos / total) * 100).to_frame(q_var)

    # Join with metadata (just disese group for now)
    perc = perc.join(meta[["sample_id", "imc_sample_id", group_var]])
    # Work only with samples where a disease group is assigned
    perc = perc.dropna(subset=["phenotypes"])
    # perc = perc.dropna(subset=["phenotypes", "imc_sample_id"])

    # # quantigy positive per mm2
    # TODO: get exact scale from images rather than using magnification
    # areas = [np.multiply(*i.image.shape[:2]) for i in col.images]
    areas = [1 if "40x" in i.name else 2 for i in col.images]
    mm2 = ((pos / areas) * 1e6).to_frame(q_var)

    # Join with metadata (just disese group for now)
    mm2 = mm2.join(meta[["sample_id", "imc_sample_id", group_var]])
    # Work only with samples where a disease group is assigned
    mm2 = mm2.dropna(subset=["phenotypes"])
    # mm2 = mm2.dropna(subset=["phenotypes", "imc_sample_id"])

    # Plot
    for df, vt in [
        (means, "intensity"),
        (perc, "percentage"),
        (mm2, "absolute"),
    ]:
        k = dict(value_type=vt, prefix="")
        Analysis.plot_sample_image_numbers(df, **k)
        Analysis.plot_comparison_between_groups(df, **k)
        Analysis.plot_example_top_bottom_images(df, col, **k)
        Analysis.plot_gating(df, **k)


class Analysis:
    @staticmethod
    def plot_sample_image_numbers(df, value_type="intensity", prefix=""):
        # Illustrate number of samples and images for each marker and disease group
        group_var = "phenotypes"
        combs = [
            ("count", "phenotypes", "marker", "by_phenotypes"),
            ("count", "marker", "phenotypes", "by_marker"),
        ]
        for x, y, h, label in combs:
            fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 1 * 4), sharey=True)
            # # samples per group
            p = (
                df.groupby(["marker", group_var])["sample_id"]
                .nunique()
                .rename("count")
                .reset_index()
            )
            # # images per group
            p2 = (
                df.groupby(["marker", group_var])
                .size()
                .rename("count")
                .reset_index()
            )
            for ax, df2, xlab in zip(
                axes, [p, p2], ["Unique samples", "Images"]
            ):
                df2["phenotypes"] = pd.Categorical(
                    df2["phenotypes"], categories=phenotype_order, ordered=True
                )
                sns.barplot(
                    data=df2,
                    x=x,
                    y=y,
                    hue=h,
                    orient="horiz",
                    ax=ax,
                    palette=globals()[h[0] + "_palette"],
                )
                ax.set(xlabel=xlab)
            fig.savefig(
                results_dir / f"ihc.{prefix}{value_type}.images_{label}.svg",
                **figkws,
            )

    @staticmethod
    def plot_comparison_between_groups(df, value_type="intensity", prefix=""):
        # Compare marker expression across disease groups (DAB intensity)
        for y, hue in [("phenotypes", "marker"), ("marker", "phenotypes")]:
            pal = globals()[hue[0] + "_palette"]
            fig, axes = plt.subplots(1, 1, figsize=(4, 4))
            sns.barplot(
                data=df.reset_index(),
                x=q_var,
                y=y,
                orient="horiz",
                hue=hue,
                ax=axes,
                palette=pal,
            )
            fig.savefig(
                results_dir / f"ihc.{prefix}{value_type}.by_{y}.barplot.svg",
                **figkws,
            )

            fig, stats = swarmboxenplot(
                data=df.reset_index(),
                y=q_var,
                x=y,
                hue=hue,
                plot_kws=dict(palette=pal),
            )
            fig.savefig(
                results_dir
                / f"ihc.{prefix}{value_type}.by_{y}.swarmboxenplot.svg",
                **figkws,
            )
            # plot also separately
            for g in df.reset_index()[hue].unique():
                p = df.reset_index().query(f"{hue} == '{g}'")
                p["phenotypes"] = p["phenotypes"].cat.remove_unused_categories()
                fig, stats = swarmboxenplot(
                    data=p,
                    y=q_var,
                    x=y,
                    plot_kws=dict(palette=globals()[y[0] + "_palette"]),
                )
                fig.savefig(
                    results_dir
                    / f"ihc.{prefix}{value_type}.by_{hue}.{g}.swarmboxenplot.svg",
                    **figkws,
                )

    @staticmethod
    def plot_example_top_bottom_images(
        df, col, n: int = 2, value_type: str = "intensity", prefix=""
    ):
        # Exemplify images with most/least stain
        nrows = len(phenotype_order)
        ncols = 2 * 2

        def nlarg(x):
            return x.nlargest(n)

        def nsmal(x):
            return x.nsmallest(n)

        for marker in col.files.keys():
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(ncols * 4, nrows * 4)
            )
            for pheno, ax in zip(phenotype_order, axes):
                img_names = (
                    df.loc[marker]
                    .query(f"phenotypes == '{pheno}'")["diaminobenzidine"]
                    .agg([nsmal, nlarg])
                    .index
                )
                imgs = [
                    i
                    for n in img_names
                    for i in col.images
                    if i.name == n and i.marker == marker
                ]
                for a, img in zip(ax, imgs):
                    a.imshow(img.image)
                    a.set_xticks([])
                    a.set_yticks([])
                    a.set_xticklabels([])
                    a.set_yticklabels([])
                    v = df.loc[(marker, img.name), "diaminobenzidine"]
                    a.set(title=f"{img.name}\n{v:.2f}")
                ax[0].set_ylabel(pheno)

            fig.savefig(
                results_dir
                / f"ihc.{prefix}{value_type}_top-bottom_{n}_per_group.{marker}.svg",
                **figkws,
            )

    @staticmethod
    def gate_with_gmm_by_marker(df, values="diaminobenzidine"):
        df["pos"] = np.nan
        for marker in col.markers:
            sel = df["marker"] == marker
            pos = get_population(df.loc[sel, values])
            df.loc[sel, "pos"] = pos
        return df

    @staticmethod
    def plot_gating(df, value_type="intensity", prefix=""):
        x, y = "hematoxilyn", "diaminobenzidine"
        fig, axes = plt.subplots(
            1,
            len(col.markers),
            figsize=(4 * len(col.markers), 4),
            sharex=True,
            sharey=True,
        )
        for ax, marker in zip(axes, col.markers):
            q = df.query(f"marker == '{marker}'")
            ax.axhline(0.3, linestyle="--", color="grey")
            ax.scatter(q[x], q[y], s=1, alpha=0.1, rasterized=True)
            ax.set(title=f"{marker}\n(n = {q.shape[0]:})", xlabel=x, ylabel=y)
            ax.scatter(
                q.loc[pos, x],
                q.loc[pos, y],
                s=2,
                alpha=0.1,
                rasterized=True,
                color="red",
            )
        fig.savefig(
            results_dir
            / f"ihc.{prefix}{value_type}.gating.by_marker.scatterplot.svg",
            **figkws,
        )

        # # plot also as histogram
        fig, axes = plt.subplots(
            1,
            len(col.markers),
            figsize=(4 * len(col.markers), 4),
            sharex=True,
            sharey=True,
        )
        for ax, marker in zip(axes, col.markers):
            q = df.query(f"marker == '{marker}'")
            ax.axhline(0.3, linestyle="--", color="grey")
            sns.distplot(q[y], kde=False, ax=ax)
            ax.set(
                title=f"{marker}\n(n = {q.shape[0]:,})",
                xlabel=x,
                ylabel=y,
            )
            sns.distplot(
                q.loc[q["pos"] == True, y], color="red", kde=False, ax=ax
            )
        fig.savefig(
            results_dir
            / f"ihc_image.{value_type}.gating.by_marker.histplot.svg",
            **figkws,
        )

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
        file.parent.mkdir()
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

    def decompose_hdab(self, normalize: bool = True):
        from skimage.color import separate_stains, hdx_from_rgb

        ihc = np.moveaxis(separate_stains(self.image, hdx_from_rgb), -1, 0)
        if not normalize:
            return np.stack([ihc[0], ihc[2]])
        x = np.stack([minmax_scale(ihc[0]), minmax_scale(ihc[1])])
        return x

        # i = ihc.mean((1, 2)).argmax()
        # o = 0 if i == 1 else 1
        # x[i] = x[i] + x[o] * (x[o].mean() / x[i].mean())
        # hema = minmax_scale(x[0])
        # dab = minmax_scale(x[1])

        # fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
        # axes[0].imshow(self.image)
        # axes[1].imshow(ihc[..., 0], cmap=cmap_hema)
        # axes[2].imshow(ihc[..., 1], cmap=cmap_dab)
        # axes[3].imshow(ihc[..., 2])

        # hema = minmax_scale(ihc[0] / ihc.sum(0))
        # dab = minmax_scale(ihc[2] / ihc.sum(0))
        # hema2 = hema + dab * 0.33
        # dab2 = dab + hema * 0.33
        # hema = minmax_scale(hema2)
        # dab = minmax_scale(dab2)
        # return np.stack([dab, hema])

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

    @property
    def markers(self):
        return sorted(np.unique([i.marker for i in col.images]).tolist())

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
            quants.index = quants.index.astype(int)
        else:
            quants = pd.DataFrame(
                index=pd.Series(name="cell_id", dtype=int),
                columns=["hematoxilyn", "diaminobenzidine", "image", "marker"],
            )
        return quants

    def quantify(
        self,
        force_refresh: bool = False,
        save: bool = True,
        transform_func: Callable = None,
    ):
        # import multiprocessing

        # _quants = list()
        # for image in tqdm(images):
        #     q = image.quantify()
        #     q['hematoxilyn'] = transform_func(q['hematoxilyn'])
        #     q['diaminobenzidine'] = transform_func(q['diaminobenzidine'])
        #     _quants.append(q)
        # quants = pd.concat(_quants)

        quants = self.quantification
        _quants = list()
        for image in tqdm(self.images):
            e = quants.query(
                f"marker == '{image.marker}' & image == '{image.name}'"
            )
            if e.empty or force_refresh:
                tqdm.write(image.name)
                q = image.quantify()
                if transform_func is not None:
                    q["hematoxilyn"] = transform_func(q["hematoxilyn"])
                    q["diaminobenzidine"] = transform_func(
                        q["diaminobenzidine"]
                    )
                _quants.append(q)
        if force_refresh:
            quants = pd.concat(_quants)
        else:
            quants = pd.concat([quants] + _quants)
        if save:
            quants.to_csv(self.quant_file)
        return quants


def files_to_dataframe(files: Dict[str, Dict[str, str]]) -> DataFrame:
    """
    Convert the nested dict of image markers, IDS and URLs into a dataframe.
    """
    f = [pd.DataFrame(v).T.assign(marker=k) for k, v in files.items()]
    return (
        pd.concat(f)
        .reset_index()
        .rename(columns={"image": "image_url", "mask": "mask_url"})
    )


def join_metadata(file_df: DataFrame) -> DataFrame:
    """
    Join information of each IHC image with the clinical metadata of the respective patient.
    """
    df = file_df.copy()

    # the image name strings need to be standardized
    # in order to create a dataframe if split by space
    repl = lambda m: f"{m.group('c')} {m.group('r')}."
    idx = df["index"].str.replace(
        r"(?P<c>alveolar|airway|vessel)(?P<r>\d+).", repl
    )

    annot = pd.DataFrame(
        map(
            pd.Series,
            pd.Series(
                [
                    n.replace("x -", "x-")
                    .replace("nl6699", "nl 6699")
                    .replace("nl113", "nl 113")
                    .replace("nl111", "nl 111")
                    .replace("nl114", "nl 114")
                    .replace("dad ards", "dad_ards")
                    .replace("flu19-23", "flu 19-23")
                    .replace("flu20-5", "flu 20-5")
                    .replace("pneumonia100", "pneumonia 100")
                    for n in idx
                ]
            ).str.split(" "),
        )
    )
    annot.columns = ["disease", "ihc_patient_id", "location", "replicate"]
    assert annot.isnull().sum().all() == False

    # further separate some fields (magnification only exists for certain images, e.g. MPO)
    annot["replicate"] = annot["replicate"].str.replace(".tif", "")
    sel = annot["replicate"].str.contains("x")
    annot["magnification"] = np.nan
    annot.loc[sel, ["magnification", "replicate"]] = (
        annot.loc[sel, "replicate"].str.extract(r"(\d+x)-(\d+)").values
    )

    # cleanup IDs but keep original under "ihc_patient_id"
    annot["sample_id"] = annot["disease"] + annot["ihc_patient_id"]
    annot["ihc_patient_id"] = annot["ihc_patient_id"].str.replace("a", "")
    annot["disease"] = (
        annot["disease"].replace("dad_ards", "ards").replace("nl", "normal")
    )
    # join the field dataframe with the original containing markers, URLs and image names
    df = df.join(annot.drop(["disease"], 1))
    df["image"] = df["index"].str.replace(".tif", "")

    # This was used once to create a reduced dataframe for manual annotation - no longer needed
    # red_annot = (
    #     annot[["disease", "ihc_patient_id"]]
    #     .drop_duplicates()
    #     .sort_values(["disease", "ihc_patient_id"])
    # )
    # red_annot.to_csv(metadata_dir / "ihc_images.only_patient_ids.csv")

    # match non-covid based on autopsy ID
    # # non matches try to add a "a" before ID.
    # # non matches try to add a "s19-" before ID.
    # # non matches try to add a "archoi" before ID (normal samples).
    # not matched:
    # # covid: 5, 8, 9, 26, 29, 32

    # join with clinical and IMC metadata
    # # this is a manual annotation of IHC IDs to IMC samples
    meta = pd.read_csv(metadata_dir / "ihc_metadata.id_match_to_imc.csv")
    clinical = pd.read_csv(
        metadata_dir / "clinical_annotation.csv", index_col=0
    )
    # # drop IMC-specific stuff
    clinical = clinical.drop(
        [
            "phenotypes",  # this one is droped so ihc one is used
            "acquisition_name",
            "acquisition_date",
            "acquisition_id",
            "instrument",
            "panel_annotation_file",
            "panel_version",
            "observations",
            "mcd_file",
        ],
        axis=1,
    )
    meta = meta.merge(
        clinical, left_on="imc_sample_id", right_index=True, how="left"
    )

    full = df.merge(meta, on="ihc_patient_id", how="left")
    assert (df["index"] == full["index"]).all()
    full.to_csv(metadata_dir / "ihc_metadata.csv", index=False)

    return full.set_index(["marker", "image"])


def get_box_folder() -> BoxFolder:
    """
    Get the root Box.com folder with a new connection.
    """
    secret_params = json.load(open(SECRETS_FILE, "r"))
    oauth = OAuth2(**secret_params)
    client = Client(oauth)

    return client.folder(ROOT_BOX_FOLDER)


@cache
def get_image_from_url(url: str = None) -> Array:
    import requests

    if url is None:
        url = urls[0]
    with requests.get(url) as req:
        return tifffile.imread(io.BytesIO(req.content))


def download_all_files(
    files: Dict[str, Dict[str, str]], exclude_subfolders: List[str] = None
) -> None:
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


def get_urls(
    query_string: str = "", file_type: str = "tif"
) -> Dict[str, Dict[str, str]]:
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

    files: Dict[str, Dict[str, str]] = dict()
    for sf, sfmask in tqdm(subfolders, desc="marker"):
        files[sf.name] = dict()
        fss = list(sf.get_items())
        if sfmask is not None:
            masks = list(sfmask.get_items())
        for image in tqdm(fss, desc="image"):
            add = {}
            if sfmask is not None:
                masks = [
                    m
                    for m in masks
                    if m.name.replace(".stardist_mask.tiff", ".tif")
                    == image.name
                ]
                if masks:
                    mask = masks[0]
                    add = {"mask": mask.get_shared_link_download_url()}
                else:
                    print(
                        f"Image still does not have mask: '{sf}/{image.name}'"
                    )
            files[sf.name][image.name] = {  # type: ignore[assignment]
                "image": image.get_shared_link_download_url(),
                **add,
            }

    return files


def segment_stardist_imagej(
    images: List[Image],
    exclude_markers: List[str] = None,
    overwrite: bool = False,
) -> None:
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
) -> None:  # TODO: get actuall return type
    root_folder = get_box_folder()
    subfolders = root_folder.get_items()
    subfolders = [
        f for f in subfolders if f.name == subfolder_name + subfolder_suffix
    ]
    if subfolders:
        subfolder = subfolders[0]
    else:
        subfolder = root_folder.create_subfolder(
            subfolder_name + subfolder_suffix
        )
    tmp_file = tempfile.NamedTemporaryFile()
    tifffile.imwrite(tmp_file, img)

    return subfolder.upload(tmp_file.name, file_name=file_name)


def quantify_cell_intensity(
    stack: Array, mask: Array, red_func: str = "mean"
) -> DataFrame:
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


def plot_decomposition_segmentation(col: ImageCollection, n=6):
    # s = qp.sort_values("diaminobenzidine")
    # images = s.head(3)["image"].tolist() + s.tail(3)["image"].tolist()
    import matplotlib

    try:
        markers = col.markers
    except:
        markers = np.unique([i.marker for i in col.images])

    for marker in markers:
        sel_images = np.random.choice(
            [i for i in col.images if i.marker == marker], n
        )
        n_top = len(sel_images)
        labels = [
            "Original image",
            "Hematoxilyn",
            "Diaminobenzidine",
            "Segmentation masks",
            "Segmentation masks (overlay)",
        ]

        fig, axes = plt.subplots(
            n_top,
            5,
            figsize=(5 * 4, n_top * 4),
            gridspec_kw=dict(hspace=0, wspace=0.1),
            sharex="col",
            sharey="col",
        )
        for axs, image in zip(axes, sel_images):
            img = image.image
            mask = image.mask
            hema, dab = image.decompose_hdab()

            axs[0].set_ylabel(image.name)
            axs[0].imshow(img, rasterized=True)
            axs[1].imshow(hema, cmap=cmap_hema)  # , vmin=0.35)
            axs[2].imshow(dab, cmap=cmap_dab)  # , vmin=0.4)
            axs[3].imshow(mask, rasterized=True, cmap=get_random_label_cmap())
            axs[4].imshow(img, rasterized=True)
            axs[4].contour(
                mask, levels=2, cmap="Reds", vmin=-0.2, vmax=0, linewidths=0.5
            )
            for c in axs[4].get_children():
                if isinstance(c, matplotlib.collections.LineCollection):
                    c.set_rasterized(True)
        for ax in axes.flat:
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        for ax, lab in zip(axes[0], labels):
            ax.set_title(lab)
        fig.savefig(
            results_dir
            / f"he_dab.{marker}.decomposition_segmentation.illustration.svg",
            **figkws,
        )


def _plot_example_():
    from skimage.measure import regionprops_table

    examples = [
        ("MPO", "flu 20-5 vessel 1", (1400, 1100), (0, 250)),
        (
            "Cleaved caspase 3",
            "covid 2 alveolar 1-1",
            (1200, 1670),
            (1670, 1200),
        ),
        ("cd163", "covid 1 airway 1-1", (100, 500), (1000, 500)),
        (
            "Cleaved caspase 3",
            "nl114 alveolar 1-2",
            (1200, 1670),
            (1670, 1200),
        ),
        (
            "Cleaved caspase 3",
            "nl114 alveolar 1-2",
            (1350, 1600),
            (1650, 1500),
        ),
    ]
    for marker, name, start, end in examples:
        image = [
            i for i in col.images if i.marker == marker and i.name == name
        ][0]
        img = image.image
        mask = image.mask
        hema, dab = image.decompose_hdab()

        # old quantification
        quant = col.quantification.query(
            f"image == '{image.name}' and marker == '{marker}'"
        )

        props = pd.DataFrame(
            regionprops_table(mask, properties=["centroid"]), index=quant.index
        )
        props.columns = ["y", "x"]
        props = props.sort_index(axis=1)

        fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
        axs[0].set_ylabel(image.name)
        axs[0].imshow(img, rasterized=True)
        axs[1].imshow(hema, cmap=cmap_hema)  # , vmin=0.35)
        axs[2].imshow(dab, cmap=cmap_dab)  # , vmin=0.4)
        axs[3].imshow(mask, rasterized=True, cmap=get_random_label_cmap())
        axs[4].imshow(img, rasterized=True)
        axs[4].contour(
            mask, levels=1, cmap="Reds", vmin=-0.2, vmax=0, linewidths=0.5
        )

        x = (props["x"] >= start[0]) & (props["x"] <= start[1])
        y = (props["y"] >= end[0]) & (props["y"] <= end[1])

        for cell in props.loc[x | y].index:
            xy = props.loc[cell]
            axs[4].text(*xy, f"{quant.loc[cell, 'diaminobenzidine']:.2f}")

        for ax in axs:
            ax.set(xlim=start, ylim=end)


class Extra:
    @staticmethod
    def _move_box_folders() -> None:
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

    @staticmethod
    def _remove_wrong_masks(col) -> None:
        """
        At some point some original images got copied to the mask file.
        This removes them.
        """
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

    @staticmethod
    def _remove_wrong_masks2(col) -> None:
        """"""
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
