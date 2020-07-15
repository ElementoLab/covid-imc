#!/usr/bin/env python

"""
Generate simple illustrations of channels and ROIs for visual inspection.
"""

from tqdm import tqdm
import parmap

from imc.operations import measure_channel_background

from src.config import *


# QC
# # Signal per channel for all ROIs
for c in tqdm(channels_include):
    if (qc_dir / prj.name + f".all_rois.{c}.pdf").exists():
        continue
    print(c)
    fig = prj.plot_channels(c, save=True)
    plt.close(fig)

for roi in prj.rois:
    f = qc_dir / roi.name + ".all_channels.pdf"
    if f.exists():
        continue
    fig = roi.plot_channels(roi.channel_labels.tolist())
    fig.savefig(f)
    plt.close(fig)


def plot_illustrations(roi):
    for colors, chs in illustration_channel_list:
        label = "-".join([f"{k}:{v}" for k, v in zip(colors, chs)])
        _f = illustration_dir / roi.name + f".{label}.pdf"
        if _f.exists():
            continue
        _fig = roi.plot_channels(chs, output_colors=colors, merged=True)
        _fig.savefig(f, dpi=600, bbox_inches="tight")
        plt.close(_fig)


parmap.map(plot_illustrations, prj.rois)


# # Segmentation
for sample in prj.samples:
    f = (
        prj.results_dir / "qc" / sample.name
        + ".probabilities_and_segmentation.pdf"
    )
    if f.exists():
        continue
    fig = sample.plot_probabilities_and_segmentation()
    fig.savefig(f)
    plt.close(fig)


# # Signal
for sample in prj.samples:
    measure_channel_background(
        sample.rois, output_prefix=prj.results_dir / "qc" / sample.name
    )
    plt.close("all")
