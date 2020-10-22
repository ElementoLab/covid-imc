"""
Generate simple illustrations of channels and ROIs for visual inspection.
"""

from tqdm import tqdm
import parmap

from imc.operations import measure_channel_background

from src.config import *


output_dir = results_dir / "illustration"
output_dir.mkdir(exist_ok=True, parents=True)


# QC
# # Signal per channel for all ROIs
for c in tqdm(channels_include):
    f = output_dir / prj.name + f".all_rois.{c}.pdf"
    if f.exists():
        continue
    print(c)
    fig = prj.plot_channels(c, save=True)
    fig.savefig(f, **figkws)
    plt.close(fig)

for roi in prj.rois:
    f = output_dir / roi.name + ".all_channels.pdf"
    if f.exists():
        continue
    fig = roi.plot_channels(roi.channel_labels.tolist())
    fig.savefig(f, **figkws)
    plt.close(fig)


for roi in prj.rois:
    f = output_dir / roi.name + ".mean.svgz"
    # if f.exists():
    # continue
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    roi.plot_channel("mean", ax=ax)
    fig.savefig(f, **figkws)
    plt.close(fig)
# Plot combination of markers
output_dir = results_dir / "marker_illustration"
output_dir.mkdir(exist_ok=True, parents=True)


def plot_illustrations(roi):
    for colors, chs in illustration_channel_list:
        label = "-".join([f"{k}:{v}" for k, v in zip(colors, chs)])
        _f = output_dir / roi.name + f".{label}.pdf"
        if _f.exists():
            continue
        _fig = roi.plot_channels(chs, output_colors=colors, merged=True)
        _fig.savefig(_f, dpi=600, bbox_inches="tight")
        plt.close(_fig)


parmap.map(plot_illustrations, prj.rois)


# # Segmentation
output_dir = results_dir / "segmentation_illustration"
output_dir.mkdir(exist_ok=True, parents=True)
for sample in prj.samples:
    f = output_dir / sample.name + ".probabilities_and_segmentation.pdf"
    if f.exists():
        continue
    fig = sample.plot_probabilities_and_segmentation()
    fig.savefig(f)
    plt.close(fig)


# # Signal
output_dir = results_dir / "marker_illustration"
output_dir.mkdir(exist_ok=True, parents=True)

for sample in prj.samples:
    measure_channel_background(
        sample.rois, output_prefix=output_dir / sample.name
    )
    plt.close("all")
