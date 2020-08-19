#!/usr/bin/env python

"""
Some samples were split in multiple MCD files.
They were processed separately but will now be merged into the same directory.

A new sample annotation will also be written to disk: metadata/samples.csv
"""

from glob import glob
import shutil
from pathlib import Path

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]

from src.config import processed_dir, metadata_dir

origi = (
    pd.read_csv(metadata_dir / "samples.initial.csv")
    .set_index("sample_name")
    .query("toggle == True")
    .drop(["toggle", "complete"], axis=1)
)

# Remove duplicated from CSV
nulls = origi.index.str.extract("(.*)_2$")[0].isnull()
new = origi.loc[nulls.values]
new.to_csv(metadata_dir / "samples.csv")


duplicates = origi.index[~nulls]


file_endings = [
    "_full.csv",
    "_full.tiff",
    "_ilastik_s2.h5",
    "_Probabilities.tiff",
    "_full_mask.tiff",
    "_full_nucmask.tiff",
]

for sample in duplicates:
    ori = sample[:-2]
    # determine ending of first MCD rois
    new_tiff_dir = processed_dir / ori / "tiffs"
    old_tiff_dir = processed_dir / sample / "tiffs"
    last_roi_name = sorted(glob(str(new_tiff_dir / "*_full.tiff")))[-1]
    last_roi_numb = int(last_roi_name.replace("_full.tiff", "")[-2:])

    # now rename and move the second MCD rois
    rois = sorted(glob(str(old_tiff_dir / "*_full.tiff")))

    for roi in rois:
        old_prefix = roi[:-10]
        roi_numb = int(old_prefix[-2:])
        new_numb = last_roi_numb + roi_numb
        new_prefix = old_prefix[:-2].replace("_2", "") + str(new_numb).zfill(2)

        for end in file_endings:
            old = Path(old_prefix + end)
            new = Path(new_prefix + end)
            print(old.exists(), new.exists())
            shutil.move(old, new)

    # Move ilastik training files
    ilastik_files = glob(str(old_tiff_dir.parent / "ilastik/*"))
    for f in ilastik_files:
        shutil.move(f, new_tiff_dir.parent / "ilastik")

    # Remove directory
    shutil.rmtree(processed_dir / sample)
