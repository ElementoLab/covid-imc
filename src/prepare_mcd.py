#!/usr/bin/env python

import argparse

import pandas as pd

from imc.types import Path
from imc.utils import mcd_to_dir


parser = argparse.ArgumentParser()
parser.add_argument(dest="mcd_file", type=Path)
parser.add_argument(dest="pannel_csv", type=Path)
parser.add_argument("-o", "--output-dir", dest="output_dir", type=Path)
parser.add_argument("-n", "--sample-name", dest="sample_name", type=str)
parser.add_argument("-p", "--partition-panels", dest="partition_panels", action="store_true")
parser.add_argument("--filter-full", dest="filter_full", action="store_true")
parser.add_argument("--overwrite", dest="overwrite", action="store_true")
parser.add_argument("--no-empty-rois", dest="allow_empty_rois", action="store_false")
parser.add_argument("--only-crops", dest="only_crops", action="store_true")
parser.add_argument("--n-crops", dest="n_crops", type=int)
parser.add_argument("--crop-width", dest="crop_width", type=int)
parser.add_argument("--crop-height", dest="crop_height", type=int)
parser.add_argument(
    "-k", "--keep-original-names", dest="keep_original_roi_names", action="store_true"
)
args = parser.parse_args()

print("Started!")

mcd_to_dir(**{k: v for k, v in parser.parse_args().__dict__.items() if v is not None})

print("Finished!")
