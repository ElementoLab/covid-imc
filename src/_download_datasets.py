# coding: utf-8

"""
This script downloads all raw/processed data necessary to re-run most analysis.

Please note that it may be impractical to download everything serially as here,
so this script is more of inspiration on which files should be placed where.


Feel free to download datasets manually from Zenodo as well:

 - IMC dataset, raw MCD files: https://zenodo.org/deposit/4110560
 - IMC dataset, cell masks: https://zenodo.org/deposit/4139443
 - IHC dataset: https://zenodo.org/record/4633906
 - GeoMx dataset: https://zenodo.org/record/4635286

"""

import io, sys, json, tempfile
from typing import Tuple, Dict, List, Optional, Callable

from tqdm import tqdm
import requests
import numpy as np
import pandas as pd

from imc.types import Path

metadata_dir = Path("metadata")
metadata_dir.mkdir(exist_ok=True)
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
processed_dir = Path("processed")
processed_dir.mkdir(exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

ihc_dir = data_dir / "ihc"
ihc_dir.mkdir(exist_ok=True)
geomx_dir = data_dir / "geomx"
geomx_dir.mkdir(exist_ok=True)


def main():
    ...


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\t - Exiting due to user interruption.")
        sys.exit(1)
