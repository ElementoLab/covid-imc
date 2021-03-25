# coding: utf-8

"""
This script downloads all raw/processed data necessary to re-run most analysis.

Please note that it may be impractical to download everything serially as here,
so this script is more of inspiration on which files should be placed where.

You will need to create a new access token from Zenodo
(https://zenodo.org/account/settings/applications/tokens/new/), and add this to a
file "~/.zenodo.auth.json" as a simple key: value pair e.g.:
    {'access_token': '123asd123asd123asd123asd123asd'}

Feel free to download datasets manually from Zenodo directly/manually as well:

 - IMC dataset, raw MCD files: https://zenodo.org/record/4110560
 - IMC dataset, cell masks: https://zenodo.org/record/4139443
 - IMC activation panel dataset, MCD and masks: https://zenodo.org/record/4637034
 - IHC dataset: https://zenodo.org/record/4633906
 - GeoMx dataset: https://zenodo.org/record/4635286

"""

import io, sys, json, tempfile, argparse
from typing import Tuple, Dict, List, Optional, Callable, Any
import hashlib

from tqdm import tqdm
import requests
import numpy as np
import pandas as pd

from imc.types import Path

paths = [
    Path("metadata"),
    Path("data"),
    Path("data") / "ihc",
    Path("data") / "geomx",
    Path("processed"),
    Path("results"),
]
for path in paths:
    path.mkdir(exist_ok=True)


api_root = "https://zenodo.org/api/"
headers = {"Content-Type": "application/json"}
secrets_file = Path("~/.zenodo.auth.json").expanduser()
secrets = json.load(open(secrets_file))

kws = dict(params=secrets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    deps = {
        "imc_raw": 4110560,
        "imc_proc": 4110560,
        "imc_activation": 4637033,
        "ihc": 4633906,
        "geomx": 4635286,
    }
    for name, dep_id in deps.items():
        print(f"Downloading '{name}' dataset.")
        dep = get(dep_id)
        for file in dep["files"]:
            output_file = Path(file["filename"])
            output_file.parent.mkdir(parents=True)
            if output_file.exists() and not args.overwrite:
                print(f"File '{output_file}' already exists, skipping.")
                continue
            print(f"Downloading '{output_file}'")
            while True:
                download_file(file["links"]["download"], output_file)
                if get_checksum(output_file) == file["checksum"]:
                    break
                print("Checksum failed to match. Re-trying download.")
    return 0


def get(deposit_id: int) -> Dict[str, Any]:
    return requests.get(
        f"{api_root}deposit/depositions/{deposit_id}", **kws
    ).json()


def download_file(url: str, output_file: Path) -> None:
    with requests.get(url, stream=True, **kws) as r:
        r.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def get_checksum(filename: Path, algo: str = "md5") -> str:
    """Return checksum of file contents."""
    import hashlib

    with open(filename, "rb") as f:
        file_hash = getattr(hashlib, algo)()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\t - Exiting due to user interruption.")
        sys.exit(1)
