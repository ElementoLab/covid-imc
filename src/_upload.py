#!/usr/bin/env python

"""
This script was used to upload IMC data to Zenodo
upon the release of the manuscript as a preprint.

After peer-review, the entries were updated to match
the accepted manuscript using the `_upload_update.py` script.

"""

import sys, json, requests, hashlib
from typing import Dict, Any

import pandas as pd

from imc.types import Path

from src.config import roi_exclude_strings


secrets_file = Path("~/.zenodo.auth.json").expanduser()
secrets = json.load(open(secrets_file))
zenodo_json = Path("zenodo.deposition.proc.json")
api_root = "https://zenodo.org/api/"
headers = {"Content-Type": "application/json"}
kws = dict(params=secrets)


def main():
    # Test connection
    req = requests.get(api_root + "deposit/depositions", **kws)
    assert req.ok

    # Get a new bucket or load existing
    if not zenodo_json.exists():
        req = requests.post(
            api_root + "deposit/depositions",
            json={},
            **kws,
        )
        json.dump(req.json(), open(zenodo_json, "w"))
    dep = json.load(open(zenodo_json, "r"))
    # dep = {"id": 4110560}
    # dep = {"id": 4139443}

    # renew the metadata:
    dep = get()

    # Add metadata
    authors_meta = pd.read_csv("metadata/authors.csv")
    if (
        "creators" not in dep["metadata"]
        or len(dep["metadata"]["creators"]) != authors_meta.shape[0]
    ):
        metadata = json.load(open("metadata/zenodo_metadata.json"))
        authors = authors_meta[["name", "affiliation", "orcid"]].T.to_dict()
        authors = [v for k, v in authors.items()]
        metadata["metadata"]["creators"] = authors
        r = requests.put(
            api_root + f"deposit/depositions/{dep['id']}",
            data=json.dumps(metadata),
            headers=headers,
            **kws,
        )
        assert r.ok

    # Upload files
    samples = pd.read_csv("metadata/samples.csv", index_col=0)
    bucket_url = dep["links"]["bucket"] + "/"
    # 'https://zenodo.org/api/files/0a6fcd1c-58a5-4a91-a85e-7b8e44c3f44f/'
    # 'https://zenodo.org/api/files/d19ebdf6-b560-4b22-89b5-b0d9fdce08d3/'

    # # Upload MCD files
    mcd_files = pd.Series(Path("data").glob("*/*.mcd"))
    mcd_files = mcd_files.loc[
        mcd_files.astype(str).isin(samples["mcd_file"])
    ].astype(str)

    for file in mcd_files:
        upload(file)

    # # # Upload Stacks
    # stack_files = pd.Series(Path("processed").glob("*/tiffs/*_full.tiff"))
    # # match to annotation
    # stack_files = stack_files[
    #     stack_files.apply(lambda x: x.parts[1]).isin(samples.index)
    # ].astype(str)
    # # exclude excluded ROIs (3/240 in total)
    # stack_files = [
    #     x for x in stack_files if not any([y in x for y in roi_exclude_strings])
    # ]
    # existing_files = [x["filename"] for x in dep["files"]]
    # for file in [f for f in stack_files if f not in existing_files]:
    #     upload(file)

    # # # Unfortunately the current quota is not enough to upload all files :'(
    # for file in [f for f in stack_files if f in existing_files]:
    #     delete(file)

    # # Upload Masks
    masks = pd.Series(Path("processed").glob("*/tiffs/*_full_mask.tiff"))
    # match to annotation
    masks = masks[masks.apply(lambda x: x.parts[1]).isin(samples.index)].astype(
        str
    )
    # exclude excluded ROIs (3/240 in total)
    masks = [x for x in masks if not any([y in x for y in roi_exclude_strings])]
    existing_files = [x["filename"] for x in dep["files"]]
    for file in [f for f in masks if f not in existing_files]:
        upload(file)

    # Upload h5ad
    upload("results/covid-imc.h5ad")


def get() -> Dict[str, Any]:
    return requests.get(
        api_root + f"deposit/depositions/{dep['id']}", **kws
    ).json()


def get_file_md5sum(filename: str, chunk_size: int = 8192) -> str:
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def upload(file: str, refresh: bool = False) -> None:
    if refresh:
        exists = [x["filename"] for x in get()["files"]]
    else:
        try:
            exists = dep["existing_files"]
        except KeyError:
            exists = []
    if file in exists:
        print(f"File '{file}' already uploaded.")
        return
    print(f"Uploading '{file}'.")
    with open(file, "rb") as handle:
        r = requests.put(bucket_url + file, data=handle, **kws)
    assert r.ok, f"Error uploading file '{file}': {r.json()['message']}."
    print(f"Successfuly uploaded '{file}'.")

    f = r.json()["checksum"].replace("md5:", "")
    g = get_file_md5sum(file)
    assert f == g, f"MD5 checksum does not match for file '{file}'."
    print(f"Checksum match for '{file}'.")


def delete(file: str, refresh: bool = False) -> None:
    print(f"Deleting '{file}'.")
    if refresh:
        files = get()["files"]
    else:
        files = dep["files"]
    file_ids = [f["id"] for f in files if f["filename"] == file]
    # ^^ this should always be one but just in case
    for file_id in file_ids:
        r = requests.delete(
            api_root + f"deposit/depositions/{dep['id']}/files/{file_id}", **kws
        )
    assert r.ok, f"Error deleting file '{file}', with id '{file_id}'."


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
