#!/usr/bin/env python

"""
This script was used to upload data to Zenodo following the revision of the manuscript.

It updates the metadata of the previous Zenodo submission(s),
uploads a IHC dataset, a GeoMx dataset, and the IMC activation panel.
"""

import sys, json, requests, hashlib, time
from typing import Dict, Any

import pandas as pd

from imc.types import Path

from src.config import roi_exclude_strings


secrets_file = Path("~/.zenodo.auth.json").expanduser()
secrets = json.load(open(secrets_file))
zenodo_ihc_json = Path("zenodo.deposition.ihc.json")
zenodo_geomx_json = Path("zenodo.deposition.geomx.json")
api_root = "https://zenodo.org/api/"
headers = {"Content-Type": "application/json"}
kws = dict(params=secrets)

title = "The spatial landscape of lung pathology during COVID-19 progression"
abstract = "Recent studies have provided insights into the pathology and immune response to coronavirus disease 2019 (COVID-19). However thorough interrogation of the interplay between infected cells and the immune system at sites of infection is lacking. We use high parameter imaging mass cytometry9 targeting the expression of 36 proteins, to investigate at single cell resolution, the cellular composition and spatial architecture of human acute lung injury including SARS-CoV-2. This spatially resolved, single-cell data unravels the disordered structure of the infected and injured lung alongside the distribution of extensive immune infiltration. Neutrophil and macrophage infiltration are hallmarks of bacterial pneumonia and COVID-19, respectively. We provide evidence that SARS-CoV-2 infects predominantly alveolar epithelial cells and induces a localized hyper-inflammatory cell state associated with lung damage. By leveraging the temporal range of COVID-19 severe fatal disease in relation to the time of symptom onset, we observe increased macrophage extravasation, mesenchymal cells, and fibroblasts abundance concomitant with increased proximity between these cell types as the disease progresses, possibly as an attempt to repair the damaged lung tissue. This spatially resolved single-cell data allowed us to develop a biologically interpretable landscape of lung pathology from a structural, immunological and clinical standpoint. This spatial single-cell landscape enabled the pathophysiological characterization of the human lung from its macroscopic presentation to the single-cell, providing an important basis for the understanding of COVID-19, and lung pathology in general."


def main() -> int:
    # Test connection
    req = requests.get(api_root + "deposit/depositions", **kws)
    assert req.ok

    update_metadata()

    upload_ihc()

    return 0


def update_metadata() -> None:
    # Update bucket metadata
    deps = [("raw data", {"id": 4110560}), ("processed data", {"id": 4139443})]
    for name, dep in deps:
        # renew the metadata:
        dep = get()

        # Update metadata

        # # Title
        if dep["metadata"]["title"] != f"{title} - {name}":
            dep["metadata"]["title"] = f"{title} - {name}"
            put(dep)

        # # Abstract
        if dep["metadata"]["description"] != abstract:
            dep["metadata"]["description"] = abstract
            put(dep)

        # # Authors
        authors_meta = pd.read_csv("metadata/authors.csv")
        if len(dep["metadata"]["creators"]) != authors_meta.shape[0]:
            authors = authors_meta[["name", "affiliation", "orcid"]].T.to_dict()
            authors = [
                {k2: v2 for k2, v2 in v.items() if not pd.isnull(v2)}
                for k, v in authors.items()
            ]
            dep["metadata"]["creators"] = authors
            put(dep)


def upload_ihc():
    # Get a new bucket or load existing
    if not zenodo_ihc_json.exists():
        req = requests.post(
            api_root + "deposit/depositions",
            json={},
            **kws,
        )
        json.dump(req.json(), open(zenodo_ihc_json, "w"))
    dep = json.load(open(zenodo_ihc_json, "r"))
    # renew the metadata:
    dep = get()  # {"id": 4633905}

    # Add metadata
    authors_meta = pd.read_csv("metadata/authors.csv")
    dep["metadata"] = json.load(open("metadata/zenodo_metadata.ihc.json"))[
        "metadata"
    ]
    authors = authors_meta[["name", "affiliation", "orcid"]].T.to_dict()
    authors = [
        {k2: v2 for k2, v2 in v.items() if not pd.isnull(v2)}
        for k, v in authors.items()
    ]
    dep["metadata"]["creators"] = authors
    put(dep)

    # Upload files
    bucket_url = dep["links"]["bucket"] + "/"
    # 'https://zenodo.org/api/files/6a4ac068-d7e5-419c-9e05-77bf99ad780a/'

    # # Upload OME-TIFF files and masks
    data_dir = Path("data")
    ihc_files = sorted(map(str, (data_dir / "ihc").glob("*/*.tif*")))
    ihc_files = [f for f in ihc_files if "MPO" in f or "cd163" in f]
    finished = False
    while not finished:
        dep = get()
        uploaded_files = [x["filename"] for x in dep["files"]]
        ihc_files = [x for x in ihc_files if x not in uploaded_files]
        if not ihc_files:
            break
        file = ihc_files[0]
        for file in tqdm(ihc_files[ihc_files.index(file) :]):
            try:
                upload(file)
                if file == ihc_files[-1]:
                    finished = True
            except requests.exceptions.ConnectionError:
                pass
        time.sleep(5)

    # # Upload metadata
    upload("metadata/ihc_metadata.csv")

    # # Upload quantification
    upload("data/ihc/quantification_hdab.gated_by_image.csv")


def upload_geomx() -> None:
    # Get a new bucket or load existing
    if not zenodo_geomx_json.exists():
        req = requests.post(
            api_root + "deposit/depositions",
            json={},
            **kws,
        )
        json.dump(req.json(), open(zenodo_geomx_json, "w"))
    dep = json.load(open(zenodo_geomx_json, "r"))
    # renew the metadata:
    dep = get()  # {"id": 4635286}

    # Add metadata
    authors_meta = pd.read_csv("metadata/authors.csv")
    dep["metadata"] = json.load(open("metadata/zenodo_metadata.geomx.json"))[
        "metadata"
    ]
    authors = authors_meta[["name", "affiliation", "orcid"]].T.to_dict()
    authors = [
        {k2: v2 for k2, v2 in v.items() if not pd.isnull(v2)}
        for k, v in authors.items()
    ]
    dep["metadata"]["creators"] = authors
    put(dep)

    # Upload files
    bucket_url = dep["links"]["bucket"] + "/"
    # 'https://zenodo.org/api/files/a41d8925-e4ec-454c-93c2-7cb97f7ee854/'

    # # Upload metadata
    upload("data/geomx/metadata_matrix.pq")

    # # Upload data
    upload("data/geomx/expression_matrix.pq")


def upload_imc_activation() -> None:
    ...


def get() -> Dict[str, Any]:
    return requests.get(
        api_root + f"deposit/depositions/{dep['id']}", **kws
    ).json()


def put(payload: Dict, check: bool = True) -> requests.models.Response:
    """
    Raises:
        `AssertionError` if `check` is True and response not ok.
    """
    r = requests.put(
        api_root + f"deposit/depositions/{payload['id']}",
        data=json.dumps(payload),
        headers=headers,
        **kws,
    )
    if check:
        assert r.ok
    return r


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
