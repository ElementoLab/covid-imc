#!/usr/bin/env python

"""
Preprocess images and save as uint8 npyz for upload and serving in Heroku app.

Disk space is limited so this script will handle lack of disk space gracefuly
and may be run several times to process/upload/delete the stacks in parts.
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm
import parmap
import skimage as ski
from skimage.exposure import equalize_hist as eq
from boxsdk import (
    OAuth2,
    Client,
    BoxOAuthException,
    BoxAPIException,
)

sys.path.append(".")  # IPython adds current dir to PATH automatically
from src.config import prj, metadata_dir, processed_dir

INTMAX = (2 ** 8) - 1


def minmax_scale(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - x.min()) / (x.max() - x.min())


def process_roi(roi: "ROI") -> int:
    if roi.np_stack_file.exists():
        return 0
    stack = np.asarray(
        [minmax_scale(eq(x)) for x in roi.stack[~roi.channel_exclude]]
    )
    try:
        np.savez(
            roi.np_stack_file, **{"stack": (stack * INTMAX).astype("uint8")}
        )
    # # I ran out of disk space a couple times.
    # # This ensures no corrupted files are left on
    # # the file system if that happens
    except OSError:
        roi.np_stack_file.unlink(missing_ok=True)
        return 1
    return 0


# Path to processed stack file
for roi in prj.rois:
    roi.np_stack_file = processed_dir / roi.sample.name / roi.name + ".npz"


# Store upload metadata as JSON
uploads_file = metadata_dir / "processed_stack.upload_info.json"
try:
    uploads = json.load(open(uploads_file, "r"))
except FileNotFoundError:
    uploads = dict()


# Store list of channel names as JSON
channels_file = metadata_dir / "processed_stack.channel_info.json"
with open(channels_file, "w") as h:
    json.dump(
        roi.channel_labels[~roi.channel_exclude.values].tolist(),
        h,
        sort_keys=True,
        indent=4,
    )


# Process ROI stacks
missing = [
    r
    for r in prj.rois
    if not r.np_stack_file.exists() and r.name not in uploads
]
err = parmap.map(process_roi, missing, pm_pbar=True)


# Upload
# # This is done in a way that accepts ROIs where processed files are missing

# # Read credentials and connect to server
secrets_file = Path("~/.imctransfer.auth.json").expanduser()
secret_params = json.load(open(secrets_file, "r"))
oauth = OAuth2(**secret_params)
client = Client(oauth)

# # This is the folder ID corresponding to "/processed/stacks"
folder_id = "127030520434"


# # Upload ROIs with processed stacks but missing uploads,
# # while updating JSON as it goes.
rois = [
    r for r in prj.rois if r.np_stack_file.exists() and r.name not in uploads
]

for roi in tqdm(rois):
    try:
        f = client.folder(folder_id).upload(roi.np_stack_file)
    except BoxAPIException:
        continue
    uploads[roi.name] = {
        "url": f.get_url(),
        "shared_link": f.get_shared_link(),
        "download_link": f.get_download_url(),
    }

    with open(uploads_file, "w") as h:
        json.dump(uploads, h, sort_keys=True, indent=4)


# Delete stacks of uploaded ROIs
for roi in prj.rois:
    if roi.np_stack_file.exists() and roi.name in uploads:
        roi.np_stack_file.unlink()


# Get one more link
f = client.folder(folder_id)
itms = list(f.get_items())  # this is just so tqdm knows the size
for i in tqdm(itms):
    uploads[i.name.replace(".npz", "")][
        "shared_download_url"
    ] = i.get_shared_link_download_url(access="open")
with open(uploads_file, "w") as h:
    json.dump(uploads, h, sort_keys=True, indent=4)
