# COVID19 profiling of lung tissue with imaging mass cytometry (IMC)

[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.4110560.svg)](https://doi.org/10.5281/zenodo.4110560)
[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.4139443.svg)](https://doi.org/10.5281/zenodo.4139443)
[![PEP compatible](http://pepkit.github.io/img/PEP-compatible-green.svg)](http://pep.databio.org/)

[![medRxiv badge](https://zenodo.org/badge/doi/10.1101/2020.10.26.20219584.svg)](https://doi.org/10.1101/2020.10.26.20219584) ⬅️ read the preprint here

## Organization

- The [metadata](metadata) directory contains metadata relevant to annotate the samples
- [This CSV file](metadata/samples.csv) is the master record of all analyzed samples
- The [src](src) directory contains source code used to analyze the data
- Raw data (i.e. MCD files) will be under the `data` directory.
- Processing of the data will create TIFF files under the `processed`  directory.
- Outputs from the analysis will be present in a `results` directory, with subfolders pertaining to each part of the analysis as described below.


Raw data in the form of MCD files are hosted on WCM's enterprise version of Box.com. An account is needed to download the files, which can be made programmatically with the [imctransfer](https://github.com/ElementoLab/imctransfer) program.
For now you'll need a developer token to connect to box.com programmatically. Place the credentials in a JSON file in `~/.imctransfer.auth.json`.

Pre-processing of the MCD files into images is done with [imcpipeline](https://github.com/ElementoLab/imcpipeline).
Be sure to make the file read-only (e.g. `chmod 400 ~/.imctransfer.auth.json`).

## Reproducibility

### Running

To see all available steps type:
```bash
$ make
```

Steps used for the initiall processing of raw data are marked with the `[dev]` label.
```
Makefile for the covid-imc project/package.
Available commands:
help                Display help and quit
requirements        Install Python requirements
transfer            [dev] Transfer data from wcm.box.com to local environment (to run internally at WCM)
prepare             [dev] Run first step of conversion of MCD to various files (should be done only when processing files from MCD files)
process_local       [dev] Run IMC pipeline locally (should be done only when processing files from MCD files)
process_scu         [dev] Run IMC pipeline on SCU (should be done only when processing files from MCD files)
run_locally         [dev] Alternative way to run the processing on a local computer (should be done only when processing files from MCD files)
run                 [dev] Alternative way to run the processing on a SLURM cluster (should be done only when processing files from MCD files)
checkfailure        [dev] Check whether any samples failed during preprocessing
fail                [dev] Check whether any samples failed during preprocessing
checksuccess        [dev] Check which samples succeded during preprocessing
succ                [dev] Check which samples succeded during preprocessing
rename_forward      [dev] Rename outputs from CellProfiler output to values expected by `imc`
rename_back         [dev] Rename outputs from values expected by `imc` to CellProfiler
merge_runs          [dev] Merge images from the same acquisition that were in multiple MCD files (should be done only when processing files from MCD files)
sync                [dev] Sync data/code to SCU server (should be done only when processing files from MCD files)
upload_data         [dev] Upload processed files to Zenodo (TODO: upload image stacks)
download_data       [TODO!] Download processed data from Zenodo (for reproducibility)
analysis            Run the actual analysis
```

To reproduce analysis using the pre-preocessed data, one would so:

```bash
$ make help
$ make requirements   # install python requirements using pip
$ make download_data  # download stacks and masks from Zenodo
$ make analysus       # run the analysis scripts
```

#### Requirements

- Python 3.7+ (was run on 3.8.2)
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`.

Feel free to use some virtualization or compartimentalization software such as virtual environments or conda to install the requirements.

#### Virtual environment

It is recommended to compartimentalize the analysis software from the system's using virtual environments, for example.

Here's how to create one with the repository and installed requirements:

```bash
git clone git@github.com:ElementoLab/covid-imc.git
cd covid-imc
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Additional datasets

### Desai GeoMx

### IHC

660 H-DAB images for 4 markers (MPO, CD163, CD8 and Cleaved Caspase 3) across all disease groups are available.

Raw images and segmentation masks are available from the [following file](metadata/ihc_files.image_mask_urls.json).

In order to reduce disk space consumption, files are kept online and only downloaded when needed.

The workflow is the following:
Single nucleus are segmentated with [Stardist](https://github.com/mpicbg-csbd/stardist) using the [**he_heavy_augment** model](https://github.com/stardist/stardist-imagej/blob/master/src/main/resources/models/2D/he_heavy_augment.zip).

Images are decomposed into Hematoxylin and DAB components and each cell is quantified for the abundance of either marker. Positive cells are declared using a mixture of gaussian models. Intensity and percentage of positive cells are compared between patients, compartments within the tisse and disease groups.
