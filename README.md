# COVID19 profiling of lung tissue with imaging mass cytometry (IMC)

[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.4110560.svg)](https://doi.org/10.5281/zenodo.4110560) ⬅️ Raw IMC data <br>
[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.4139443.svg)](https://doi.org/10.5281/zenodo.4139443) ⬅️ Processed IMC data <br>
[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.4637034.svg)](https://doi.org/10.5281/zenodo.4637034) ⬅️ 2nd IMC panel data <br>
[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.4633905.svg)](https://doi.org/10.5281/zenodo.4633905) ⬅️ Immunohistochemistry data <br>
[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.4635285.svg)](https://doi.org/10.5281/zenodo.4635285) ⬅️ Targeted spatial transcriptomics data <br>

[![medRxiv badge](https://zenodo.org/badge/doi/10.1101/2020.10.26.20219584.svg)](https://doi.org/10.1101/2020.10.26.20219584) ⬅️ read the preprint here

[![PEP compatible](http://pepkit.github.io/img/PEP-compatible-green.svg)](http://pep.databio.org/)

## Organization

- The [metadata](metadata) directory contains metadata relevant to annotate the samples
- [This CSV file](metadata/samples.csv) is the master record of all analyzed samples
- The [src](src) directory contains source code used to analyze the data
- Raw data (i.e. MCD files) will be under the `data` directory.
- Processing of the data will create TIFF files under the `processed`  directory.
- Outputs from the analysis will be present in a `results` directory, with subfolders pertaining to each part of the analysis as described below.

To download files from Zenodo programatically create an access token (https://zenodo.org/account/settings/applications/tokens/new/), and add this to a file `~/.zenodo.auth.json` as a simple key: value pair e.g.: `{'access_token': '123asd123asd123asd123asd123asd'}`.
Be sure to make the file read-only (e.g. `chmod 400 ~/.zenodo.auth.json`).


## Reproducibility

### Running

To see all available steps type:
```bash
$ make
```
```
Makefile for the covid-imc project.
Available commands:
help                Display help and quit
requirements        Install Python requirements
download_data       Download all data from Zenodo
analysis            Run the actual analysis
```

To reproduce analysis using the pre-preocessed data, one would so:

```bash
$ make requirements   # install python requirements using pip
$ make download_data  # download data from Zenodo
$ make analysis       # run the analysis scripts
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

### Immunohistochemistry (IHC)

383 H-DAB images for two markers (MPO, and CD163) across all disease groups are available.

Raw images and segmentation masks are available here: https://doi.org/10.5281/zenodo.4633905.

The workflow is the following:
Single nucleus are segmentated with [Stardist](https://github.com/mpicbg-csbd/stardist) using the *2D_versatile_he* model.

Images are decomposed into Hematoxylin and DAB components and each cell is quantified for the abundance of either marker. Positive cells are declared using a mixture of gaussian models. Intensity and percentage of positive cells are compared between patients, compartments within the tisse and disease groups.

### Targeted spatial transcriptomics (GeoMx)

Newly generated data is available here: https://doi.org/10.5281/zenodo.4635285.
A script used to load and analyze the dataset is available here: [src/geomx.py](src/geomx.py).

### Reanalysis of targeted spatial transcriptomics data from [Desai et al](https://doi.org/10.1038/s41467-020-20139-7)

A script used to get the dataset and analise it is available here: [src/geomx_desai.py](src/geomx_desai.py).
