# COVID19 profiling of lung tissue with imaging mass cytometry (IMC)

[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.4110560.svg)](https://doi.org/10.5281/zenodo.4110560)
[![PEP compatible](http://pepkit.github.io/img/PEP-compatible-green.svg)](http://pep.databio.org/)


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
$ make help
```
```
Makefile for the covid-imc project/package.
Available commands:
help            Display help and quit
requirements    Install Python requirements
transfer        Transfer data from wcm.box.com to local environment
prepare         Run first step of convertion of MCD to various files
process_local   Run IMC pipeline locally
process_scu     Run IMC pipeline on SCU
checkfailure    Check whether any samples failed during preprocessing
fail            Check whether any samples failed during preprocessing
checksuccess    Check which samples succeded during preprocessing
succ            Check which samples succeded during preprocessing
rename_outputs  Rename outputs from CellProfiler output to values expected by `imc`
rename_outputs_back     Rename outputs from values expected by `imc` to CellProfiler
sync            Sync code to SCU server
```

To reproduce analysis, simply do:

```bash
$ make requirements
$ make
```

### Requirements

- Python 3.7+ (was run on 3.8.2)
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`.


### Virtual environment

It is recommended to compartimentalize the analysis software from the system's using virtual environments, for example.

Here's how to create one with the repository and installed requirements:

```bash
git clone git@github.com:ElementoLab/covid-imc.git
cd covid-imc
virtualenv .venv
source activate .venv/bin/activate
pip install -r requirements.txt
```
