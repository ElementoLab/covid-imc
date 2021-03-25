# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# Makefile for the "covid-imc" project.
# This specifies the steps to run and their order and allows running them.
# Type `make` for instructions. Type make <command> to execute a command.
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.DEFAULT_GOAL := help

NAME=$(shell basename `pwd`)
SAMPLES=$(shell ls data)

help:  ## Display help and quit
	@echo Makefile for the $(NAME) project/package.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'

requirements:  ## Install Python requirements
	pip install -r requirements.txt

transfer:  ## [dev] Transfer data from wcm.box.com to local environment (to run internally at WCM)
	imctransfer -q 2020  # Query for files produced in 2020 only

prepare:  ## [dev] Run first step of conversion of MCD to various files (should be done only when processing files from MCD files)
	@echo "Running prepare step for samples: $(SAMPLES)"
	for SAMPLE in $(SAMPLES); do \
	python src/prepare_mcd.py \
			--n-crops 1 \
			data/$${SAMPLE}/$${SAMPLE}.mcd \
			metadata/panel_markers.COVID19-2.csv \
			-o processed/$${SAMPLE}; \
	done

process_local:  ## [dev] Run IMC pipeline locally (should be done only when processing files from MCD files)
	for SAMPLE in $(SAMPLES); do
		# python ~/projects/imcpipeline/imcpipeline/pipeline.py \
		python -m imcpipeline.pipeline \
		--ilastik-model _models/COVID19-2/COVID19-2.ilp \
		--csv-pannel metadata/panel_markers.COVID19-2.csv \
		--container docker \
		-i data/$${SAMPLE} \
		-o processed/$${SAMPLE} \
		-s predict,segment
	done

process_scu:  ## [dev] Run IMC pipeline on SCU (should be done only when processing files from MCD files)
	for SAMPLE in $(SAMPLES); do
		# python ~/projects/imcpipeline/imcpipeline/pipeline.py \
		python -m imcpipeline.pipeline \
		--ilastik-model _models/COVID19-2/COVID19-2.ilp \
		--csv-pannel metadata/panel_markers.COVID19-2.csv \
		--cellprofiler-exec \
			"source ~/.miniconda2/bin/activate && conda activate cellprofiler && cellprofiler" \
		-i data/$${SAMPLE} \
		-o processed/$${SAMPLE} \
		-s predict,segment
	done

run_locally: ##  [dev] Alternative way to run the processing on a local computer (should be done only when processing files from MCD files)
	imcrunner \
		--divvy local \
		metadata/samples.initial.csv \
			--ilastik-model _models/COVID19-2/COVID19-2.ilp \
			--csv-pannel metadata/panel_markers.COVID19-2.csv \
			--container docker

run: ## [dev] Alternative way to run the processing on a SLURM cluster (should be done only when processing files from MCD files)
	imcrunner \
		--divvy slurm \
		metadata/samples.initial.csv \
			--ilastik-model _models/COVID19-2/COVID19-2.ilp \
			--csv-pannel metadata/panel_markers.COVID19-2.csv \
			--cellprofiler-exec \
				"source ~/.miniconda2/bin/activate && conda activate cellprofiler && cellprofiler"


checkfailure:  ## [dev] Check whether any samples failed during preprocessing
	grep -H "Killed" submission/*.log && \
	grep -H "Error" submission/*.log && \
	grep -H "CANCELLED" submission/*.log && \
	grep -H "exceeded" submission/*.log

fail: checkfailure  ## [dev] Check whether any samples failed during preprocessing

checksuccess:  ## [dev] Check which samples succeded during preprocessing
	ls -hl processed/*/cpout/cell.csv

succ: checksuccess  ## [dev] Check which samples succeded during preprocessing


rename_forward:  ## [dev] Rename outputs from CellProfiler output to values expected by `imc`
	find processed \
		-name "*_ilastik_s2_Probabilities.tiff" \
		-exec rename "s/_ilastik_s2_Probabilities/_Probabilities/g" \
		{} \;
	find processed \
		-name "*_ilastik_s2_Probabilities_mask.tiff" \
		-exec rename "s/_ilastik_s2_Probabilities_mask/_full_mask/g" \
		{} \;
	find processed \
		-name "*_ilastik_s2_Probabilities_NucMask.tiff" \
		-exec rename "s/_ilastik_s2_Probabilities_NucMask/_full_nucmask/g" \
		{} \;

rename_back:  ## [dev] Rename outputs from values expected by `imc` to CellProfiler
	find processed \
		-name "*_Probabilities.tiff" \
		-exec rename "s/_Probabilities/_ilastik_s2_Probabilities/g" \
		{} \;
	find processed \
		-name "*_full_mask.tiff" \
		-exec rename "s/_full_mask/_ilastik_s2_Probabilities_mask/g" \
		{} \;
	find processed \
		-name "*_full_nucmask.tiff" \
		-exec rename "s/_full_nucmask/_ilastik_s2_Probabilities_NucMask/g" \
		{} \;

merge_runs:  ## [dev] Merge images from the same acquisition that were in multiple MCD files (should be done only when processing files from MCD files)
	python -u src/_merge_runs.py

backup_time:
	echo "Last backup: " `date` >> _backup_time
	chmod 700 _backup_time

_sync:
	rsync --copy-links --progress -r \
	. afr4001@pascal.med.cornell.edu:projects/$(NAME)

sync: _sync backup_time ## [dev] Sync data/code to SCU server (should be done only when processing files from MCD files)


upload_data: ## [dev] Upload processed files to Zenodo (TODO: upload image stacks, activation IMC)
	@echo "Warning: this step is not meant to be run, but simply details how datasets were uploaded."
	python -u src/_upload.py  # Used in the first data deposition in Sept 2020
	python -u src/_upload_update.py ## Update metadata and add validation datasets (2021)

download_data: ## Download processed data from Zenodo (for reproducibility)
	@echo "Not yet implemented!"
	python -u src/_download_datasets.py

analysis:  ## Run the actual analysis
	@echo "Running analysis!"

	# Analysis of first submission
	# # Global look at the clinical data [Fig1b]
	python -u src/clinical.py
	# # General analysis of lung tissue [Fig1 c-f]
	python -u src/pathology.py
	# # Phenotyping by clustering single cells [Fig1 g-o]
	python -u src/cell_type.py
	python -u src/macrophage_diversity.py
	# # Declare marker positivity [Fig2 a-e]
	python -u src/gating.py
	# # Cell-cell interactions [Fig2 f-l]
	python -u src/interaction.py
	# # Unsupervised analysis [Fig3]
	python -u src/unsupervised.py
	# # Various illustrations of IMC data
	python -u src/illustration.py
	# # Supplementary Tables [TODO: not committed yet]
	python -u src/supplement.py


	# Revision work:
	# # Analysis of commorbidities [reviewer figure]
	python -u src/supervised.py

	# # GeoMx data:
	python -u src/geomx.py
	python -u src/geomx_desai.py
	# # Bulk RNA-seq data:
	python -u src/rnaseq.py

	# # IHC data:
	python -u src/ihc.py

	# # Additional IMC data
	python -u src/imc_activation_panel.py


figures:  ## Produce figures in various formats
	cd figures; bash process.sh


.PHONY : help \
	requirements \
	transfer \
	prepare \
	process_local process_scu run_locally run \
	checkfailure fail checksuccess succ \
	rename_forward rename_back \
	merge_runs \
	sync \
	upload_data \
	download_data \
	analysis \
	figures
