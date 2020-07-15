.DEFAULT_GOAL := all

NAME=$(shell basename `pwd`)
SAMPLES=$(shell ls data)

help:  ## Display help and quit
	@echo Makefile for the $(NAME) project/package.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'

all: install clean test

requirements:  ## Install Python requirements
	pip install -r requirements.txt

transfer:  ## Transfer data from wcm.box.com to local environment
	imctransfer -q 202007

prepare:  ##  Run first step of convertion of MCD to various files
	@echo "Running prepare step for samples: $(SAMPLES)"
	for SAMPLE in $(SAMPLES); do \
	python src/prepare_mcd.py \
			--n-crops 1 \
			data/$${SAMPLE}/$${SAMPLE}.mcd \
			metadata/panel_markers.COVID19-2.csv \
			-o processed/$${SAMPLE}; \
	done

process_local:  ## Run IMC pipeline locally
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

process_scu:  ## Run IMC pipeline on SCU
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

run:
	imcrunner \
		--divvy slurm \
		metadata/samples.csv \
			--ilastik-model _models/COVID19-2/COVID19-2.ilp \
			--csv-pannel metadata/panel_markers.COVID19-2.csv \
			--cellprofiler-exec \
				"source ~/.miniconda2/bin/activate && conda activate cellprofiler && cellprofiler"

run_locally:
	imcrunner \
		--divvy local \
		metadata/samples.csv \
			--ilastik-model _models/COVID19-2/COVID19-2.ilp \
			--csv-pannel metadata/panel_markers.COVID19-2.csv \
			--container docker


# singularity pull shub://arcsUVA/cellprofiler:3.1.8
# singularity exec shub://arcsUVA/cellprofiler:3.1.8 cellprofiler

# imcpipeline \
# --container singularity \
# --image arcsUVA/cellprofiler:3.1.8 \
# -i projects/$(NAME)-data/data/20200612_FLU_1923/ \
# -o projects/$(NAME)-data/processed/20200612_FLU_1923


checkfailure:  ## Check whether any samples failed during preprocessing
	grep -H "Killed" submission/*.log && \
	grep -H "Error" submission/*.log && \
	grep -H "CANCELLED" submission/*.log && \
	grep -H "exceeded" submission/*.log

fail: checkfailure  ## Check whether any samples failed during preprocessing

checksuccess:  ## Check which samples succeded during preprocessing
	ls -hl processed/*/cpout/cell.csv

succ: checksuccess  ## Check which samples succeded during preprocessing


rename_outputs:  ## Rename outputs from CellProfiler output to values expected by `imc`
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

rename_outputs_back:  ## Rename outputs from values expected by `imc` to CellProfiler
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

sync:  ## Sync code to SCU server
	rsync --copy-links --progress -r \
	. afr4001@pascal.med.cornell.edu:projects/$(NAME)


.PHONY : move_models_out move_models_in clean_build clean_dist clean_eggs \
clean _install install clean_docs docs run run_locally checkfailure fail checksuccess succ sync
