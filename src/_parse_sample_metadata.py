#!/usr/bin/env python

"""
Parse clinical metadata from excel file and
annotate samples in a standardized way.
"""

import sys
import json

import numpy as np
import pandas as pd

from src.config import metadata_dir


def main() -> int:
    true_values = ["Y", "yes", "YES", "TRUE", "true", "True"]
    false_values = ["N", "no", "NO", "FALSE", "false", "False"]

    origi = pd.read_csv(metadata_dir / "samples.csv")
    # .query("toggle == True")
    # .drop(["toggle", "complete"], axis=1)

    annot = pd.read_excel(
        metadata_dir
        / "original"
        / "Autopsy Collected Data_210114 without MRN.xlsx",
        na_values=["na", "NA"],
        true_values=true_values,
        false_values=false_values,
    )

    # expand lab data across tissues per patient
    new_lab_vars = ["ESR", "CRP", "Procalcitonin", "IL-6"]
    for code in annot["AutopsyCode"]:
        sel = annot["AutopsyCode"] == code
        v = annot.loc[sel, new_lab_vars].dropna(how="all")
        if not v.empty:
            annot.loc[sel, new_lab_vars] = v.values
    annot["IL6"] = annot["IL-6"]

    # Keep only lung data and cases with autopsy number
    annot = annot.dropna(subset=["AutopsyCode"])
    annot = annot.loc[(annot["Organ"] == "Lung") | (pd.isna(annot["Organ"]))]

    annot["AutopsyCode"] = (
        annot["AutopsyCode"]
        .str.split(" ")
        .apply(lambda x: x[0])
        .str.strip()
        .str.capitalize()
        .str.replace("=", "-")
    )

    annot = annot.drop_duplicates(subset=["AutopsyCode"])

    meta = origi.merge(  # type: ignore[operator]
        annot, how="left", left_on="autopsy_code", right_on="AutopsyCode"
    )

    # Start processing variables in groups of relevance
    # # Clinical
    meta["disease"] = pd.Categorical(
        meta["disease"],
        categories=["Healthy", "FLU", "ARDS", "COVID19"],
        ordered=True,
    )
    meta["organ"] = pd.Categorical(meta["organ"])
    meta["phenotypes"] = pd.Categorical(
        meta["phenotypes"],
        categories=[
            "Healthy",
            "Flu",
            "ARDS",
            "Pneumonia",
            "COVID19_early",
            "COVID19_late",
        ],
        ordered=True,
    )
    meta["classification"] = meta["SampleClassification"]
    meta["cause_of_death_text"] = meta["CauseOfDeath"]

    # # unroll
    death = (
        meta["cause_of_death_text"]
        .str.lower()
        .str.replace(",", "")
        .rename("cause_of_death")
        .rename_axis("sample")
        .to_frame()
        .assign(count=1)
        .pivot_table(
            index="sample",
            columns="cause_of_death",
            values="count",
        )
        .reindex(meta.index)
        .fillna(0)
        .astype(bool)
    ).drop("unknown", axis=1)
    death.columns = "cause_of_death:" + death.columns
    meta = meta.join(death)

    # comorbidities
    meta["comorbidities_exist"] = pd.Categorical(
        meta["Comorbidity"]
        .replace({"N": False, "Y": True})
        .astype(pd.BooleanDtype())
    )
    meta["comorbidities_text"] = meta["ComorbiditySpec"]

    # # unroll
    coms = (
        meta["comorbidities_text"]
        .str.replace("breast ca", "cancer")
        .str.replace("pancreatic ca", "cancer")
        .str.replace("Lung cancer", "cancer")
        .str.replace("valve disease", "valve_disease")
        .str.replace(" Type 2", "")
        .str.replace("Cholangiocancerrcinoma", "Cholangiocarcinoma")
        .str.replace("Sickle cell", "Sickle_cell")
        .str.replace(" and ", ", ")
        .str.replace(" ", ", ")
        .str.split(", ")
        .apply(pd.Series)
        .stack()
        .str.replace(",", "")
        .str.replace(r"^ca$", "cancer", regex=True)
        .str.replace(r"^cancerD$", "cancer", regex=True)
        .str.lower()
        .str.replace("dm2", "diabetes")
        .reset_index(level=1, drop=True)
        .rename("comorbidities")
        .rename_axis("sample")
        .to_frame()
        .assign(count=1)
        .pivot_table(
            index="sample",
            columns="comorbidities",
            values="count",
        )
        .reindex(meta.index)
        .fillna(0)
        .astype(bool)
    ).drop("n", axis=1)
    coms.columns = "comorbidity:" + coms.columns
    meta = meta.join(coms)

    # treatment
    meta["treated"] = pd.Categorical(~meta["Treatment"].isnull(), ordered=True)
    meta["treatment_text"] = meta["Treatment"].replace("None", np.nan)

    # # unroll treatments
    treatments = (
        meta["treatment_text"]
        .str.replace("enoxaparin/heparin HC", "enoxaparin and heparin and HC")
        .str.replace(" and ", ", ")
        .str.split(", ")
        .apply(pd.Series)
        .stack()
        .str.lower()
        .reset_index(level=1, drop=True)
        .rename("treatments")
        .rename_axis("sample")
        .to_frame()
        .assign(count=1)
        .pivot_table(
            index="sample",
            columns="treatments",
            values="count",
        )
        .reindex(meta.index)
        .fillna(0)
        .astype(bool)
    )
    treatments.columns = "treatments:" + treatments.columns
    meta = meta.join(treatments)

    clinical_vars = (
        [
            "disease",
            "organ",
            "phenotypes",
            "classification",
            "cause_of_death_text",
        ]
        + death.columns.tolist()
        + [
            "hospitalization",
            "days_hospitalized",
            "intubated",
            "lung_weight_grams",
            "treated",
            "treatment_text",
        ]
        + treatments.columns.tolist()
        + [
            "comorbidities_exist",
            "comorbidities_text",
        ]
        + coms.columns.tolist()
    )  # type: ignore[attr-defined]

    meta["days_hospitalized"] = (
        meta["Hospitalized"]
        .astype(str)
        .str.extract(r"(\d+).?")[0]
        .fillna(-1)
        .astype(int)
        .replace(-1, np.nan)
        .astype(pd.Int64Dtype())
    )
    meta["hospitalization"] = ~(
        meta["days_hospitalized"].isnull() | meta["days_hospitalized"] == 0
    )
    meta["lung_weight_grams"] = meta["LungWeightG"].astype(pd.Int64Dtype())

    # Lung histology/lesions
    pathology_vars = pd.Series(
        [
            "Airway",
            "AlveolarPMN",
            "ChronicAlveolarInflammationMacrophage",
            "AcuteAlveolarWallInflammation",
            "ChronicAlveolarWallInflammation",
            "Microthrombi",
            "Large thrombi",
            "Hyaline membranes",
            "Type 2 hyperplasia only",
            "Type 2 with fibroblasts",
            "Organizing pneumonia",
            "Squamous metaplasia",
            # "OtherLungLesions",
        ]
    )

    for col in pathology_vars:
        meta["pathology:" + col + "_clean"] = (
            meta[col]
            .astype(str)
            .str.replace(",", "")
            .str.split(" ")
            .apply(lambda x: x[0] if isinstance(x, list) else x)
            .replace(
                {k: True for k in true_values + ["FOCAL", "focal", "BIZARRE"]}
            )
            .replace({k: False for k in false_values})
            .replace("nan", np.nan)
            .astype(pd.BooleanDtype())
        )
        meta["pathology:" + col + "_focal"] = (
            meta[col].astype(str).str.contains("focal", case=False)
        )
        meta["pathology:" + col + "_bizarre"] = (
            meta[col].astype(str).str.contains("bizarre", case=False)
        )

    for col in pathology_vars:
        meta["pathology:" + col] = meta["pathology:" + col + "_clean"]
        meta = meta.drop("pathology:" + col + "_clean", axis=1)

    # # unroll other lung lesions
    lesions = (
        meta["OtherLungLesions"]
        .str.replace(" and ", ", ")
        .str.split(", ")
        .apply(pd.Series)
        .stack()
        .reset_index(level=1, drop=True)
        .rename("lesions")
        .rename_axis("sample")
        .to_frame()
        .assign(count=1)
        .pivot_table(
            index="sample",
            columns="lesions",
            values="count",
        )
        .reindex(meta.index)
        .fillna(0)
        .astype(bool)
    ).drop("N", axis=1)
    lesions.columns = [
        x.replace(" ", "_") for x in "lesions:" + lesions.columns
    ]
    meta = meta.join(lesions)

    #  Convert classes to Categorical
    demographic_vars = [
        "age",
        "gender",
        "race",
        "smoker",
        "race:Black",
        "race:Hispanic",
        "race:White",
    ]
    meta["age"] = meta["Age"].astype(pd.Int64Dtype())
    meta["gender"] = pd.Categorical(
        meta["Gender"]
        .replace("M", "Male")
        .replace("F", "Female")
        .replace("m", "Male")
        .replace("f", "Female"),
        ordered=True,
        categories=["Male", "Female"],
    )
    meta["race"] = meta["Race"].replace("W", "white").str.capitalize()
    races = pd.get_dummies(meta["race"]).astype(bool)
    races.columns = "race:" + races.columns
    meta = meta.join(races)

    meta["smoker"] = (
        meta["Smoke"]
        .str.split(" ")
        .apply(lambda x: x[0] if isinstance(x, list) else x)
        .replace("never", "No")
        .replace("N", "No")
        .replace("former", "Former")
    )

    symptoms = [
        "fever",
        "fever_max_temperature",
        "cough",
        "shortness_of_breath",
    ]
    meta["fever_max_temperature"] = (
        meta["FeverTMax"]
        .astype(str)
        .str.extract(r"(\d+).?")[0]
        .fillna(-1)
        .replace(-1, np.nan)
        .astype(float)
    )
    meta["fever"] = np.nan
    meta.loc[~meta["fever_max_temperature"].isnull(), "fever"] = meta[
        "FeverTMax"
    ].str.contains("reported", case=False).fillna(False) | (
        meta["fever_max_temperature"] >= 38
    )
    meta["cough"] = (
        meta["Cough"]
        .replace("y", True)
        .replace("Y", True)
        .replace("YES", True)
        .replace("N", False)
        .replace("n", False)
        .replace("NO", False)
    )
    meta["shortness_of_breath"] = (
        meta["ShortNessOfBreath"]
        .replace("y", True)
        .replace("Y", True)
        .replace("YES", True)
        .replace("N", False)
        .replace("n", False)
        .replace("NO", False)
    )

    for sympt in ["fever", "cough", "shortness_of_breath"]:
        meta[sympt] = pd.Categorical(meta[sympt], ordered=True)
    meta["smoker"] = pd.Categorical(
        meta["smoker"], categories=["No", "Former"], ordered=True
    )

    # # Temporal
    temporal_vars = ["days_intubated", "days_of_disease", "days_in_hospital"]
    meta["days_intubated"] = meta["DaysIntubated"].astype(pd.Int64Dtype())
    meta["days_of_disease"] = meta["DaysOfDisease"]
    meta["days_in_hospital"] = meta["DaysInHspital"]
    meta["intubated"] = meta["days_intubated"] > 0

    # Assemble vars
    ids = [
        "sample_name",
        "wcmc_code",
        "autopsy_code",
        "acquisition_name",
        "acquisition_date",
        "acquisition_id",
    ]
    lab_vars = [
        "PLTpermL",
        "Ddimer_mgperL",
        "Ddimer_mgperL_max",
        "WBC",
        "Lypct",
        "PMNpct",
        "ESR",
        "CRP",
        "Procalcitonin",
        "IL6",
    ]
    technical_vars = [
        "instrument",
        "panel_annotation_file",
        "panel_version",
        "panel_file",
        "observations",
        "mcd_file",
    ]
    # Save
    cols = (
        ids
        + technical_vars
        + demographic_vars
        + clinical_vars
        + temporal_vars
        + symptoms
        + ("pathology:" + pathology_vars).tolist()
        + lesions.columns.tolist()
        + ("pathology:" + pathology_vars + "_focal").tolist()
        + ("pathology:" + pathology_vars + "_bizarre").tolist()
        + lab_vars
    )
    # meta = meta.set_index("sample_name").reindex(origi["sample_name"]).reset_index()
    meta[cols].to_csv(metadata_dir / "clinical_annotation.csv", index=False)
    meta[cols].to_parquet(metadata_dir / "clinical_annotation.pq", index=False)

    # Write metadata variables to json
    variables = {
        "ids": ids,
        "technical": technical_vars,
        "demographics": demographic_vars,
        "clinical": clinical_vars,
        "temporal": temporal_vars,
        "symptoms": symptoms,
        "pathology": ("pathology:" + pathology_vars).tolist()
        + ("pathology:" + pathology_vars + "_focal").tolist()
        + ("pathology:" + pathology_vars + "_bizarre").tolist(),
        "lab": lab_vars,
    }
    json.dump(
        variables,
        open(metadata_dir / "variables.class_to_variable.json", "w"),
        indent=4,
    )
    json.dump(
        {x: k for k, v in variables.items() for x in v},
        open(metadata_dir / "variables.variable_to_class.json", "w"),
        indent=4,
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
