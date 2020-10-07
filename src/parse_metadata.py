#!/usr/bin/env python

"""
Parse clinical metadata from excel file and
annotate samples in a standardized way.
"""

import json

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]

from src.config import metadata_dir

true_values = ["yes", "YES", "TRUE", "true", "True"]
false_values = ["no", "NO", "FALSE", "false", "False"]

origi = pd.read_csv(metadata_dir / "samples.csv")
# .query("toggle == True")
# .drop(["toggle", "complete"], axis=1)

annot = pd.read_excel(
    metadata_dir / "original" / "Hyperion samples.xlsx",
    na_values=["na", "NA"],
    true_values=true_values,
    false_values=false_values,
)

annot["Autopsy Code"] = (
    annot["Autopsy Code"]
    .str.split(" ")
    .apply(lambda x: x[0])
    .str.strip()
    .str.capitalize()
)


meta = origi.merge(  # type: ignore[operator]
    annot, how="left", left_on="autopsy_code", right_on="Autopsy Code"
)

clinical = [
    "disease",
    "organ",
    "phenotypes",
    "classification",
    "cause_of_death",
    "hospitalization",
    "days_hospitalized",
    "intubated",
    "lung_weight_grams",
    "comorbidities",
    "treated",
    "treatment",
]

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
meta["classification"] = meta["Sample Classification"]
meta["cause_of_death"] = meta["Cause of death"]
meta["comorbidities"] = meta["COMORBIDITY (Y/N; spec)"]
meta["treated"] = pd.Categorical(~meta["TREATMENT"].isnull(), ordered=True)
meta["treatment"] = meta["TREATMENT"].replace("None", np.nan)


meta["days_hospitalized"] = (
    meta["hospitalized"]
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

meta["lung_weight_grams"] = meta["lung WEIGHT g"].astype(pd.Int64Dtype())

# Lung histology/lesions
pathology = pd.Series(
    [
        "Airway",
        "Alveolar pmn",
        "Chronic Alveolar inflammation/ Macrophage",
        "Acute alveolar wall inflammation",
        "Chronic alveolar wall inflammation",
        "Microthrombi",
        "Large thrombi",
        "Hyaline membranes",
        "Type 2 hyperplasia only",
        "Type 2 with fibroblasts",
        "Organizing pneumonia",
        "Squamous metaplasia",
        # "Other lung lesions",
    ]
)

for col in pathology:
    meta[col + "_clean"] = (
        meta[col]
        .astype(str)
        .str.replace(",", "")
        .str.split(" ")
        .apply(lambda x: x[0] if isinstance(x, list) else x)
        .replace({k: True for k in true_values + ["FOCAL", "focal"]})
        .replace({k: False for k in false_values})
        .replace("nan", np.nan)
        .astype(pd.BooleanDtype())
    )
    meta[col + "_focal"] = (
        meta[col].astype(str).str.contains("focal", case=False)
    )

for col in pathology:
    meta[col] = meta[col + "_clean"]
    meta = meta.drop(col + "_clean", axis=1)

#  Convert classes to Categorical
demographics = ["age", "sex", "race", "smoker"]
meta["age"] = meta["AGE (years)"].astype(pd.Int64Dtype())
meta["sex"] = (
    meta["GENDER (M/F)"]
    .replace("M", "Male")
    .replace("F", "Female")
    .replace("m", "Male")
    .replace("f", "Female")
)
meta["race"] = meta["RACE"].replace("W", "white").str.capitalize()
meta["smoker"] = (
    meta["SMOKE (Y/N)"]
    .str.split(" ")
    .apply(lambda x: x[0] if isinstance(x, list) else x)
    .replace("never", "No")
    .replace("N", "No")
    .replace("former", "Former")
)

symptoms = [
    "fever",
    "fever_temperature_celsius",
    "cough",
    "shortness_of_breath",
]
meta["fever_temperature_celsius"] = (
    meta["Fever (Tmax)"]
    .astype(str)
    .str.extract(r"(\d+).?")[0]
    .fillna(-1)
    .replace(-1, np.nan)
    .astype(float)
)
meta["fever"] = np.nan
meta.loc[~meta["fever_temperature_celsius"].isnull(), "fever"] = meta[
    "Fever (Tmax)"
].str.contains("reported", case=False).fillna(False) | (
    meta["fever_temperature_celsius"] >= 38
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
    meta["Shortness of breath"]
    .replace("y", True)
    .replace("Y", True)
    .replace("YES", True)
    .replace("N", False)
    .replace("NO", False)
)

for sympt in filter(lambda x: not "celsius" in x, symptoms + ["smoker"]):
    # parquet serialization does not yet support categorical boolean
    meta[sympt] = pd.Categorical(meta[sympt].astype(str), ordered=True)

meta["smoker"] = pd.Categorical(
    meta["smoker"], categories=["No", "Former"], ordered=True
)

temporal = pd.Series(["Days Intubated", "Days of disease", "Days in hospital"])
for var in temporal:
    meta[var.lower().replace(" ", "_")] = meta[var].astype(pd.Int64Dtype())

meta["intubated"] = meta["days_intubated"] > 0

lab = ["PLT/mL", "D-dimer (mg/L)", "WBC", "LY%", "PMN %"]

# TODO: check D-dimer value for 20200722_COVID_28_EARLY
meta["D-dimer (mg/L)"] = meta["D-dimer (mg/L)"].replace("7192 (15032)", 15032)


ids = [
    "sample_name",
    "wcmc_code",
    "autopsy_code",
    "acquisition_name",
    "acquisition_date",
    "acquisition_id",
]
technical = [
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
    + technical
    + demographics
    + clinical
    + temporal.str.lower().str.replace(" ", "_").tolist()
    + symptoms
    + pathology.tolist()
    + (pathology + "_focal").tolist()
    + ["Other lung lesions"]
    + lab
)
meta[cols].to_csv(metadata_dir / "clinical_annotation.csv", index=False)
meta[cols].to_parquet(metadata_dir / "clinical_annotation.pq", index=False)


# Write metadata variables to json
variables = {
    "ids": ids,
    "technical": technical,
    "demographics": demographics,
    "clinical": clinical,
    "temporal": temporal.str.lower().str.replace(" ", "_").tolist(),
    "symptoms": symptoms,
    "pathology": pathology.tolist()
    + (pathology + "_focal").tolist()
    + ["Other lung lesions"],
    "lab": lab,
}
json.dump(
    variables, open(metadata_dir / "variables.class_to_variable.json", "w")
)
json.dump(
    {x: k for k, v in variables.items() for x in v},
    open(metadata_dir / "variables.variable_to_class.json", "w"),
)
