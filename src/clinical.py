#!/usr/bin/env python

"""
Visualize the clinical data/metadata of the patient cohort.
"""

from seaborn_extensions import swarmboxenplot, clustermap

from src.config import *

output_dir = results_dir / "clinical"
output_dir.mkdir()

meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq")
variables = json.load(open(metadata_dir / "variables.class_to_variable.json"))

subsets = [
    "demographics",
    "clinical",
    "temporal",
    "symptoms",
    "pathology",
    "lab",
]
subvars = [
    x
    for y in [variables[c] for c in subsets]
    for x in y
    if (not x.endswith("_text"))
    and (meta[x].dtype != object)
    and (x not in ["disease", "phenotypes"])
]
meta_s = meta.query("disease != 'Healthy'")[
    subvars + ["sample_name"]
].set_index("sample_name")

# Get
# # For continuous variables
cont_vars = meta_s.columns[
    list(map(lambda x: x.name.lower() in ["float64", "int64"], meta_s.dtypes))
].tolist()
cont_meta = meta_s.loc[:, cont_vars].astype(float)

# # For categoricals
cat_vars = meta_s.columns[
    list(
        map(
            lambda x: x.name.lower() in ["category", "bool", "boolean"],
            meta_s.dtypes,
        )
    )
].tolist()
cat_meta = meta_s.loc[:, cat_vars]

# # # convert categoricals
cat_meta = pd.DataFrame(
    {
        x: cat_meta[x].astype(float)
        if cat_meta[x].dtype.name in ["bool", "boolean"]
        else cat_meta[x].cat.codes
        for x in cat_meta.columns
    }
)
cat_meta = cat_meta.loc[:, cat_meta.nunique() > 1]

clustermap(cont_meta.fillna(-1), mask=cont_meta.isnull(), config="z")
clustermap(cat_meta.fillna(-1), mask=cat_meta.isnull(), config="z")


# FigS1 - cohort description
m = meta.set_index("sample_name").sort_values(["phenotypes", "age"])
grid = clustermap(
    m.assign(a=0)[["a"]].T,
    col_colors=m[
        variables["clinical"]
        + variables["demographics"]
        + variables["temporal"]
        + variables["symptoms"]
    ],
    col_cluster=False,
    row_cluster=False,
    cmap="binary",
)
grid.fig.savefig(output_dir / "cohort_description.svg")

variables["demographics"] + variables["clinical"]


# develop clinical score
continuous_risk_variables = {
    # variable: direction
    "PLT/mL": 1,
    "D-dimer (mg/L)": 1,
    # "WBC": 1,
    "LY%": -1,
    # "PMN %": 1,
    "lung_weight_grams": 1,
    "fever_temperature_celsius": 1,
}
boolean_risk_variables = [
    "cough",
    "shortness_of_breath",
    "comorbidities",
]
cov_meta = meta.query("disease == 'COVID19'").set_index("sample_name")
from imc.utils import z_score

_scores = dict()
for var, direction in continuous_risk_variables.items():
    _scores[var] = z_score(cov_meta[var]) * direction
scores = pd.DataFrame(_scores).mean(1).rename("clinical_score")
increment = 0.1
for var in boolean_risk_variables:
    if var != "comorbidities":
        scores += (
            cov_meta[var]
            .dropna()
            .astype(str)
            .replace({"True": increment, "False": 0})
        )
    else:
        scores += (
            cov_meta[var].dropna().str.startswith("Y ").astype(int) * increment
        )


# swarmboxenplots for each variable
_stat = list()
for var in variables["demographics"] + variables["clinical"]:
    if meta[var].dtype.name == "Int64":
        meta[var] = meta[var].astype(float)
    for grouping in ["disease", "phenotypes"]:
        if var == grouping:
            continue
        y = var
        hue = None
        boxen = True
        if meta[var].dtype.name in ["object", "category", "boolean"]:
            df = (
                meta.groupby(grouping)[var]
                .value_counts()
                .rename("count")
                .reset_index()
            )
            hue = var
            y = "count"
            boxen = False
        else:
            df = meta

        fig, stats = swarmboxenplot(
            data=df,
            x=grouping,
            y=y,
            hue=hue,
            boxen=boxen,
            test_kws=dict(parametric=False),
            plot_kws=dict(palette=colors.get(grouping)),
        )
        fig.savefig(output_dir / f"{var}.by_{grouping}.svg", **figkws)
        plt.close(fig)
        _stat.append(stats.assign(variable=var, grouping=grouping))
stats = pd.concat(_stat)
stats.to_csv(output_dir / "stats.csv")

#

#

# Stratify lung weight by disease and gender (Fig1b)
meta["lung_weight_grams"] = meta["lung_weight_grams"].astype(float)
fig, stats = swarmboxenplot(
    data=meta,
    x="phenotypes",
    y="lung_weight_grams",
    hue="gender",
    test_kws=dict(parametric=False),
)


#

#

# See if lung weight can be a function of disease progression

cmeta = meta.loc[meta["disease"] == "COVID19"]


# # Lung weight is different between genderes
sns.swarmplot(x=meta["gender"], y=meta["lung_weight_grams"].astype(float))

# # Weight of lungs does not correlate with days of disease
# # but since we don't have the total weight of the patients it is not possible to conclude anything
plt.scatter(
    cmeta["days_of_disease"],
    cmeta["lung_weight_grams"].astype(float),
    c=cmeta["gender"] == "Female",
)
