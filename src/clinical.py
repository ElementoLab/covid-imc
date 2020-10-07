#!/usr/bin/env python

"""
"""

from seaborn_extensions import swarmboxenplot, activate_annotated_clustermap

from src.config import *

activate_annotated_clustermap()

output_dir = results_dir / "clinical"
output_dir.mkdir()


meta = pd.read_parquet(metadata_dir / "clinical_annotation.pq")
variables = json.load(open(metadata_dir / "variables.class_to_variable.json"))


# FigS1 - cohort description
m = meta.set_index("sample_name").sort_values(["phenotypes", "age"])
grid = sns.clustermap(
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

# Stratify lung weight by disease and sex
meta["lung_weight_grams"] = meta["lung_weight_grams"].astype(float)
fig, stats = swarmboxenplot(
    data=meta,
    x="phenotypes",
    y="lung_weight_grams",
    hue="sex",
    test_kws=dict(parametric=False),
)


#

#

# See if lung weight can be a function of disease progression

cmeta = meta.loc[meta["disease"] == "COVID19"]


# # Lung weight is different between sexes
sns.swarmplot(x=meta["sex"], y=meta["lung_weight_grams"].astype(float))

# # Weight of lungs does not correlate with days of disease
# # but since we don't have the total weight of the patients it is not possible to conclude anything
plt.scatter(
    cmeta["days_of_disease"],
    cmeta["lung_weight_grams"].astype(float),
    c=cmeta["sex"] == "Female",
)
